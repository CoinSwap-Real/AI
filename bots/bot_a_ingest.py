"""
bots/bot_a_ingest.py — 패턴 A/B 봇

[데이터 흐름]
  GET /chart/ohlc (BTC 풀) ┐
  GET /chart/ohlc (ETH 풀) ┘ 병렬 수집
          ↓
  FeatureBuilder.build() → (N, 8) scaler 적용 완료
          ↓
  각 엔진에 load_features()
          ↓
  infer() 병렬 실행 → (btc_ret, eth_ret) × 3모델
          ↓
  BTC 앙상블 / ETH 앙상블 각각 side·confidence 계산
          ↓
  AIInferenceResult × 2 (BTC, ETH)
          ↓
  IngestService.upload_all()
          ↓
  POST /ai/ingest/signals · predictions · sentiment

[horizon 매핑 — 현재 모델은 1초 뒤 예측]
  scalper  (seq=10)  → 1h  : 단기 반응성 기반 예측
  swing    (seq=60)  → 24h : 중기 패턴 기반 예측
  longterm (seq=120) → 7d  : 장기 맥락 기반 예측
  ※ 모델 재학습 없이 최선의 매핑. 정밀도는 horizon 이 길수록 낮아지므로
     confidence 를 horizon 길이에 비례해 감쇠합니다.
"""

from __future__ import annotations

import asyncio
import logging
import math
import traceback
from typing import Optional

from config import settings
from core.swapgo_client import SwapGoClient, StaleQuoteError
from ai.feature_builder import FeatureBuilder
from ai.ai_engine import AIEngine
from services.ingest_service import IngestService
from schemas.models import AIInferenceResult, HorizonPrediction

logger = logging.getLogger(__name__)

_STALE_MAX_RETRY = 3

# horizon 별 신뢰구간 반폭(±%) — 단기일수록 좁게
_CI_HALF = {"1h": 0.005, "24h": 0.015, "7d": 0.030}

# horizon 별 confidence 감쇠율 — 먼 미래일수록 불확실
_CONF_DECAY = {"1h": 1.0, "24h": 0.75, "7d": 0.50}

# BTC 심볼 → settings.symbols 의 첫 번째 (대문자 일치 필요)
_BTC_SYMBOL = "BTC"
_ETH_SYMBOL = "ETH"


class BotA_Ingest:
    def __init__(
        self,
        client: SwapGoClient,
        feature_builder: FeatureBuilder,
        ai_scalper: AIEngine,   # seq=10  → horizon "1h"
        ai_swing: AIEngine,     # seq=60  → horizon "24h"
        ai_longterm: AIEngine,  # seq=120 → horizon "7d"
        candle_cache=None,      # CandleCache | None (주입 시 REST 중복 제거)
    ):
        self._client  = client
        self._fb      = feature_builder
        self._cache   = candle_cache   # None 이면 직접 REST 호출 방식 유지
        self._ingest  = IngestService(client)
        self._engines: list[tuple[AIEngine, str]] = [
            (ai_scalper,  "1h"),
            (ai_swing,    "24h"),
            (ai_longterm, "7d"),
        ]
        self._tick_count  = 0
        self._trade_count = 0

    # ── 메인 루프 ────────────────────────────────────────────
    async def run(self) -> None:
        logger.info(
            f"[Bot A] 시작 | interval={settings.ingest_interval_sec}s "
            f"auto_trade={settings.enable_auto_trade}"
        )
        while True:
            try:
                await self._cycle()
            except Exception:
                logger.error(f"[Bot A] 사이클 오류\n{traceback.format_exc()}")
            await asyncio.sleep(settings.ingest_interval_sec)

    # ── 단일 사이클 ──────────────────────────────────────────
    async def _cycle(self) -> None:
        self._tick_count += 1
        logger.info(f"[Bot A] 사이클 #{self._tick_count}")

        # 1. 캔들 수집 — CandleCache 주입 시 캐시 경유, 없으면 직접 호출
        if self._cache is not None:
            # CandleCache: BTC 갱신 + ETH TTL 자동 갱신
            await self._cache.refresh_btc_from_rest()
            ticker_result = await self._client.get_ticker(settings.pool_id)
            ticker = ticker_result if not isinstance(ticker_result, Exception) else {}
            features = await self._cache.get_features()
            if features is None:
                logger.info("[Bot A] 피처 빌드 실패 (캔들 부족) — skip")
                return
            btc_last_price = _safe_float(ticker.get("last_price"))
            eth_last_price = self._cache.latest_eth_close()
            btc_candles    = self._cache.btc_candles()
        else:
            # Fallback: CandleCache 없이 직접 REST 호출
            btc_candles, eth_candles, ticker = await asyncio.gather(
                self._client.get_ohlc(
                    settings.pool_id,
                    interval=settings.candle_interval,
                    limit=settings.candle_limit,
                ),
                self._client.get_ohlc(
                    settings.eth_pool_id,
                    interval=settings.candle_interval,
                    limit=settings.candle_limit,
                ),
                self._client.get_ticker(settings.pool_id),
                return_exceptions=True,
            )
            if isinstance(btc_candles, Exception):
                logger.error(f"[Bot A] BTC 캔들 수집 실패: {btc_candles}")
                return
            if isinstance(eth_candles, Exception):
                logger.error(f"[Bot A] ETH 캔들 수집 실패: {eth_candles}")
                return
            if isinstance(ticker, Exception):
                logger.warning(f"[Bot A] ticker 수집 실패, 계속 진행: {ticker}")
                ticker = {}
            btc_last_price = _safe_float(ticker.get("last_price"))
            eth_last_price = _safe_float(
                eth_candles[-1].get("close") if eth_candles else None
            )
            features = self._fb.build(btc_candles, eth_candles)
            if features is None:
                logger.info("[Bot A] 피처 빌드 실패 (캔들 부족) — skip")
                return

        # 3. 모든 엔진에 피처 적재
        for engine, _ in self._engines:
            engine.load_features(features)

        # 4. 병렬 추론 → (btc_ret, eth_ret) | None
        raw_preds = await asyncio.gather(
            *[engine.infer() for engine, _ in self._engines]
        )

        # 5. 결과 유효성 확인
        if all(p is None for p in raw_preds):
            logger.info("[Bot A] 모든 모델 워밍업 중 — skip")
            return

        # 6. BTC / ETH 각각 AIInferenceResult 구성
        btc_result = self._build_result(
            symbol=_BTC_SYMBOL,
            raw_preds=raw_preds,
            ret_idx=0,                    # output[0] = BTC ret
            last_price=btc_last_price,
            btc_candles=btc_candles,
        )
        eth_result = self._build_result(
            symbol=_ETH_SYMBOL,
            raw_preds=raw_preds,
            ret_idx=1,                    # output[1] = ETH ret
            last_price=eth_last_price,
            btc_candles=btc_candles,      # 공통 지표(RSI·MACD·MA)는 BTC 캔들 기준
        )

        results = [r for r in [btc_result, eth_result] if r is not None]
        if not results:
            return

        # 7. ingest 업로드
        await self._ingest.upload_all(results)

        # 8. 자동거래 (패턴 B, BTC 신호 기준)
        if (
            settings.enable_auto_trade
            and btc_result is not None
            and btc_result.confidence >= settings.trade_confidence_threshold
        ):
            await self._maybe_trade(btc_result.side, btc_result.confidence)

    # ── AIInferenceResult 조립 ───────────────────────────────
    def _build_result(
        self,
        symbol: str,
        raw_preds: list[Optional[tuple[float, float]]],
        ret_idx: int,
        last_price: float,
        btc_candles: list[dict],
    ) -> Optional[AIInferenceResult]:
        """
        3 모델의 예측값에서 해당 심볼(ret_idx)의 로그수익률을 추출해
        AIInferenceResult 를 조립합니다.
        """
        horizon_predictions: list[HorizonPrediction] = []
        valid_rets: list[float] = []

        for (engine, horizon), pred_pair in zip(self._engines, raw_preds):
            # 워밍업 미완료 엔진: 워밍업 완료 후 값으로 채움 (없으면 0)
            if pred_pair is None:
                log_ret = 0.0
                conf    = 0.5
            else:
                log_ret = pred_pair[ret_idx]
                conf    = _ret_to_confidence(log_ret)
                valid_rets.append(log_ret)

            # 로그수익률 → 실제 가격
            if last_price and last_price > 0:
                pred_price = last_price * math.exp(log_ret)
            else:
                pred_price = 0.0

            ci = _CI_HALF[horizon]
            decay = _CONF_DECAY[horizon]
            horizon_predictions.append(
                HorizonPrediction(
                    horizon=horizon,
                    predicted_price_human=f"{pred_price:.4f}" if pred_price > 0 else "0.0000",
                    lower_bound_human=f"{pred_price * (1 - ci):.4f}" if pred_price > 0 else None,
                    upper_bound_human=f"{pred_price * (1 + ci):.4f}" if pred_price > 0 else None,
                    confidence=round(conf * decay, 4),
                )
            )

        if not valid_rets:
            return None

        ensemble_ret   = sum(valid_rets) / len(valid_rets)
        side, ens_conf = _ret_to_signal(ensemble_ret)

        return AIInferenceResult(
            symbol=symbol,
            side=side,
            confidence=ens_conf,
            predictions=horizon_predictions,
            sentiment_score=_ret_to_sentiment(ensemble_ret),
            rsi=_calc_rsi(btc_candles),
            macd=_calc_macd(btc_candles),
            ma7_human=_calc_ma(btc_candles, 7),
            ma25_human=_calc_ma(btc_candles, 25),
            model_tag=settings.model_tag,
        )

    # ── 자동거래 (패턴 B) ────────────────────────────────────
    async def _maybe_trade(self, side: str, confidence: float) -> None:
        if side == "hold":
            return

        swap_side   = "quote_to_base" if side == "buy" else "base_to_quote"
        quote_body  = {
            "pool_id": settings.pool_id,
            "side": swap_side,
            "amount_in_human": settings.trade_amount_human,
            "slippage_tolerance_bps": settings.trade_slippage_bps,
        }

        for attempt in range(1, _STALE_MAX_RETRY + 1):
            try:
                quote         = await self._client.quote_swap(quote_body)
                slippage_level = quote.get("slippage_level", "safe")

                if settings.trade_slippage_danger_skip and slippage_level == "danger":
                    logger.warning(f"[Bot A] slippage=danger → 거래 건너뜀")
                    return

                exec_body = {
                    "pool_id": settings.pool_id,
                    "side": swap_side,
                    "amount_in_human": settings.trade_amount_human,
                    "min_amount_out": quote["amount_out_min"],
                    "slippage_tolerance_bps": quote.get(
                        "slippage_threshold_used_bps", settings.trade_slippage_bps
                    ),
                    "expected_revision": quote["pool_after"]["revision"],
                }
                result = await self._client.execute_swap(exec_body)
                self._trade_count += 1
                logger.info(
                    f"[Bot A] 자동거래 #{self._trade_count} 완료 | "
                    f"{side.upper()} conf={confidence:.2f} "
                    f"slippage={slippage_level} "
                    f"out={result.get('amount_out_human', '?')}"
                )
                return

            except StaleQuoteError:
                logger.warning(f"[Bot A] StaleQuote(409) → 재견적 {attempt}/{_STALE_MAX_RETRY}")
                if attempt == _STALE_MAX_RETRY:
                    logger.error("[Bot A] StaleQuote 한도 초과 — 이번 거래 포기")
            except Exception as e:
                logger.error(f"[Bot A] 자동거래 오류: {e}")
                return

    # ── 상태 조회 ────────────────────────────────────────────
    def get_stats(self) -> dict:
        return {
            "tick_count": self._tick_count,
            "trade_count": self._trade_count,
            "auto_trade_enabled": settings.enable_auto_trade,
            "engines": [
                {"horizon": h, **engine.get_info()}
                for engine, h in self._engines
            ],
        }


# ════════════════════════════════════════════════════════════
# 순수 함수 헬퍼
# ════════════════════════════════════════════════════════════

def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _ret_to_signal(log_ret: float) -> tuple[str, float]:
    """
    로그수익률 → (side, confidence)
    threshold: 0.0001 ≈ 0.01% 이상 움직임을 신호로 간주
    """
    conf = _ret_to_confidence(log_ret)
    if abs(log_ret) < 0.0001:
        return "hold", conf
    return ("buy" if log_ret > 0 else "sell"), conf


def _ret_to_confidence(log_ret: float) -> float:
    """로그수익률 절댓값 → 0.5~1.0 confidence (tanh 정규화)"""
    return round(0.5 + 0.5 * math.tanh(abs(log_ret) / 0.001), 4)


def _ret_to_sentiment(log_ret: float) -> int:
    """로그수익률 → sentiment_score (-100~100 정수)"""
    score = math.tanh(log_ret / 0.001) * 100
    return max(-100, min(100, int(round(score))))


def _calc_ma(candles: list[dict], n: int) -> Optional[str]:
    closes = [float(c["close"]) for c in candles if _safe_float(c.get("close")) > 0]
    if len(closes) < n:
        return None
    return f"{sum(closes[-n:]) / n:.4f}"


def _calc_rsi(candles: list[dict], period: int = 14) -> Optional[float]:
    closes = [float(c["close"]) for c in candles if _safe_float(c.get("close")) > 0]
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, period + 1):
        d = closes[-i] - closes[-i - 1]
        (gains if d > 0 else losses).append(abs(d))
    avg_g = sum(gains) / period
    avg_l = sum(losses) / period
    if avg_l == 0:
        return 100.0
    return round(100.0 - 100.0 / (1.0 + avg_g / avg_l), 2)


def _calc_macd(candles: list[dict], fast=12, slow=26, signal=9) -> Optional[float]:
    closes = [float(c["close"]) for c in candles if _safe_float(c.get("close")) > 0]
    if len(closes) < slow + signal:
        return None

    def ema(vals: list[float], p: int) -> list[float]:
        k, res = 2 / (p + 1), [vals[0]]
        for v in vals[1:]:
            res.append(v * k + res[-1] * (1 - k))
        return res

    ef = ema(closes, fast)
    es = ema(closes, slow)
    ml = [f - s for f, s in zip(ef[slow - fast:], es)]
    sl = ema(ml, signal)
    return round(ml[-1] - sl[-1], 6)