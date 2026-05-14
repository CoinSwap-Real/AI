"""
bots/bot_a_ingest.py — 패턴 A/B 봇

[데이터 흐름]
  GET /chart/ohlc + /chart/ticker
      ↓
  load_candles() → 각 엔진 버퍼 적재
      ↓
  infer() 병렬 실행
      scalper  → predicted_price(1h)
      swing    → predicted_price(24h)
      longterm → predicted_price(7d)
      ↓
  앙상블 side/confidence + MA 계산
      ↓
  AIInferenceResult(predictions=[1h, 24h, 7d]) 구성
      ↓
  IngestService.upload_all()
      ↓
  POST /ai/ingest/signals · predictions · sentiment

[ai_ingest_service.py 로부터 확인된 규칙]
  - predicted_price_human : to_base_units(value, asset.decimals) 로 저장
                            → 실제 사람단위 소수 문자열만 허용
  - ma7_human / ma25_human: 동일하게 to_base_units 적용
                            → 캔들 close 배열 기반 실제 MA 계산값만 허용
  - horizon 3종 동시 필요  : 프론트 /ai 페이지 그리드 완성 조건
"""

from __future__ import annotations

import asyncio
import logging
import math
import traceback
from typing import Optional

from config import settings
from core.swapgo_client import SwapGoClient, StaleQuoteError
from ai.ai_engine import AIEngine
from services.ingest_service import IngestService
from schemas.models import AIInferenceResult, HorizonPrediction

logger = logging.getLogger(__name__)

_STALE_MAX_RETRY = 3
# 모델별 horizon 매핑 — 순서는 AIEngine 생성 순서와 동일해야 함
_HORIZONS = ["1h", "24h", "7d"]
# 신뢰구간 폭 (±%) — 단기일수록 좁게
_CI_HALF_PCT = {"1h": 0.005, "24h": 0.015, "7d": 0.030}


class BotA_Ingest:
    def __init__(
        self,
        client: SwapGoClient,
        ai_scalper: AIEngine,  # seq_len=10  → horizon 1h
        ai_swing: AIEngine,  # seq_len=60  → horizon 24h
        ai_longterm: AIEngine,  # seq_len=120 → horizon 7d
    ):
        self._client = client
        self._ingest = IngestService(client)
        # (engine, horizon) 쌍 — 인덱스로 매핑
        self._engine_horizon: list[tuple[AIEngine, str]] = [
            (ai_scalper, "1h"),
            (ai_swing, "24h"),
            (ai_longterm, "7d"),
        ]
        self._tick_count = 0
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

        # 1. 캔들·ticker 병렬 수집
        candles, ticker = await asyncio.gather(
            self._client.get_ohlc(
                settings.pool_id,
                interval=settings.candle_interval,
                limit=settings.candle_limit,
            ),
            self._client.get_ticker(settings.pool_id),
            return_exceptions=True,
        )
        if isinstance(candles, Exception):
            logger.error(f"[Bot A] 캔들 수집 실패, 사이클 중단: {candles}")
            return
        if isinstance(ticker, Exception):
            logger.warning(f"[Bot A] ticker 수집 실패, 계속 진행: {ticker}")
            ticker = {}

        last_price = _safe_float(ticker.get("last_price"))

        # 2. 각 엔진에 캔들 일괄 적재
        for engine, _ in self._engine_horizon:
            engine.load_candles(candles)

        # 3. 병렬 추론
        preds = await asyncio.gather(
            *[engine.infer() for engine, _ in self._engine_horizon]
        )

        # 4. 워밍업 미완료 엔진 필터링
        ready = [
            (pred, horizon)
            for (pred, (_, horizon)) in zip(preds, self._engine_horizon)
            if pred is not None
        ]
        if not ready:
            logger.info("[Bot A] 모든 모델 워밍업 중 — skip")
            return

        # 5. 앙상블 신호
        pred_vals = [p for p, _ in ready]
        ensemble_pred = sum(pred_vals) / len(pred_vals)
        side, ensemble_conf = _pred_to_signal(ensemble_pred)

        # 6. 모델별 HorizonPrediction 구성
        #    각 모델의 예측 변화율(pred)을 last_price 에 적용해 실제 가격 문자열 생성
        horizon_predictions: list[HorizonPrediction] = []
        for (engine, horizon), pred in zip(self._engine_horizon, preds):
            # 아직 워밍업 안 된 모델은 last_price 그대로 사용 (최소한 값은 채움)
            effective_pred = pred if pred is not None else 0.0
            if last_price and last_price > 0:
                price = last_price * (1.0 + effective_pred / settings.pred_output_scale)
            else:
                price = 0.0
            conf = _pred_confidence(effective_pred)
            ci_half = _CI_HALF_PCT[horizon]
            horizon_predictions.append(
                HorizonPrediction(
                    horizon=horizon,
                    predicted_price_human=f"{price:.4f}",
                    lower_bound_human=(
                        f"{price * (1 - ci_half):.4f}" if price > 0 else None
                    ),
                    upper_bound_human=(
                        f"{price * (1 + ci_half):.4f}" if price > 0 else None
                    ),
                    confidence=conf,
                )
            )

        # 7. 보조 지표 계산 — 실제 캔들 close 기반
        rsi = _calc_rsi(candles)
        macd = _calc_macd(candles)
        ma7 = _calc_ma(candles, 7)
        ma25 = _calc_ma(candles, 25)

        # 8. AIInferenceResult 구성
        results = [
            AIInferenceResult(
                symbol=sym,
                side=side,
                confidence=ensemble_conf,
                predictions=horizon_predictions,
                sentiment_score=_pred_to_sentiment(ensemble_pred),
                rsi=rsi,
                macd=macd,
                ma7_human=ma7,
                ma25_human=ma25,
                model_tag=settings.model_tag,
            )
            for sym in settings.symbols
        ]

        # 9. 업로드
        await self._ingest.upload_all(results)

        # 10. 자동거래 (패턴 B)
        if (
            settings.enable_auto_trade
            and ensemble_conf >= settings.trade_confidence_threshold
        ):
            await self._maybe_trade(side, ensemble_conf)

    # ── 자동거래 (패턴 B) ────────────────────────────────────
    async def _maybe_trade(self, side: str, confidence: float) -> None:
        if side == "hold":
            return

        swap_side = "quote_to_base" if side == "buy" else "base_to_quote"
        quote_body = {
            "pool_id": settings.pool_id,
            "side": swap_side,
            "amount_in_human": settings.trade_amount_human,
            "slippage_tolerance_bps": settings.trade_slippage_bps,
        }

        for attempt in range(1, _STALE_MAX_RETRY + 1):
            try:
                # (1) 견적
                quote = await self._client.quote_swap(quote_body)
                slippage_level = quote.get("slippage_level", "safe")

                if settings.trade_slippage_danger_skip and slippage_level == "danger":
                    logger.warning(
                        f"[Bot A] slippage=danger → 거래 건너뜀 "
                        f"(side={side}, conf={confidence:.2f})"
                    )
                    return

                # (2) 실행 — expected_revision 으로 옵티미스틱 락
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
                logger.warning(
                    f"[Bot A] StaleQuote(409) → 재견적 {attempt}/{_STALE_MAX_RETRY}"
                )
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
                for engine, h in self._engine_horizon
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


def _pred_to_signal(pred: float) -> tuple[str, float]:
    """
    앙상블 예측값(연속) → (side, confidence)
    confidence: tanh 기반 0.5~1.0 정규화
    """
    abs_pred = abs(pred)
    confidence = round(0.5 + 0.5 * math.tanh(abs_pred / 0.5), 4)
    if abs_pred < 0.05:
        return "hold", confidence
    return ("buy" if pred > 0 else "sell"), confidence


def _pred_confidence(pred: float) -> float:
    return round(0.5 + 0.5 * math.tanh(abs(pred) / 0.5), 4)


def _pred_to_sentiment(pred: float) -> int:
    score = math.tanh(pred / 0.5) * 100
    return max(-100, min(100, int(round(score))))


def _calc_ma(candles: list[dict], n: int) -> Optional[str]:
    """
    캔들 close 배열로 단순이동평균(SMA) 계산.
    반환: 사람단위 소수 문자열 (백엔드 to_base_units 입력용)
    """
    closes = [float(c["close"]) for c in candles if _safe_float(c.get("close")) > 0]
    if len(closes) < n:
        return None
    ma = sum(closes[-n:]) / n
    return f"{ma:.4f}"


def _calc_rsi(candles: list[dict], period: int = 14) -> Optional[float]:
    """Wilder RSI"""
    closes = [float(c["close"]) for c in candles if _safe_float(c.get("close")) > 0]
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, period + 1):
        delta = closes[-i] - closes[-i - 1]
        if delta > 0:
            gains.append(delta)
        else:
            losses.append(abs(delta))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    return round(100.0 - (100.0 / (1.0 + avg_gain / avg_loss)), 2)


def _calc_macd(
    candles: list[dict],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Optional[float]:
    """MACD 히스토그램 (fast EMA - slow EMA - signal EMA)"""
    closes = [float(c["close"]) for c in candles if _safe_float(c.get("close")) > 0]
    if len(closes) < slow + signal:
        return None

    def ema(values: list[float], period: int) -> list[float]:
        k = 2 / (period + 1)
        result = [values[0]]
        for v in values[1:]:
            result.append(v * k + result[-1] * (1 - k))
        return result

    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = [f - s for f, s in zip(ema_fast[slow - fast :], ema_slow)]
    signal_line = ema(macd_line, signal)
    histogram = macd_line[-1] - signal_line[-1]
    return round(histogram, 6)
