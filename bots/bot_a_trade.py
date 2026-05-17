"""
bots/bot_a_trade.py — 시스템 B: AI 방향 예측 기반 즉시 거래 봇

[역할]
  model_trade.onnx (1분 뒤 로그수익률 예측) 를 사용해
  캔들 완성마다 방향을 판단하고 /swap/execute 를 직접 호출합니다.

[데이터 흐름]
  WS 캔들 완성 이벤트 (or REST 폴링)
      ↓
  FeatureBuilder.build() → (N, 8)
      ↓
  AIEngine(model_trade).push_feature_row() → infer()
      ↓  (btc_log_ret, eth_log_ret)
  |ret| > trade_min_log_return 이고
  confidence > trade_min_confidence 이고
  cooldown 경과
      ↓
  POST /swap/quote → POST /swap/execute
  StaleQuote(409) → 최대 trade_stale_retry 회 재견적

[ingest 봇과의 차이]
  - ingest 봇: 신호를 대시보드에 업로드 (60초 주기, 장기 예측)
  - 거래 봇  : 스왑 직접 실행 (캔들 단위, 단기 예측)
  - 두 봇은 같은 FeatureBuilder·scaler 를 공유하지만 AIEngine 은 별개
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import traceback
from collections import deque
from typing import Optional

from config import settings
from core.swapgo_client import SwapGoClient, StaleQuoteError
from core.ws_client import WsClient
from ai.feature_builder import FeatureBuilder
from ai.ai_engine import AIEngine

logger = logging.getLogger(__name__)

# ETH 캔들 REST 갱신 주기 (BotB_WS 와 동일 전략)
_ETH_REFRESH_EVERY = 10


class BotA_Trade:
    """
    model_trade.onnx 기반 AI 차익거래 봇.
    BTC 예측 방향으로 BTC 풀에서 스왑을 실행합니다.
    ETH 예측 방향은 미래 확장용으로 로그만 남깁니다.
    """

    def __init__(
        self,
        client: SwapGoClient,
        feature_builder: FeatureBuilder,
        ai_trade: AIEngine,       # model_trade.onnx, seq_len=30
    ):
        self._client  = client
        self._fb      = feature_builder
        self._engine  = ai_trade

        # 슬라이딩 캔들 윈도우 (BotB_WS 와 동일 방식)
        self._btc_window: deque[dict] = deque(maxlen=settings.candle_limit)
        self._eth_cache:  list[dict]  = []
        self._candle_count  = 0

        # 상태
        self._last_trade_ts: float = 0.0
        self._trade_count   = 0
        self._skip_count    = 0

        # WS (use_websocket=True 시 활성화)
        self._ws = WsClient(pool_id=settings.pool_id)
        self._ws.on_candle(self._on_candle)

    # ── 진입점 ──────────────────────────────────────────────
    async def run(self) -> None:
        logger.info(
            f"[BotA_Trade] 시작 | "
            f"threshold={settings.trade_min_log_return} "
            f"confidence>={settings.trade_min_confidence} "
            f"cooldown={settings.trade_cooldown_sec}s"
        )
        await self._prefetch()

        if settings.use_websocket:
            await self._ws.run()          # 재접속 루프 포함
        else:
            await self._polling_loop()    # REST 폴백

    # ── 초기 캔들 프리페치 ───────────────────────────────────
    async def _prefetch(self) -> None:
        try:
            btc, eth = await asyncio.gather(
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
            )
            for c in btc:
                self._btc_window.append(c)
            self._eth_cache = eth
            logger.info(
                f"[BotA_Trade] 프리페치 완료 "
                f"BTC={len(self._btc_window)} ETH={len(self._eth_cache)}"
            )
        except Exception as e:
            logger.warning(f"[BotA_Trade] 프리페치 실패: {e}")

    # ── REST 폴링 모드 (WS 비활성화 시) ─────────────────────
    async def _polling_loop(self) -> None:
        logger.info(
            f"[BotA_Trade] REST 폴링 모드 "
            f"(interval={settings.trade_poll_interval_sec}s)"
        )
        while True:
            try:
                btc_candles, eth_candles = await asyncio.gather(
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
                    return_exceptions=True,
                )
                if isinstance(btc_candles, Exception) or isinstance(eth_candles, Exception):
                    logger.warning("[BotA_Trade] 캔들 수집 실패, 재시도 대기")
                else:
                    # 전체 새 캔들로 피처 빌드 후 추론
                    features = self._fb.build(btc_candles, eth_candles)
                    if features is not None:
                        self._engine.load_features(features)
                        result = await self._engine.infer()
                        if result:
                            btc_ret, eth_ret = result
                            last_price = _safe_float(btc_candles[-1].get("close"))
                            await self._maybe_trade(btc_ret, eth_ret, last_price)
            except Exception:
                logger.error(f"[BotA_Trade] 폴링 오류\n{traceback.format_exc()}")
            await asyncio.sleep(settings.trade_poll_interval_sec)

    # ── WS 캔들 완성 콜백 ────────────────────────────────────
    async def _on_candle(self, candle: dict) -> None:
        try:
            close = _safe_float(candle.get("close"))
            if close <= 0:
                return

            self._btc_window.append(candle)
            self._candle_count += 1

            # ETH 주기적 갱신
            if self._candle_count % _ETH_REFRESH_EVERY == 1 or not self._eth_cache:
                await self._refresh_eth()
            if not self._eth_cache:
                return

            # 피처 빌드 → 마지막 행 push
            features = self._fb.build(list(self._btc_window), self._eth_cache)
            if features is None:
                return
            self._engine.push_feature_row(features[-1])

            # 추론
            result = await self._engine.infer()
            if result is None:
                return

            btc_ret, eth_ret = result
            await self._maybe_trade(btc_ret, eth_ret, close)

        except Exception:
            logger.error(f"[BotA_Trade] 캔들 처리 오류\n{traceback.format_exc()}")

    # ── ETH 캔들 갱신 ────────────────────────────────────────
    async def _refresh_eth(self) -> None:
        try:
            self._eth_cache = await self._client.get_ohlc(
                settings.eth_pool_id,
                interval=settings.candle_interval,
                limit=settings.candle_limit,
            )
        except Exception as e:
            logger.warning(f"[BotA_Trade] ETH 갱신 실패: {e}")

    # ── 거래 결정 ────────────────────────────────────────────
    async def _maybe_trade(
        self,
        btc_ret: float,
        eth_ret: float,
        last_price: float,
    ) -> None:
        """
        BTC 예측 방향이 임계값을 넘고 cooldown 이 경과했을 때 스왑 실행.
        ETH 방향은 추후 ETH 풀 거래 봇 확장 시 사용.
        """
        # 쿨다운 체크
        now = time.monotonic()
        if now - self._last_trade_ts < settings.trade_cooldown_sec:
            self._skip_count += 1
            return

        # 신호 강도 체크
        abs_ret    = abs(btc_ret)
        confidence = _ret_to_confidence(btc_ret)

        if abs_ret < settings.trade_min_log_return:
            return
        if confidence < settings.trade_min_confidence:
            return

        is_buy    = btc_ret > 0
        swap_side = "quote_to_base" if is_buy else "base_to_quote"
        direction = "BUY " if is_buy else "SELL"

        logger.info(
            f"[BotA_Trade] {direction} 신호 | "
            f"ret={btc_ret:+.5f}  conf={confidence:.3f}  "
            f"price={last_price:.4f}"
        )

        await self._execute_with_retry(swap_side, direction)
        self._last_trade_ts = time.monotonic()

    # ── 스왑 실행 (StaleQuote 재시도 포함) ───────────────────
    async def _execute_with_retry(self, swap_side: str, direction: str) -> None:
        quote_body = {
            "pool_id": settings.pool_id,
            "side":    swap_side,
            "amount_in_human":       settings.trade_execute_amount_human,
            "slippage_tolerance_bps": settings.trade_execute_slippage_bps,
        }

        for attempt in range(1, settings.trade_stale_retry + 1):
            try:
                # (1) 견적
                quote          = await self._client.quote_swap(quote_body)
                slippage_level = quote.get("slippage_level", "safe")

                if slippage_level == "danger":
                    logger.warning(f"[BotA_Trade] slippage=danger → 건너뜀")
                    return

                # (2) 실행
                exec_body = {
                    "pool_id":               settings.pool_id,
                    "side":                  swap_side,
                    "amount_in_human":       settings.trade_execute_amount_human,
                    "min_amount_out":        quote["amount_out_min"],
                    "slippage_tolerance_bps": quote.get(
                        "slippage_threshold_used_bps",
                        settings.trade_execute_slippage_bps,
                    ),
                    "expected_revision":     quote["pool_after"]["revision"],
                }
                result = await self._client.execute_swap(exec_body)
                self._trade_count += 1
                logger.info(
                    f"[BotA_Trade] ✓ {direction} #{self._trade_count} 완료 | "
                    f"out={result.get('amount_out_human', '?')} "
                    f"slippage={slippage_level}"
                )
                return

            except StaleQuoteError:
                logger.warning(
                    f"[BotA_Trade] StaleQuote(409) → 재견적 "
                    f"{attempt}/{settings.trade_stale_retry}"
                )
                if attempt == settings.trade_stale_retry:
                    logger.error("[BotA_Trade] 재견적 한도 초과 — 이번 거래 포기")
                else:
                    await asyncio.sleep(0.5 * attempt)  # 짧은 backoff

            except Exception as e:
                logger.error(f"[BotA_Trade] 거래 실패: {e}")
                return

    # ── 상태 조회 ────────────────────────────────────────────
    def get_stats(self) -> dict:
        return {
            "trade_count":   self._trade_count,
            "skip_count":    self._skip_count,
            "candle_count":  self._candle_count,
            "btc_window":    len(self._btc_window),
            "engine":        self._engine.get_info(),
        }


# ════════════════════════════════════════════════════════════
# 순수 함수
# ════════════════════════════════════════════════════════════

def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _ret_to_confidence(log_ret: float) -> float:
    """로그수익률 절댓값 → 0.5~1.0 confidence"""
    return round(0.5 + 0.5 * math.tanh(abs(log_ret) / 0.001), 4)