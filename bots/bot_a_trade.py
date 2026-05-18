"""
bots/bot_a_trade.py — 시스템 B: AI 방향 예측 기반 즉시 거래 봇

[경량화 적용 사항]
  이전 구조                      →  개선 구조
  ─────────────────────────────────────────────────────────
  자체 WsClient 생성              →  CandleEventBus.subscribe(on_candle) 으로 대체
  자체 _btc_window deque          →  CandleCache.btc_candles() 읽기
  자체 _eth_cache + _refresh_eth  →  CandleCache.get_features() 에 ETH 갱신 위임
  자체 _prefetch() REST 2회 호출  →  CandleCache.prefetch() 공유 (main 에서 1회)
  피처 계산 캔들마다 직접 호출    →  CandleCache.get_features() 캐시 재사용

[데이터 흐름]
  CandleEventBus → on_candle(candle)
      ↓
  cache.get_features() — 이미 계산된 (N,8) 행렬 반환 (캐시 히트 시 O(1))
      ↓
  engine.push_feature_row(features[-1]) → engine.infer()
      ↓  (btc_log_ret, eth_log_ret)
  |ret| > threshold AND confidence > min AND cooldown 경과
      ↓
  POST /swap/quote → POST /swap/execute
  StaleQuote(409) → 최대 trade_stale_retry 회 재견적
      ↓
  trade_event_publisher(payload) — SSE /stream/trades 로 즉시 푸시

[REST 폴링 모드]  use_websocket=False 시
  trade_poll_interval_sec 주기로 CandleCache.refresh_btc_from_rest() 호출
  → 이후 cache.get_features() 로 피처 획득 (동일 코드 경로 공유)
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import traceback
from typing import Callable, Optional

from config import settings
from core.swapgo_client import SwapGoClient, StaleQuoteError
from ai.ai_engine import AIEngine
from services.candle_cache import CandleCache

logger = logging.getLogger(__name__)


class BotA_Trade:
    """
    model_trade.onnx 기반 AI 차익거래 봇.
    캔들 완성마다 BTC 예측 방향을 판단하고 /swap/execute 를 직접 호출합니다.
    """

    def __init__(
        self,
        client: SwapGoClient,
        ai_trade: AIEngine,
        candle_cache: CandleCache,
        trade_event_publisher: Optional[Callable[[dict], None]] = None,
    ):
        self._client    = client
        self._engine    = ai_trade
        self._cache     = candle_cache
        self._publish   = trade_event_publisher  # SSE 이벤트 발행 콜백

        self._last_trade_ts: float = 0.0
        self._trade_count   = 0
        self._skip_count    = 0
        self._candle_count  = 0

    # ════════════════════════════════════════════════════════
    # 진입점
    # ════════════════════════════════════════════════════════

    async def run(self) -> None:
        logger.info(
            f"[BotA_Trade] 시작 | "
            f"threshold={settings.trade_min_log_return} "
            f"conf>={settings.trade_min_confidence} "
            f"cooldown={settings.trade_cooldown_sec}s "
            f"mode={'WS' if settings.use_websocket else 'REST 폴링'}"
        )
        if not settings.use_websocket:
            # WS 모드가 아닐 때만 독립 폴링 루프 실행.
            # WS 모드에서는 main.py 가 EventBus.subscribe(on_candle) 으로 연결하므로
            # run() 은 폴링 루프를 실행하지 않고 그냥 종료해도 무방하지만
            # 태스크가 살아있어야 /health 에 표시되므로 대기 루프를 유지합니다.
            await self._polling_loop()
        else:
            # WS 모드: on_candle 이 EventBus 에서 직접 호출됨.
            # 태스크가 종료되지 않도록 무한 대기.
            while True:
                await asyncio.sleep(3600)

    # ════════════════════════════════════════════════════════
    # REST 폴링 모드 루프
    # ════════════════════════════════════════════════════════

    async def _polling_loop(self) -> None:
        logger.info(
            f"[BotA_Trade] REST 폴링 (interval={settings.trade_poll_interval_sec}s)"
        )
        while True:
            try:
                # CandleCache 를 통해 BTC 갱신 (ETH 는 TTL 기반 자동 갱신)
                await self._cache.refresh_btc_from_rest()
                await self._run_inference_and_trade()
            except Exception:
                logger.error(f"[BotA_Trade] 폴링 오류\n{traceback.format_exc()}")
            await asyncio.sleep(settings.trade_poll_interval_sec)

    # ════════════════════════════════════════════════════════
    # WS 모드 캔들 콜백 — CandleEventBus 가 호출
    # ════════════════════════════════════════════════════════

    async def on_candle(self, candle: dict) -> None:
        """
        CandleEventBus 구독 콜백.
        main.py: event_bus.subscribe(bot_a_trade.on_candle)

        CandleCache.push_btc() 는 EventBus 가 별도로 먼저 호출하므로
        여기서는 캐시를 읽기만 합니다.
        """
        try:
            if _safe_float(candle.get("close")) <= 0:
                return
            self._candle_count += 1
            await self._run_inference_and_trade()
        except Exception:
            logger.error(f"[BotA_Trade] on_candle 오류\n{traceback.format_exc()}")

    # ════════════════════════════════════════════════════════
    # 공통 추론 → 거래 결정 경로
    # ════════════════════════════════════════════════════════

    async def _run_inference_and_trade(self) -> None:
        # 1. 피처 행렬 — 캐시 재사용 (계산 비용 없음)
        features = await self._cache.get_features()
        if features is None:
            return

        # 2. 마지막 행을 엔진 버퍼에 push
        self._engine.push_feature_row(features[-1])

        # 3. 추론
        result = await self._engine.infer()
        if result is None:
            return

        btc_ret, eth_ret = result
        last_price = self._cache.latest_btc_close()
        await self._maybe_trade(btc_ret, eth_ret, last_price)

    # ════════════════════════════════════════════════════════
    # 거래 결정
    # ════════════════════════════════════════════════════════

    async def _maybe_trade(
        self,
        btc_ret: float,
        eth_ret: float,
        last_price: float,
    ) -> None:
        # 쿨다운
        if time.monotonic() - self._last_trade_ts < settings.trade_cooldown_sec:
            self._skip_count += 1
            return

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
            f"ret={btc_ret:+.5f}  conf={confidence:.3f}  price={last_price:.4f}"
        )

        amount_out, slippage_level = await self._execute_with_retry(swap_side)
        if amount_out is not None:
            self._trade_count += 1
            self._last_trade_ts = time.monotonic()

            # SSE 이벤트 발행
            if self._publish:
                self._publish({
                    "ts":         time.time(),
                    "bot":        "BotA_Trade",
                    "side":       direction.strip(),
                    "amount_in":  settings.trade_execute_amount_human,
                    "amount_out": amount_out,
                    "pool_id":    settings.pool_id,
                    "slippage":   slippage_level,
                })

    # ════════════════════════════════════════════════════════
    # 스왑 실행 (StaleQuote 재시도 포함)
    # ════════════════════════════════════════════════════════

    async def _execute_with_retry(
        self, swap_side: str
    ) -> tuple[Optional[str], Optional[str]]:
        """
        성공 시 (amount_out_human, slippage_level) 반환.
        실패 시 (None, None) 반환.
        """
        quote_body = {
            "pool_id":                settings.pool_id,
            "side":                   swap_side,
            "amount_in_human":        settings.trade_execute_amount_human,
            "slippage_tolerance_bps": settings.trade_execute_slippage_bps,
        }

        for attempt in range(1, settings.trade_stale_retry + 1):
            try:
                quote          = await self._client.quote_swap(quote_body)
                slippage_level = quote.get("slippage_level", "safe")

                if slippage_level == "danger":
                    logger.warning("[BotA_Trade] slippage=danger → 건너뜀")
                    return None, None

                exec_body = {
                    "pool_id":                settings.pool_id,
                    "side":                   swap_side,
                    "amount_in_human":        settings.trade_execute_amount_human,
                    "min_amount_out":         quote["amount_out_min"],
                    "slippage_tolerance_bps": quote.get(
                        "slippage_threshold_used_bps",
                        settings.trade_execute_slippage_bps,
                    ),
                    "expected_revision":      quote["pool_after"]["revision"],
                }
                result     = await self._client.execute_swap(exec_body)
                amount_out = result.get("amount_out_human", "?")

                logger.info(
                    f"[BotA_Trade] ✓ #{self._trade_count + 1} 완료 | "
                    f"out={amount_out}  slippage={slippage_level}"
                )
                return amount_out, slippage_level

            except StaleQuoteError:
                logger.warning(
                    f"[BotA_Trade] StaleQuote(409) → 재견적 "
                    f"{attempt}/{settings.trade_stale_retry}"
                )
                if attempt == settings.trade_stale_retry:
                    logger.error("[BotA_Trade] 재견적 한도 초과 — 포기")
                    return None, None
                await asyncio.sleep(0.5 * attempt)

            except Exception as e:
                logger.error(f"[BotA_Trade] 거래 실패: {e}")
                return None, None

        return None, None

    # ════════════════════════════════════════════════════════
    # 상태 조회
    # ════════════════════════════════════════════════════════

    def get_stats(self) -> dict:
        return {
            "trade_count":  self._trade_count,
            "skip_count":   self._skip_count,
            "candle_count": self._candle_count,
            "engine":       self._engine.get_info(),
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
    return round(0.5 + 0.5 * math.tanh(abs(log_ret) / 0.001), 4)