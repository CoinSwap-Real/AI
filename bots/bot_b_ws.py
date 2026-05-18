"""
bots/bot_b_ws.py — 시스템 A 보조: WebSocket 실시간 ingest 봇

[경량화 적용 사항]
  이전 구조                        →  개선 구조
  ──────────────────────────────────────────────────────────
  자체 WsClient 생성 + run()        →  CandleEventBus.subscribe(on_candle) 으로 대체
  자체 _btc_window deque            →  CandleCache.btc_candles() 읽기
  자체 ETH_REFRESH_EVERY REST 호출  →  CandleCache.get_features() 에 ETH TTL 위임
  자체 _prefetch() REST 2회         →  CandleCache.prefetch() 공유
  피처 캔들마다 직접 계산           →  CandleCache.get_features() 캐시 재사용

[역할]
  캔들 완성 이벤트마다 3개 ingest 엔진(scalper/swing/longterm) 을 추론하고
  signals / predictions / sentiment 를 SwapGo ingest API 에 즉시 업로드합니다.

[EventBus 구독 방법]  main.py 에서:
  event_bus.subscribe(bot_b_ws.on_candle)

[run() 동작]
  WS 모드에서 CandleEventBus 가 on_candle 을 직접 호출하므로
  run() 은 태스크 생존용 무한 대기만 수행합니다.
  WS 비활성 시 이 봇 자체가 main.py 에서 생성되지 않습니다.
"""

from __future__ import annotations

import asyncio
import logging
import math
import traceback
from typing import Optional

from config import settings
from core.swapgo_client import SwapGoClient
from ai.ai_engine import AIEngine
from services.candle_cache import CandleCache
from services.ingest_service import IngestService
from schemas.models import AIInferenceResult, HorizonPrediction
from bots.bot_a_ingest import (
    _ret_to_signal,
    _ret_to_confidence,
    _ret_to_sentiment,
    _calc_rsi,
    _calc_macd,
    _calc_ma,
    _CI_HALF,
    _CONF_DECAY,
    _BTC_SYMBOL,
    _ETH_SYMBOL,
)

logger = logging.getLogger(__name__)


class BotB_WS:
    """
    WebSocket 실시간 ingest 봇.
    캔들 완성 이벤트마다 ingest API 에 신호·예측·심리를 업로드합니다.
    캔들 버퍼·피처 계산은 모두 CandleCache 에 위임합니다.
    """

    def __init__(
        self,
        client: SwapGoClient,
        ai_scalper: AIEngine,
        ai_swing: AIEngine,
        ai_longterm: AIEngine,
        candle_cache: CandleCache,
    ):
        self._client  = client
        self._ingest  = IngestService(client)
        self._cache   = candle_cache
        self._engines: list[tuple[AIEngine, str]] = [
            (ai_scalper,  "1h"),
            (ai_swing,    "24h"),
            (ai_longterm, "7d"),
        ]
        self._candle_count = 0

    # ════════════════════════════════════════════════════════
    # 진입점 — 태스크 생존용 루프 (실제 처리는 on_candle 콜백)
    # ════════════════════════════════════════════════════════

    async def run(self) -> None:
        logger.info("[BotB_WS] 시작 (EventBus 구독 대기 중)")
        while True:
            await asyncio.sleep(3600)

    # ════════════════════════════════════════════════════════
    # CandleEventBus 구독 콜백 (public)
    # main.py: event_bus.subscribe(bot_b_ws.on_candle)
    # ════════════════════════════════════════════════════════

    async def on_candle(self, candle: dict) -> None:
        """
        CandleEventBus 가 캔들 완성 시마다 호출합니다.

        순서:
          1. CandleCache 에서 공유 피처 행렬 획득 (캐시 히트 시 O(1))
          2. 마지막 행을 각 엔진 버퍼에 push
          3. 병렬 추론
          4. AIInferenceResult 조립 후 ingest 업로드
        """
        try:
            close_btc = float(candle.get("close", 0))
            if close_btc <= 0:
                return
            self._candle_count += 1

            # 1. CandleCache 에서 피처 획득
            #    cache.push_btc(candle) 은 EventBus 가 먼저 호출하므로 여기선 읽기만
            features = await self._cache.get_features()
            if features is None:
                return

            # 2. 마지막 피처 행 → 각 엔진 push
            last_row = features[-1]
            for engine, _ in self._engines:
                engine.push_feature_row(last_row)

            # 3. 병렬 추론
            raw_preds = await asyncio.gather(
                *[engine.infer() for engine, _ in self._engines]
            )
            if all(p is None for p in raw_preds):
                return

            # 4. BTC / ETH 결과 조립 후 ingest 업로드
            btc_last = close_btc
            eth_last = self._cache.latest_eth_close()
            btc_list = self._cache.btc_candles()

            results = []
            for symbol, ret_idx, last_price in [
                (_BTC_SYMBOL, 0, btc_last),
                (_ETH_SYMBOL, 1, eth_last),
            ]:
                result = self._build_result(
                    symbol, raw_preds, ret_idx, last_price, btc_list
                )
                if result:
                    results.append(result)

            if results:
                await self._ingest.upload_all(results)

        except Exception:
            logger.error(f"[BotB_WS] on_candle 오류\n{traceback.format_exc()}")

    # ════════════════════════════════════════════════════════
    # AIInferenceResult 조립
    # ════════════════════════════════════════════════════════

    def _build_result(
        self,
        symbol: str,
        raw_preds: list,
        ret_idx: int,
        last_price: float,
        btc_list: list[dict],
    ) -> Optional[AIInferenceResult]:
        horizon_preds: list[HorizonPrediction] = []
        valid_rets: list[float] = []

        for (engine, horizon), pred_pair in zip(self._engines, raw_preds):
            if pred_pair is None:
                log_ret, conf = 0.0, 0.5
            else:
                log_ret = pred_pair[ret_idx]
                conf    = _ret_to_confidence(log_ret)
                valid_rets.append(log_ret)

            pred_price = (last_price * math.exp(log_ret)) if last_price > 0 else 0.0
            ci         = _CI_HALF[horizon]
            horizon_preds.append(HorizonPrediction(
                horizon=horizon,
                predicted_price_human=(
                    f"{pred_price:.4f}" if pred_price > 0 else "0.0000"
                ),
                lower_bound_human=(
                    f"{pred_price * (1 - ci):.4f}" if pred_price > 0 else None
                ),
                upper_bound_human=(
                    f"{pred_price * (1 + ci):.4f}" if pred_price > 0 else None
                ),
                confidence=round(conf * _CONF_DECAY[horizon], 4),
            ))

        if not valid_rets:
            return None

        ens_ret        = sum(valid_rets) / len(valid_rets)
        side, ens_conf = _ret_to_signal(ens_ret)

        return AIInferenceResult(
            symbol=symbol,
            side=side,
            confidence=ens_conf,
            predictions=horizon_preds,
            sentiment_score=_ret_to_sentiment(ens_ret),
            rsi=_calc_rsi(btc_list),
            macd=_calc_macd(btc_list),
            ma7_human=_calc_ma(btc_list, 7),
            ma25_human=_calc_ma(btc_list, 25),
            model_tag=settings.model_tag,
        )

    # ════════════════════════════════════════════════════════
    # 상태 조회
    # ════════════════════════════════════════════════════════

    def get_stats(self) -> dict:
        return {
            "candle_count": self._candle_count,
            "cache_info":   self._cache.get_info(),
            "engines": [
                {"horizon": h, **engine.get_info()}
                for engine, h in self._engines
            ],
        }