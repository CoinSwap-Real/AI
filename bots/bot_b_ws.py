"""
bots/bot_b_ws.py — WebSocket 실시간 캔들 수신 봇

use_websocket=True 설정 시 활성화됩니다.

[동작 방식]
  1. BTC 풀의 ohlc:{pool_id}:1m 채널 구독
  2. 캔들 완성 이벤트마다:
     a. BTC 캔들 슬라이딩 윈도우 갱신
     b. ETH 캔들 REST 갱신 (N 캔들마다 1회, 캐시 사용)
     c. FeatureBuilder.build() → (N, 8) 피처 행렬
     d. 각 엔진에 push_feature_row() (마지막 행만)
     e. infer() → (btc_ret, eth_ret)
     f. ingest 업로드

[ETH 캔들 갱신 전략]
  - ETH 풀도 WS 를 동시에 구독하면 이상적이지만 구현 복잡도↑
  - 현실적 타협: BTC 캔들 ETH_REFRESH_EVERY 개마다 REST 1회 호출
  - 기본값 10 → 1m 캔들 기준 약 10분마다 ETH 갱신
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from collections import deque
from typing import Optional

from config import settings
from core.swapgo_client import SwapGoClient
from core.ws_client import WsClient
from ai.feature_builder import FeatureBuilder
from ai.ai_engine import AIEngine
from services.ingest_service import IngestService
from schemas.models import AIInferenceResult, HorizonPrediction, CandleData
from bots.bot_a_ingest import (
    _ret_to_signal, _ret_to_confidence, _ret_to_sentiment,
    _calc_rsi, _calc_macd, _calc_ma, _safe_float,
    _CI_HALF, _CONF_DECAY, _BTC_SYMBOL, _ETH_SYMBOL,
)
import math

logger = logging.getLogger(__name__)

ETH_REFRESH_EVERY = 10  # BTC 캔들 N개마다 ETH 캔들 REST 갱신


class BotB_WS:
    def __init__(
        self,
        client: SwapGoClient,
        feature_builder: FeatureBuilder,
        ai_scalper: AIEngine,
        ai_swing: AIEngine,
        ai_longterm: AIEngine,
    ):
        self._client  = client
        self._fb      = feature_builder
        self._ingest  = IngestService(client)
        self._engines: list[tuple[AIEngine, str]] = [
            (ai_scalper,  "1h"),
            (ai_swing,    "24h"),
            (ai_longterm, "7d"),
        ]

        # 슬라이딩 캔들 윈도우
        max_seq = max(e.seq_len for e, _ in self._engines)
        self._btc_window: deque[dict] = deque(maxlen=settings.candle_limit)
        self._eth_cache:  list[dict]  = []
        self._candle_count = 0

        # WS 클라이언트
        self._ws = WsClient(pool_id=settings.pool_id)
        self._ws.on_candle(self._on_btc_candle)

    async def run(self) -> None:
        logger.info("[Bot B WS] 시작")
        # 초기 캔들 REST 프리페치
        await self._prefetch()
        await self._ws.run()

    # ── 초기 프리페치 ────────────────────────────────────────
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
                f"[Bot B WS] 프리페치 완료 "
                f"BTC={len(self._btc_window)} ETH={len(self._eth_cache)}"
            )
        except Exception as e:
            logger.warning(f"[Bot B WS] 프리페치 실패: {e}")

    # ── BTC 캔들 완성 콜백 ───────────────────────────────────
    async def _on_btc_candle(self, candle: dict) -> None:
        try:
            if _safe_float(candle.get("close")) <= 0:
                return

            self._btc_window.append(candle)
            self._candle_count += 1

            # ETH 캔들 갱신 (주기적으로)
            if self._candle_count % ETH_REFRESH_EVERY == 1 or not self._eth_cache:
                await self._refresh_eth()

            if not self._eth_cache:
                return

            # 피처 빌드
            features = self._fb.build(
                list(self._btc_window),
                self._eth_cache,
            )
            if features is None:
                return

            # 마지막 피처 행을 각 엔진에 push
            last_row = features[-1]
            for engine, _ in self._engines:
                engine.push_feature_row(last_row)

            # 병렬 추론
            raw_preds = await asyncio.gather(
                *[engine.infer() for engine, _ in self._engines]
            )
            if all(p is None for p in raw_preds):
                return

            btc_last = _safe_float(candle.get("close"))
            eth_last = _safe_float(
                self._eth_cache[-1].get("close") if self._eth_cache else None
            )

            results = []
            for symbol, ret_idx, last_price in [
                (_BTC_SYMBOL, 0, btc_last),
                (_ETH_SYMBOL, 1, eth_last),
            ]:
                result = self._build_result(symbol, raw_preds, ret_idx, last_price)
                if result:
                    results.append(result)

            if results:
                await self._ingest.upload_all(results)

        except Exception:
            logger.error(f"[Bot B WS] 캔들 처리 오류\n{traceback.format_exc()}")

    # ── ETH 갱신 ─────────────────────────────────────────────
    async def _refresh_eth(self) -> None:
        try:
            self._eth_cache = await self._client.get_ohlc(
                settings.eth_pool_id,
                interval=settings.candle_interval,
                limit=settings.candle_limit,
            )
        except Exception as e:
            logger.warning(f"[Bot B WS] ETH 캔들 갱신 실패: {e}")

    # ── AIInferenceResult 조립 (bot_a 와 동일 로직) ──────────
    def _build_result(
        self,
        symbol: str,
        raw_preds: list,
        ret_idx: int,
        last_price: float,
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
            ci = _CI_HALF[horizon]
            horizon_preds.append(HorizonPrediction(
                horizon=horizon,
                predicted_price_human=f"{pred_price:.4f}" if pred_price > 0 else "0.0000",
                lower_bound_human=f"{pred_price*(1-ci):.4f}" if pred_price > 0 else None,
                upper_bound_human=f"{pred_price*(1+ci):.4f}" if pred_price > 0 else None,
                confidence=round(conf * _CONF_DECAY[horizon], 4),
            ))

        if not valid_rets:
            return None

        ens_ret        = sum(valid_rets) / len(valid_rets)
        side, ens_conf = _ret_to_signal(ens_ret)
        btc_list       = list(self._btc_window)

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

    def get_stats(self) -> dict:
        return {
            "candle_count": self._candle_count,
            "btc_window": len(self._btc_window),
            "eth_cache":  len(self._eth_cache),
            "engines": [{"horizon": h, **e.get_info()} for e, h in self._engines],
        }