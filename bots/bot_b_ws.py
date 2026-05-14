"""
bots/bot_b_ws.py — WebSocket 실시간 캔들 수신 봇 (가이드 섹션 6)

use_websocket=True 설정 시 활성화됩니다.
폴링(Bot A)과 달리, 캔들이 완성되는 즉시 추론 → ingest 를 수행합니다.

채널: ohlc:{pool_id}:1m — 1분 캔들 완성 시 push
활용 패턴: "새 캔들이 도착할 때마다 모델 forward → /ai/ingest/signals 로 즉시 업로드"
"""

from __future__ import annotations

import asyncio
import logging
import traceback

from config import settings
from core.swapgo_client import SwapGoClient
from core.ws_client import WsClient
from ai.ai_engine import AIEngine
from services.ingest_service import IngestService
from schemas.models import AIInferenceResult

logger = logging.getLogger(__name__)


class BotB_WS:
    """
    WebSocket 실시간 봇.
    캔들 완성 이벤트마다 AI 추론 → ingest 업로드를 수행합니다.
    """

    def __init__(
        self,
        client: SwapGoClient,
        ai_scalper: AIEngine,
        ai_swing: AIEngine,
        ai_longterm: AIEngine,
    ):
        self._client = client
        self._ingest = IngestService(client)
        self._engines = [ai_scalper, ai_swing, ai_longterm]
        self._ws = WsClient(pool_id=settings.pool_id)
        self._ws.on_candle(self._on_candle)
        self._candle_count = 0

    async def run(self) -> None:
        logger.info("[Bot B WS] WebSocket 실시간 봇 시작")
        await self._ws.run()  # 재접속 루프 포함

    # ── 캔들 완성 콜백 ──────────────────────────────────────
    async def _on_candle(self, candle: dict) -> None:
        """ohlc 채널에서 완성된 캔들을 수신할 때마다 호출됩니다."""
        try:
            close = float(candle.get("close", 0))
            if close <= 0:
                return

            # 엔진에 가격 push
            for engine in self._engines:
                engine.push_price(close)

            # 병렬 추론
            preds = await asyncio.gather(*[e.infer() for e in self._engines])
            warm_preds = [p for p in preds if p is not None]
            if not warm_preds:
                return

            self._candle_count += 1
            ensemble_pred = sum(warm_preds) / len(warm_preds)
            side, confidence = _pred_to_signal(ensemble_pred)
            predicted_price = close * (1 + ensemble_pred / settings.pred_output_scale)
            sentiment_score = _pred_to_sentiment(ensemble_pred)

            results = [
                AIInferenceResult(
                    symbol=sym,
                    side=side,
                    confidence=confidence,
                    predicted_price_human=f"{predicted_price:.4f}",
                    sentiment_score=sentiment_score,
                    model_tag=settings.model_tag,
                )
                for sym in settings.symbols
            ]
            await self._ingest.upload_all(results)

        except Exception:
            logger.error(f"[Bot B WS] 캔들 처리 오류\n{traceback.format_exc()}")

    def get_stats(self) -> dict:
        return {
            "candle_count": self._candle_count,
            "engines": [e.get_info() for e in self._engines],
        }


# ── 공유 헬퍼 (bot_a 와 동일 로직) ──────────────────────────
import math


def _pred_to_signal(pred: float) -> tuple[str, float]:
    abs_pred = abs(pred)
    confidence = 0.5 + 0.5 * math.tanh(abs_pred / 0.5)
    if abs_pred < 0.05:
        return "hold", round(confidence, 4)
    return ("buy" if pred > 0 else "sell"), round(confidence, 4)


def _pred_to_sentiment(pred: float) -> int:
    score = math.tanh(pred / 0.5) * 100
    return max(-100, min(100, int(round(score))))
