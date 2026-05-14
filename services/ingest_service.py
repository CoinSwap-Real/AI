"""
services/ingest_service.py — AI 추론 결과를 SwapGo ingest API 형식으로 변환·업로드

백엔드 ai_ingest_service.py 분석 결과 적용:

  [predictions]
    to_base_units(predicted_price_human, asset.decimals) 로 정수 저장.
    → predicted_price_human 은 반드시 실제 사람단위 소수 문자열.
    → 각 모델이 자신의 horizon 예측가를 직접 계산 (임의 스케일 금지).
    → model.scalper → "1h", swing → "24h", longterm → "7d" 1:1 매핑.

  [sentiments]
    ma7_human / ma25_human 도 to_base_units 로 저장.
    → 반드시 실제 캔들 close 배열 기반 MA 계산값을 문자열로 전달.
    → 근사치(last_price * 0.998 등) 사용 금지.

  [signals]
    source 필드는 백엔드 API 레이어가 봇 키로부터 자동 주입.
    → 봇에서 별도로 보낼 필요 없음.
"""

from __future__ import annotations

import logging
from typing import Optional

from config import settings
from core.swapgo_client import SwapGoClient
from schemas.models import (
    AIInferenceResult,
    SignalItem,
    PredictionItem,
    SentimentItem,
)

logger = logging.getLogger(__name__)


class IngestService:
    def __init__(self, client: SwapGoClient):
        self._client = client

    # ════════════════════════════════════════════════════════
    # 공개 메서드
    # ════════════════════════════════════════════════════════

    async def upload_all(self, results: list[AIInferenceResult]) -> None:
        """signals / predictions / sentiment 를 순서대로 업로드합니다."""
        await self._upload_signals(results)
        await self._upload_predictions(results)
        await self._upload_sentiment(results)

    # ════════════════════════════════════════════════════════
    # 개별 업로드
    # ════════════════════════════════════════════════════════

    async def _upload_signals(self, results: list[AIInferenceResult]) -> None:
        items = [
            SignalItem(
                symbol=r.symbol,
                side=r.side,
                confidence=r.confidence,
                reason=r.reason or _auto_reason(r),
                expires_in_sec=int(settings.ingest_interval_sec * 3),
            ).model_dump(exclude_none=True)
            for r in results
        ]
        try:
            resp = await self._client.ingest_signals(items)
            inserted = resp.get("data", {}).get("inserted", "?")
            logger.info(f"[Ingest] signals 업로드 완료 (inserted={inserted})")
        except Exception as e:
            logger.error(f"[Ingest] signals 업로드 실패: {e}")

    async def _upload_predictions(self, results: list[AIInferenceResult]) -> None:
        """
        각 AIInferenceResult.predictions 에는 이미 1h / 24h / 7d 세 항목이 담겨 있음.
        (bot_a_ingest 에서 scalper→1h, swing→24h, longterm→7d 로 직접 생성)
        여기서는 타입 변환만 수행.
        """
        items = []
        for r in results:
            for pred in r.predictions:
                items.append(
                    PredictionItem(
                        symbol=r.symbol,
                        horizon=pred.horizon,
                        predicted_price_human=pred.predicted_price_human,
                        lower_bound_human=pred.lower_bound_human,
                        upper_bound_human=pred.upper_bound_human,
                        confidence=pred.confidence,
                        model_tag=r.model_tag or settings.model_tag,
                    ).model_dump(exclude_none=True)
                )
        if not items:
            return
        try:
            resp = await self._client.ingest_predictions(items)
            inserted = resp.get("data", {}).get("inserted", "?")
            logger.info(f"[Ingest] predictions 업로드 완료 (inserted={inserted})")
        except Exception as e:
            logger.error(f"[Ingest] predictions 업로드 실패: {e}")

    async def _upload_sentiment(self, results: list[AIInferenceResult]) -> None:
        """
        ma7_human / ma25_human 은 AIInferenceResult 에 이미 실제 캔들 MA 값이 담겨 있음.
        (bot_a_ingest 의 _calc_ma() 결과)
        """
        items = []
        for r in results:
            item = SentimentItem(
                symbol=r.symbol,
                sentiment_score=r.sentiment_score,
                rsi=r.rsi,
                macd=r.macd,
                ma7_human=r.ma7_human,
                ma25_human=r.ma25_human,
            ).model_dump(exclude_none=True)
            items.append(item)
        if not items:
            return
        try:
            resp = await self._client.ingest_sentiment(items)
            inserted = resp.get("data", {}).get("inserted", "?")
            logger.info(f"[Ingest] sentiment 업로드 완료 (inserted={inserted})")
        except Exception as e:
            logger.error(f"[Ingest] sentiment 업로드 실패: {e}")


# ════════════════════════════════════════════════════════════
# 헬퍼
# ════════════════════════════════════════════════════════════


def _auto_reason(r: AIInferenceResult) -> str:
    direction = {"buy": "상승", "sell": "하락", "hold": "중립"}[r.side]
    rsi_str = f", RSI {r.rsi:.1f}" if r.rsi is not None else ""
    return f"GRU 앙상블 {direction} 예측 (신뢰도 {r.confidence * 100:.0f}%{rsi_str})"
