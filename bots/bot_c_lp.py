"""
bots/bot_c_lp.py — 패턴 C: 시장조성형 LP 봇 (가이드 섹션 8)

enable_lp_bot=True 설정 시 활성화됩니다.
reserve_*, mid 가격, 24h 변동률을 보고 /liquidity/add 로 유동성 공급합니다.
scope: bot:lp 필요.
"""

from __future__ import annotations

import asyncio
import logging
import traceback

from config import settings
from core.swapgo_client import SwapGoClient

logger = logging.getLogger(__name__)

# LP 봇 파라미터 (필요 시 config 로 이동)
_LP_CHECK_INTERVAL = 300.0   # 5분마다 상태 점검
_SPREAD_THRESHOLD = 0.005    # 스프레드 0.5% 초과 시 유동성 공급 검토
_LP_AMOUNT_HUMAN = "50"      # 공급 기준 금액 (사람단위)


class BotC_LP:
    def __init__(self, client: SwapGoClient):
        self._client = client
        self._add_count = 0

    async def run(self) -> None:
        logger.info("[Bot C LP] 시장조성 LP 봇 시작")
        while True:
            try:
                await self._cycle()
            except Exception:
                logger.error(f"[Bot C LP] 오류\n{traceback.format_exc()}")
            await asyncio.sleep(_LP_CHECK_INTERVAL)

    async def _cycle(self) -> None:
        pool = await self._client.get_pool(settings.pool_id)
        ticker = await self._client.get_ticker(settings.pool_id)

        # 풀 가격 vs 24h 중간가격 비교로 불균형 탐지
        last_price = float(ticker.get("last_price", 0))
        high_24h = float(ticker.get("high_24h", last_price))
        low_24h = float(ticker.get("low_24h", last_price))
        mid_24h = (high_24h + low_24h) / 2 if high_24h and low_24h else last_price

        if last_price <= 0 or mid_24h <= 0:
            return

        spread = abs(last_price - mid_24h) / mid_24h

        if spread > _SPREAD_THRESHOLD:
            logger.info(
                f"[Bot C LP] 스프레드 {spread*100:.2f}% 감지 → 유동성 공급 검토"
            )
            await self._add_liquidity(pool)

    async def _add_liquidity(self, pool: dict) -> None:
        try:
            # 1. 견적
            quote_resp = await self._client._request(
                "POST",
                "/liquidity/quote-add",
                json={
                    "pool_id": settings.pool_id,
                    "amount_quote_human": _LP_AMOUNT_HUMAN,
                },
            )
            # 2. 공급
            add_resp = await self._client._request(
                "POST",
                "/liquidity/add",
                json={
                    "pool_id": settings.pool_id,
                    "amount_quote_human": _LP_AMOUNT_HUMAN,
                    "min_lp_tokens": quote_resp.get("min_lp_tokens", "0"),
                    "expected_revision": quote_resp.get("expected_revision", 0),
                },
            )
            self._add_count += 1
            logger.info(
                f"[Bot C LP] 유동성 공급 완료 #{self._add_count}: "
                f"lp_tokens={add_resp.get('lp_tokens_minted_human', '?')}"
            )
        except Exception as e:
            logger.error(f"[Bot C LP] 유동성 공급 실패: {e}")

    def get_stats(self) -> dict:
        return {"lp_add_count": self._add_count}
