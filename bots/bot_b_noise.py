"""
bots/bot_b_noise.py — 시스템 B: 랜덤 노이즈 유동성 봇

[역할]
  AI 예측 없이 무작위 방향·금액으로 /swap/execute 를 반복 호출합니다.
  BotA_Trade 와 같은 풀에서 충돌하며 가상 시장 유동성을 조성합니다.

[설계 원칙]
  - 완전 무작위: 방향(BUY/SELL) 50/50, 금액 균등분포
  - ingest 호출 없음: swap/execute 만 사용
  - 슬리피지 상한 내에서만 거래: danger 수준이면 건너뜀
  - StaleQuote(409) 는 즉시 포기 (노이즈 봇은 타이밍보다 빈도가 중요)
"""

from __future__ import annotations

import asyncio
import logging
import random
import traceback
from typing import Callable, Optional

from config import settings
from core.swapgo_client import SwapGoClient, StaleQuoteError

logger = logging.getLogger(__name__)


class BotB_Noise:
    """
    랜덤 노이즈 유동성 봇.
    설정된 간격으로 무작위 매수/매도를 반복해 풀에 거래량을 공급합니다.
    """

    def __init__(
        self,
        client: SwapGoClient,
        trade_event_publisher: Optional[Callable[[dict], None]] = None,
    ):
        self._client = client
        self._trade_count = 0
        self._skip_count = 0
        self._publish = trade_event_publisher

    # ── 메인 루프 ────────────────────────────────────────────
    async def run(self) -> None:
        logger.info(
            f"[BotB_Noise] 시작 | "
            f"interval=[{settings.noise_interval_min}, {settings.noise_interval_max}]s "
            f"amount=[{settings.noise_amount_min}, {settings.noise_amount_max}]"
        )
        while True:
            try:
                delay = random.uniform(
                    settings.noise_interval_min,
                    settings.noise_interval_max,
                )
                await asyncio.sleep(delay)
                await self._execute_random_swap()

            except Exception:
                logger.error(f"[BotB_Noise] 오류\n{traceback.format_exc()}")
                await asyncio.sleep(2.0)

    # ── 단일 랜덤 스왑 ───────────────────────────────────────
    async def _execute_random_swap(self) -> None:
        amount_min = float(settings.noise_amount_min)
        amount_max = float(settings.noise_amount_max)
        amount = round(random.uniform(amount_min, amount_max), 6)
        is_buy = random.choice([True, False])
        swap_side = "quote_to_base" if is_buy else "base_to_quote"
        direction = "BUY " if is_buy else "SELL"

        quote_body = {
            "pool_id": settings.pool_id,
            "side": swap_side,
            "amount_in_human": str(amount),
            "slippage_tolerance_bps": settings.noise_slippage_bps,
        }

        try:
            # 견적
            raw_quote = await self._client.quote_swap(quote_body)

            # 🌟 [추가된 부분] 백엔드 응답 봉투에서 알맹이("data")만 쏙 빼냅니다.
            quote = raw_quote.get("data", raw_quote)

            slippage_level = quote.get("slippage_level", "safe")

            if slippage_level == "danger":
                self._skip_count += 1
                logger.debug(
                    f"[BotB_Noise] slippage=danger → 건너뜀 (총 {self._skip_count}회)"
                )
                return

            # 실행
            exec_body = {
                "pool_id": settings.pool_id,
                "side": swap_side,
                "amount_in_human": str(amount),
                "min_amount_out": quote["amount_out_min"],
                "slippage_tolerance_bps": quote.get(
                    "slippage_threshold_used_bps",
                    settings.noise_slippage_bps,
                ),
                "expected_revision": quote["pool_after"]["revision"],
            }
            result = await self._client.execute_swap(exec_body)
            self._trade_count += 1

            logger.debug(
                f"[BotB_Noise] {direction} #{self._trade_count} | "
                f"in={amount:.6f}  out={result.get('amount_out_human', '?')}  "
                f"slippage={slippage_level}"
            )

        except StaleQuoteError:
            # 노이즈 봇은 재시도 없이 즉시 포기 — 다음 사이클에 재시도
            logger.debug("[BotB_Noise] StaleQuote(409) → 포기 (다음 사이클에 재시도)")
            self._skip_count += 1

        except Exception as e:
            logger.warning(f"[BotB_Noise] 스왑 실패: {e}")

    # ── 상태 조회 ────────────────────────────────────────────
    def get_stats(self) -> dict:
        return {
            "trade_count": self._trade_count,
            "skip_count": self._skip_count,
        }
