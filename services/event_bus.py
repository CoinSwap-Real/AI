"""
services/event_bus.py — 단일 WebSocket → 다중 구독자 팬아웃

[해결하는 문제]
  BotA_Trade, BotB_WS 가 각자 WsClient 를 생성 → SwapGo 에 WS 연결 2개 유지.
  CandleEventBus 는 단 1개의 WS 연결을 유지하고, 수신한 캔들을
  asyncio.Queue 를 통해 모든 구독자에게 분배합니다.

[사용 방법]
  bus = CandleEventBus(pool_id=1)
  bus.subscribe(my_async_callback)   # async def callback(candle: dict)
  asyncio.create_task(bus.run())     # WS 수신 루프 (재접속 포함)

[설계 원칙]
  - 구독자별 Queue 분리: 한 구독자의 처리 지연이 다른 구독자를 차단하지 않음
  - 큐 포화(maxsize) 시 최신 캔들 유지: 느린 구독자는 오래된 캔들을 건너뜀
  - 구독자 예외 격리: 한 구독자 오류가 WS 연결에 영향 없음
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Callable

from config import settings
from core.ws_client import WsClient

logger = logging.getLogger(__name__)

_QUEUE_SIZE = 8  # 구독자당 최대 미처리 캔들 수


class CandleEventBus:
    """
    단일 WS 연결을 공유하는 캔들 이벤트 버스.
    구독자는 각자 asyncio.Queue 를 통해 독립적으로 캔들을 수신합니다.
    """

    def __init__(self, pool_id: int | None = None):
        self._ws = WsClient(pool_id=pool_id or settings.pool_id)
        self._ws.on_candle(self._on_candle)

        # 구독자별 Queue (dict key = 구독 ID)
        self._queues: dict[int, asyncio.Queue] = {}
        self._next_id = 0
        self._candle_count = 0

    # ── 구독 등록 ────────────────────────────────────────────
    def subscribe(self, callback: Callable) -> None:
        """
        캔들 완성 이벤트를 수신할 async 콜백 함수를 등록합니다.
        콜백은 별도 Task 에서 실행되므로 느려도 버스를 차단하지 않습니다.
        """
        sub_id = self._next_id
        self._next_id += 1
        queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_SIZE)
        self._queues[sub_id] = queue

        # 각 구독자마다 독립 소비 태스크 생성
        asyncio.create_task(
            self._consumer(sub_id, queue, callback),
            name=f"event_bus_consumer_{sub_id}",
        )
        logger.debug(
            f"[EventBus] 구독 등록 #{sub_id}: "
            f"{getattr(callback, '__qualname__', repr(callback))}"
        )

    # ── WS 수신 루프 ─────────────────────────────────────────
    async def run(self) -> None:
        """WS 재접속 루프. main.py 에서 asyncio.create_task 로 실행."""
        logger.info("[EventBus] WS 수신 루프 시작")
        await self._ws.run()

    # ── 내부: 캔들 수신 → 큐 분배 ───────────────────────────
    async def _on_candle(self, candle: dict) -> None:
        self._candle_count += 1
        for sub_id, queue in self._queues.items():
            if queue.full():
                # 포화 시 가장 오래된 캔들을 버리고 최신 캔들을 넣음
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                logger.debug(f"[EventBus] 구독자 #{sub_id} 큐 포화 → 오래된 캔들 드롭")
            await queue.put(candle)

    # ── 내부: 구독자 소비 루프 ───────────────────────────────
    async def _consumer(
        self,
        sub_id: int,
        queue: asyncio.Queue,
        callback: Callable,
    ) -> None:
        while True:
            try:
                candle = await queue.get()
                await callback(candle)
                queue.task_done()
            except Exception:
                name = getattr(callback, "__qualname__", repr(callback))
                logger.error(
                    f"[EventBus] 구독자 #{sub_id} ({name}) 오류\n"
                    f"{traceback.format_exc()}"
                )

    # ── 상태 ─────────────────────────────────────────────────
    def get_info(self) -> dict:
        return {
            "subscriber_count": len(self._queues),
            "candle_count": self._candle_count,
            "queue_sizes": {str(sid): q.qsize() for sid, q in self._queues.items()},
        }
