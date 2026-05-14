"""
core/ws_client.py — SwapGo WebSocket 클라이언트 (가이드 섹션 6)

채널: ohlc:{pool_id}:1m  — 1분 캔들 완성 시 push
      trades:{pool_id}   — 체결 발생 시 push

사용 패턴:
    ws = WsClient(pool_id=1)
    ws.on_candle(my_callback)   # async def my_callback(candle: dict)
    ws.on_trade(my_callback)
    await ws.run()              # 재접속 루프 포함
"""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from typing import Callable, Optional

import websockets
from websockets.exceptions import ConnectionClosedError, WebSocketException

from config import settings

logger = logging.getLogger(__name__)


class WsClient:
    def __init__(self, pool_id: Optional[int] = None):
        self.pool_id = pool_id or settings.pool_id
        self._candle_callbacks: list[Callable] = []
        self._trade_callbacks: list[Callable] = []
        self._is_running = False

    # ── 콜백 등록 ────────────────────────────────────────────
    def on_candle(self, cb: Callable) -> None:
        """완성된 캔들 수신 시 호출될 async 함수 등록"""
        self._candle_callbacks.append(cb)

    def on_trade(self, cb: Callable) -> None:
        """체결 수신 시 호출될 async 함수 등록"""
        self._trade_callbacks.append(cb)

    # ── 메인 루프 ────────────────────────────────────────────
    async def run(self) -> None:
        """자동 재접속 포함 WebSocket 수신 루프"""
        ws_url = settings.swapgo_base_url.replace("http", "ws") + "/ws"
        self._is_running = True
        logger.info(f"[WsClient] 연결 시도: {ws_url}")

        while self._is_running:
            try:
                async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
                    logger.info("[WsClient] 연결 성공")
                    await self._subscribe(ws)
                    async for raw in ws:
                        await self._dispatch(raw)

            except (ConnectionClosedError, WebSocketException, OSError) as e:
                logger.warning(
                    f"[WsClient] 연결 끊김 ({e}), "
                    f"{settings.ws_reconnect_delay}초 후 재접속..."
                )
                await asyncio.sleep(settings.ws_reconnect_delay)
            except Exception:
                logger.error(f"[WsClient] 예상치 못한 오류\n{traceback.format_exc()}")
                await asyncio.sleep(settings.ws_reconnect_delay)

    async def stop(self) -> None:
        self._is_running = False

    # ── 내부 헬퍼 ────────────────────────────────────────────
    async def _subscribe(self, ws) -> None:
        channels = [
            f"ohlc:{self.pool_id}:1m",
            f"trades:{self.pool_id}",
        ]
        await ws.send(json.dumps({"op": "subscribe", "channels": channels}))
        logger.info(f"[WsClient] 채널 구독: {channels}")

    async def _dispatch(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        channel = msg.get("channel", "")

        if channel.startswith("ohlc:"):
            for cb in self._candle_callbacks:
                await self._safe_call(cb, msg.get("data", msg))
        elif channel.startswith("trades:"):
            for cb in self._trade_callbacks:
                await self._safe_call(cb, msg.get("data", msg))

    @staticmethod
    async def _safe_call(cb: Callable, data: dict) -> None:
        try:
            await cb(data)
        except Exception:
            name = getattr(cb, "__qualname__", repr(cb))
            logger.error(f"[WsClient] 콜백 '{name}' 오류\n{traceback.format_exc()}")
