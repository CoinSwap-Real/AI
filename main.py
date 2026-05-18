"""
main.py — SwapGo AI 봇 시스템 통합 진입점 v6.0

[경량화 적용 사항]
  1. CandleEventBus  : WS 연결 2개 → 1개 (단일 공유 버스)
  2. CandleCache     : ETH REST 중복 호출 제거 + 피처 계산 1회/캔들
  3. SSE 엔드포인트  : 프런트엔드 폴링 → 서버 푸시로 전환
     /stream/status  : 1초 주기 상태 요약 (가격·예측·거래수)
     /stream/trades  : 거래 발생 즉시 이벤트 푸시

실행:
    uvicorn main:app --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import settings
from core.swapgo_client import SwapGoClient
from ai.feature_builder import FeatureBuilder
from ai.ai_engine import AIEngine
from services.candle_cache import CandleCache
from services.event_bus import CandleEventBus
from bots.bot_a_ingest import BotA_Ingest
from bots.bot_a_trade import BotA_Trade
from bots.bot_b_noise import BotB_Noise
from bots.bot_b_ws import BotB_WS
from bots.bot_c_lp import BotC_LP
from schemas.responses import HealthResponse, TaskInfo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# SSE 거래 이벤트 브로드캐스트용 큐 (trade 이벤트를 SSE 스트림으로 전달)
_trade_event_queues: list[asyncio.Queue] = []


def publish_trade_event(payload: dict) -> None:
    """BotA_Trade / BotB_Noise 에서 거래 완료 시 호출. SSE 구독자에게 전달."""
    for q in _trade_event_queues:
        if not q.full():
            q.put_nowait(payload)


# ════════════════════════════════════════════════════════════
# 생명주기
# ════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("═" * 52)
    logger.info("  SwapGo AI 봇 시스템 v6.0 (경량화 적용)")
    logger.info(f"  서버        : {settings.swapgo_base_url}")
    logger.info(f"  BTC 풀      : {settings.pool_id}  ETH 풀: {settings.eth_pool_id}")
    logger.info(f"  거래 봇     : {settings.enable_trade_bots}")
    logger.info(f"  WebSocket   : {settings.use_websocket}")
    logger.info("═" * 52)

    # ── 공유 컴포넌트 ────────────────────────────────────────
    client = SwapGoClient()
    fb     = FeatureBuilder()

    # CandleCache: BTC+ETH 캔들 공유 버퍼
    cache  = CandleCache(client, fb)
    await cache.prefetch()

    # ── 시스템 A AI 엔진 ─────────────────────────────────────
    ai_scalper  = AIEngine("Scalper",  settings.model_scalper_path,  seq_len=10)
    ai_swing    = AIEngine("Swing",    settings.model_swing_path,    seq_len=60)
    ai_longterm = AIEngine("Longterm", settings.model_longterm_path, seq_len=120)

    # ── 시스템 A 봇 (REST 폴링, CandleCache 사용) ────────────
    bot_a_ingest = BotA_Ingest(client, fb, ai_scalper, ai_swing, ai_longterm,
                               candle_cache=cache)
    tasks: list[asyncio.Task] = [
        asyncio.create_task(bot_a_ingest.run(), name="bot_a_ingest"),
    ]

    # ── CandleEventBus (WS 단일 연결) ────────────────────────
    event_bus: CandleEventBus | None = None
    bot_b_ws:  BotB_WS | None       = None

    if settings.use_websocket:
        event_bus = CandleEventBus(pool_id=settings.pool_id)

        # EventBus → CandleCache 피드
        event_bus.subscribe(cache.push_btc)

        # WS 기반 ingest 봇
        bot_b_ws = BotB_WS(client, fb, ai_scalper, ai_swing, ai_longterm,
                           candle_cache=cache, event_bus=event_bus)
        event_bus.subscribe(bot_b_ws.on_candle)

        tasks.append(asyncio.create_task(event_bus.run(), name="event_bus"))

    # ── 시스템 B (거래 봇) ───────────────────────────────────
    bot_a_trade: BotA_Trade | None  = None
    bot_b_noise: BotB_Noise | None  = None

    if settings.enable_trade_bots:
        ai_trade = AIEngine("Trade", settings.model_trade_path, seq_len=30)
        bot_a_trade = BotA_Trade(client, fb, ai_trade,
                                 candle_cache=cache,
                                 trade_event_publisher=publish_trade_event)
        bot_b_noise = BotB_Noise(client,
                                 trade_event_publisher=publish_trade_event)

        if event_bus:
            event_bus.subscribe(bot_a_trade.on_candle)

        tasks.append(asyncio.create_task(bot_a_trade.run(), name="bot_a_trade"))
        tasks.append(asyncio.create_task(bot_b_noise.run(), name="bot_b_noise"))

    # ── 시스템 C LP 봇 ───────────────────────────────────────
    bot_c_lp: BotC_LP | None = None
    if settings.enable_lp_bot:
        bot_c_lp = BotC_LP(client)
        tasks.append(asyncio.create_task(bot_c_lp.run(), name="bot_c_lp"))

    # ── app.state ────────────────────────────────────────────
    app.state.client          = client
    app.state.cache           = cache
    app.state.event_bus       = event_bus
    app.state.bot_a_ingest    = bot_a_ingest
    app.state.bot_b_ws        = bot_b_ws
    app.state.bot_a_trade     = bot_a_trade
    app.state.bot_b_noise     = bot_b_noise
    app.state.bot_c_lp        = bot_c_lp
    app.state.ingest_engines  = [ai_scalper, ai_swing, ai_longterm]
    app.state.trade_engine    = ai_trade if settings.enable_trade_bots else None
    app.state.tasks           = tasks

    logger.info(f"  실행 태스크: {[t.get_name() for t in tasks]}")
    logger.info("  ✅ 봇 가동 완료")

    yield

    logger.info("  종료 중...")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await client.close()
    logger.info("  종료 완료")


# ════════════════════════════════════════════════════════════
# FastAPI 앱
# ════════════════════════════════════════════════════════════

app = FastAPI(
    title="SwapGo AI Bot System",
    version="6.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════════════════════
# REST 모니터링 엔드포인트
# ════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health():
    tasks: list[asyncio.Task] = app.state.tasks
    all_ok = all(not t.done() for t in tasks)
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        tasks=[
            TaskInfo(
                name=t.get_name(),
                running=not t.done(),
                failed=t.done() and not t.cancelled() and t.exception() is not None,
            )
            for t in tasks
        ],
    )


@app.get("/status")
async def status():
    cache:     CandleCache          = app.state.cache
    event_bus: CandleEventBus | None = app.state.event_bus
    bot_a_i    = app.state.bot_a_ingest
    bot_b_ws   = app.state.bot_b_ws
    bot_a_t    = app.state.bot_a_trade
    bot_b_n    = app.state.bot_b_noise
    bot_c      = app.state.bot_c_lp

    return {
        "config": {
            "swapgo_base_url":    settings.swapgo_base_url,
            "pool_id":            settings.pool_id,
            "eth_pool_id":        settings.eth_pool_id,
            "candle_interval":    settings.candle_interval,
            "ingest_interval_sec": settings.ingest_interval_sec,
            "use_websocket":      settings.use_websocket,
            "enable_trade_bots":  settings.enable_trade_bots,
        },
        "candle_cache": cache.get_info(),
        "event_bus":    event_bus.get_info() if event_bus else None,
        "ingest_engines": [e.get_info() for e in app.state.ingest_engines],
        "trade_engine":   app.state.trade_engine.get_info()
                          if app.state.trade_engine else None,
        "system_a": {
            "bot_a_ingest": bot_a_i.get_stats(),
            "bot_b_ws":     bot_b_ws.get_stats() if bot_b_ws else None,
        },
        "system_b": {
            "bot_a_trade": bot_a_t.get_stats() if bot_a_t else None,
            "bot_b_noise": bot_b_n.get_stats() if bot_b_n else None,
        },
        "system_c": {
            "bot_c_lp": bot_c.get_stats() if bot_c else None,
        },
    }


# ════════════════════════════════════════════════════════════
# SSE 스트리밍 엔드포인트 — 프런트엔드 폴링 대체
# ════════════════════════════════════════════════════════════

@app.get(
    "/stream/status",
    summary="봇 상태 SSE 스트림 (1초 주기)",
    response_class=StreamingResponse,
)
async def stream_status():
    """
    Server-Sent Events 로 봇 상태 요약을 1초마다 프런트엔드에 푸시합니다.
    EventSource 연결이 끊기면 자동으로 generator 를 종료합니다.

    프런트엔드 사용 예시:
        const es = new EventSource('http://localhost:9000/stream/status');
        es.onmessage = (e) => { const d = JSON.parse(e.data); ... };
    """
    async def generator() -> AsyncGenerator[str, None]:
        cache:     CandleCache           = app.state.cache
        event_bus: CandleEventBus | None = app.state.event_bus
        bot_a_t    = app.state.bot_a_trade
        bot_b_n    = app.state.bot_b_noise
        engine     = app.state.trade_engine

        while True:
            try:
                payload = {
                    "ts":            time.time(),
                    "btc_close":     cache.latest_btc_close(),
                    "eth_close":     cache.latest_eth_close(),
                    "ema_btc":       engine.get_info()["ema_btc"] if engine else None,
                    "ema_eth":       engine.get_info()["ema_eth"] if engine else None,
                    "trade_count_a": bot_a_t.get_stats()["trade_count"] if bot_a_t else 0,
                    "trade_count_b": bot_b_n.get_stats()["trade_count"] if bot_b_n else 0,
                    "candle_count":  event_bus.get_info()["candle_count"] if event_bus else 0,
                    "is_warm":       engine.is_warm if engine else False,
                }
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception:
                yield f"data: {json.dumps({'error': 'internal'})}\n\n"
                await asyncio.sleep(1.0)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # nginx 버퍼링 비활성화
        },
    )


@app.get(
    "/stream/trades",
    summary="거래 이벤트 SSE 스트림 (즉시 푸시)",
    response_class=StreamingResponse,
)
async def stream_trades():
    """
    BotA_Trade / BotB_Noise 가 거래를 완료할 때마다 즉시 이벤트를 푸시합니다.
    폴링 없이 실시간으로 체결 현황을 확인할 수 있습니다.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=32)
    _trade_event_queues.append(queue)

    async def generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"  # 연결 유지용 주석 라인
        except asyncio.CancelledError:
            pass
        finally:
            _trade_event_queues.remove(queue)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )