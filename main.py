"""
main.py — SwapGo AI 봇 시스템 진입점

역할:
  - 봇·클라이언트 조립 및 생명주기 관리 (lifespan)
  - 모니터링 REST API (/status, /health)
  - 봇 프로세스는 uvicorn 내부 asyncio 루프에서 비동기 태스크로 실행

실행:
    uvicorn main:app --host 0.0.0.0 --port 9000 --reload
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from core.swapgo_client import SwapGoClient
from ai.ai_engine import AIEngine
from bots.bot_a_ingest import BotA_Ingest
from bots.bot_b_ws import BotB_WS
from bots.bot_c_lp import BotC_LP

# ── 로깅 설정 ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# 생명주기 관리
# ════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("═══ SwapGo AI 봇 시스템 시작 ═══")
    logger.info(f"  대상 서버  : {settings.swapgo_base_url}")
    logger.info(f"  풀 ID      : {settings.pool_id}")
    logger.info(f"  심볼       : {settings.symbols}")
    logger.info(f"  자동거래   : {settings.enable_auto_trade}")
    logger.info(f"  WebSocket  : {settings.use_websocket}")
    logger.info(f"  LP 봇      : {settings.enable_lp_bot}")

    # ── 1. 공유 HTTP 클라이언트 ─────────────────────────────
    client = SwapGoClient()

    # ── 2. AI 엔진 (단기/스윙/장기) ─────────────────────────
    ai_scalper  = AIEngine("Scalper",  settings.model_scalper_path,  seq_len=10)
    ai_swing    = AIEngine("Swing",    settings.model_swing_path,    seq_len=60)
    ai_longterm = AIEngine("Longterm", settings.model_longterm_path, seq_len=120)

    # ── 3. 봇 조립 ──────────────────────────────────────────
    bot_a = BotA_Ingest(client, ai_scalper, ai_swing, ai_longterm)
    tasks: list[asyncio.Task] = []

    # 항상 실행: REST 폴링 기반 ingest 봇
    tasks.append(asyncio.create_task(bot_a.run(), name="bot_a_ingest"))

    # 선택: WebSocket 실시간 추론 봇
    bot_b: BotB_WS | None = None
    if settings.use_websocket:
        bot_b = BotB_WS(client, ai_scalper, ai_swing, ai_longterm)
        tasks.append(asyncio.create_task(bot_b.run(), name="bot_b_ws"))

    # 선택: LP 시장조성 봇
    bot_c: BotC_LP | None = None
    if settings.enable_lp_bot:
        bot_c = BotC_LP(client)
        tasks.append(asyncio.create_task(bot_c.run(), name="bot_c_lp"))

    # ── 4. app.state 에 참조 보관 ───────────────────────────
    app.state.client   = client
    app.state.bot_a    = bot_a
    app.state.bot_b    = bot_b
    app.state.bot_c    = bot_c
    app.state.engines  = [ai_scalper, ai_swing, ai_longterm]
    app.state.tasks    = tasks

    logger.info(f"  실행 태스크: {[t.get_name() for t in tasks]}")
    logger.info("═══ 봇 가동 완료 ═══")

    yield  # ← 요청 처리 구간

    # ── Shutdown ────────────────────────────────────────────
    logger.info("═══ 시스템 종료 중 ═══")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await client.close()
    logger.info("═══ 종료 완료 ═══")


# ════════════════════════════════════════════════════════════
# FastAPI 앱
# ════════════════════════════════════════════════════════════

app = FastAPI(
    title="SwapGo AI Bot System",
    description="GRU 앙상블 기반 AI 신호 봇 — SwapGo 백엔드 연동",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════════════════════
# 모니터링 엔드포인트
# ════════════════════════════════════════════════════════════

@app.get("/health", summary="헬스체크")
async def health():
    tasks = app.state.tasks
    return {
        "status": "ok",
        "tasks": [
            {"name": t.get_name(), "done": t.done(), "cancelled": t.cancelled()}
            for t in tasks
        ],
    }


@app.get("/status", summary="전체 봇 상태")
async def status():
    bot_a: BotA_Ingest = app.state.bot_a
    bot_b: BotB_WS | None = app.state.bot_b
    bot_c: BotC_LP | None = app.state.bot_c

    return {
        "config": {
            "swapgo_base_url": settings.swapgo_base_url,
            "pool_id": settings.pool_id,
            "symbols": settings.symbols,
            "ingest_interval_sec": settings.ingest_interval_sec,
            "auto_trade": settings.enable_auto_trade,
            "websocket": settings.use_websocket,
            "lp_bot": settings.enable_lp_bot,
        },
        "bot_a": bot_a.get_stats(),
        "bot_b": bot_b.get_stats() if bot_b else None,
        "bot_c": bot_c.get_stats() if bot_c else None,
        "ai_engines": [e.get_info() for e in app.state.engines],
    }


@app.get("/ai/info", summary="AI 엔진 상태")
async def ai_info():
    return [e.get_info() for e in app.state.engines]
