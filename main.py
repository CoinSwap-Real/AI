"""
main.py — SwapGo AI 봇 시스템 통합 진입점

[시스템 A: 대시보드 신호 봇]
  BotA_Ingest  — 60초 주기로 1h/24h/7d 예측 신호를 ingest API 에 업로드
  BotB_WS      — WS 모드 시 캔들 완성마다 즉시 ingest (선택)

[시스템 B: 가상 시장 조성 봇]  enable_trade_bots=True 시 활성화
  BotA_Trade   — model_trade.onnx 예측 방향으로 swap/execute 직접 실행
  BotB_Noise   — 랜덤 매수/매도로 풀 유동성 조성

[LP 봇 (시스템 C)]  enable_lp_bot=True 시 활성화
  BotC_LP      — 스프레드 감지 시 유동성 공급

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
from ai.feature_builder import FeatureBuilder
from ai.ai_engine import AIEngine
from bots.bot_a_ingest import BotA_Ingest
from bots.bot_a_trade import BotA_Trade
from bots.bot_b_noise import BotB_Noise
from bots.bot_b_ws import BotB_WS
from bots.bot_c_lp import BotC_LP

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
    logger.info("═" * 50)
    logger.info("  SwapGo AI 봇 시스템 v5.0")
    logger.info(f"  서버       : {settings.swapgo_base_url}")
    logger.info(f"  BTC 풀     : {settings.pool_id}")
    logger.info(f"  ETH 풀     : {settings.eth_pool_id}")
    logger.info(f"  심볼       : {settings.symbols}")
    logger.info("─" * 50)
    logger.info("  [시스템 A] 대시보드 신호 봇")
    logger.info(f"    ingest 주기 : {settings.ingest_interval_sec}s")
    logger.info(f"    WS 보조     : {settings.use_websocket}")
    logger.info("─" * 50)
    logger.info("  [시스템 B] 가상 시장 조성 봇")
    logger.info(f"    활성화      : {settings.enable_trade_bots}")
    logger.info(f"    거래 쿨다운 : {settings.trade_cooldown_sec}s")
    logger.info(f"    노이즈 간격 : {settings.noise_interval_min}~{settings.noise_interval_max}s")
    logger.info("═" * 50)

    # ── 공유 컴포넌트 ────────────────────────────────────────
    client = SwapGoClient()
    fb     = FeatureBuilder()
    logger.info(f"  FeatureBuilder: {fb.get_info()}")

    # ── 시스템 A AI 엔진 (ingest 전용: 장기 예측) ────────────
    ai_scalper   = AIEngine("Scalper",  settings.model_scalper_path,  seq_len=10)
    ai_swing     = AIEngine("Swing",    settings.model_swing_path,    seq_len=60)
    ai_longterm  = AIEngine("Longterm", settings.model_longterm_path, seq_len=120)

    # ── 시스템 A 봇 조립 ─────────────────────────────────────
    bot_a_ingest = BotA_Ingest(client, fb, ai_scalper, ai_swing, ai_longterm)
    tasks: list[asyncio.Task] = [
        asyncio.create_task(bot_a_ingest.run(), name="bot_a_ingest"),
    ]

    # [선택] WS 실시간 ingest 보조 봇
    bot_b_ws: BotB_WS | None = None
    if settings.use_websocket:
        bot_b_ws = BotB_WS(client, fb, ai_scalper, ai_swing, ai_longterm)
        tasks.append(asyncio.create_task(bot_b_ws.run(), name="bot_b_ws"))

    # ── 시스템 B 봇 조립 ─────────────────────────────────────
    bot_a_trade: BotA_Trade | None = None
    bot_b_noise: BotB_Noise | None = None

    if settings.enable_trade_bots:
        # 거래 봇 전용 AI 엔진 (단기 예측)
        ai_trade = AIEngine("Trade", settings.model_trade_path, seq_len=30)
        bot_a_trade = BotA_Trade(client, fb, ai_trade)
        bot_b_noise = BotB_Noise(client)
        tasks.append(asyncio.create_task(bot_a_trade.run(), name="bot_a_trade"))
        tasks.append(asyncio.create_task(bot_b_noise.run(), name="bot_b_noise"))

    # ── 시스템 C LP 봇 ───────────────────────────────────────
    bot_c_lp: BotC_LP | None = None
    if settings.enable_lp_bot:
        bot_c_lp = BotC_LP(client)
        tasks.append(asyncio.create_task(bot_c_lp.run(), name="bot_c_lp"))

    # ── app.state 보관 ───────────────────────────────────────
    app.state.client       = client
    app.state.fb           = fb
    app.state.bot_a_ingest = bot_a_ingest
    app.state.bot_b_ws     = bot_b_ws
    app.state.bot_a_trade  = bot_a_trade
    app.state.bot_b_noise  = bot_b_noise
    app.state.bot_c_lp     = bot_c_lp
    app.state.ingest_engines = [ai_scalper, ai_swing, ai_longterm]
    app.state.tasks        = tasks

    logger.info(f"  실행 태스크: {[t.get_name() for t in tasks]}")
    logger.info("  ✅ 봇 가동 완료")

    yield  # ← 요청 처리 구간

    # Shutdown
    logger.info("  종료 신호 수신 → 태스크 정리 중...")
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
    description=(
        "GRU 앙상블 기반 AI 봇 시스템\n\n"
        "- 시스템 A: 대시보드 신호 ingest (1h/24h/7d)\n"
        "- 시스템 B: 가상 시장 조성 봇 (직접 스왑 실행)"
    ),
    version="5.0.0",
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
    tasks: list[asyncio.Task] = app.state.tasks
    return {
        "status": "ok",
        "tasks": [
            {
                "name":      t.get_name(),
                "running":   not t.done(),
                "failed":    t.done() and not t.cancelled() and t.exception() is not None,
            }
            for t in tasks
        ],
    }


@app.get("/status", summary="전체 시스템 상태")
async def status():
    bot_a_ingest: BotA_Ingest      = app.state.bot_a_ingest
    bot_b_ws:     BotB_WS | None   = app.state.bot_b_ws
    bot_a_trade:  BotA_Trade | None = app.state.bot_a_trade
    bot_b_noise:  BotB_Noise | None = app.state.bot_b_noise
    bot_c_lp:     BotC_LP | None   = app.state.bot_c_lp
    fb:           FeatureBuilder   = app.state.fb

    return {
        "config": {
            "swapgo_base_url":    settings.swapgo_base_url,
            "pool_id":            settings.pool_id,
            "eth_pool_id":        settings.eth_pool_id,
            "candle_interval":    settings.candle_interval,
            "ingest_interval_sec": settings.ingest_interval_sec,
            "use_websocket":      settings.use_websocket,
            "enable_trade_bots":  settings.enable_trade_bots,
            "enable_lp_bot":      settings.enable_lp_bot,
        },
        "feature_builder": fb.get_info(),
        "system_a": {
            "bot_a_ingest": bot_a_ingest.get_stats(),
            "bot_b_ws":     bot_b_ws.get_stats() if bot_b_ws else None,
        },
        "system_b": {
            "bot_a_trade": bot_a_trade.get_stats() if bot_a_trade else None,
            "bot_b_noise": bot_b_noise.get_stats() if bot_b_noise else None,
        },
        "system_c": {
            "bot_c_lp": bot_c_lp.get_stats() if bot_c_lp else None,
        },
        "ingest_engines": [e.get_info() for e in app.state.ingest_engines],
    }


@app.get("/ai/info", summary="AI 엔진 상태")
async def ai_info():
    return [e.get_info() for e in app.state.ingest_engines]
