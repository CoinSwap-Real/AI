"""
schemas/responses.py — API 응답 및 SSE 스트림 페이로드 스키마

models.py 가 ingest 계약(백엔드↔봇)을 정의한다면,
이 파일은 봇 서버의 관찰 인터페이스(프런트엔드↔봇 서버)를 정의합니다.

엔드포인트별 응답 모델:
  GET /health              → HealthResponse
  GET /status              → StatusResponse
  GET /stream/status (SSE) → SseStatusPayload  (1초마다 푸시)
  GET /stream/trades (SSE) → SseTradePayload   (거래 발생 시 푸시)
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel


# ════════════════════════════════════════════════════════════
# /health
# ════════════════════════════════════════════════════════════

class TaskInfo(BaseModel):
    name:      str
    running:   bool
    failed:    bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    tasks:  list[TaskInfo]


# ════════════════════════════════════════════════════════════
# /status
# ════════════════════════════════════════════════════════════

class EngineInfo(BaseModel):
    model:        str
    seq_len:      int
    buffer_fill:  int
    is_warm:      bool
    ema_btc:      Optional[float]
    ema_eth:      Optional[float]
    infer_count:  int
    mock_mode:    bool


class CandleCacheInfo(BaseModel):
    btc_count:       int
    eth_count:       int
    eth_age_sec:     float
    features_shape:  Optional[list[int]]
    features_age_sec: float


class EventBusInfo(BaseModel):
    subscriber_count: int
    candle_count:     int
    queue_sizes:      dict[str, int]


class BotIngestStats(BaseModel):
    tick_count:          int
    trade_count:         int
    auto_trade_enabled:  bool


class BotTradeStats(BaseModel):
    trade_count:  int
    skip_count:   int
    candle_count: int


class BotNoiseStats(BaseModel):
    trade_count: int
    skip_count:  int


class SystemConfig(BaseModel):
    swapgo_base_url:    str
    pool_id:            int
    eth_pool_id:        int
    candle_interval:    str
    ingest_interval_sec: float
    use_websocket:      bool
    enable_trade_bots:  bool
    enable_lp_bot:      bool


class StatusResponse(BaseModel):
    config:        SystemConfig
    candle_cache:  CandleCacheInfo
    event_bus:     Optional[EventBusInfo]
    ingest_engines: list[EngineInfo]
    system_a: dict
    system_b: dict
    system_c: dict


# ════════════════════════════════════════════════════════════
# SSE 페이로드 — GET /stream/status
# 프런트엔드가 폴링 없이 실시간으로 봇 상태를 수신합니다.
# ════════════════════════════════════════════════════════════

class SseStatusPayload(BaseModel):
    """1초마다 SSE 로 전송되는 봇 상태 요약."""
    ts:            float             # UNIX timestamp
    btc_close:     float             # 최근 BTC 캔들 close
    eth_close:     float             # 최근 ETH 캔들 close
    ema_btc:       Optional[float]   # 거래 모델 EMA 예측 (BTC)
    ema_eth:       Optional[float]   # 거래 모델 EMA 예측 (ETH)
    trade_count_a: int               # BotA_Trade 누적 거래 수
    trade_count_b: int               # BotB_Noise 누적 거래 수
    candle_count:  int               # EventBus 수신 캔들 수
    is_warm:       bool              # 거래 모델 워밍업 완료 여부


# ════════════════════════════════════════════════════════════
# SSE 페이로드 — GET /stream/trades
# 거래 발생 즉시 프런트엔드에 푸시합니다.
# ════════════════════════════════════════════════════════════

class SseTradePayload(BaseModel):
    """거래 발생 시 SSE 로 전송되는 개별 거래 이벤트."""
    ts:         float                              # 거래 시각
    bot:        Literal["BotA_Trade", "BotB_Noise"]
    side:       Literal["BUY", "SELL"]
    amount_in:  str                                # 사람단위 투입량
    amount_out: str                                # 사람단위 수령량
    pool_id:    int
    slippage:   Optional[str]                      # "safe" | "warning" | "danger"