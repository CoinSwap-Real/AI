"""
config.py — SwapGo AI 봇 시스템 전역 설정 (단일 진실 공급원)

모든 설정은 환경변수 또는 .env 파일로 주입합니다.
코드/Git에 시크릿(BOT_KEY 등)을 절대 하드코딩하지 마세요.
"""

from __future__ import annotations
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── SwapGo 백엔드 접속 ───────────────────────────────────
    swapgo_base_url: str = Field("http://localhost:8000")
    bot_key: str = Field(..., description="봇 API 키. 환경변수 필수.")

    # ── 대상 풀 / 심볼 ───────────────────────────────────────
    pool_id: int = Field(1, description="BTC 풀 ID")
    eth_pool_id: int = Field(2, description="ETH 풀 ID")
    symbols: list[str] = Field(["BTC", "ETH"])

    # ── AI 모델 경로 ─────────────────────────────────────────
    # 4개 모델 모두 5m 캔들 Binance 데이터로 학습합니다.
    # 봇 런타임도 CANDLE_INTERVAL=5m 을 사용합니다.
    model_scalper_path:  str = Field("models/model_scalper.onnx",  description="1h  신호용 seq_len=10")
    model_swing_path:    str = Field("models/model_swing.onnx",    description="24h 신호용 seq_len=60")
    model_longterm_path: str = Field("models/model_longterm.onnx", description="7d  신호용 seq_len=120")
    model_trade_path:    str = Field("models/model_trade.onnx",    description="거래봇용  seq_len=30")
    model_tag: str = Field("gru-v1")

    # ── AI 엔진 seq_len ──────────────────────────────────────
    seq_len_scalper:  int = Field(10,  ge=1)
    seq_len_swing:    int = Field(60,  ge=1)
    seq_len_longterm: int = Field(120, ge=1)
    seq_len_trade:    int = Field(30,  ge=1)

    # ── Scaler ──────────────────────────────────────────────
    scaler_path: str = Field("models/scaler.pkl", description="4개 모델 공통 8피처 scaler")

    # ── 출력 EMA 평활 ────────────────────────────────────────
    ema_alpha: float = Field(0.3)

    # ── 캔들 수집 ────────────────────────────────────────────
    candle_interval: Literal["1m", "5m", "1h", "1d"] = Field("5m", description="학습 데이터와 동일한 5m 캔들")
    candle_limit: int = Field(200, ge=30, le=1000)

    # ── ingest 봇 주기 (시스템 A) ────────────────────────────
    ingest_interval_sec: float = Field(60.0, ge=30.0)

    # ── ingest 봇 자동거래 (선택) ────────────────────────────
    enable_auto_trade: bool = Field(False)
    trade_confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    trade_amount_human: str = Field("10")
    trade_slippage_bps: int = Field(100)
    trade_slippage_danger_skip: bool = Field(True)

    # ── 시스템 B: 거래 봇 ────────────────────────────────────
    enable_trade_bots: bool = Field(
        False,
        description="True 시 BotA_Trade + BotB_Noise 활성화 (가상 시장 조성)",
    )

    # BotA_Trade 파라미터
    trade_min_log_return: float = Field(
        0.0002,
        description="거래 트리거 최소 |log_return| (0.02% ≈ 0.0002)",
    )
    trade_min_confidence: float = Field(
        0.55,
        ge=0.5,
        le=1.0,
        description="거래 트리거 최소 confidence",
    )
    trade_execute_amount_human: str = Field("5", description="BotA_Trade 1회 거래 수량")
    trade_execute_slippage_bps: int = Field(200, description="거래 봇 슬리피지 허용치 (더 관대하게)")
    trade_cooldown_sec: float = Field(
        5.0,
        ge=1.0,
        description="BotA_Trade 연속 거래 최소 대기(초). StaleQuote 방지.",
    )
    trade_poll_interval_sec: float = Field(
        30.0,
        ge=10.0,
        description="WS 비활성화 시 BotA_Trade REST 폴링 주기(초)",
    )
    trade_stale_retry: int = Field(3, description="StaleQuote 재시도 횟수")

    # BotB_Noise 파라미터
    noise_interval_min: float = Field(3.0, ge=1.0, description="노이즈 봇 최소 대기(초)")
    noise_interval_max: float = Field(8.0, ge=1.0, description="노이즈 봇 최대 대기(초)")
    noise_amount_min: str = Field("0.001", description="노이즈 봇 최소 거래량 (사람단위)")
    noise_amount_max: str = Field("0.05",  description="노이즈 봇 최대 거래량 (사람단위)")
    noise_slippage_bps: int = Field(500, description="노이즈 봇 슬리피지 (매우 관대)")

    # ── WebSocket ────────────────────────────────────────────
    use_websocket: bool = Field(False, description="True=WS 실시간 캔들")
    ws_reconnect_delay: float = Field(5.0)

    # ── LP 봇 ────────────────────────────────────────────────
    enable_lp_bot: bool = Field(False)

    # ── HTTP ─────────────────────────────────────────────────
    http_timeout_sec: float = Field(10.0)

    @field_validator("bot_key")
    @classmethod
    def key_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("bot_key 가 비어 있습니다. BOT_KEY 환경변수를 설정하세요.")
        return v.strip()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True


settings = Settings()
