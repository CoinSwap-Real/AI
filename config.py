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
    swapgo_base_url: str = Field(
        "http://localhost:8000",
        description="SwapGo FastAPI 서버 주소",
    )
    bot_key: str = Field(
        ...,
        description="봇 API 키 (콘솔에 한 번 출력된 raw 값). 환경변수 필수.",
    )

    # ── 대상 풀 / 심볼 ───────────────────────────────────────
    pool_id: int = Field(1, description="기본 감시 풀 ID")
    symbols: list[str] = Field(
        ["BTC", "ETH"],
        description="ingest 대상 심볼 목록 (/market/coins 의 symbol 그대로)",
    )

    # ── AI 모델 경로 ─────────────────────────────────────────
    model_scalper_path: str = Field("models/model_scalper.onnx")
    model_swing_path: str = Field("models/model_swing.onnx")
    model_longterm_path: str = Field("models/model_longterm.onnx")
    model_tag: str = Field("gru-v1", description="ingest predictions 에 기록되는 모델 식별자")

    # ── AI 스케일링 ──────────────────────────────────────────
    # 훈련 시 사용한 scaler 의 max 가격으로 교체 필수
    price_input_scale: float = Field(100_000.0, description="입력 정규화 분모")
    pred_output_scale: float = Field(1_000.0, description="출력 역정규화 분모")
    ema_alpha: float = Field(0.3, description="예측 EMA 평활 계수")

    # ── 캔들 수집 ────────────────────────────────────────────
    candle_interval: Literal["1m", "5m", "1h", "1d"] = Field("5m")
    candle_limit: int = Field(200, ge=10, le=1000)

    # ── ingest 주기 ──────────────────────────────────────────
    ingest_interval_sec: float = Field(
        60.0,
        ge=30.0,
        description="시그널/예측/심리 업로드 주기(초). 최소 30초 권장 (SQLite WAL)",
    )

    # ── Bot B: 자동거래 (Pattern B) ──────────────────────────
    enable_auto_trade: bool = Field(False, description="True 시 confidence 임계 초과 시 자동 스왑")
    trade_confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    trade_amount_human: str = Field("10", description="1회 스왑 수량 (사람단위 문자열)")
    trade_slippage_bps: int = Field(100, description="슬리피지 허용치 (basis points, 1%=100)")
    trade_slippage_danger_skip: bool = Field(True, description="danger 등급이면 거래 건너뜀")

    # ── Bot C: LP 봇 (Pattern C) — 선택 ─────────────────────
    enable_lp_bot: bool = Field(False, description="True 시 LP 봇 활성화")

    # ── WebSocket ────────────────────────────────────────────
    use_websocket: bool = Field(
        False,
        description="True=WS 실시간 캔들, False=REST 폴링 (기본값)",
    )
    ws_reconnect_delay: float = Field(5.0, description="WS 재접속 대기(초)")

    # ── 요청 타임아웃 ────────────────────────────────────────
    http_timeout_sec: float = Field(10.0)

    @field_validator("bot_key")
    @classmethod
    def key_not_empty(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("bot_key 가 비어 있습니다. SWAPGO_BOT_KEY 환경변수를 설정하세요.")
        return v.strip()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # 환경변수 접두어 없이 그대로 매핑 (BOT_KEY → bot_key)
        populate_by_name = True


settings = Settings()
