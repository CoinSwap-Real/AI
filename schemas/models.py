"""
schemas/models.py — SwapGo 백엔드 ai.py 와 정확히 대응하는 Pydantic 스키마

백엔드 참조:
  app/schemas/ai.py                 → SignalIn / PredictionIn / SentimentIn
  app/services/ai_ingest_service.py → decimals 변환 규칙

핵심 계약 (422 선방어):
  - confidence        : 0.0~1.0 (퍼센트 금지)
  - sentiment_score   : 정수 -100~100
  - horizon           : 정확히 "1h" | "24h" | "7d"
  - *_human 가격 필드 : 사람단위 소수 문자열 ("43210.55")
                        백엔드가 to_base_units(value, asset.decimals) 로 정수 변환
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ════════════════════════════════════════════════════════════
# SwapGo /chart/ohlc 캔들 응답 스키마
# ════════════════════════════════════════════════════════════

class CandleData(BaseModel):
    """
    GET /chart/ohlc 응답의 개별 캔들.
    8피처 계산에 사용하는 필드:
      close        → Ret_btc / Ret_eth (로그 수익률)
      high/low     → BB_Width (볼린저 밴드 너비)
      volume_base  → Volume_btc / Volume_eth, Log_Pow_btc (체결강도 근사)
      trades_count → Log_Trades_btc
    """
    bucket_start: str       # ISO8601 타임스탬프
    open: float
    high: float
    low: float
    close: float
    volume_base: float      # 기간 내 Base Asset 총 거래량 (BTC/ETH 수량)
    volume_quote: float     # 기간 내 Quote Asset 총 거래량 (원화/달러 등)
    trades_count: int       # 기간 내 체결 건수

    @classmethod
    def from_dict(cls, d: dict) -> "CandleData":
        """API 응답 dict 를 안전하게 파싱합니다."""
        return cls(
            bucket_start=str(d.get("bucket_start", "")),
            open=float(d.get("open", 0)),
            high=float(d.get("high", 0)),
            low=float(d.get("low", 0)),
            close=float(d.get("close", 0)),
            volume_base=float(d.get("volume_base", 0)),
            volume_quote=float(d.get("volume_quote", 0)),
            trades_count=int(d.get("trades_count", 0)),
        )


# ════════════════════════════════════════════════════════════
# ingest 요청 바디 — 백엔드 SignalIn / PredictionIn / SentimentIn 과 1:1
# ════════════════════════════════════════════════════════════

class SignalItem(BaseModel):
    """백엔드 SignalIn 과 동일"""
    symbol: str
    side: Literal["buy", "sell", "hold"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: Optional[str] = None
    expires_in_sec: Optional[int] = None


class PredictionItem(BaseModel):
    """
    백엔드 PredictionIn 과 동일.
    *_human 필드는 to_base_units(value, asset.decimals) 로 변환되어 저장됨.
    반드시 사람단위 소수 문자열 (예: "43210.55") 이어야 함.
    """
    symbol: str
    horizon: Literal["1h", "24h", "7d"]
    predicted_price_human: str
    lower_bound_human: Optional[str] = None
    upper_bound_human: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_tag: Optional[str] = None

    @field_validator(
        "predicted_price_human", "lower_bound_human", "upper_bound_human",
        mode="before",
    )
    @classmethod
    def _must_be_positive_numeric_string(cls, v: object) -> Optional[str]:
        if v is None:
            return None
        try:
            val = float(str(v))
            if val < 0:
                raise ValueError("가격은 0 이상이어야 합니다.")
        except (TypeError, ValueError) as e:
            raise ValueError(f"사람단위 소수 문자열이어야 합니다: {v!r} — {e}") from e
        return str(v)


class SentimentItem(BaseModel):
    """
    백엔드 SentimentIn 과 동일.
    ma7_human / ma25_human 도 to_base_units 로 정수 변환됨.
    반드시 캔들 close 배열 기반 실제 MA 계산값이어야 함.
    """
    symbol: str
    sentiment_score: int = Field(..., ge=-100, le=100)
    rsi: Optional[float] = Field(None, ge=0.0, le=100.0)
    macd: Optional[float] = None
    ma7_human: Optional[str] = None
    ma25_human: Optional[str] = None
    extra_json: Optional[str] = None

    @field_validator("ma7_human", "ma25_human", mode="before")
    @classmethod
    def _must_be_numeric_string(cls, v: object) -> Optional[str]:
        if v is None:
            return None
        try:
            float(str(v))
        except (TypeError, ValueError):
            raise ValueError(f"MA 는 사람단위 소수 문자열이어야 합니다: {v!r}")
        return str(v)


# ════════════════════════════════════════════════════════════
# 봇 내부 도메인 모델 — AI 추론 결과 DTO
# ════════════════════════════════════════════════════════════

class HorizonPrediction(BaseModel):
    """
    단일 모델의 단일 horizon 예측.
    scalper → "1h" / swing → "24h" / longterm → "7d" 로 1:1 매핑.
    """
    horizon: Literal["1h", "24h", "7d"]
    predicted_price_human: str   # 실제 가격, 사람단위 문자열 (예: "43210.5500")
    lower_bound_human: Optional[str] = None
    upper_bound_human: Optional[str] = None
    confidence: float            # 해당 모델의 독립 confidence


class AIInferenceResult(BaseModel):
    """
    한 symbol 에 대한 전체 AI 추론 결과.
      signal      : 앙상블 side / confidence
      predictions : 모델별 horizon 예측 3종 (1h / 24h / 7d)
      sentiment   : score, rsi, macd, ma7_human, ma25_human
    """
    symbol: str

    # 매매 신호 (앙상블)
    side: Literal["buy", "sell", "hold"]
    confidence: float

    # 가격 예측 — 반드시 1h / 24h / 7d 세 가지 모두 포함
    predictions: list[HorizonPrediction]

    # 시장 심리
    sentiment_score: int = Field(0, ge=-100, le=100)
    rsi: Optional[float] = None
    macd: Optional[float] = None
    ma7_human: Optional[str] = None
    ma25_human: Optional[str] = None

    # 메타
    reason: Optional[str] = None
    model_tag: Optional[str] = None

    @model_validator(mode="after")
    def _clamp_confidence(self) -> "AIInferenceResult":
        self.confidence = round(max(0.0, min(1.0, self.confidence)), 4)
        return self

    @model_validator(mode="after")
    def _check_horizons(self) -> "AIInferenceResult":
        """프론트 그리드가 빈칸 없이 채워지려면 세 horizon 이 모두 있어야 합니다."""
        horizons = {p.horizon for p in self.predictions}
        missing = {"1h", "24h", "7d"} - horizons
        if missing:
            raise ValueError(
                f"predictions 에 horizon {missing} 가 빠져 있습니다. "
                "1h / 24h / 7d 세 가지가 모두 있어야 합니다."
            )
        return self
