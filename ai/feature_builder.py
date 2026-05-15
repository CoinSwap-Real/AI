"""
ai/feature_builder.py — 캔들 데이터 → 모델 입력 8피처 변환기

[피처 정의 (모델 학습 순서와 정확히 일치)]
  idx  이름             계산 방법
  0    Ret_btc          log(close_t / close_t-1)                — BTC 로그 수익률
  1    Ret_eth          log(eth_close_t / eth_close_t-1)        — ETH 로그 수익률
  2    Log_Pow_btc      log(volume_base + 1)                    — BTC 체결강도 근사
                        ※ 정확한 체결강도는 /market/trades의 side 정보가 필요하나
                           SwapGo OHLC 캔들에는 없으므로 volume_base 로 근사
  3    Log_Trades_btc   log(trades_count + 1)                   — BTC 체결건수
  4    Volat_btc        std(Ret_btc[-VOLAT_WINDOW:])             — BTC 단기 변동성
  5    BB_Width         (upper - lower) / middle                — 볼린저 밴드 너비
                        upper/lower = SMA±2σ (BB_PERIOD 캔들 기준)
  6    Volume_btc       volume_base (raw, scaler 가 정규화)     — BTC 거래량
  7    Volume_eth       eth_volume_base (raw)                   — ETH 거래량

[입출력]
  입력: BTC 캔들 리스트 + ETH 캔들 리스트 (동일 interval, 동일 limit)
  출력: np.ndarray shape (N, 8), scaler 적용 완료

[scaler.pkl]
  joblib 으로 직렬화된 sklearn scaler (MinMaxScaler / StandardScaler 등).
  학습 시 사용한 것과 동일한 파일을 models/scaler.pkl 에 위치시켜야 합니다.
  파일이 없으면 Mock 모드 (정규화 없이 raw 값 사용, 개발 환경 전용).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from config import settings
from schemas.models import CandleData

logger = logging.getLogger(__name__)

# 피처 계산 파라미터
_VOLAT_WINDOW = 10    # 변동성 계산 rolling 윈도우
_BB_PERIOD    = 20    # 볼린저 밴드 SMA 기간
_LOG_EPS      = 1e-9  # log(0) 방지


class FeatureBuilder:
    """
    BTC + ETH 캔들 시퀀스를 받아 (N, 8) 피처 행렬을 반환합니다.
    scaler.pkl 이 있으면 자동 적용, 없으면 raw 값으로 Mock 모드 동작.
    """

    def __init__(self):
        self._scaler = None
        self._mock   = False
        self._load_scaler()

    # ── 초기화 ───────────────────────────────────────────────
    def _load_scaler(self) -> None:
        try:
            import joblib
            self._scaler = joblib.load(settings.scaler_path)
            logger.info(f"[FeatureBuilder] scaler 로드 완료: {settings.scaler_path}")
        except Exception as e:
            logger.warning(
                f"[FeatureBuilder] scaler 없음 → Mock 모드 (정규화 생략): {e}"
            )
            self._mock = True

    # ── 공개 메서드 ──────────────────────────────────────────
    def build(
        self,
        btc_candles: list[dict],
        eth_candles: list[dict],
    ) -> Optional[np.ndarray]:
        """
        BTC·ETH 캔들 → (N, 8) 피처 행렬 (scaler 적용 완료).

        반환값:
          np.ndarray shape (N, 8) — N은 두 캔들 리스트 중 짧은 쪽 길이
          None — 유효 캔들이 부족할 때
        """
        btc = [CandleData.from_dict(c) for c in btc_candles if float(c.get("close", 0)) > 0]
        eth = [CandleData.from_dict(c) for c in eth_candles if float(c.get("close", 0)) > 0]

        # 두 시퀀스 길이를 맞춤 (짧은 쪽 기준)
        n = min(len(btc), len(eth))
        if n < 2:
            logger.debug("[FeatureBuilder] 캔들 부족 (n<2)")
            return None

        btc = btc[-n:]
        eth = eth[-n:]

        rows: list[list[float]] = []
        for i in range(1, n):   # i=0 은 이전 봉이 없으므로 건너뜀
            row = self._build_row(btc, eth, i)
            rows.append(row)

        if not rows:
            return None

        X = np.array(rows, dtype=np.float32)   # (N-1, 8)

        # NaN/Inf 를 0 으로 대체
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if self._mock or self._scaler is None:
            return X

        try:
            X_scaled = self._scaler.transform(X).astype(np.float32)
            return X_scaled
        except Exception as e:
            logger.error(f"[FeatureBuilder] scaler.transform 실패: {e}")
            return X  # fallback: 비정규화 raw 반환

    # ── 단일 행 계산 ─────────────────────────────────────────
    def _build_row(
        self,
        btc: list[CandleData],
        eth: list[CandleData],
        i: int,
    ) -> list[float]:
        """인덱스 i 의 피처 벡터 (8개) 반환"""

        # ① Ret_btc
        ret_btc = _log_return(btc[i - 1].close, btc[i].close)

        # ② Ret_eth
        ret_eth = _log_return(eth[i - 1].close, eth[i].close)

        # ③ Log_Pow_btc (체결강도 근사: log volume)
        log_pow_btc = math.log(btc[i].volume_base + _LOG_EPS)

        # ④ Log_Trades_btc
        log_trades_btc = math.log(btc[i].trades_count + _LOG_EPS)

        # ⑤ Volat_btc — 최근 VOLAT_WINDOW 개 수익률의 표준편차
        start = max(1, i - _VOLAT_WINDOW + 1)
        recent_rets = [
            _log_return(btc[j - 1].close, btc[j].close)
            for j in range(start, i + 1)
        ]
        volat_btc = float(np.std(recent_rets)) if len(recent_rets) > 1 else 0.0

        # ⑥ BB_Width — SMA±2σ 기반 볼린저 밴드 너비
        bb_start = max(0, i - _BB_PERIOD + 1)
        closes = [btc[j].close for j in range(bb_start, i + 1)]
        bb_width = _bollinger_width(closes)

        # ⑦ Volume_btc
        volume_btc = btc[i].volume_base

        # ⑧ Volume_eth
        volume_eth = eth[i].volume_base

        return [ret_btc, ret_eth, log_pow_btc, log_trades_btc,
                volat_btc, bb_width, volume_btc, volume_eth]

    # ── 상태 조회 ────────────────────────────────────────────
    def get_info(self) -> dict:
        return {
            "mock_mode": self._mock,
            "scaler_path": settings.scaler_path,
            "scaler_type": type(self._scaler).__name__ if self._scaler else None,
        }


# ════════════════════════════════════════════════════════════
# 순수 함수 헬퍼
# ════════════════════════════════════════════════════════════

def _log_return(prev: float, curr: float) -> float:
    if prev <= 0 or curr <= 0:
        return 0.0
    return math.log(curr / prev)


def _bollinger_width(closes: list[float]) -> float:
    """볼린저 밴드 너비 = (upper - lower) / middle = 4σ / SMA"""
    if len(closes) < 2:
        return 0.0
    arr = np.array(closes, dtype=np.float64)
    sma = arr.mean()
    if sma == 0:
        return 0.0
    std = arr.std()
    return float(4.0 * std / sma)