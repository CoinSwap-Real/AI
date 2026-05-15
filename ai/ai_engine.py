"""
ai/ai_engine.py — ONNX GRU 추론 엔진

[모델 I/O 스펙]
  입력: (batch=1, seq_len, 8) — FeatureBuilder 가 scaler 적용한 행렬
  출력: (1, 2)                — [BTC 1초 뒤 로그수익률, ETH 1초 뒤 로그수익률]

[설계 원칙]
  - 피처 계산·정규화는 FeatureBuilder 에 위임. AIEngine 은 추론만 담당.
  - EMA 로 출력 평활. BTC / ETH 각각 독립적으로 관리.
  - Mock 모드: ONNX 파일 없을 때 전 틱 대비 수익률을 그대로 반환.

[사용 방법]
  # 폴링 (REST)
  features = feature_builder.build(btc_candles, eth_candles)  # (N, 8)
  engine.load_features(features)
  btc_ret, eth_ret = await engine.infer()   # 로그 수익률

  # 스트리밍 (WebSocket)
  engine.push_feature_row(row)              # (8,) 행 하나
  btc_ret, eth_ret = await engine.infer()
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Optional

import numpy as np

from config import settings

logger = logging.getLogger(__name__)

# 모델 출력 인덱스
_IDX_BTC = 0
_IDX_ETH = 1


class AIEngine:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        seq_len: int,
    ):
        self.model_name = model_name
        self.seq_len     = seq_len
        self.ema_alpha   = settings.ema_alpha

        # 피처 버퍼: 각 원소는 shape (8,) numpy 배열
        self._buffer: deque[np.ndarray] = deque(maxlen=seq_len)

        # EMA 캐시 (BTC, ETH 각각)
        self._ema_btc: Optional[float] = None
        self._ema_eth: Optional[float] = None
        self._infer_count = 0

        # ONNX 세션
        self.session    = None
        self.input_name: Optional[str] = None
        self._mock      = False
        self._init_onnx(model_path)

    # ── ONNX 초기화 ──────────────────────────────────────────
    def _init_onnx(self, path: str) -> None:
        try:
            import onnxruntime as ort
            self.session    = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            # 입력 shape 검증
            expected = self.session.get_inputs()[0].shape
            logger.info(
                f"[{self.model_name}] ONNX 로드 완료: {path}  "
                f"입력shape={expected}"
            )
        except Exception as e:
            logger.warning(f"[{self.model_name}] 모델 없음 → Mock 모드: {e}")
            self._mock = True

    # ════════════════════════════════════════════════════════
    # 버퍼 적재 (폴링 방식)
    # ════════════════════════════════════════════════════════

    def load_features(self, features: np.ndarray) -> None:
        """
        FeatureBuilder.build() 결과 (N, 8) 배열을 받아 버퍼를 갱신합니다.
        매 ingest 사이클 시작 시 호출.
        """
        if features is None or features.ndim != 2 or features.shape[1] != 8:
            logger.warning(f"[{self.model_name}] 잘못된 피처 shape: {getattr(features, 'shape', None)}")
            return
        self._buffer.clear()
        for row in features[-self.seq_len:]:
            self._buffer.append(row.astype(np.float32))

    # ════════════════════════════════════════════════════════
    # 버퍼 적재 (WebSocket 스트리밍 방식)
    # ════════════════════════════════════════════════════════

    def push_feature_row(self, row: np.ndarray) -> None:
        """
        완성된 캔들 1개에서 계산한 피처 벡터 (8,) 를 버퍼에 추가합니다.
        WS 모드에서 캔들 완성 콜백마다 호출.
        """
        if row is None or row.shape != (8,):
            return
        self._buffer.append(row.astype(np.float32))

    # ════════════════════════════════════════════════════════
    # 추론
    # ════════════════════════════════════════════════════════

    async def infer(self) -> Optional[tuple[float, float]]:
        """
        버퍼가 seq_len 만큼 채워지면 추론 실행.

        반환값:
          (btc_log_return, eth_log_return) — EMA 평활된 로그 수익률
          None — 버퍼 미충전
        """
        if not self.is_warm:
            return None

        if self._mock:
            return self._mock_predict()

        arr = np.array([list(self._buffer)], dtype=np.float32)  # (1, seq_len, 8)
        try:
            output = await asyncio.to_thread(
                self.session.run, None, {self.input_name: arr}
            )
            # output shape: (1, 2)
            btc_raw = float(output[0][0][_IDX_BTC])
            eth_raw = float(output[0][0][_IDX_ETH])

            self._ema_btc = _ema_update(self._ema_btc, btc_raw, self.ema_alpha)
            self._ema_eth = _ema_update(self._ema_eth, eth_raw, self.ema_alpha)
            self._infer_count += 1

            return self._ema_btc, self._ema_eth

        except Exception as e:
            logger.error(f"[{self.model_name}] 추론 실패: {e}")
            return None

    def _mock_predict(self) -> tuple[float, float]:
        """개발 Mock: 버퍼 마지막 두 행의 수익률 그대로 반환"""
        buf = list(self._buffer)
        # 피처 idx 0 = Ret_btc, idx 1 = Ret_eth (scaler 이전 값 기준)
        btc_raw = float(buf[-1][0]) if buf else 0.0
        eth_raw = float(buf[-1][1]) if buf else 0.0
        self._ema_btc = _ema_update(self._ema_btc, btc_raw, self.ema_alpha)
        self._ema_eth = _ema_update(self._ema_eth, eth_raw, self.ema_alpha)
        return self._ema_btc, self._ema_eth  # type: ignore[return-value]

    # ════════════════════════════════════════════════════════
    # 상태 조회
    # ════════════════════════════════════════════════════════

    @property
    def is_warm(self) -> bool:
        return len(self._buffer) >= self.seq_len

    def get_info(self) -> dict:
        return {
            "model": self.model_name,
            "seq_len": self.seq_len,
            "buffer_fill": len(self._buffer),
            "is_warm": self.is_warm,
            "ema_btc": self._ema_btc,
            "ema_eth": self._ema_eth,
            "infer_count": self._infer_count,
            "mock_mode": self._mock,
        }


# ════════════════════════════════════════════════════════════
# 순수 함수
# ════════════════════════════════════════════════════════════

def _ema_update(prev: Optional[float], new: float, alpha: float) -> float:
    if prev is None:
        return new
    return alpha * new + (1.0 - alpha) * prev
