"""
ai/ai_engine.py — ONNX GRU 추론 엔진

입력: SwapGo /chart/ohlc 캔들의 close 가격 시퀀스
출력: 다음 틱 가격 변화 예측 (EMA 평활)

가이드 섹션 9 의 팁:
  - reserve_*, amount_* 는 raw 문자열 → int(x)/10**decimals 로 변환
  - bucket_start 는 ISO8601 → pd.to_datetime
  - confidence 0~1 그대로 사용

Mock 모드: ONNX 파일 없을 때 개발/테스트 환경에서도 전체 파이프라인 동작 가능
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Optional

import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class AIEngine:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        seq_len: int,
    ):
        self.model_name = model_name
        self.seq_len = seq_len
        self.input_scale = settings.price_input_scale
        self.output_scale = settings.pred_output_scale
        self.ema_alpha = settings.ema_alpha

        self.buffer: deque = deque(maxlen=seq_len)
        self._raw_pred: Optional[float] = None
        self._ema_pred: Optional[float] = None
        self._infer_count: int = 0

        self.session = None
        self.input_name: Optional[str] = None
        self._mock = False
        self._init_onnx(model_path)

    # ── 초기화 ───────────────────────────────────────────────
    def _init_onnx(self, path: str) -> None:
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"[{self.model_name}] ONNX 모델 로드 완료: {path}")
        except Exception as e:
            logger.warning(f"[{self.model_name}] 모델 없음 → Mock 모드: {e}")
            self._mock = True

    # ── 캔들 시퀀스 일괄 입력 (폴링 방식) ────────────────────
    def load_candles(self, candles: list[dict]) -> None:
        """
        REST /chart/ohlc 응답 캔들을 버퍼에 적재합니다.
        매 ingest 주기 시작 시 최신 N개를 통째로 넣어 버퍼를 갱신합니다.
        """
        self.buffer.clear()
        for c in candles[-self.seq_len :]:
            close = float(c.get("close", 0))
            if close > 0:
                self.buffer.append([close / self.input_scale])

    # ── 단일 가격 스트리밍 입력 (WS 방식) ───────────────────
    def push_price(self, close: float) -> None:
        """WebSocket 캔들 완성 시 가격 하나를 버퍼에 추가합니다."""
        if close > 0:
            self.buffer.append([close / self.input_scale])

    # ── 비동기 추론 ──────────────────────────────────────────
    async def infer(self) -> Optional[float]:
        """
        버퍼가 충분히 채워졌을 때 추론 실행.
        반환값: EMA 평활된 예측 변화율 (양수=상승, 음수=하락)
        """
        if len(self.buffer) < self.seq_len:
            return None

        if self._mock:
            prices = [v[0] for v in self.buffer]
            raw = (prices[-1] - prices[-2]) * self.output_scale
            self._update_ema(raw)
            return self._ema_pred

        arr = np.array([list(self.buffer)], dtype=np.float32)
        try:
            output = await asyncio.to_thread(
                self.session.run, None, {self.input_name: arr}
            )
            raw = float(output[0][0][0])
            self._raw_pred = raw
            self._update_ema(raw)
            self._infer_count += 1
            return self._ema_pred
        except Exception as e:
            logger.error(f"[{self.model_name}] 추론 실패: {e}")
            return None

    # ── EMA 평활 ─────────────────────────────────────────────
    def _update_ema(self, val: float) -> None:
        if self._ema_pred is None:
            self._ema_pred = val
        else:
            self._ema_pred = self.ema_alpha * val + (1 - self.ema_alpha) * self._ema_pred

    # ── 상태 조회 ────────────────────────────────────────────
    @property
    def is_warm(self) -> bool:
        return len(self.buffer) >= self.seq_len

    @property
    def direction(self) -> Optional[int]:
        """예측 방향: +1(상승) / -1(하락) / None(워밍업 중)"""
        if self._ema_pred is None:
            return None
        return 1 if self._ema_pred > 0 else -1

    def get_info(self) -> dict:
        return {
            "model": self.model_name,
            "seq_len": self.seq_len,
            "buffer_fill": len(self.buffer),
            "is_warm": self.is_warm,
            "ema_prediction": self._ema_pred,
            "infer_count": self._infer_count,
            "mock_mode": self._mock,
        }
