"""
services/candle_cache.py — 공유 캔들 버퍼 + 피처 캐시

[해결하는 문제]
  - BotA_Trade / BotB_WS 가 ETH 캔들을 각자 독립적으로 REST 호출 → 중복 제거
  - 캔들마다 FeatureBuilder.build() 를 여러 봇이 각자 호출 → 1회로 통합
  - BotA_Ingest 의 REST 폴링도 동일 캐시를 읽어 중복 API 호출 방지

[캐시 전략]
  - BTC 캔들: WS 푸시(실시간) 또는 REST 갱신(폴링) 양쪽 모두 지원
  - ETH 캔들: TTL 기반 갱신 (기본 60초). 여러 봇이 동시에 만료 감지해도
    asyncio.Lock 으로 REST 호출은 단 1회만 발생
  - 피처 행렬: 캔들 업데이트마다 1회 계산 후 캐시. 동일 캔들 상태이면 재사용
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from collections import deque
from typing import Optional

import numpy as np

from config import settings
from ai.feature_builder import FeatureBuilder

logger = logging.getLogger(__name__)

_ETH_TTL_SEC = 60.0          # ETH 캔들 REST 갱신 주기
_FEATURE_STALE_SEC = 5.0     # 피처 캐시 유효 시간 (동일 캔들 내 재요청 방지)


class CandleCache:
    """
    모든 봇이 공유하는 캔들 버퍼 + 피처 캐시 싱글턴.
    생성 후 main.py 에서 app.state.candle_cache 에 보관하고
    각 봇이 주입받아 사용합니다.
    """

    def __init__(self, client, feature_builder: FeatureBuilder):
        self._client = client
        self._fb     = feature_builder

        self._btc: deque[dict] = deque(maxlen=settings.candle_limit)
        self._eth: list[dict]  = []

        self._eth_ts:      float = 0.0   # 마지막 ETH 갱신 시각
        self._eth_lock     = asyncio.Lock()

        # 피처 캐시
        self._features:    Optional[np.ndarray] = None
        self._features_ts: float = 0.0
        self._feature_lock = asyncio.Lock()

    # ════════════════════════════════════════════════════════
    # BTC 캔들 업데이트
    # ════════════════════════════════════════════════════════

    def push_btc(self, candle: dict) -> None:
        """WS 모드: CandleEventBus 가 캔들 완성 시마다 호출."""
        if float(candle.get("close", 0)) > 0:
            self._btc.append(candle)
            self._features_ts = 0.0  # 피처 캐시 무효화

    async def refresh_btc_from_rest(self) -> None:
        """REST 폴링 모드: BotA_Ingest 의 60초 사이클 시 호출."""
        try:
            candles = await self._client.get_ohlc(
                settings.pool_id,
                interval=settings.candle_interval,
                limit=settings.candle_limit,
            )
            self._btc.clear()
            for c in candles:
                if float(c.get("close", 0)) > 0:
                    self._btc.append(c)
            self._features_ts = 0.0
            logger.debug(f"[CandleCache] BTC REST 갱신: {len(self._btc)}개")
        except Exception:
            logger.error(f"[CandleCache] BTC REST 갱신 실패\n{traceback.format_exc()}")

    # ════════════════════════════════════════════════════════
    # ETH 캔들 — TTL 기반 자동 갱신
    # ════════════════════════════════════════════════════════

    async def _ensure_eth(self) -> None:
        """ETH 캔들이 TTL 초과 시 REST 갱신. Lock 으로 중복 호출 방지."""
        if time.monotonic() - self._eth_ts < _ETH_TTL_SEC:
            return
        async with self._eth_lock:
            # double-check: 다른 코루틴이 Lock 보유 중 이미 갱신했을 수 있음
            if time.monotonic() - self._eth_ts < _ETH_TTL_SEC:
                return
            try:
                self._eth = await self._client.get_ohlc(
                    settings.eth_pool_id,
                    interval=settings.candle_interval,
                    limit=settings.candle_limit,
                )
                self._eth_ts = time.monotonic()
                self._features_ts = 0.0
                logger.debug(f"[CandleCache] ETH 갱신: {len(self._eth)}개")
            except Exception:
                logger.warning(f"[CandleCache] ETH 갱신 실패\n{traceback.format_exc()}")

    # ════════════════════════════════════════════════════════
    # 피처 행렬 — 캔들 변경 시 1회 계산 후 캐시
    # ════════════════════════════════════════════════════════

    async def get_features(self) -> Optional[np.ndarray]:
        """
        (N, 8) scaler 적용 피처 행렬 반환.
        최근 캔들 업데이트 이후 최초 1회만 계산하고 이후 캐시 반환.
        """
        await self._ensure_eth()

        now = time.monotonic()
        if self._features is not None and now - self._features_ts < _FEATURE_STALE_SEC:
            return self._features

        async with self._feature_lock:
            # double-check
            if self._features is not None and time.monotonic() - self._features_ts < _FEATURE_STALE_SEC:
                return self._features

            btc_list = list(self._btc)
            eth_list = self._eth

            if len(btc_list) < 2 or len(eth_list) < 2:
                return None

            try:
                features = await asyncio.to_thread(
                    self._fb.build, btc_list, eth_list
                )
                self._features    = features
                self._features_ts = time.monotonic()
                return features
            except Exception:
                logger.error(f"[CandleCache] 피처 빌드 실패\n{traceback.format_exc()}")
                return None

    # ════════════════════════════════════════════════════════
    # 편의 메서드
    # ════════════════════════════════════════════════════════

    def latest_btc_close(self) -> float:
        return float(self._btc[-1].get("close", 0)) if self._btc else 0.0

    def latest_eth_close(self) -> float:
        return float(self._eth[-1].get("close", 0)) if self._eth else 0.0

    def btc_candles(self) -> list[dict]:
        return list(self._btc)

    def get_info(self) -> dict:
        return {
            "btc_count":      len(self._btc),
            "eth_count":      len(self._eth),
            "eth_age_sec":    round(time.monotonic() - self._eth_ts, 1),
            "features_shape": list(self._features.shape) if self._features is not None else None,
            "features_age_sec": round(time.monotonic() - self._features_ts, 1),
        }

    # ════════════════════════════════════════════════════════
    # 초기 프리페치 (시스템 시작 시)
    # ════════════════════════════════════════════════════════

    async def prefetch(self) -> None:
        """시스템 시작 시 한 번 호출. BTC + ETH 캔들을 동시에 수집합니다."""
        try:
            btc, eth = await asyncio.gather(
                self._client.get_ohlc(
                    settings.pool_id,
                    interval=settings.candle_interval,
                    limit=settings.candle_limit,
                ),
                self._client.get_ohlc(
                    settings.eth_pool_id,
                    interval=settings.candle_interval,
                    limit=settings.candle_limit,
                ),
            )
            self._btc.clear()
            for c in btc:
                if float(c.get("close", 0)) > 0:
                    self._btc.append(c)
            self._eth    = eth
            self._eth_ts = time.monotonic()
            logger.info(
                f"[CandleCache] 프리페치 완료 "
                f"BTC={len(self._btc)} ETH={len(self._eth)}"
            )
        except Exception:
            logger.error(f"[CandleCache] 프리페치 실패\n{traceback.format_exc()}")