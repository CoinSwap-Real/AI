"""
core/swapgo_client.py — SwapGo 백엔드 REST API 클라이언트

가이드 섹션 2·3·4·5 를 구현합니다.
- GET 엔드포인트: 인증 불필요 (시장 데이터, 캔들, 풀 상태 등)
- POST 엔드포인트: X-Bot-Key 헤더 필수 (ingest, swap)
- 재시도: 일시적 오류(5xx, 연결 끊김)는 최대 3회 backoff 재시도
- StaleQuote(409): 호출 측에서 반드시 재견적 처리 필요 → 예외로 노출
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)

_RETRY_CODES = {500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # 초


class StaleQuoteError(Exception):
    """409 STALE_QUOTE — 풀이 변경됨. 호출 측에서 /quote 부터 재시도해야 합니다."""


class SwapGoClient:
    """
    SwapGo 백엔드와 통신하는 비동기 HTTP 클라이언트.
    httpx.AsyncClient 를 싱글턴으로 보유하므로 close() 를 명시적으로 호출하거나
    async context manager 로 사용하세요.
    """

    def __init__(self):
        self._headers = {
            "X-Bot-Key": settings.bot_key,
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(
            base_url=settings.swapgo_base_url,
            headers=self._headers,
            timeout=settings.http_timeout_sec,
        )

    async def close(self) -> None:
        await self._client.aclose()

    # ── 내부 재시도 래퍼 ─────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict] = None,
        json: Optional[dict] = None,
    ) -> Any:
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = await self._client.request(
                    method, path, params=params, json=json
                )
                if resp.status_code == 409:
                    raise StaleQuoteError(resp.text)
                if resp.status_code == 401:
                    raise PermissionError("401 UNAUTHORIZED — X-Bot-Key 가 잘못됐거나 누락됐습니다.")
                if resp.status_code == 403:
                    raise PermissionError(f"403 FORBIDDEN — 스코프 부족: {resp.text}")
                if resp.status_code in _RETRY_CODES:
                    raise httpx.HTTPStatusError(
                        f"서버 오류 {resp.status_code}", request=resp.request, response=resp
                    )
                resp.raise_for_status()
                return resp.json()

            except (StaleQuoteError, PermissionError):
                raise  # 재시도 없이 즉시 상위로

            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_exc = e
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"[SwapGoClient] {method} {path} 실패 (시도 {attempt}/{_MAX_RETRIES}), "
                    f"{delay:.1f}초 후 재시도: {e}"
                )
                await asyncio.sleep(delay)

        raise last_exc  # type: ignore[misc]

    # ════════════════════════════════════════════════════════
    # 데이터 조회 (인증 불필요 — GET, 가이드 섹션 2)
    # ════════════════════════════════════════════════════════

    async def get_coins(self) -> list[dict]:
        """GET /market/coins → 코인 카드 리스트"""
        data = await self._request("GET", "/market/coins")
        return data.get("coins", [])

    async def get_ohlc(
        self,
        pool_id: int,
        interval: str = "5m",
        limit: int = 200,
        from_dt: Optional[str] = None,
        to_dt: Optional[str] = None,
    ) -> list[dict]:
        """GET /chart/ohlc → 캔들 리스트"""
        params: dict = {"pool_id": pool_id, "interval": interval, "limit": limit}
        if from_dt:
            params["from"] = from_dt
        if to_dt:
            params["to"] = to_dt
        data = await self._request("GET", "/chart/ohlc", params=params)
        return data.get("candles", [])

    async def get_ticker(self, pool_id: int) -> dict:
        """GET /chart/ticker → 24h 시세 요약"""
        data = await self._request("GET", "/chart/ticker", params={"pool_id": pool_id})
        # 백엔드가 { ok, data: { last_price, ... } } 형태로 감쌈
        return data.get("data", data)

    async def get_pool(self, pool_id: int) -> dict:
        """GET /pools/{id} → 풀 reserve, fee, revision"""
        return await self._request("GET", f"/pools/{pool_id}")

    async def get_trades(self, pool_id: int, limit: int = 200) -> list[dict]:
        """GET /market/trades → 최근 체결 (슬리피지 등급 포함)"""
        data = await self._request(
            "GET", "/market/trades", params={"pool_id": pool_id, "limit": limit}
        )
        return data.get("trades", data) if isinstance(data, dict) else data

    async def get_wallet(self) -> dict:
        """GET /wallet/me → 봇 지갑 잔고 (봇 키 인증)"""
        return await self._request("GET", "/wallet/me")

    # ════════════════════════════════════════════════════════
    # Ingest API (봇 키 필수, 가이드 섹션 3)
    # ════════════════════════════════════════════════════════

    async def ingest_signals(self, items: list[dict]) -> dict:
        """POST /ai/ingest/signals"""
        return await self._request("POST", "/ai/ingest/signals", json={"items": items})

    async def ingest_predictions(self, items: list[dict]) -> dict:
        """POST /ai/ingest/predictions"""
        return await self._request("POST", "/ai/ingest/predictions", json={"items": items})

    async def ingest_sentiment(self, items: list[dict]) -> dict:
        """POST /ai/ingest/sentiment"""
        return await self._request("POST", "/ai/ingest/sentiment", json={"items": items})

    # ════════════════════════════════════════════════════════
    # 스왑 견적·실행 (가이드 섹션 4)
    # ════════════════════════════════════════════════════════

    async def quote_swap(self, body: dict) -> dict:
        """POST /swap/quote → amount_out_min, slippage_level, expected_revision"""
        return await self._request("POST", "/swap/quote", json=body)

    async def execute_swap(self, body: dict) -> dict:
        """
        POST /swap/execute (scope: bot:trade)
        StaleQuoteError(409) 발생 시 호출 측에서 quote_swap 부터 재시도 필요.
        """
        return await self._request("POST", "/swap/execute", json=body)

    async def deposit_mock(self, symbol: str, amount_human: str) -> dict:
        """POST /wallet/deposit/mock → 모의 입금 (개발용)"""
        return await self._request(
            "POST",
            "/wallet/deposit/mock",
            json={"symbol": symbol, "amount_human": amount_human},
        )
