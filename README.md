제공해주신 가이드라인 문서를 바탕으로 GitHub 레포지토리에 바로 적용할 수 있는 `README.md` 파일을 작성했습니다.

---

# SwapGo AI Integration Guide

SwapGo 백엔드를 활용하여 시장 데이터를 수집하고, AI 모델을 학습 및 추론하며, 그 결과를 프론트엔드에 반영하거나 자동 모의 거래를 실행하기 위한 AI팀 통합 가이드입니다.

## 📌 주요 통합 흐름

* 
**데이터 수집 (Data IN)**: 인증 없이 `/market/*`, `/chart/*`, `/pools`, `/explorer/*` 엔드포인트를 통해 데이터를 가져옵니다.


* 
**결과 업로드 (Data OUT)**: `bot:ingest` 권한이 있는 봇 API 키를 사용하여 `/ai/ingest/*` 로 매매 신호, 예측, 심리 지표를 전송합니다.


* 
**거래 실행 (Optional)**: `bot:trade` 권한을 통해 `/swap/execute` 로 실제 봇 자동 모의 거래를 실행할 수 있습니다.


* 
**실시간 통신 (WebSocket)**: `/ws` 에 연결하여 풀 상태, 체결 내역, 캔들 데이터 등을 실시간으로 수신합니다.



---

## 🔑 인증 및 API 키 발급

서버 첫 구동 시 콘솔에 기본 봇 API 키가 한 번만 표시됩니다. 이 raw 키를 환경변수나 시크릿 매니저에 안전하게 보관하세요.

* 
**헤더 사용법**: `X-Bot-Key: <raw>` 또는 `Authorization: Bearer <raw>`.


* 
**권한(Scope) 관리**: 용도에 맞게 최소한의 스코프만 부여하는 것을 권장합니다.


* 
`bot:ingest`: AI 신호/예측 데이터 업로드용.


* 
`bot:trade`: 자동 거래 실행용.


* 
`bot:lp`: 유동성 공급/회수용.





---

## ⚠️ [중요] 데이터 단위(Decimals) 처리 규칙

백엔드는 내부에서 모든 수량을 **raw 정수**로 저장합니다 (예: 1 USDT = 1_000_000, 1 BTC = 100_000_000). 잘못된 단위 사용은 가장 흔한 버그의 원인입니다.

* 
`/ai/ingest/*` 호출 시 `ma7_human`, `predicted_price_human` 등의 필드는 반드시 **사람 단위 소수 문자열** (예: "2950.5")로 입력해야 합니다.


* 
`confidence` 필드는 퍼센트(%)가 아닌 **0.0 ~ 1.0 사이의 값**을 사용합니다.


* 
`sentiment_score` 필드는 **-100 ~ 100 사이의 정수**를 입력합니다.



---

## 📡 API 레퍼런스

### 1. 데이터 조회 (인증 불필요)

* 
**마켓/통계**: `GET /market/coins`, `GET /market/global`, `GET /market/trades`.


* 
**차트(캔들)**: `GET /chart/ohlc?pool_id=1&interval=1m&limit=200`.


* 
**호가창/풀 상태**: `GET /market/orderbook`, `GET /pools`.


* 
**원장 데이터**: `GET /explorer/blocks`, `GET /explorer/pools/{pool_id}/history` (무결성이 보장된 학습용 원시 데이터).



### 2. 결과 업로드 (POST, `bot:ingest` 인증 필요)

* 
**매매 신호**: `/ai/ingest/signals` (buy/sell/hold 방향 및 신뢰도).


* 
**가격 예측**: `/ai/ingest/predictions` (1h/24h/7d 기간별 예측 가격).


* 
**시장 심리**: `/ai/ingest/sentiment` (강세/약세 스코어 및 보조 지표).



참고: 인서트 즉시 프론트엔드의 `/ai` 페이지에 반영됩니다.

---

## 🚀 빠른 시작 (Quick Start)

아래의 5분 파이썬 스크립트를 실행하면 즉시 프론트엔드 UI에 반영되는 미니 봇을 구동할 수 있습니다.

```python
# pip install httpx
import httpx, random, time, os

BASE = os.environ.get("SWAPGO_BASE", "http://localhost:8000")
KEY  = os.environ["SWAPGO_BOT_KEY"]
H = {"X-Bot-Key": KEY, "Content-Type": "application/json"}

with httpx.Client(base_url=BASE, headers=H, timeout=10) as cli:
    while True:
        symbols = ["BTC", "ETH"]

        # 1) 시그널 업로드
        items = [{
            "symbol": s,
            "side": random.choice(["buy", "sell", "hold"]),
            "confidence": round(random.uniform(0.4, 0.9), 2),
            "reason": "데모 신호 (random)",
            "expires_in_sec": 600,
        } for s in symbols]
        cli.post("/ai/ingest/signals", json={"items": items}).raise_for_status()

        # 2) 예측 업로드
        ticker = cli.get("/chart/ticker", params={"pool_id": 1}).json()
        last = float(ticker["data"]["last_price"]) if ticker["ok"] else 43000
        items = [{
            "symbol": "BTC",
            "horizon": h,
            "predicted_price_human": f"{last * (1 + random.uniform(-0.02, 0.02)):.2f}",
            "confidence": round(random.uniform(0.5, 0.85), 2),
            "model_tag": "demo-rng",
        } for h in ["1h", "24h", "7d"]]
        cli.post("/ai/ingest/predictions", json={"items": items}).raise_for_status()

        # 3) 심리 지표 업로드
        items = [{
            "symbol": s,
            "sentiment_score": random.randint(-60, 60),
            "rsi": round(random.uniform(20, 80), 1),
            "macd": round(random.uniform(-1, 1), 2),
        } for s in symbols]
        cli.post("/ai/ingest/sentiment", json={"items": items}).raise_for_status()

        time.sleep(5)

```



---

## 🛠 흔한 오류 (Troubleshooting)

* 
`401 UNAUTHORIZED`: X-Bot-Key 헤더가 누락되었거나 폐기된 키를 사용했습니다.


* 
`403 FORBIDDEN`: API 키에 필요한 스코프(예: `bot:ingest`)가 없습니다.


* `409 STALE_QUOTE`: 견적 조회 이후 풀 상태가 변경되었습니다. 재견적이 필요합니다.


* 
`422 VALIDATION_ERROR`: 데이터 타입이나 범위(예: horizon, confidence)를 위반했습니다.



## 📂 코드 레퍼런스

더 자세한 스키마 및 동작 방식은 다음 소스코드를 참고하세요:

* 엔드포인트 정의: `app/api/v1/ai.py` 


* 데이터 모델: `app/schemas/ai.py`, `app/db/models/ai_*.py` 


* 프론트엔드 UI 연동: `swapgo-frontend/src/app/ai/page.tsx`
