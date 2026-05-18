# SwapGo AI Bot System (2026.05.18 기준)

GRU 기반 앙상블 모델로 실시간 가상 거래 환경을 조성하고, SwapGo 프론트엔드 대시보드에 AI 투자 신호를 공급하는 비동기 봇 서버입니다.

---

## 목차

- [동작 원리](#동작-원리)
- [시스템 구조](#시스템-구조)
- [AI 모델](#ai-모델)
- [디렉토리 구조](#디렉토리-구조)
- [API 목록](#api-목록)
- [빠른 시작](#빠른-시작)
- [환경변수 레퍼런스](#환경변수-레퍼런스)
- [모델 학습](#모델-학습)

---

## 동작 원리

이 서버는 두 가지 독립적인 목적을 동시에 수행합니다.

**시스템 A — 대시보드 신호 봇**

SwapGo `/chart/ohlc` API에서 BTC·ETH 캔들을 수집하고, 8개의 기술적 피처로 변환한 뒤 GRU 모델 3개(scalper / swing / longterm)를 통해 1h·24h·7d 가격을 예측합니다. 예측 결과는 60초 주기로 SwapGo ingest API에 업로드되어 프론트엔드 AI 대시보드에 표시됩니다.

**시스템 B — 가상 시장 조성 봇**

별도의 단기 예측 모델(model_trade)이 캔들 완성마다 즉시 추론을 실행합니다. 예측 방향이 임계값을 넘으면 `BotA_Trade`가 `/swap/execute`를 직접 호출하고, `BotB_Noise`는 무작위 방향으로 지속적으로 스왑을 실행해 유동성을 공급합니다. 두 봇이 같은 풀에서 충돌하며 지속적인 가상 거래 환경을 조성합니다.

### 데이터 흐름 요약

```
[SwapGo 백엔드]
  GET /chart/ohlc (BTC + ETH)
  WS  ohlc:{pool_id}:1m
          │
          ▼
  ┌─────────────────────────────────────────────────────┐
  │                   CandleCache                       │
  │  BTC 버퍼  │  ETH 버퍼(TTL 60s)  │  피처 캐시(5s)   │
  └────────────────────┬────────────────────────────────┘
                       │ (N, 8) scaler 적용 피처 행렬
              ┌────────┴────────┐
              ▼                 ▼
    [시스템 A]              [시스템 B]
    scalper  → 1h 예측      model_trade → BTC/ETH log_ret
    swing    → 24h 예측         │
    longterm → 7d 예측      |ret| > threshold?
              │                 │
              ▼                 ├── YES → /swap/execute (BotA_Trade)
    POST /ai/ingest/*           └── NO  → 건너뜀
    (signals/predictions        BotB_Noise → 무작위 /swap/execute
     /sentiment)
```

---

## 시스템 구조

### 컴포넌트 관계도

```
main.py (FastAPI + lifespan)
  │
  ├── CandleCache           # 공유 캔들 버퍼 + 피처 캐시
  │     ├── FeatureBuilder  # 캔들 → 8피처 변환 + scaler.pkl 적용
  │     └── SwapGoClient    # REST API 클라이언트
  │
  ├── CandleEventBus        # 단일 WS 연결 → 다중 구독자 팬아웃
  │     ├── → CandleCache.push_btc()
  │     ├── → BotB_WS.on_candle()
  │     └── → BotA_Trade.on_candle()
  │
  ├── [시스템 A]
  │     ├── BotA_Ingest     # 60s 주기 REST 폴링 → ingest 업로드
  │     └── BotB_WS         # WS 실시간 ingest 보조 (선택)
  │
  ├── [시스템 B]
  │     ├── BotA_Trade      # AI 예측 기반 즉시 스왑 실행
  │     └── BotB_Noise      # 랜덤 노이즈 유동성 공급
  │
  └── [시스템 C]
        └── BotC_LP         # 시장조성 LP 봇 (선택)
```

### 경량화 설계 원칙

| 항목 | 이전 구조 | 현재 구조 |
|------|-----------|-----------|
| WebSocket 연결 수 | 봇별 독립 (2개+) | `CandleEventBus` 공유 (1개) |
| ETH 캔들 REST 호출 | 캔들마다 봇별 개별 호출 | `CandleCache` TTL 기반 (60s) |
| 피처 계산 횟수 | 봇별 캔들마다 재계산 | 캔들당 1회, 5초 캐시 재사용 |
| 프론트엔드 통신 | REST 반복 폴링 | SSE 단방향 스트림 |

---

## AI 모델

4개의 ONNX 모델과 1개의 scaler를 `models/` 디렉토리에 배치합니다.

| 파일 | seq_len | 학습 타깃 | 담당 봇 | 용도 |
|------|---------|-----------|---------|------|
| `model_trade.onnx` | 30 | 1분 후 로그수익률 | `BotA_Trade` | 시스템 B 즉시 거래 |
| `model_scalper.onnx` | 10 | 60분 후 로그수익률 | `BotA_Ingest` | 1h 신호 |
| `model_swing.onnx` | 60 | 1440분 후 로그수익률 | `BotA_Ingest` | 24h 신호 |
| `model_longterm.onnx` | 120 | 10080분 후 로그수익률 | `BotA_Ingest` | 7d 신호 |
| `scaler.pkl` | — | 8피처 StandardScaler | 4개 모델 공통 | 입력 정규화 |

**모델 입출력 스펙 (공통)**

```
입력:  (1, seq_len, 8)  — scaler 적용된 8피처 시퀀스
출력:  (1, 2)           — [BTC 로그수익률, ETH 로그수익률]
```

**8개 입력 피처 (순서 고정)**

| 인덱스 | 피처명 | 계산 방법 |
|--------|--------|-----------|
| 0 | `Ret_btc` | `log(close_t / close_t-1)` |
| 1 | `Ret_eth` | ETH 풀 동일 계산 |
| 2 | `Log_Pow_btc` | `log(volume_base + ε)` (체결강도 근사) |
| 3 | `Log_Trades_btc` | `log(trades_count + ε)` |
| 4 | `Volat_btc` | rolling std(log_returns, window=10) |
| 5 | `BB_Width` | `4σ / SMA` (볼린저 밴드 너비, period=20) |
| 6 | `Volume_btc` | `volume_base` (raw) |
| 7 | `Volume_eth` | ETH 풀 `volume_base` (raw) |

---

## 디렉토리 구조

```
project/
├── main.py                          # FastAPI 진입점, 생명주기 관리
├── config.py                        # 전역 설정 (pydantic-settings)
├── .env.example                     # 환경변수 템플릿
│
├── core/
│   ├── swapgo_client.py             # SwapGo REST API 클라이언트
│   └── ws_client.py                 # WebSocket 클라이언트 (자동 재접속)
│
├── ai/
│   ├── feature_builder.py           # 캔들 → 8피처 변환 + scaler 적용
│   └── ai_engine.py                 # ONNX 추론 엔진 (EMA 평활)
│
├── services/
│   ├── candle_cache.py              # 공유 캔들 버퍼 + 피처 캐시
│   ├── event_bus.py                 # 단일 WS → 다중 구독자 팬아웃
│   └── ingest_service.py            # AI 결과 → ingest API 변환·업로드
│
├── bots/
│   ├── bot_a_ingest.py              # [시스템 A] REST 폴링 ingest 봇
│   ├── bot_b_ws.py                  # [시스템 A] WS 실시간 ingest 보조
│   ├── bot_a_trade.py               # [시스템 B] AI 기반 즉시 거래 봇
│   ├── bot_b_noise.py               # [시스템 B] 랜덤 노이즈 유동성 봇
│   └── bot_c_lp.py                  # [시스템 C] 시장조성 LP 봇 (선택)
│
├── schemas/
│   ├── models.py                    # ingest 요청/응답 + 내부 도메인 모델
│   └── responses.py                 # REST·SSE 응답 스키마
│
├── models/                          # ONNX 모델 배치 위치 (Git 제외)
│   ├── model_trade.onnx
│   ├── model_scalper.onnx
│   ├── model_swing.onnx
│   ├── model_longterm.onnx
│   └── scaler.pkl
│
└── train/                           # 모델 학습 스크립트
    ├── 01_build_dataset.py          # 캔들 데이터 → 피처·타깃 생성
    ├── 02_train_model.py            # GRU 학습 (--horizon trade|1h|24h|7d)
    ├── 03_export_onnx.py            # PyTorch → ONNX 변환·검증
    └── requirements_train.txt       # 학습 전용 의존성
```

---

## API 목록

서버 기본 포트: `9000`  
자동 문서: [`http://localhost:9000/docs`](http://localhost:9000/docs)

### REST

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/health` | 태스크 생존 상태 확인. `status: ok \| degraded` |
| `GET` | `/status` | 전체 시스템 상태 (캐시·엔진·봇별 통계) |
| `GET` | `/ai/info` | ingest AI 엔진 3개의 워밍업·예측·추론횟수 |

#### `/health` 응답 예시

```json
{
  "status": "ok",
  "tasks": [
    { "name": "bot_a_ingest", "running": true, "failed": false },
    { "name": "event_bus",    "running": true, "failed": false },
    { "name": "bot_a_trade",  "running": true, "failed": false },
    { "name": "bot_b_noise",  "running": true, "failed": false }
  ]
}
```

#### `/status` 응답 구조

```json
{
  "config":         { "pool_id": 1, "eth_pool_id": 2, "candle_interval": "1m", ... },
  "candle_cache":   { "btc_count": 200, "eth_count": 200, "eth_age_sec": 12.4, ... },
  "event_bus":      { "subscriber_count": 3, "candle_count": 847, ... },
  "ingest_engines": [ { "model": "Scalper", "is_warm": true, "ema_btc": 0.00012, ... }, ... ],
  "trade_engine":   { "model": "Trade", "is_warm": true, "ema_btc": -0.00008, ... },
  "system_a":       { "bot_a_ingest": { "tick_count": 42, ... }, "bot_b_ws": null },
  "system_b":       { "bot_a_trade": { "trade_count": 18, ... }, "bot_b_noise": { "trade_count": 231, ... } },
  "system_c":       { "bot_c_lp": null }
}
```

---

### SSE (Server-Sent Events)

프론트엔드 폴링 없이 실시간 데이터를 수신합니다.

#### `GET /stream/status` — 1초 주기 상태 스트림

봇 상태 요약을 1초마다 푸시합니다.

```javascript
const es = new EventSource('http://localhost:9000/stream/status');
es.onmessage = (e) => {
  const d = JSON.parse(e.data);
  // d.ts, d.btc_close, d.eth_close,
  // d.ema_btc, d.ema_eth,
  // d.trade_count_a, d.trade_count_b,
  // d.candle_count, d.is_warm
};
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `ts` | `float` | UNIX timestamp |
| `btc_close` | `float` | 최근 BTC 캔들 종가 |
| `eth_close` | `float` | 최근 ETH 캔들 종가 |
| `ema_btc` | `float\|null` | 거래 모델 EMA 예측 (BTC 로그수익률) |
| `ema_eth` | `float\|null` | 거래 모델 EMA 예측 (ETH 로그수익률) |
| `trade_count_a` | `int` | `BotA_Trade` 누적 거래 수 |
| `trade_count_b` | `int` | `BotB_Noise` 누적 거래 수 |
| `candle_count` | `int` | EventBus 수신 캔들 수 |
| `is_warm` | `bool` | 거래 모델 워밍업 완료 여부 |

#### `GET /stream/trades` — 거래 이벤트 즉시 푸시

`BotA_Trade` 또는 `BotB_Noise`의 스왑이 완료될 때마다 이벤트를 전송합니다.  
연결 유지를 위해 30초마다 `: keepalive` 주석 라인을 전송합니다.

```javascript
const es = new EventSource('http://localhost:9000/stream/trades');
es.onmessage = (e) => {
  const d = JSON.parse(e.data);
  // d.ts, d.bot, d.side, d.amount_in, d.amount_out,
  // d.pool_id, d.slippage
};
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `ts` | `float` | 거래 완료 시각 (UNIX timestamp) |
| `bot` | `string` | `"BotA_Trade"` 또는 `"BotB_Noise"` |
| `side` | `string` | `"BUY"` 또는 `"SELL"` |
| `amount_in` | `string` | 투입 수량 (사람단위) |
| `amount_out` | `string` | 수령 수량 (사람단위) |
| `pool_id` | `int` | 스왑 대상 풀 ID |
| `slippage` | `string\|null` | `"safe"`, `"warning"`, `"danger"` |

---

## 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정

```bash
cp .env.example .env
# .env 파일을 열어 BOT_KEY와 SWAPGO_BASE_URL 입력
```

### 3. 모델 배치

학습된 ONNX 파일과 scaler를 `models/` 디렉토리에 배치합니다.  
파일이 없으면 자동으로 **Mock 모드**로 동작합니다 (전 틱 수익률을 예측값으로 대체, 개발 환경 전용).

```
models/
├── model_trade.onnx
├── model_scalper.onnx
├── model_swing.onnx
├── model_longterm.onnx
└── scaler.pkl
```

### 4. 서버 실행

```bash
# 기본 (시스템 A만 활성화)
uvicorn main:app --host 0.0.0.0 --port 9000

# 가상 시장 조성 포함 (시스템 A + B)
ENABLE_TRADE_BOTS=true uvicorn main:app --host 0.0.0.0 --port 9000

# WebSocket 실시간 모드
USE_WEBSOCKET=true ENABLE_TRADE_BOTS=true uvicorn main:app --port 9000
```

---

## 환경변수 레퍼런스

`.env.example` 파일을 기준으로 합니다. 필수 항목만 표시하며 나머지는 기본값이 있습니다.

### 필수

| 변수 | 설명 |
|------|------|
| `BOT_KEY` | SwapGo 봇 API 키 (콘솔에서 발급) |
| `SWAPGO_BASE_URL` | SwapGo 백엔드 서버 주소 |

### 풀 / 심볼

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `POOL_ID` | `1` | BTC 풀 ID |
| `ETH_POOL_ID` | `2` | ETH 풀 ID |
| `SYMBOLS` | `["BTC","ETH"]` | ingest 대상 심볼 |

### 시스템 A — 신호 봇

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `CANDLE_INTERVAL` | `1m` | 캔들 주기 |
| `CANDLE_LIMIT` | `200` | 수집 캔들 수 |
| `INGEST_INTERVAL_SEC` | `60` | ingest 업로드 주기 (최소 30) |
| `USE_WEBSOCKET` | `false` | WS 실시간 모드 활성화 |

### 시스템 B — 거래 봇

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `ENABLE_TRADE_BOTS` | `false` | 시스템 B 활성화 여부 |
| `TRADE_MIN_LOG_RETURN` | `0.0002` | 거래 트리거 최소 예측 변화율 |
| `TRADE_MIN_CONFIDENCE` | `0.55` | 거래 트리거 최소 신뢰도 |
| `TRADE_EXECUTE_AMOUNT_HUMAN` | `5` | BotA_Trade 1회 거래 수량 |
| `TRADE_COOLDOWN_SEC` | `5.0` | 연속 거래 최소 대기 (초) |
| `TRADE_POLL_INTERVAL_SEC` | `30.0` | REST 폴링 주기 (WS 비활성 시) |
| `NOISE_INTERVAL_MIN` | `3.0` | 노이즈 봇 최소 대기 (초) |
| `NOISE_INTERVAL_MAX` | `8.0` | 노이즈 봇 최대 대기 (초) |

### AI 모델

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `SCALER_PATH` | `models/scaler.pkl` | 4개 모델 공통 scaler |
| `MODEL_TRADE_PATH` | `models/model_trade.onnx` | 거래 봇 모델 |
| `MODEL_SCALPER_PATH` | `models/model_scalper.onnx` | 1h 신호 모델 |
| `MODEL_SWING_PATH` | `models/model_swing.onnx` | 24h 신호 모델 |
| `MODEL_LONGTERM_PATH` | `models/model_longterm.onnx` | 7d 신호 모델 |
| `EMA_ALPHA` | `0.3` | 예측값 EMA 평활 계수 |

---

## 모델 학습

`train/` 디렉토리의 스크립트를 순서대로 실행합니다.

```bash
pip install -r train/requirements_train.txt
```

### Step 1 — 데이터셋 구축

`train/01_build_dataset.py` 상단의 `HORIZON_CANDLES`와 `OUTPUT_TAG`를 수정해 4번 실행합니다.  
첫 실행(`trade`)에서 `models/scaler.pkl`이 생성되며 이후 실행에서 공유합니다.

```bash
# HORIZON_CANDLES=1,    OUTPUT_TAG="trade"
# HORIZON_CANDLES=60,   OUTPUT_TAG="1h"
# HORIZON_CANDLES=1440, OUTPUT_TAG="24h"
# HORIZON_CANDLES=10080,OUTPUT_TAG="7d"
python train/01_build_dataset.py
```

### Step 2 — 모델 학습

```bash
python train/02_train_model.py --horizon trade
python train/02_train_model.py --horizon 1h
python train/02_train_model.py --horizon 24h
python train/02_train_model.py --horizon 7d
```

### Step 3 — ONNX 변환 및 검증

```bash
python train/03_export_onnx.py --horizon trade
python train/03_export_onnx.py --horizon 1h
python train/03_export_onnx.py --horizon 24h
python train/03_export_onnx.py --horizon 7d
```

변환 스크립트는 다음 세 항목을 자동 검증합니다.

- PyTorch 출력과 ONNX 출력의 수치 차이 < 1e-5
- ONNX 입출력 shape `(1, seq_len, 8)` → `(1, 2)` 일치
- 실제 봇 코드(`FeatureBuilder` + `AIEngine`) 엔드투엔드 통과

---

## 주의사항

- `.env` 파일은 절대 Git에 커밋하지 마세요. `.gitignore`에 반드시 포함해야 합니다.
- `scaler.pkl`은 학습 데이터의 앞 80%에만 `fit`하고 저장해야 data leakage를 방지할 수 있습니다.
- `FeatureBuilder`의 `_VOLAT_WINDOW`, `_BB_PERIOD` 상수와 학습 스크립트의 동일 파라미터가 반드시 일치해야 합니다. 불일치 시 모델 성능이 급격히 저하됩니다.
- Mock 모드(모델 파일 없음)는 개발·테스트 전용입니다. 프로덕션에서는 반드시 실제 ONNX 파일을 배치하세요.
