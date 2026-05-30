# SwapGo AI 모델 학습 가이드

> **5m Binance 캔들 → GRU 4모델 → ONNX 변환 → 봇 배포**  
> Google Colab (무료 T4 GPU) 기준, 전체 소요 시간 약 **1~2시간**

---

## 모델 설계 개요

| 모델 | seq_len | 타깃 horizon | 학습 데이터 | Colab 예상 시간 |
|------|---------|-------------|-----------|--------------|
| `model_trade` | 30 | 5분 후 (1봉) | 2년 5m | ~5분 |
| `model_scalper` | 10 | 1시간 후 (12봉) | 2년 5m | ~3분 |
| `model_swing` | 60 | 24시간 후 (288봉) | 3년 5m | ~12분 |
| `model_longterm` | 120 | 7일 후 (2016봉) | 3년 5m | ~20분 |

- **왜 5m?** 1m 대비 학습 행 수 5배 감소, 봇 단일 설정(`CANDLE_INTERVAL=5m`) 유지
- **왜 1s/30m/1h 아닌 5m?** 1s·3m는 노이즈 과다, 15m 이상은 trade/scalper 단기 패턴 손실

---

## STEP 1 — 로컬에서 Binance 데이터 다운로드

> 인터넷이 연결된 **로컬 PC에서** 실행합니다.  
> Colab 에서 직접 다운로드하면 세션 종료 시 데이터가 사라집니다.

```bash
cd swapgo_engine

# 의존성 설치 (처음 한 번만)
pip install requests pandas scikit-learn joblib numpy

# 3년치 BTC+ETH 5m 데이터 다운로드 (약 30~60분, 파일 크기 ~200MB)
python train/00_download_binance.py --years 3
```

완료 후 확인:
```
data/raw/BTCUSDT_5m.csv   → ~315,000행
data/raw/ETHUSDT_5m.csv   → ~315,000행
```

---

## STEP 2 — 로컬에서 학습 데이터셋 생성

```bash
python train/01_build_dataset.py --model all
```

생성 파일:
```
data/
├── trade/
│   ├── features_scaled.npy  (학습 입력)
│   ├── targets_btc.npy
│   └── targets_eth.npy
├── scalper/   (동일 구조)
├── swing/     (동일 구조)
└── longterm/  (동일 구조)
models/
├── scaler.pkl           ← 봇 런타임용
├── scaler_trade.pkl
├── scaler_scalper.pkl
├── scaler_swing.pkl
└── scaler_longterm.pkl
```

---

## STEP 3 — Google Drive에 업로드

Colab 세션은 휘발성이므로 Google Drive 에 올려둡니다.

```
내 드라이브/
└── swapgo_train/
    ├── data/
    │   ├── trade/
    │   ├── scalper/
    │   ├── swing/
    │   └── longterm/
    ├── train/
    │   ├── 02_train_model.py
    │   ├── 03_export_onnx.py
    │   └── train_02_helpers.py
    └── models/         ← 빈 폴더 (학습 결과 저장용)
```

> **팁**: `data/` 폴더가 클 경우 zip 으로 압축 후 Colab에서 해제하면 빠릅니다.

---

## STEP 4 — Google Colab 학습

새 Colab 노트북을 열고 아래 셀을 순서대로 실행합니다.

### 셀 1: 환경 설정

```python
# GPU 연결 확인 (런타임 → 런타임 유형 변경 → T4 GPU 선택 후 실행)
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음 (CPU)")

# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 작업 디렉토리 설정
import os, sys
BASE = "/content/drive/MyDrive/swapgo_train"
os.chdir(BASE)
sys.path.insert(0, BASE)
print("작업 디렉토리:", os.getcwd())
```

### 셀 2: 의존성 설치

```python
!pip install onnx onnxruntime scikit-learn joblib -q
```

### 셀 3: 모델 학습 (4회 반복)

```python
# trade 모델 학습
!python train/02_train_model.py --horizon trade
```

```python
# scalper 모델 학습
!python train/02_train_model.py --horizon scalper
```

```python
# swing 모델 학습
!python train/02_train_model.py --horizon swing
```

```python
# longterm 모델 학습
!python train/02_train_model.py --horizon longterm
```

> 각 셀 실행 후 로그에서 **방향 정확도** 를 확인하세요.
> 목표: BTC 55% 이상, ETH 53% 이상 (랜덤 기준선 50%)

### 셀 4: ONNX 변환 및 검증

```python
!python train/03_export_onnx.py --horizon trade
!python train/03_export_onnx.py --horizon scalper
!python train/03_export_onnx.py --horizon swing
!python train/03_export_onnx.py --horizon longterm
```

변환 완료 후 `models/` 에 다음 파일이 생성됩니다:
```
models/
├── model_trade.onnx
├── model_scalper.onnx
├── model_swing.onnx
├── model_longterm.onnx
└── scaler.pkl          ← 이미 있으면 재생성 불필요
```

### 셀 5: Drive 동기화 확인

```python
import os
for f in os.listdir("models"):
    size = os.path.getsize(f"models/{f}") / 1024
    print(f"  {f}: {size:.1f} KB")
```

---

## STEP 5 — 로컬 서버에 모델 배포

Google Drive 에서 `models/` 폴더를 다운로드해서 `swapgo_engine/models/` 에 복사합니다.

```
swapgo_engine/
└── models/
    ├── model_trade.onnx      ✅
    ├── model_scalper.onnx    ✅
    ├── model_swing.onnx      ✅
    ├── model_longterm.onnx   ✅
    └── scaler.pkl            ✅
```

`.env` 설정 확인:
```env
CANDLE_INTERVAL=5m          # 반드시 5m (학습 데이터와 동일)
SEQ_LEN_TRADE=30
SEQ_LEN_SCALPER=10
SEQ_LEN_SWING=60
SEQ_LEN_LONGTERM=120
```

런처에서 AI 엔진 시작하면 완료입니다.

---

## 학습 품질 기준

| 지표 | 기준값 | 설명 |
|------|-------|------|
| 방향 정확도 | ≥ 55% | 매수/매도 방향 예측 정확도 |
| 테스트 RMSE | 타깃 std의 70% 이하 | 실제 수익률 대비 오차 |
| Val Loss 수렴 | 20에폭 이내 개선 없음 | Early stopping 발동 |

방향 정확도가 52% 미만이면 학습 데이터 기간을 늘리거나 하이퍼파라미터를 조정하세요.

---

## 하이퍼파라미터 조정 (02_train_model.py)

```python
HIDDEN_SIZE  = 64    # 128로 늘리면 정확도↑, 학습 시간↑
NUM_LAYERS   = 2     # 3으로 늘리면 표현력↑, 과적합 위험↑
DROPOUT      = 0.2   # 과적합 시 0.3으로 증가
BATCH_SIZE   = 512   # 메모리 부족 시 256으로 감소
EPOCHS       = 100   # Early stopping 이 자동 조절
LR           = 1e-3  # 0.0005로 줄이면 안정적 학습
PATIENCE     = 10    # Early stopping 인내심
```

---

## 자주 발생하는 오류

**`FileNotFoundError: data/raw/BTCUSDT_5m.csv`**
→ STEP 1 실행 누락. `00_download_binance.py` 먼저 실행하세요.

**`AssertionError: 데이터가 너무 짧습니다`**
→ `--years 3` 으로 재다운로드 후 `01_build_dataset.py` 재실행.

**`CUDA out of memory`**
→ `BATCH_SIZE = 256` 으로 줄이거나, Colab Pro (A100) 사용.

**방향 정확도 50%대 초반**
→ 정상입니다. 금융 시계열 예측은 55% 달성도 우수한 수준입니다.

**ONNX 수치 차이 > 1e-5**
→ `opset_version=15` 로 낮춰서 재시도.