"""
train/01_build_dataset.py — 학습 데이터셋 구축

[실행 순서]
  python train/01_build_dataset.py

[출력 파일]
  data/features_raw.npy     shape (T, 8)   scaler 적용 전 원본 피처
  data/features_scaled.npy  shape (T, 8)   scaler 적용 후 정규화 피처  ← 모델 입력
  data/targets_btc.npy      shape (T,)     BTC 로그수익률 타깃
  data/targets_eth.npy      shape (T,)     ETH 로그수익률 타깃
  data/timestamps.npy       shape (T,)     각 행의 타임스탬프 (str)
  models/scaler.pkl                        봇 추론에 사용할 scaler

[타깃 정의]
  모델은 "현재 시점 t 에서 HORIZON 뒤의 로그수익률"을 예측합니다.
  HORIZON_CANDLES 를 바꾸면 scalper/swing/longterm 각각의 타깃이 됩니다.

  현재 설정 (candle_interval=1m 기준):
    scalper  → HORIZON_CANDLES =  60  (1h  = 60분)
    swing    → HORIZON_CANDLES = 1440 (24h = 1440분)
    longterm → HORIZON_CANDLES = 10080 (7d  = 10080분)

  같은 스크립트를 HORIZON_CANDLES 만 바꿔서 3번 실행하면 됩니다.
  출력 디렉토리를 data/1h/, data/24h/, data/7d/ 로 분리하세요.
"""

import math
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── 설정 ──────────────────────────────────────────────────────
# 이 값만 바꿔서 4번 실행:
#
#  모델          HORIZON_CANDLES  OUTPUT_TAG  용도
#  ──────────────────────────────────────────────────────────
#  model_trade       1            "trade"     시스템 B 거래 봇 (다음 1분 캔들 예측)
#  model_scalper    60            "1h"        시스템 A 1h 신호
#  model_swing    1440            "24h"       시스템 A 24h 신호
#  model_longterm 10080           "7d"        시스템 A 7d 신호
#
# scaler.pkl 은 OUTPUT_TAG="trade" 실행 시 최초 1회만 저장.
# 이후 실행에서는 저장 라인을 주석 처리하거나 덮어쓰기 허용.
HORIZON_CANDLES = 1            # ← 여기를 바꿔 실행
OUTPUT_TAG      = "trade"      # ← 여기를 바꿔 실행

# 데이터 소스 (CSV 경로 또는 API 선택)
BTC_CSV = "data/raw/btc_1m.csv"   # 컬럼: timestamp, open, high, low, close, volume_base, trades_count
ETH_CSV = "data/raw/eth_1m.csv"   # 동일 컬럼

# 피처 계산 파라미터 — feature_builder.py 와 반드시 일치
VOLAT_WINDOW = 10
BB_PERIOD    = 20
LOG_EPS      = 1e-9

# 출력 경로
OUT_DIR = Path(f"data/{OUTPUT_TAG}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
Path("models").mkdir(exist_ok=True)
# ─────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════
# Step 1. 데이터 로드 및 정렬
# ════════════════════════════════════════════════════════════

def load_candles(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    # 결측값 전진 채움
    df["close"]        = df["close"].ffill()
    df["volume_base"]  = df["volume_base"].fillna(0)
    df["trades_count"] = df["trades_count"].fillna(0)
    return df


print("[1/5] 캔들 데이터 로드 중...")
btc = load_candles(BTC_CSV)
eth = load_candles(ETH_CSV)

# timestamp 기준 내부 조인 — 두 심볼에 모두 캔들이 있는 시점만 사용
merged = pd.merge(
    btc[["timestamp", "close", "volume_base", "trades_count"]].add_suffix("_btc").rename(columns={"timestamp_btc": "timestamp"}),
    eth[["timestamp", "close", "volume_base"]].add_suffix("_eth").rename(columns={"timestamp_eth": "timestamp"}),
    on="timestamp",
    how="inner",
).sort_values("timestamp").reset_index(drop=True)

print(f"  병합 후 총 캔들 수: {len(merged):,} (BTC {len(btc):,} / ETH {len(eth):,})")
assert len(merged) > BB_PERIOD + VOLAT_WINDOW + HORIZON_CANDLES + 100, \
    "데이터가 너무 짧습니다."


# ════════════════════════════════════════════════════════════
# Step 2. 피처 계산 — feature_builder.py 와 동일한 로직
# ════════════════════════════════════════════════════════════

print("[2/5] 8피처 계산 중...")

close_btc = merged["close_btc"].values.astype(np.float64)
close_eth = merged["close_eth"].values.astype(np.float64)
vol_btc   = merged["volume_base_btc"].values.astype(np.float64)
vol_eth   = merged["volume_base_eth"].values.astype(np.float64)
trades    = merged["trades_count_btc"].values.astype(np.float64)

T = len(merged)
features_raw = np.zeros((T, 8), dtype=np.float32)

for i in range(1, T):
    # ① Ret_btc
    ret_btc = math.log(close_btc[i] / close_btc[i-1] + LOG_EPS)

    # ② Ret_eth
    ret_eth = math.log(close_eth[i] / close_eth[i-1] + LOG_EPS)

    # ③ Log_Pow_btc (volume 근사)
    log_pow = math.log(vol_btc[i] + LOG_EPS)

    # ④ Log_Trades_btc
    log_trades = math.log(trades[i] + LOG_EPS)

    # ⑤ Volat_btc — rolling std of log returns
    start = max(1, i - VOLAT_WINDOW + 1)
    recent_rets = [
        math.log(close_btc[j] / close_btc[j-1] + LOG_EPS)
        for j in range(start, i + 1)
    ]
    volat = float(np.std(recent_rets)) if len(recent_rets) > 1 else 0.0

    # ⑥ BB_Width
    bb_start = max(0, i - BB_PERIOD + 1)
    closes_w = close_btc[bb_start: i + 1]
    sma = closes_w.mean()
    bb_width = (4.0 * closes_w.std() / sma) if sma > 0 else 0.0

    # ⑦ Volume_btc
    volume_btc = vol_btc[i]

    # ⑧ Volume_eth
    volume_eth = vol_eth[i]

    features_raw[i] = [ret_btc, ret_eth, log_pow, log_trades,
                       volat, bb_width, volume_btc, volume_eth]

# i=0 은 이전 봉 없으므로 첫 행 제거
features_raw = features_raw[1:]
timestamps   = merged["timestamp"].values[1:]
close_btc_t  = close_btc[1:]
close_eth_t  = close_eth[1:]

# NaN / Inf 제거
features_raw = np.nan_to_num(features_raw, nan=0.0, posinf=0.0, neginf=0.0)
print(f"  피처 행렬 shape: {features_raw.shape}")


# ════════════════════════════════════════════════════════════
# Step 3. 타깃 계산
#   y_btc[t] = log(close_btc[t + HORIZON] / close_btc[t])
#   y_eth[t] = log(close_eth[t + HORIZON] / close_eth[t])
# ════════════════════════════════════════════════════════════

print(f"[3/5] 타깃 계산 중 (horizon={HORIZON_CANDLES} 캔들)...")

T_feat = len(features_raw)
# 타깃을 만들 수 있는 마지막 인덱스 = T_feat - HORIZON_CANDLES - 1
valid_end = T_feat - HORIZON_CANDLES

targets_btc = np.zeros(valid_end, dtype=np.float32)
targets_eth = np.zeros(valid_end, dtype=np.float32)

for i in range(valid_end):
    future_btc = close_btc_t[i + HORIZON_CANDLES]
    future_eth = close_eth_t[i + HORIZON_CANDLES]
    targets_btc[i] = math.log(future_btc / close_btc_t[i] + LOG_EPS)
    targets_eth[i] = math.log(future_eth / close_eth_t[i] + LOG_EPS)

# 피처도 동일 구간으로 자르기
features_raw  = features_raw[:valid_end]
timestamps    = timestamps[:valid_end]

print(f"  최종 샘플 수: {len(features_raw):,}")
print(f"  BTC 타깃 분포 — mean={targets_btc.mean():.5f}  std={targets_btc.std():.5f}")
print(f"  ETH 타깃 분포 — mean={targets_eth.mean():.5f}  std={targets_eth.std():.5f}")


# ════════════════════════════════════════════════════════════
# Step 4. Scaler 학습 및 저장
#   훈련 데이터(앞 80%)에만 fit → 이후 transform
# ════════════════════════════════════════════════════════════

print("[4/5] Scaler 학습 및 저장 중...")

train_end = int(len(features_raw) * 0.8)
scaler = StandardScaler()
scaler.fit(features_raw[:train_end])

features_scaled = scaler.transform(features_raw).astype(np.float32)
features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)

# 봇에서 사용할 scaler 저장 (horizon 공통 — 피처가 동일하므로)
scaler_path = "models/scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"  scaler 저장: {scaler_path}")
print(f"  scaler mean (8): {scaler.mean_.round(4)}")
print(f"  scaler std  (8): {scaler.scale_.round(4)}")


# ════════════════════════════════════════════════════════════
# Step 5. 저장
# ════════════════════════════════════════════════════════════

print("[5/5] 데이터셋 저장 중...")

np.save(OUT_DIR / "features_raw.npy",    features_raw)
np.save(OUT_DIR / "features_scaled.npy", features_scaled)
np.save(OUT_DIR / "targets_btc.npy",     targets_btc)
np.save(OUT_DIR / "targets_eth.npy",     targets_eth)
np.save(OUT_DIR / "timestamps.npy",      timestamps)

print(f"\n✅ 완료! 저장 위치: {OUT_DIR}/")
print(f"  features_raw.npy    {features_raw.shape}")
print(f"  features_scaled.npy {features_scaled.shape}")
print(f"  targets_btc.npy     {targets_btc.shape}")
print(f"  targets_eth.npy     {targets_eth.shape}")
print()
print("  다음 단계: python train/02_train_model.py --horizon 1h")