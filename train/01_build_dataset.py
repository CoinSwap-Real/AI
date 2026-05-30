"""
train/01_build_dataset.py — 5m Binance 데이터 → 4모델 학습 데이터셋 생성

[실행]
  python train/01_build_dataset.py --model all      # 4개 모두
  python train/01_build_dataset.py --model trade    # 거래봇만
  python train/01_build_dataset.py --model scalper
  python train/01_build_dataset.py --model swing
  python train/01_build_dataset.py --model longterm

[모델별 설계]
  ┌──────────┬────────┬──────────┬───────────────┬────────────┐
  │ 모델     │ seq_len│ horizon  │ horizon(실제) │ 최소 데이터 │
  ├──────────┼────────┼──────────┼───────────────┼────────────┤
  │ trade    │   30   │  1봉     │ 5분 후        │  2년       │
  │ scalper  │   10   │ 12봉     │ 1시간 후      │  2년       │
  │ swing    │   60   │ 288봉    │ 24시간 후     │  3년       │
  │ longterm │  120   │ 2016봉   │ 7일 후        │  3년       │
  └──────────┴────────┴──────────┴───────────────┴────────────┘
  모두 5m 캔들 기준. 봇 런타임도 CANDLE_INTERVAL=5m 사용.

[입력]
  data/raw/BTCUSDT_5m.csv   (00_download_binance.py 출력)
  data/raw/ETHUSDT_5m.csv

[출력]
  data/{model}/features_raw.npy     (T, 8)  — scaler 적용 전
  data/{model}/features_scaled.npy  (T, 8)  — scaler 적용 후 ← 학습 입력
  data/{model}/targets_btc.npy      (T,)    — BTC 로그수익률 타깃
  data/{model}/targets_eth.npy      (T,)    — ETH 로그수익률 타깃
  data/{model}/timestamps.npy       (T,)
  models/scaler.pkl                          — 봇 런타임용 (trade 실행 시 생성)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── CLI ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["trade", "scalper", "swing", "longterm", "all"],
    default="all",
)
args = parser.parse_args()

TARGETS = (
    ["trade", "scalper", "swing", "longterm"]
    if args.model == "all"
    else [args.model]
)

# ── 모델별 설정 ───────────────────────────────────────────────
MODEL_CFG: dict[str, dict] = {
    "trade":    {"seq_len": 30,  "horizon": 1,    "min_years": 2},
    "scalper":  {"seq_len": 10,  "horizon": 12,   "min_years": 2},
    "swing":    {"seq_len": 60,  "horizon": 288,  "min_years": 3},
    "longterm": {"seq_len": 120, "horizon": 2016, "min_years": 3},
}

# ── 피처 파라미터 — feature_builder.py 와 반드시 일치 ──────────
VOLAT_WINDOW = 10
BB_PERIOD    = 20
LOG_EPS      = 1e-9

Path("models").mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════
# 공통 함수
# ════════════════════════════════════════════════════════════

def load_csv(path: str) -> pd.DataFrame:
    """Binance 5m CSV 로드 및 컬럼 통일."""
    df = pd.read_csv(path)

    # timestamp: ms 정수 또는 ISO 문자열 모두 처리
    if df["timestamp"].dtype in ["int64", "float64"]:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # volume 컬럼 통일 (volume_base 또는 volume)
    if "volume_base" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"volume_base": "volume"})
    if "trades_count" not in df.columns and "trade_count" in df.columns:
        df = df.rename(columns={"trade_count": "trades_count"})

    df["close"]        = df["close"].ffill().astype(float)
    df["volume"]       = df["volume"].fillna(0).astype(float)
    df["trades_count"] = df["trades_count"].fillna(0).astype(int)
    return df.sort_values("timestamp").reset_index(drop=True)


def build_features(merged: pd.DataFrame) -> np.ndarray:
    """(T, 8) 피처 행렬 계산 — feature_builder.py 와 동일 로직."""
    close_btc = merged["close_btc"].values.astype(np.float64)
    close_eth = merged["close_eth"].values.astype(np.float64)
    vol_btc   = merged["volume_btc"].values.astype(np.float64)
    vol_eth   = merged["volume_eth"].values.astype(np.float64)
    trades    = merged["trades_count_btc"].values.astype(np.float64)
    T = len(merged)
    X = np.zeros((T, 8), dtype=np.float32)

    for i in range(1, T):
        ret_btc    = math.log(close_btc[i] / close_btc[i-1] + LOG_EPS)
        ret_eth    = math.log(close_eth[i] / close_eth[i-1] + LOG_EPS)
        log_pow    = math.log(vol_btc[i] + LOG_EPS)
        log_trades = math.log(trades[i] + LOG_EPS)

        s  = max(1, i - VOLAT_WINDOW + 1)
        rr = [math.log(close_btc[j]/close_btc[j-1]+LOG_EPS) for j in range(s, i+1)]
        volat = float(np.std(rr)) if len(rr) > 1 else 0.0

        bbs  = max(0, i - BB_PERIOD + 1)
        cw   = close_btc[bbs:i+1]
        sma  = cw.mean()
        bbw  = (4.0 * cw.std() / sma) if sma > 0 else 0.0

        X[i] = [ret_btc, ret_eth, log_pow, log_trades, volat, bbw,
                vol_btc[i], vol_eth[i]]

    return np.nan_to_num(X[1:], nan=0.0, posinf=0.0, neginf=0.0)


def build_targets(
    close_btc: np.ndarray,
    close_eth: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = len(close_btc) - horizon
    y_btc = np.array(
        [math.log(close_btc[i+horizon] / close_btc[i] + LOG_EPS) for i in range(valid)],
        dtype=np.float32,
    )
    y_eth = np.array(
        [math.log(close_eth[i+horizon] / close_eth[i] + LOG_EPS) for i in range(valid)],
        dtype=np.float32,
    )
    return y_btc, y_eth


# ════════════════════════════════════════════════════════════
# 데이터 로드 (1회)
# ════════════════════════════════════════════════════════════

print("=" * 56)
print("  SwapGo AI 데이터셋 빌더 — 5m Binance 기반")
print("=" * 56)

btc_path = "data/raw/BTCUSDT_5m.csv"
eth_path = "data/raw/ETHUSDT_5m.csv"

for p in [btc_path, eth_path]:
    if not Path(p).exists():
        raise FileNotFoundError(
            f"{p} 없음. 먼저 실행하세요:\n"
            "  python train/00_download_binance.py --years 3"
        )

print("\n[데이터 로드]")
btc = load_csv(btc_path)
eth = load_csv(eth_path)
print(f"  BTC: {len(btc):,}행  ({btc['timestamp'].iloc[0]:%Y-%m-%d} ~ {btc['timestamp'].iloc[-1]:%Y-%m-%d})")
print(f"  ETH: {len(eth):,}행  ({eth['timestamp'].iloc[0]:%Y-%m-%d} ~ {eth['timestamp'].iloc[-1]:%Y-%m-%d})")

# timestamp 기준 내부 조인
merged_all = pd.merge(
    btc[["timestamp","close","volume","trades_count"]]
      .add_suffix("_btc").rename(columns={"timestamp_btc":"timestamp"}),
    eth[["timestamp","close","volume"]]
      .add_suffix("_eth").rename(columns={"timestamp_eth":"timestamp"}),
    on="timestamp", how="inner",
).sort_values("timestamp").reset_index(drop=True)
print(f"  병합: {len(merged_all):,}행")

# 8피처 (1회만 계산 — 모든 모델 공유)
print("\n[8피처 계산 중...]")
features_all = build_features(merged_all)          # (T-1, 8)
timestamps_all = merged_all["timestamp"].values[1:]
close_btc_all  = merged_all["close_btc"].values[1:].astype(np.float64)
close_eth_all  = merged_all["close_eth"].values[1:].astype(np.float64)
print(f"  피처 shape: {features_all.shape}")


# ════════════════════════════════════════════════════════════
# 각 모델별 데이터셋 생성
# ════════════════════════════════════════════════════════════

scaler_saved = False   # scaler.pkl 은 trade 모델 처리 시 1회 저장

for model_name in TARGETS:
    cfg     = MODEL_CFG[model_name]
    horizon = cfg["horizon"]
    min_yrs = cfg["min_years"]
    print(f"\n{'─'*56}")
    print(f"  모델: {model_name}  seq={cfg['seq_len']}  horizon={horizon}봉 ({horizon*5}분 = "
          f"{'%.1f' % (horizon*5/60)}h)")

    # 기간 필터: min_years 전 이후 데이터만 사용
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.DateOffset(years=min_yrs)
    mask   = timestamps_all >= np.datetime64(cutoff)
    feat   = features_all[mask]
    ts     = timestamps_all[mask]
    cb     = close_btc_all[mask]
    ce     = close_eth_all[mask]
    print(f"  기간 필터 ({min_yrs}년): {len(feat):,}행")

    if len(feat) < cfg["seq_len"] + horizon + 1000:
        print(f"  ⚠️  데이터 부족 — 건너뜀 (다운로드 기간을 늘리세요)")
        continue

    # 타깃
    y_btc, y_eth = build_targets(cb, ce, horizon)
    feat = feat[:len(y_btc)]
    ts   = ts[:len(y_btc)]
    print(f"  최종 샘플: {len(feat):,}")
    print(f"  BTC 타깃  mean={y_btc.mean():.5f}  std={y_btc.std():.5f}")
    print(f"  ETH 타깃  mean={y_eth.mean():.5f}  std={y_eth.std():.5f}")

    # Scaler fit (훈련 80% 기준)
    te  = int(len(feat) * 0.8)
    scl = StandardScaler()
    scl.fit(feat[:te])
    feat_scaled = np.nan_to_num(
        scl.transform(feat).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )

    # 모델별 scaler 저장
    model_scaler_path = f"models/scaler_{model_name}.pkl"
    joblib.dump(scl, model_scaler_path)
    print(f"  scaler: {model_scaler_path}")

    # 봇 런타임용 scaler.pkl = trade 모델 scaler (5m 단기 분포 기준)
    if model_name == "trade" and not scaler_saved:
        joblib.dump(scl, "models/scaler.pkl")
        print(f"  scaler: models/scaler.pkl  (봇 런타임용, trade 기준)")
        scaler_saved = True

    # 저장
    out_dir = Path(f"data/{model_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "features_raw.npy",    feat)
    np.save(out_dir / "features_scaled.npy", feat_scaled)
    np.save(out_dir / "targets_btc.npy",     y_btc)
    np.save(out_dir / "targets_eth.npy",     y_eth)
    np.save(out_dir / "timestamps.npy",      ts)
    print(f"  저장: data/{model_name}/  ✅")

print(f"\n{'='*56}")
print("  데이터셋 생성 완료!")
print("  다음 단계: Google Colab 에 data/ 와 train/ 폴더를 업로드하고")
print("  train/colab_train.ipynb 를 실행하세요.")
print(f"{'='*56}")