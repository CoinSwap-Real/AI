"""
train/00_download_binance.py — Binance Vision 5m 캔들 데이터 다운로드

[사용법]
  python train/00_download_binance.py           # 기본: 최근 3년치
  python train/00_download_binance.py --years 2  # 최근 2년치

[출력]
  data/raw/BTCUSDT_5m.csv
  data/raw/ETHUSDT_5m.csv
  컬럼: timestamp(ms), open, high, low, close, volume, trades_count

[Binance Vision 주소]
  https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/5m/
  월별 ZIP 파일을 순서대로 다운로드 후 병합합니다.
  인터넷 연결만 있으면 됩니다 (API 키 불필요).
"""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# ── 설정 ─────────────────────────────────────────────────────
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "5m"
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
OUT_DIR = Path("data/raw")
RETRY = 3
RETRY_SEC = 5.0

# Binance Klines 컬럼 (공식 순서)
COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades_count",
    "taker_base",
    "taker_quote",
    "ignore",
]

parser = argparse.ArgumentParser()
parser.add_argument("--years", type=int, default=3, help="다운로드할 연도 수 (기본: 3)")
args = parser.parse_args()


def _months_back(n_years: int) -> list[tuple[int, int]]:
    """현재 월 기준 n_years 전부터 지난달까지의 (year, month) 목록."""
    now = datetime.utcnow().replace(day=1)
    start = now - timedelta(days=365 * n_years)
    result = []
    cur = start.replace(day=1)
    end = (now - timedelta(days=1)).replace(day=1)
    while cur <= end:
        result.append((cur.year, cur.month))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return result


def download_month(symbol: str, year: int, month: int) -> pd.DataFrame | None:
    fn = f"{symbol}-{INTERVAL}-{year:04d}-{month:02d}.zip"
    url = f"{BASE_URL}/{symbol}/{INTERVAL}/{fn}"

    for attempt in range(1, RETRY + 1):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                print(f"    없음(404): {fn}")
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, header=None, names=COLS)
            return df
        except Exception as e:
            print(f"    시도 {attempt}/{RETRY} 실패: {e}")
            if attempt < RETRY:
                time.sleep(RETRY_SEC)
    return None


OUT_DIR.mkdir(parents=True, exist_ok=True)
months = _months_back(args.years)
print(
    f"다운로드 기간: {months[0][0]}-{months[0][1]:02d} ~ "
    f"{months[-1][0]}-{months[-1][1]:02d}  ({len(months)}개월)"
)

for symbol in SYMBOLS:
    print(f"\n[{symbol}] 다운로드 시작...")
    frames: list[pd.DataFrame] = []

    for year, month in months:
        tag = f"{year}-{month:02d}"
        df = download_month(symbol, year, month)
        if df is None:
            continue
        frames.append(df)
        print(f"  {tag}  {len(df):,}행")
        time.sleep(0.2)  # 서버 부하 방지

    if not frames:
        print(f"  ⚠️  {symbol} 데이터 없음")
        continue

    merged = pd.concat(frames, ignore_index=True)

    # 정제
    out = pd.DataFrame(
        {
            "timestamp": merged["open_time"].astype("int64"),  # ms
            "open": merged["open"].astype(float),
            "high": merged["high"].astype(float),
            "low": merged["low"].astype(float),
            "close": merged["close"].astype(float),
            "volume": merged["volume"].astype(float),  # = volume_base
            "trades_count": merged["trades_count"].astype(int),
        }
    )

    # 🚀 (수정된 핵심 코드) 단위 강제 통일 (us -> ms)
    # 3조(대략 2065년)보다 큰 값이면 마이크로초(us)로 간주하고 1000으로 나누어 밀리초(ms)로 맞춥니다.
    out.loc[out["timestamp"] > 3000000000000, "timestamp"] = out["timestamp"] // 1000

    out = (
        out.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    )

    path = OUT_DIR / f"{symbol}_{INTERVAL}.csv"
    out.to_csv(path, index=False)

    # 1. NumPy 타입을 순수 파이썬 float 타입으로 강제 변환 후 1000으로 나누어 초 단위로 변경
    start_ts = float(out["timestamp"].iloc[0]) / 1000.0
    end_ts = float(out["timestamp"].iloc[-1]) / 1000.0

    # 2. Pandas를 우회하여 파이썬 내장 datetime 함수로 직접 변환
    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)

    print(f"  저장: {path}  ({len(out):,}행, {start_dt:%Y-%m-%d} ~ {end_dt:%Y-%m-%d})")

print("\n✅ 다운로드 완료!")
print("다음 단계: python train/01_build_dataset.py --model all")
