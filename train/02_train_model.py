"""
train/02_train_model.py — GRU 모델 학습

[실행 예시]
  python train/02_train_model.py --horizon trade
  python train/02_train_model.py --horizon 1h
  python train/02_train_model.py --horizon 24h
  python train/02_train_model.py --horizon 7d

[모델 I/O — ai_engine.py 와 정확히 일치]
  입력: (batch, seq_len, 8)   — scaler 적용 완료 피처
  출력: (batch, 2)             — [BTC 로그수익률, ETH 로그수익률]

[horizon 별 용도]
  trade → seq_len=30  model_trade.onnx    시스템 B BotA_Trade (다음 1분 캔들 예측)
  1h    → seq_len=10  model_scalper.onnx  시스템 A ingest (1h 신호)
  24h   → seq_len=60  model_swing.onnx    시스템 A ingest (24h 신호)
  7d    → seq_len=120 model_longterm.onnx 시스템 A ingest (7d 신호)
"""

import argparse
import math
from pathlib import Path

import joblib
# 학습 시: models/scaler_{model_name}.pkl 사용 (01_build_dataset.py 생성)
# 봇 런타임: models/scaler.pkl (trade 기준, 5m 분포)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── CLI ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--horizon", choices=["trade","scalper","swing","longterm"], required=True)
args = parser.parse_args()

# ── horizon → seq_len / 모델 이름 매핑 ───────────────────────
HORIZON_CFG = {
    "trade":    {"seq_len": 30,  "model_name": "model_trade",    "data_tag": "trade"},
    "scalper":  {"seq_len": 10,  "model_name": "model_scalper",  "data_tag": "scalper"},
    "swing":    {"seq_len": 60,  "model_name": "model_swing",    "data_tag": "swing"},
    "longterm": {"seq_len": 120, "model_name": "model_longterm", "data_tag": "longterm"},
}
cfg     = HORIZON_CFG[args.horizon]
SEQ_LEN = cfg["seq_len"]
NAME    = cfg["model_name"]
DATA_DIR = Path(f"data/{cfg['data_tag']}")

# ── 하이퍼파라미터 ────────────────────────────────────────────
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
DROPOUT      = 0.2
BATCH_SIZE   = 512
EPOCHS       = 100
LR           = 1e-3
PATIENCE     = 10      # early stopping
GRAD_CLIP    = 1.0
TRAIN_RATIO  = 0.7
VAL_RATIO    = 0.15
# (나머지 0.15 = test)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[{NAME}] device={DEVICE}  seq_len={SEQ_LEN}  horizon={args.horizon}")


# ════════════════════════════════════════════════════════════
# Step 1. 데이터 로드 및 시퀀스 윈도우 생성
# ════════════════════════════════════════════════════════════

print("[1/4] 데이터 로드 중...")
X_all = np.load(DATA_DIR / "features_scaled.npy")   # (T, 8)
y_btc = np.load(DATA_DIR / "targets_btc.npy")       # (T,)
y_eth = np.load(DATA_DIR / "targets_eth.npy")       # (T,)

T = len(X_all)
assert T > SEQ_LEN, f"데이터({T})가 seq_len({SEQ_LEN}) 보다 짧습니다."


def make_sequences(X, y_btc, y_eth, seq_len):
    """
    슬라이딩 윈도우로 (N, seq_len, 8) 입력과 (N, 2) 타깃 생성.
    타깃은 윈도우 마지막 시점의 BTC/ETH 로그수익률.
    """
    N = len(X) - seq_len
    Xs = np.zeros((N, seq_len, 8), dtype=np.float32)
    ys = np.zeros((N, 2),          dtype=np.float32)
    for i in range(N):
        Xs[i] = X[i: i + seq_len]
        ys[i, 0] = y_btc[i + seq_len - 1]
        ys[i, 1] = y_eth[i + seq_len - 1]
    return Xs, ys


X_seq, y_seq = make_sequences(X_all, y_btc, y_eth, SEQ_LEN)
print(f"  시퀀스 shape — X: {X_seq.shape}  y: {y_seq.shape}")

# 시간 순서 유지 분할 (shuffle 금지)
n = len(X_seq)
t1 = int(n * TRAIN_RATIO)
t2 = int(n * (TRAIN_RATIO + VAL_RATIO))

X_train, y_train = X_seq[:t1],    y_seq[:t1]
X_val,   y_val   = X_seq[t1:t2],  y_seq[t1:t2]
X_test,  y_test  = X_seq[t2:],    y_seq[t2:]
print(f"  train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

# DataLoader
def to_loader(X, y, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, pin_memory=(DEVICE=="cuda"))

train_loader = to_loader(X_train, y_train, shuffle=True)
val_loader   = to_loader(X_val,   y_val)
test_loader  = to_loader(X_test,  y_test)


# ════════════════════════════════════════════════════════════
# Step 2. 모델 정의
# ════════════════════════════════════════════════════════════

class GRUModel(nn.Module):
    """
    ai_engine.py 의 ONNX 스펙과 일치:
      입력: (batch, seq_len, 8)
      출력: (batch, 2)  — [BTC_log_ret, ETH_log_ret]
    """
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,           # (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)  # 출력: [btc_ret, eth_ret]

    def forward(self, x):
        # x: (batch, seq_len, 8)
        out, _ = self.gru(x)            # out: (batch, seq_len, hidden)
        last   = out[:, -1, :]          # 마지막 타임스텝: (batch, hidden)
        last   = self.dropout(last)
        return self.fc(last)            # (batch, 2)


print("[2/4] 모델 초기화...")
model = GRUModel(
    input_size=8,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"  파라미터 수: {total_params:,}")

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.5, min_lr=1e-5
)


# ════════════════════════════════════════════════════════════
# Step 3. 학습 루프
# ════════════════════════════════════════════════════════════

def run_epoch(loader, train=True):
    model.train(train)
    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            pred = model(Xb)
            loss = criterion(pred, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            total_loss += loss.item() * len(Xb)
    return total_loss / len(loader.dataset)


print("[3/4] 학습 시작...")
best_val   = math.inf
no_improve = 0
Path("models").mkdir(exist_ok=True)
best_path  = f"models/{NAME}_best.pt"

for epoch in range(1, EPOCHS + 1):
    tr_loss = run_epoch(train_loader, train=True)
    va_loss = run_epoch(val_loader,   train=False)
    scheduler.step(va_loss)

    improved = va_loss < best_val
    if improved:
        best_val   = va_loss
        no_improve = 0
        torch.save(model.state_dict(), best_path)
    else:
        no_improve += 1

    if epoch % 10 == 0 or epoch <= 5 or improved:
        lr_now = optimizer.param_groups[0]["lr"]
        mark   = "✓" if improved else " "
        print(
            f"  [{mark}] epoch {epoch:03d}/{EPOCHS} | "
            f"train={tr_loss:.6f}  val={va_loss:.6f}  "
            f"best={best_val:.6f}  lr={lr_now:.2e}"
        )

    if no_improve >= PATIENCE:
        print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
        break

# 최적 가중치 로드
model.load_state_dict(torch.load(best_path, map_location=DEVICE))

# 테스트 평가
test_loss = run_epoch(test_loader, train=False)
test_rmse = math.sqrt(test_loss)
print(f"\n  테스트 MSE={test_loss:.6f}  RMSE={test_rmse:.6f}")


# ════════════════════════════════════════════════════════════
# Step 4. 방향 정확도 (선택 평가 지표)
#   예측 부호와 실제 부호의 일치율 — 매매 신호 정확도의 프록시
# ════════════════════════════════════════════════════════════

model.eval()
all_pred, all_true = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        pred = model(Xb.to(DEVICE)).cpu().numpy()
        all_pred.append(pred)
        all_true.append(yb.numpy())

all_pred = np.vstack(all_pred)
all_true = np.vstack(all_true)

dir_acc_btc = np.mean(np.sign(all_pred[:, 0]) == np.sign(all_true[:, 0]))
dir_acc_eth = np.mean(np.sign(all_pred[:, 1]) == np.sign(all_true[:, 1]))
print(f"  방향 정확도 — BTC: {dir_acc_btc*100:.1f}%  ETH: {dir_acc_eth*100:.1f}%")
print(f"  (랜덤 기준선: 50%)")

print(f"\n✅ 학습 완료! 최적 모델: {best_path}")
print("  다음 단계: python train/03_export_onnx.py --horizon 1h")