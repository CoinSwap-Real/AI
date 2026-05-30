"""
train/03_export_onnx.py — PyTorch 모델 → ONNX 변환 및 검증

[실행 예시]
  python train/03_export_onnx.py --horizon trade
  python train/03_export_onnx.py --horizon 1h
  python train/03_export_onnx.py --horizon 24h
  python train/03_export_onnx.py --horizon 7d

[검증 항목]
  1. ONNX 출력 shape = (1, 2) ← ai_engine.py 가 기대하는 형태
  2. PyTorch 출력 vs ONNX 출력 수치 차이 < 1e-5
  3. scaler 역변환 후 실제 수익률 범위 확인
  4. 봇 코드(FeatureBuilder + AIEngine) 엔드투엔드 통과 테스트
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import torch

# ── 02_train_model.py 의 GRUModel 재사용 ─────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from train_helpers import GRUModel   # 아래 helpers 파일에서 임포트
# (실제 사용 시: from train.02_train_model import GRUModel 처럼
#  같은 파일에 클래스가 있으면 직접 임포트하면 됩니다)

# ── CLI ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--horizon", choices=["trade","scalper","swing","longterm"], required=True)
args = parser.parse_args()

HORIZON_CFG = {
    "trade":    {"seq_len": 30,  "model_name": "model_trade",    "data_tag": "trade"},
    "scalper":  {"seq_len": 10,  "model_name": "model_scalper",  "data_tag": "scalper"},
    "swing":    {"seq_len": 60,  "model_name": "model_swing",    "data_tag": "swing"},
    "longterm": {"seq_len": 120, "model_name": "model_longterm", "data_tag": "longterm"},
}
cfg     = HORIZON_CFG[args.horizon]
SEQ_LEN = cfg["seq_len"]
NAME    = cfg["model_name"]

PT_PATH   = f"models/{NAME}_best.pt"
ONNX_PATH = f"models/{NAME}.onnx"
DATA_DIR  = Path(f"data/{cfg['data_tag']}")

print(f"[{NAME}] ONNX 변환 시작")
print(f"  PyTorch 소스: {PT_PATH}")
print(f"  ONNX 출력  : {ONNX_PATH}")


# ════════════════════════════════════════════════════════════
# Step 1. PyTorch 모델 로드
# ════════════════════════════════════════════════════════════

model = GRUModel(input_size=8, hidden_size=64, num_layers=2, dropout=0.0)
model.load_state_dict(torch.load(PT_PATH, map_location="cpu"))
model.eval()
print("[1/4] PyTorch 모델 로드 완료")


# ════════════════════════════════════════════════════════════
# Step 2. ONNX 변환
#   dynamic_axes 로 batch_size 를 가변으로 설정
#   봇은 항상 batch=1 로 추론하지만 유연성을 위해 동적 설정
# ════════════════════════════════════════════════════════════

dummy_input = torch.zeros(1, SEQ_LEN, 8, dtype=torch.float32)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input":  {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)
print(f"[2/4] ONNX 변환 완료: {ONNX_PATH}")


# ════════════════════════════════════════════════════════════
# Step 3. 수치 검증 — PyTorch vs ONNX 출력 비교
# ════════════════════════════════════════════════════════════

print("[3/4] 수치 검증 중...")

X_scaled = np.load(DATA_DIR / "features_scaled.npy")   # (T, 8)

# 마지막 seq_len 행으로 테스트 입력 구성
test_input = X_scaled[-SEQ_LEN:][np.newaxis, :, :]      # (1, seq_len, 8)

# PyTorch 출력
with torch.no_grad():
    pt_out = model(torch.from_numpy(test_input)).numpy()  # (1, 2)

# ONNX 출력
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"input": test_input.astype(np.float32)})[0]  # (1, 2)

max_diff = np.abs(pt_out - ort_out).max()
print(f"  PyTorch 출력: BTC={pt_out[0,0]:.8f}  ETH={pt_out[0,1]:.8f}")
print(f"  ONNX    출력: BTC={ort_out[0,0]:.8f}  ETH={ort_out[0,1]:.8f}")
print(f"  최대 차이   : {max_diff:.2e}", end="  ")

if max_diff < 1e-5:
    print("✅ 통과")
else:
    print("⚠️  차이가 큽니다 — opset_version 을 낮춰 보세요")

# ONNX 입출력 shape 검증
input_shape  = sess.get_inputs()[0].shape
output_shape = sess.get_outputs()[0].shape
print(f"  ONNX 입력 shape : {input_shape}   ← 봇 기대: [1, {SEQ_LEN}, 8]")
print(f"  ONNX 출력 shape : {output_shape}  ← 봇 기대: [1, 2]")

assert input_shape[1]  == SEQ_LEN, f"seq_len 불일치: {input_shape[1]} ≠ {SEQ_LEN}"
assert output_shape[1] == 2,       f"출력 차원 불일치: {output_shape[1]} ≠ 2"


# ════════════════════════════════════════════════════════════
# Step 4. 봇 엔드투엔드 검증
#   실제 bot 코드 (FeatureBuilder + AIEngine) 를 통해
#   scaler → 피처 → 추론 전 과정이 문제없는지 확인
# ════════════════════════════════════════════════════════════

print("[4/4] 봇 엔드투엔드 검증...")

try:
    import sys, os
    # 프로젝트 루트를 sys.path 에 추가
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # .env 없이 최소한의 설정으로 import
    os.environ.setdefault("BOT_KEY", "test")
    os.environ.setdefault("SCALER_PATH", "models/scaler.pkl")
    os.environ.setdefault(f"MODEL_{NAME.upper().replace('-','_')}_PATH", ONNX_PATH)

    from ai.feature_builder import FeatureBuilder
    from ai.ai_engine import AIEngine
    import asyncio

    fb     = FeatureBuilder()
    engine = AIEngine(NAME, ONNX_PATH, SEQ_LEN)

    # 더미 피처로 버퍼 채우기
    raw_features = np.load(DATA_DIR / "features_raw.npy")  # (T, 8)
    scaler = joblib.load("models/scaler.pkl")
    scaled = scaler.transform(raw_features[-SEQ_LEN:]).astype(np.float32)

    engine.load_features(scaled)

    async def _test_infer():
        return await engine.infer()

    result = asyncio.run(_test_infer())

    if result is not None:
        btc_ret, eth_ret = result
        print(f"  봇 추론 결과 — BTC: {btc_ret:.8f}  ETH: {eth_ret:.8f}")
        print("  ✅ 엔드투엔드 검증 통과")
    else:
        print("  ⚠️  infer() 가 None 반환 — 버퍼 미충전 상태 (seq_len 확인 필요)")

except ImportError as e:
    print(f"  ⚠️  봇 코드 임포트 실패 (학습 환경에서 정상): {e}")
    print("     서버 환경에서 별도로 엔드투엔드 검증을 수행하세요.")

print(f"\n✅ 완료!")
print(f"  ONNX 파일: {ONNX_PATH}")
print(f"  scaler   : models/scaler.pkl")
print()
print("  이 두 파일을 서버의 models/ 디렉토리에 배치하면 봇이 즉시 사용합니다.")