"""
train/train_02_helpers.py — 학습 스크립트 간 공유 클래스

03_export_onnx.py 에서 GRUModel 을 재사용하기 위해 분리합니다.
"""

import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """
    ai_engine.py 의 ONNX 스펙과 일치:
      입력  : (batch, seq_len, 8)
      출력  : (batch, 2)  — [BTC_log_ret, ETH_log_ret]
    """

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)  # (batch, seq_len, hidden)
        last = out[:, -1, :]  # (batch, hidden)
        return self.fc(self.dropout(last))  # (batch, 2)
