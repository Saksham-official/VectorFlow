"""
ForecastLSTM — Bidirectional LSTM with Attention Pooling
=========================================================
Architecture:
    Input (input_dim features, seq_len time-steps)
    └─ LayerNorm
    └─ BiLSTM (hidden_dim, num_layers=2, bidirectional=True, dropout=0.2)
    └─ Soft Attention Pooling  → context vector (hidden_dim * 2)
    └─ LayerNorm → Linear(hidden_dim*2 → 32) → GELU → Dropout → Linear(32 → 1)
    └─ Sigmoid → attack onset probability in [0, 1]

Checkpoint format (saved by training script):
    {
        "state_dict": <OrderedDict>,   # model weights
        "in_dim":     <int>,           # number of input features
        "hidden_dim": <int>,           # LSTM hidden size per direction
    }
"""

import torch
import torch.nn as nn


class ForecastLSTM(nn.Module):
    """
    Two-layer Bidirectional LSTM with soft-attention context pooling.

    Parameters
    ----------
    in_dim : int
        Number of input features per time-step (15 for VectorFlow flow windows).
    hidden_dim : int
        LSTM hidden units *per direction*. Total context size = hidden_dim * 2.
    dropout : float
        Dropout applied between LSTM layers and inside the classifier head.
    """

    def __init__(self, in_dim: int = 15, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.input_bn = nn.BatchNorm1d(in_dim)
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.attn_fc = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, in_dim)

        Returns
        -------
        Tensor of shape (batch,) — raw logits (apply sigmoid for probability).
        """
        x_t = x.transpose(1, 2)
        x_bn = self.input_bn(x_t).transpose(1, 2)
        out, _ = self.lstm(x_bn)                          # (B, T, H*2)
        norm_out = self.norm(out)
        attn_w = torch.softmax(self.attn_fc(norm_out), dim=1)  # (B, T, 1)
        context = (out * attn_w).sum(dim=1)              # (B, H*2)
        return self.head(context).squeeze(-1)            # (B,)
