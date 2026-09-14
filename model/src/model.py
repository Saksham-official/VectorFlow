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

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.in_ln = nn.LayerNorm(in_dim)
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        # Soft attention: score each time-step, then take weighted sum
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
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
        x = self.in_ln(x)                          # (B, T, F)
        out, _ = self.lstm(x)                       # (B, T, H*2)
        attn_w = torch.softmax(self.attn(out), dim=1)  # (B, T, 1)
        context = (out * attn_w).sum(dim=1)         # (B, H*2)
        return self.fc(context).squeeze(-1)         # (B,)
