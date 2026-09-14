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
    num_layers : int
        Number of recurrent layers.
    dropout : float
        Dropout applied between LSTM layers and inside the classifier head.
    bidirectional : bool
        Whether to use bidirectional recurrence.
    """

    def __init__(
        self,
        in_dim: int = 15,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        lstm_out_dim = hidden_dim * self.directions

        self.input_bn = nn.BatchNorm1d(in_dim)
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.norm = nn.LayerNorm(lstm_out_dim)
        self.attn_fc = nn.Linear(lstm_out_dim, 1, bias=False)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
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
        b, s, f = x.shape
        x = self.input_bn(x.reshape(b * s, f)).reshape(b, s, f)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.norm(lstm_out)
        scores = self.attn_fc(lstm_out).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (lstm_out * weights).sum(dim=1)
        logits = self.head(context).squeeze(-1)
        return logits

