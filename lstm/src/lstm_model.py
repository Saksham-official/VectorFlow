import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

FLOW_BASE_FEATURES: List[str] = [
    "Tot Fwd Pkts_sum",
    "Tot Bwd Pkts_sum",
    "TotLen Fwd Pkts_sum",
    "TotLen Bwd Pkts_sum",
    "Flow Duration_mean",
    "Flow Duration_std",
    "Flow IAT Mean_mean",
    "Dst Port_nunique",
    "SYN Flag Cnt_sum",
    "ACK Flag Cnt_sum",
    "RST Flag Cnt_sum",
    "FIN Flag Cnt_sum",
    "PSH Flag Cnt_sum",
    "flow_count",
    "bwd_fwd_pkt_ratio",
]

PACKET_BASE_FEATURES: List[str] = [
    "ttl_mean",
    "ttl_session_variance",
    "tcp_window_mean",
    "tcp_window_std",
    "ip_fragment_count",
    "payload_size_mean",
    "payload_size_std",
    "zero_payload_ratio",
    "tcp_retrans_count",
    "sequential_scan_score",
    "random_scan_entropy",
]

BASE_FEATURES: List[str] = FLOW_BASE_FEATURES + PACKET_BASE_FEATURES

SEED: int = int(os.getenv("RANDOM_SEED", "42"))
LSTM_SEQ_LEN: int = int(os.getenv("LSTM_SEQ_LEN", "10"))
LSTM_HIDDEN: int = int(os.getenv("LSTM_HIDDEN_DIM", "256"))
LSTM_LAYERS: int = int(os.getenv("LSTM_NUM_LAYERS", "2"))
LSTM_DROPOUT: float = float(os.getenv("LSTM_DROPOUT", "0.4"))
LSTM_LR: float = float(os.getenv("LSTM_LEARNING_RATE", "5e-4"))
LSTM_EPOCHS: int = int(os.getenv("LSTM_EPOCHS", "50"))
LSTM_BATCH: int = int(os.getenv("LSTM_BATCH_SIZE", "256"))
LSTM_PATIENCE: int = int(os.getenv("LSTM_PATIENCE", "8"))
FOCAL_GAMMA: float = float(os.getenv("LSTM_FOCAL_GAMMA", "2.0"))
USE_ALL_FEATURES: bool = os.getenv("LSTM_USE_ALL_FEATURES", "0") == "1"


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    if USE_ALL_FEATURES:
        drop = {"window_start", "date", "Future_Attack_Target", "has_attack", "is_precursor"}
        return [c for c in df.columns if c not in drop]
    return [c for c in BASE_FEATURES if c in df.columns]


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_sequences(
    df: pd.DataFrame,
    seq_len: int = LSTM_SEQ_LEN,
    base_cols: Optional[List[str]] = None,
    target_col: str = "Future_Attack_Target",
    date_col: str = "date",
) -> Tuple[np.ndarray, np.ndarray]:
    if base_cols is None:
        base_cols = BASE_FEATURES

    X_seqs, y_seqs = [], []

    for _date, group in df.groupby(date_col, sort=False):
        group = group.sort_values("window_start")
        feats = group[base_cols].values.astype(np.float32)
        labels = group[target_col].values.astype(np.int64)

        if len(group) < seq_len:
            continue

        for i in range(len(group) - seq_len + 1):
            X_seqs.append(feats[i : i + seq_len])
            y_seqs.append(labels[i + seq_len - 1])

    return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.int64)


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class AttackLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = len(BASE_FEATURES),
        hidden_dim: int = LSTM_HIDDEN,
        num_layers: int = LSTM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        lstm_out_dim = hidden_dim * self.directions

        self.input_bn = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
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

    def attention_pool(self, lstm_out: torch.Tensor) -> torch.Tensor:
        scores = self.attn_fc(lstm_out).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (lstm_out * weights).sum(dim=1)
        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, f = x.shape
        x = self.input_bn(x.reshape(b * s, f)).reshape(b, s, f)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.norm(lstm_out)
        context = self.attention_pool(lstm_out)
        logits = self.head(context)
        return logits


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = FOCAL_GAMMA, pos_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        pt = torch.exp(-bce)
        weight = torch.where(
            targets == 1,
            torch.tensor(self.pos_weight, device=logits.device),
            torch.tensor(1.0, device=logits.device),
        )
        focal = weight * ((1 - pt) ** self.gamma) * bce
        return focal.mean()


def scale_sequences(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    N_tr, S, F = X_train.shape
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, F)
    scaler.fit(X_train_flat)

    def _transform(X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        return scaler.transform(X.reshape(-1, F)).reshape(n, S, F).astype(np.float32)

    return _transform(X_train), _transform(X_val), _transform(X_test), scaler


class EarlyStopping:
    def __init__(self, patience: int = LSTM_PATIENCE, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: Optional[float] = None
        self.counter = 0
        self.best_state: Optional[dict] = None

    def step(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False

        improved = (
            (score > self.best_score + self.min_delta)
            if self.mode == "max"
            else (score < self.best_score - self.min_delta)
        )
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def build_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    class_counts = np.bincount(y)
    weights_per_class = 1.0 / np.maximum(class_counts, 1)
    sample_weights = weights_per_class[y]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(y),
        replacement=True,
    )


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device | None = None,
    epochs: int = LSTM_EPOCHS,
    batch_size: int = LSTM_BATCH,
    lr: float = LSTM_LR,
    patience: int = LSTM_PATIENCE,
) -> Tuple[AttackLSTM, dict]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    set_seed()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = X_train.shape[2]

    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)

    sampler = build_weighted_sampler(y_train)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    model = AttackLSTM(input_dim=input_dim).to(device)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, pos_weight=20.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    stopper = EarlyStopping(patience=patience, mode="max")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_pr_auc": [],
        "val_roc_auc": [],
        "val_f1": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.float().to(device)
            logits = model(X_b).squeeze(-1)
            loss = criterion(logits, y_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.float().to(device)
                logits = model(X_b).squeeze(-1)
                loss = criterion(logits, y_b)
                val_loss += loss.item()
                val_batches += 1

                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(y_b.cpu().numpy())

        val_loss /= max(val_batches, 1)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        val_pr_auc = float(average_precision_score(all_labels, all_preds))
        val_roc_auc = float(roc_auc_score(all_labels, all_preds))
        preds_bin = (all_preds >= 0.5).astype(int)
        tp = ((preds_bin == 1) & (all_labels == 1)).sum()
        fp = ((preds_bin == 1) & (all_labels == 0)).sum()
        fn = ((preds_bin == 0) & (all_labels == 1)).sum()
        val_f1 = float((2 * tp) / (2 * tp + fp + fn + 1e-12))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pr_auc"].append(val_pr_auc)
        history["val_roc_auc"].append(val_roc_auc)
        history["val_f1"].append(val_f1)

        if stopper.step(val_pr_auc, model):
            break

    stopper.restore_best(model)
    model = model.cpu()
    model.eval()

    return model, history


def predict_proba(
    model: AttackLSTM,
    X: np.ndarray,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    ds = SequenceDataset(X, np.zeros(len(X), dtype=np.int64))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    probs = []
    with torch.no_grad():
        for X_b, _ in loader:
            logits = model(X_b.to(device)).squeeze(-1)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def save_lstm(
    model: AttackLSTM,
    scaler: StandardScaler,
    threshold: float,
    model_dir: str = "models",
    history: Optional[dict] = None,
    feature_names: Optional[List[str]] = None,
):
    import json
    import joblib

    os.makedirs(model_dir, exist_ok=True)
    feats = feature_names if feature_names is not None else BASE_FEATURES

    model_path = os.path.join(model_dir, "lstm.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": model.input_bn.num_features,
            "hidden_dim": model.hidden_dim,
            "num_layers": model.num_layers,
            "bidirectional": model.bidirectional,
            "seq_len": LSTM_SEQ_LEN,
            "feature_names": feats,
        },
        model_path,
    )

    scaler_path = os.path.join(model_dir, "lstm_scaler.joblib")
    joblib.dump(scaler, scaler_path)

    thresh_path = os.path.join(model_dir, "lstm_threshold.json")
    with open(thresh_path, "w") as f:
        json.dump({"LSTM": threshold}, f, indent=2)

    meta_path = os.path.join(model_dir, "lstm_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "input_dim": model.input_bn.num_features,
                "hidden_dim": model.hidden_dim,
                "num_layers": model.num_layers,
                "bidirectional": model.bidirectional,
                "seq_len": LSTM_SEQ_LEN,
                "threshold": threshold,
                "feature_names": feats,
            },
            f,
            indent=2,
        )

    if history is not None:
        hist_path = os.path.join(model_dir, "lstm_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)


def load_lstm(model_dir: str = "models") -> Tuple[AttackLSTM, StandardScaler, float, List[str]]:
    import json
    import joblib

    ckpt_path = os.path.join(model_dir, "lstm.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = AttackLSTM(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        bidirectional=ckpt["bidirectional"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = joblib.load(os.path.join(model_dir, "lstm_scaler.joblib"))

    thresh_path = os.path.join(model_dir, "lstm_threshold.json")
    if os.path.exists(thresh_path):
        with open(thresh_path, "r") as f:
            threshold = json.load(f).get("LSTM", 0.5)
    else:
        threshold = 0.5

    feature_names = ckpt.get("feature_names", BASE_FEATURES)

    return model, scaler, threshold, feature_names
