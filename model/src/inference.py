"""
Inference pipeline for the VectorFlow BiLSTM forecasting model.
================================================================
Provides get_inference_pipeline() — the single entry point consumed by
modules/forecasting/__init__.py.

Usage
-----
    from model.src.inference import get_inference_pipeline

    pipeline = get_inference_pipeline("/path/to/model/artifacts")
    result = pipeline.predict_sequence(list_of_feature_dicts)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from .model import ForecastLSTM

logger = logging.getLogger(__name__)

_REQUIRED_FEATURES = [
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


class InsufficientHistoryError(ValueError):
    """Raised when fewer windows are available than the model's sequence length."""


class InferencePipeline:
    """
    Wraps the trained ForecastLSTM checkpoint with preprocessing and postprocessing.

    Parameters
    ----------
    model : ForecastLSTM
        Loaded model (eval mode).
    scaler : sklearn-compatible transformer
        Fitted RobustScaler used during training.
    metadata : dict
        Parsed model_metadata.json — provides threshold, sequence_length,
        feature_cols, and forecast_horizon_seconds.
    device : str
        Torch device string, e.g. "cpu" or "cuda".
    """

    def __init__(
        self,
        model: ForecastLSTM,
        scaler: Any,
        metadata: dict,
        device: str = "cpu",
    ) -> None:
        self._model = model
        self._scaler = scaler
        self._metadata = metadata
        self._device = device
        self._feature_cols: list[str] = metadata.get("feature_cols", _REQUIRED_FEATURES)
        self._threshold: float = float(metadata.get("threshold", 0.3367))
        self._sequence_length: int = int(metadata.get("sequence_length", 10))
        self._horizon: dict = metadata.get("forecast_horizon_seconds", {"min": 30, "max": 120})

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def threshold(self) -> float:
        return self._threshold

    def _extract_features(self, state_features: dict) -> list[float]:
        """Pull the ordered feature vector from a network-state feature dict."""
        row = []
        for col in self._feature_cols:
            val = state_features.get(col, 0.0)
            try:
                row.append(float(val))
            except (TypeError, ValueError):
                row.append(0.0)
        return row

    def load_artifacts(self) -> "InferencePipeline":
        """No-op verification hook to confirm model and scaler artifacts are loaded."""
        return self

    def predict_sequence(self, states: list[dict]) -> dict:
        """
        Run inference on an ordered historical sequence of window dicts.

        Parameters
        ----------
        states : list[dict]
            Ordered list of per-window feature dicts (oldest → newest).
            Must have at least 6 entries for initial warmup.

        Returns
        -------
        dict with keys:
            attack_probability : float  — sigmoid output in [0, 1]
            prediction         : bool   — True if probability >= threshold
            threshold          : float
            sequence_length    : int
            forecast_horizon_seconds : dict{"min": int, "max": int}
            model_mode         : str    — always "real"

        Raises
        ------
        InsufficientHistoryError
            When ``len(states) < 6``.
        """
        min_windows = min(6, self._sequence_length)
        if len(states) < min_windows:
            raise InsufficientHistoryError(
                f"Model requires at least {min_windows} windows of history; "
                f"only {len(states)} available. "
                f"Collecting historical context... "
                f"({len(states)}/{self._sequence_length})"
            )

        # Use the most recent sequence_length windows; pad if in warm-up (6..9 windows)
        if len(states) < self._sequence_length:
            pad_count = self._sequence_length - len(states)
            window = [states[0]] * pad_count + list(states)
        else:
            window = states[-self._sequence_length:]

        # Build raw feature matrix (seq_len, n_features)
        raw = np.array(
            [self._extract_features(s) for s in window],
            dtype=np.float32,
        )

        # Scale features using the fitted scaler
        try:
            scaled = self._scaler.transform(raw)
        except Exception as exc:  # pragma: no cover
            logger.warning("Scaler transform failed (%s); using raw features.", exc)
            scaled = raw

        # Run model inference
        tensor = torch.from_numpy(scaled).unsqueeze(0).to(self._device)  # (1, T, F)
        with torch.no_grad():
            logit = self._model(tensor)
            probability = float(torch.sigmoid(logit).item())

        prediction = probability >= self._threshold

        logger.debug(
            "Inference: prob=%.4f, pred=%s, threshold=%.4f",
            probability,
            prediction,
            self._threshold,
        )

        return {
            "attack_probability": probability,
            "prediction": prediction,
            "threshold": self._threshold,
            "sequence_length": self._sequence_length,
            "forecast_horizon_seconds": self._horizon,
            "model_mode": "real",
        }


def get_inference_pipeline(artifacts_dir: str | None = None) -> InferencePipeline:
    """
    Load the trained BiLSTM checkpoint and return a ready-to-use InferencePipeline.

    Looks for the following files inside *artifacts_dir*
    (defaults to ``model/artifacts/`` relative to the repo root):

        lstm_model.pt        — PyTorch checkpoint
        scaler.joblib        — fitted RobustScaler
        model_metadata.json  — architecture and threshold config

    Parameters
    ----------
    artifacts_dir : str | None
        Absolute or relative path to the directory containing model artifacts.

    Returns
    -------
    InferencePipeline
    """
    if artifacts_dir is None:
        repo_root = Path(__file__).resolve().parents[3]
        artifacts_dir = str(repo_root / "model" / "artifacts")

    base = Path(artifacts_dir)
    model_path = base / "lstm_model.pt"
    scaler_path = base / "scaler.joblib"
    metadata_path = base / "model_metadata.json"

    # ── Load metadata ──────────────────────────────────────────────────────────
    if not metadata_path.exists():
        raise FileNotFoundError(f"model_metadata.json not found at {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    # ── Load scaler ───────────────────────────────────────────────────────────
    if not scaler_path.exists():
        raise FileNotFoundError(f"scaler.joblib not found at {scaler_path}")
    scaler = joblib.load(scaler_path)

    # ── Load model ────────────────────────────────────────────────────────────
    if not model_path.exists():
        raise FileNotFoundError(f"lstm_model.pt not found at {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Support checkpoint formats:
    #   (a) {"model_state_dict": ..., "input_dim": ...}             ← full package format
    #   (b) {"state_dict": ..., "in_dim": ..., "hidden_dim": ...}   ← benching format
    #   (c) raw state_dict                                          ← legacy format
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        in_dim = int(ckpt.get("input_dim", metadata.get("input_dim", 15)))
        hidden_dim = int(ckpt.get("hidden_dim", metadata.get("hidden_size", 128)))
        num_layers = int(ckpt.get("num_layers", metadata.get("num_layers", 2)))
        bidirectional = bool(ckpt.get("bidirectional", metadata.get("bidirectional", True)))
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        in_dim = int(ckpt.get("in_dim", metadata.get("input_dim", 15)))
        hidden_dim = int(ckpt.get("hidden_dim", metadata.get("hidden_size", 128)))
        num_layers = int(metadata.get("num_layers", 2))
        bidirectional = bool(metadata.get("bidirectional", True))
        state_dict = ckpt["state_dict"]
    else:
        in_dim = int(metadata.get("input_dim", 15))
        hidden_dim = int(metadata.get("hidden_size", 128))
        num_layers = int(metadata.get("num_layers", 2))
        bidirectional = bool(metadata.get("bidirectional", True))
        state_dict = ckpt

    model = ForecastLSTM(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(
        "Loaded ForecastLSTM: in_dim=%d, hidden_dim=%d, seq_len=%d, threshold=%.4f",
        in_dim,
        hidden_dim,
        metadata.get("sequence_length", 10),
        metadata.get("threshold", 0.3367),
    )

    return InferencePipeline(model=model, scaler=scaler, metadata=metadata, device=device)
