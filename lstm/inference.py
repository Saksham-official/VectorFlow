import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import lstm_model as LM
from dual_pipeline import DUAL_BASE_FEATURES, process_capture
from packet_extractor import extract_pcap_features

FEATURE_COLS: list[str] = [
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
    "Tot Fwd Pkts_sum_lag1",
    "Tot Fwd Pkts_sum_lag2",
    "Tot Fwd Pkts_sum_lag3",
    "Tot Bwd Pkts_sum_lag1",
    "Tot Bwd Pkts_sum_lag2",
    "Tot Bwd Pkts_sum_lag3",
    "TotLen Fwd Pkts_sum_lag1",
    "TotLen Fwd Pkts_sum_lag2",
    "TotLen Fwd Pkts_sum_lag3",
    "TotLen Bwd Pkts_sum_lag1",
    "TotLen Bwd Pkts_sum_lag2",
    "TotLen Bwd Pkts_sum_lag3",
    "Flow Duration_mean_lag1",
    "Flow Duration_mean_lag2",
    "Flow Duration_mean_lag3",
    "Flow Duration_std_lag1",
    "Flow Duration_std_lag2",
    "Flow Duration_std_lag3",
    "Flow IAT Mean_mean_lag1",
    "Flow IAT Mean_mean_lag2",
    "Flow IAT Mean_mean_lag3",
    "Dst Port_nunique_lag1",
    "Dst Port_nunique_lag2",
    "Dst Port_nunique_lag3",
    "SYN Flag Cnt_sum_lag1",
    "SYN Flag Cnt_sum_lag2",
    "SYN Flag Cnt_sum_lag3",
    "ACK Flag Cnt_sum_lag1",
    "ACK Flag Cnt_sum_lag2",
    "ACK Flag Cnt_sum_lag3",
    "RST Flag Cnt_sum_lag1",
    "RST Flag Cnt_sum_lag2",
    "RST Flag Cnt_sum_lag3",
    "FIN Flag Cnt_sum_lag1",
    "FIN Flag Cnt_sum_lag2",
    "FIN Flag Cnt_sum_lag3",
    "PSH Flag Cnt_sum_lag1",
    "PSH Flag Cnt_sum_lag2",
    "PSH Flag Cnt_sum_lag3",
    "flow_count_lag1",
    "flow_count_lag2",
    "flow_count_lag3",
    "bwd_fwd_pkt_ratio_lag1",
    "bwd_fwd_pkt_ratio_lag2",
    "bwd_fwd_pkt_ratio_lag3",
    "Tot Fwd Pkts_sum_delta1",
    "Tot Bwd Pkts_sum_delta1",
    "TotLen Fwd Pkts_sum_delta1",
    "TotLen Bwd Pkts_sum_delta1",
    "Flow Duration_mean_delta1",
    "Flow Duration_std_delta1",
    "Flow IAT Mean_mean_delta1",
    "Dst Port_nunique_delta1",
    "SYN Flag Cnt_sum_delta1",
    "ACK Flag Cnt_sum_delta1",
    "RST Flag Cnt_sum_delta1",
    "FIN Flag Cnt_sum_delta1",
    "PSH Flag Cnt_sum_delta1",
    "flow_count_delta1",
    "bwd_fwd_pkt_ratio_delta1",
]

DEFAULT_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
MODEL_DIR = os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)
if not os.path.exists(MODEL_DIR) and os.path.exists(DEFAULT_MODEL_DIR):
    MODEL_DIR = DEFAULT_MODEL_DIR


def _load_artifacts():
    lstm_pt = os.path.join(MODEL_DIR, "lstm.pt")
    if not os.path.exists(lstm_pt):
        raise FileNotFoundError(f"Primary LSTM model not found in '{MODEL_DIR}'.")

    lstm_net, lstm_scaler, lstm_thresh, feature_names = LM.load_lstm(MODEL_DIR)
    thresholds = {"LSTM": float(lstm_thresh)}
    return lstm_net, lstm_scaler, thresholds, feature_names


_artifacts = None


def get_artifacts():
    global _artifacts
    if _artifacts is None:
        _artifacts = _load_artifacts()
    return _artifacts


def predict(
    features: dict[str, float] | pd.DataFrame,
    use_calibrated_threshold: bool = True,
    custom_threshold: float | None = None,
    include_lstm: bool = True,
) -> pd.DataFrame:
    """Run dual-level LSTM inference on flow + packet feature input."""
    lstm_net, lstm_scaler, thresholds, feature_names = get_artifacts()

    if isinstance(features, dict):
        df = pd.DataFrame([features])
    elif isinstance(features, pd.DataFrame):
        df = features.copy()
    else:
        raise TypeError("Input 'features' must be a dict or a pandas DataFrame")

    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Input is missing {len(missing)} required feature(s): {missing[:5]}")

    thresh = (
        custom_threshold
        if custom_threshold is not None
        else (thresholds.get("LSTM", 0.5) if use_calibrated_threshold else 0.5)
    )

    base_feature_names = feature_names if isinstance(feature_names, list) else LM.BASE_FEATURES
    for col in base_feature_names:
        if col not in df.columns:
            df[col] = 0.0

    X_base = df[base_feature_names].values.astype(np.float32)
    X_base = np.nan_to_num(X_base, nan=0.0)

    seq_len = LM.LSTM_SEQ_LEN
    n_samples = len(df)
    sequences = []
    for i in range(n_samples):
        start_idx = max(0, i - seq_len + 1)
        sub_seq = X_base[start_idx : i + 1]
        if len(sub_seq) < seq_len:
            pad = np.tile(sub_seq[0], (seq_len - len(sub_seq), 1))
            sub_seq = np.vstack([pad, sub_seq])
        sequences.append(sub_seq)

    X_seq = np.array(sequences, dtype=np.float32)
    orig_shape = X_seq.shape
    X_scaled = lstm_scaler.transform(X_seq.reshape(-1, orig_shape[-1])).reshape(orig_shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_net.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        logits = lstm_net(inputs).squeeze(-1)
        probas = torch.sigmoid(logits).cpu().numpy()
        if np.ndim(probas) == 0:
            probas = np.array([probas.item()])

    results = []
    for i, p in enumerate(probas):
        is_attack = bool(p >= thresh)
        results.append(
            {
                "row": i,
                "model": "LSTM",
                "probability": round(float(p), 4),
                "threshold": round(float(thresh), 4),
                "attack_predicted": is_attack,
            }
        )

    return pd.DataFrame(results)


def predict_pcap(
    pcap_path: str,
    window_seconds: float = 10.0,
    use_calibrated_threshold: bool = True,
    custom_threshold: float | None = None,
    include_lstm: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract dual-level traffic features from a raw PCAP and run LSTM forecasting."""
    features_df = process_capture(pcap_path, window_sec=window_seconds)
    if features_df.empty:
        return features_df, pd.DataFrame()

    preds_df = predict(
        features_df,
        use_calibrated_threshold=use_calibrated_threshold,
        custom_threshold=custom_threshold,
        include_lstm=include_lstm,
    )
    return features_df, preds_df
