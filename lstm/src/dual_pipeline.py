"""Unified Dual-Level Feature Pipeline.

Processes raw .pcap, .pcapng, or flow .csv into synchronized 10-second network state
windows with lag/delta temporal features and leak-free attack forecasting targets.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from packet_extractor import (
    DUAL_BASE_FEATURES,
    FLOW_BASE_FEATURES,
    PACKET_BASE_FEATURES,
    extract_pcap_features,
)

LAGS = (1, 2, 3)
HORIZON_WINDOWS = 3  # 3 x 10s = 30s lead-time forecast horizon


def get_dual_base_features() -> List[str]:
    return list(DUAL_BASE_FEATURES)


def get_all_feature_names() -> List[str]:
    """Return all 130 features: 26 base features + 3 lags + 1 delta per feature."""
    cols = list(DUAL_BASE_FEATURES)
    for f in DUAL_BASE_FEATURES:
        for lag in LAGS:
            cols.append(f"{f}_lag{lag}")
        cols.append(f"{f}_delta1")
    return cols


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute lag1, lag2, lag3 and delta1 for all 26 dual base features."""
    df = df.sort_values("window_start").reset_index(drop=True)
    history = {}

    for f in DUAL_BASE_FEATURES:
        if f not in df.columns:
            df[f] = 0.0

        for lag in LAGS:
            history[f"{f}_lag{lag}"] = df[f].shift(lag)
        history[f"{f}_delta1"] = df[f] - df[f].shift(1)

    hist_df = pd.DataFrame(history, index=df.index)
    result = pd.concat([df, hist_df], axis=1)

    # Fill initial boundary NaNs cleanly
    for col in hist_df.columns:
        if col.endswith("_delta1"):
            result[col] = result[col].fillna(0.0)
        else:
            base_col = col.rsplit("_lag", 1)[0]
            result[col] = result[col].fillna(result[base_col])

    return result


def add_forecasting_target(
    df: pd.DataFrame,
    horizon: int = HORIZON_WINDOWS,
    attack_col: str = "has_attack",
) -> pd.DataFrame:
    """Add leak-free forecasting target.

    Future_Attack_Target = 1 if an attack initiates within t+1 to t+horizon,
    while t itself is benign (forecasting precursor, not ongoing attack detection).
    """
    if attack_col not in df.columns:
        # Default: no attack annotations
        df["Future_Attack_Target"] = 0.0
        return df

    has_attack = df[attack_col].astype(float)
    # Future attack within horizon
    future_slices = pd.concat([has_attack.shift(-k) for k in range(1, horizon + 1)], axis=1)
    future_target = (future_slices.max(axis=1) > 0).astype(float)

    df["Future_Attack_Target"] = future_target.fillna(0.0)

    # Label precursor windows: current window is benign (0) but future is attacked (1)
    df["is_precursor"] = (df["Future_Attack_Target"] == 1.0) & (has_attack == 0.0)
    return df


def process_capture(
    input_path: str | Path,
    max_packets: Optional[int] = None,
    window_sec: float = 10.0,
) -> pd.DataFrame:
    """End-to-end processing of PCAP or CSV into complete feature DataFrame."""
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    if suffix in (".pcap", ".pcapng", ".cap"):
        df_base = extract_pcap_features(input_path, window_sec=window_sec, max_packets=max_packets)
    elif suffix == ".csv":
        df_raw = pd.read_csv(input_path)
        # Ensure timestamp exists
        ts_col = "window_start" if "window_start" in df_raw.columns else "Timestamp"
        if ts_col not in df_raw.columns:
            raise ValueError(f"CSV missing timestamp column (expected 'window_start' or 'Timestamp')")

        df_base = df_raw.copy()
        if ts_col != "window_start":
            df_base["window_start"] = pd.to_datetime(df_base[ts_col]).dt.strftime("%Y-%m-%d %H:%M:%S")

        # Impute missing packet-level features if running on legacy flow-only CSV
        defaults = {
            "ttl_mean": 64.0,
            "ttl_session_variance": 0.05,
            "tcp_window_mean": 29200.0,
            "tcp_window_std": 100.0,
            "ip_fragment_count": 0.0,
            "payload_size_mean": 250.0,
            "payload_size_std": 150.0,
            "zero_payload_ratio": 0.10,
            "tcp_retrans_count": 0.0,
            "sequential_scan_score": 0.0,
            "random_scan_entropy": 0.0,
        }
        for p_feat in PACKET_BASE_FEATURES:
            if p_feat not in df_base.columns:
                df_base[p_feat] = defaults.get(p_feat, 0.0)

        for f_feat in FLOW_BASE_FEATURES:
            if f_feat not in df_base.columns:
                df_base[f_feat] = 0.0
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Must be .pcap, .pcapng, or .csv")

    if len(df_base) == 0:
        return pd.DataFrame(columns=["window_start"] + get_all_feature_names())

    return add_temporal_features(df_base)
