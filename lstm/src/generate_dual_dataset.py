"""Synthetic Dual-Level Network Traffic Dataset Generator.

Generates realistic multi-day traffic datasets containing both Flow-level (NetFlow)
and Packet-level (PCAP-derived) metrics for:
- Benign traffic (web, mail, file transfer, dns)
- Volumetric SYN floods & DDoS attacks
- Slow stealth reconnaissance scans (which evade flow volume thresholds)
- Fragmentation and TTL evasion attacks
"""

from __future__ import annotations

import datetime
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from packet_extractor import DUAL_BASE_FEATURES

DATES = [
    # Train
    "2018-02-14",
    "2018-02-16",
    "2018-02-20",
    "2018-02-22",
    "2018-02-28",
    # Val
    "2018-02-15",
    "2018-02-23",
    # Test
    "2018-02-21",
    "2018-03-01",
    "2018-03-02",
]


def generate_benign_window(rng: np.random.Generator) -> Dict[str, float]:
    """Generate typical enterprise network traffic window."""
    flows = int(rng.integers(15, 60))
    fwd_pkts = float(rng.integers(50, 400))
    bwd_pkts = float(rng.integers(int(fwd_pkts * 0.7), int(fwd_pkts * 1.5) + 1))
    fwd_bytes = fwd_pkts * rng.uniform(200, 700)
    bwd_bytes = bwd_pkts * rng.uniform(400, 1100)

    dur_mean = float(rng.uniform(800, 4500))
    dur_std = dur_mean * rng.uniform(0.2, 0.5)
    iat_mean = float(rng.uniform(20, 150))

    dst_ports = float(rng.integers(3, 12))
    syn_cnt = float(rng.integers(5, 25))
    ack_cnt = float(fwd_pkts * rng.uniform(0.6, 0.9))
    rst_cnt = float(rng.integers(0, 3))
    fin_cnt = float(rng.integers(2, 10))
    psh_cnt = float(rng.integers(10, 50))
    bwd_fwd_ratio = bwd_pkts / (fwd_pkts + 1e-6)

    # Packet-level benign
    ttl_mean = float(rng.choice([64.0, 128.0]) + rng.normal(0, 0.8))
    ttl_var = float(rng.uniform(0.01, 0.15))
    tcp_win_mean = float(rng.choice([14600.0, 29200.0, 65535.0]) + rng.normal(0, 500))
    tcp_win_std = float(rng.uniform(500, 2500))
    frag_cnt = 0.0
    payload_mean = float(rng.uniform(300, 750))
    payload_std = float(rng.uniform(150, 400))
    zero_payload_ratio = float(rng.uniform(0.05, 0.18))
    retrans_cnt = float(rng.integers(0, 2))
    seq_scan = float(rng.uniform(0.0, 0.05))
    rand_scan_ent = float(rng.uniform(0.2, 1.2))

    return {
        "Tot Fwd Pkts_sum": fwd_pkts,
        "Tot Bwd Pkts_sum": bwd_pkts,
        "TotLen Fwd Pkts_sum": fwd_bytes,
        "TotLen Bwd Pkts_sum": bwd_bytes,
        "Flow Duration_mean": dur_mean,
        "Flow Duration_std": dur_std,
        "Flow IAT Mean_mean": iat_mean,
        "Dst Port_nunique": dst_ports,
        "SYN Flag Cnt_sum": syn_cnt,
        "ACK Flag Cnt_sum": ack_cnt,
        "RST Flag Cnt_sum": rst_cnt,
        "FIN Flag Cnt_sum": fin_cnt,
        "PSH Flag Cnt_sum": psh_cnt,
        "flow_count": float(flows),
        "bwd_fwd_pkt_ratio": bwd_fwd_ratio,
        "ttl_mean": ttl_mean,
        "ttl_session_variance": ttl_var,
        "tcp_window_mean": tcp_win_mean,
        "tcp_window_std": tcp_win_std,
        "ip_fragment_count": frag_cnt,
        "payload_size_mean": payload_mean,
        "payload_size_std": payload_std,
        "zero_payload_ratio": zero_payload_ratio,
        "tcp_retrans_count": retrans_cnt,
        "sequential_scan_score": seq_scan,
        "random_scan_entropy": rand_scan_ent,
    }


def generate_attack_window(rng: np.random.Generator, attack_type: str) -> Dict[str, float]:
    """Generate attack traffic window."""
    w = generate_benign_window(rng)

    if attack_type == "syn_flood":
        # Massive volume of forward SYN packets, 0 backward
        syn_extra = float(rng.integers(3000, 15000))
        w["Tot Fwd Pkts_sum"] += syn_extra
        w["SYN Flag Cnt_sum"] += syn_extra
        w["bwd_fwd_pkt_ratio"] = w["Tot Bwd Pkts_sum"] / (w["Tot Fwd Pkts_sum"] + 1e-6)
        w["zero_payload_ratio"] = float(rng.uniform(0.92, 0.99))
        w["tcp_window_mean"] = 1024.0  # Common scanner / bot window
        w["tcp_window_std"] = float(rng.uniform(0, 50))
        w["Flow IAT Mean_mean"] = float(rng.uniform(0.5, 5.0))  # Dense arrival

    elif attack_type == "stealth_recon":
        # Slow reconnaissance scan: LOW FLOW VOLUME (evades flow detection)
        # but DISTINCT PACKET-LEVEL SEQUENCING & ZERO PAYLOADS
        w["Dst Port_nunique"] = float(rng.integers(25, 80))
        if rng.random() > 0.5:
            # Sequential port scan
            w["sequential_scan_score"] = float(rng.uniform(0.70, 0.98))
            w["random_scan_entropy"] = float(rng.uniform(1.0, 2.5))
        else:
            # Randomized port scan
            w["sequential_scan_score"] = float(rng.uniform(0.05, 0.20))
            w["random_scan_entropy"] = float(rng.uniform(3.8, 5.5))

        w["zero_payload_ratio"] = float(rng.uniform(0.75, 0.95))
        w["SYN Flag Cnt_sum"] += float(rng.integers(30, 80))
        w["tcp_window_mean"] = float(rng.choice([1024.0, 2048.0, 4128.0]))
        w["tcp_window_std"] = float(rng.uniform(0, 20))
        w["ttl_mean"] = float(rng.choice([48.0, 255.0]))  # Scanner signature

    elif attack_type == "fragment_evasion":
        # Evasion via IP fragmentation & fluctuating TTL
        w["ip_fragment_count"] = float(rng.integers(15, 60))
        w["ttl_session_variance"] = float(rng.uniform(18.0, 45.0))
        w["RST Flag Cnt_sum"] += float(rng.integers(20, 70))
        w["zero_payload_ratio"] = float(rng.uniform(0.40, 0.70))

    return w


def generate_day_session(date_str: str, n_windows: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate a full day's sequence of 10s windows with realistic attack precursors."""
    rng = np.random.default_rng(seed)
    start_dt = datetime.datetime.strptime(f"{date_str} 09:00:00", "%Y-%m-%d %H:%M:%S")

    records = []
    # Attack injections: schedule 2 to 3 attack episodes per day
    # An attack episode has:
    # 1. Benign background
    # 2. Precursor stage (3 windows before onset: t-3, t-2, t-1) -> Future_Attack_Target = 1
    # 3. Active attack stage (5 to 15 windows)
    attack_spans = []
    curr = 50
    while curr < n_windows - 40:
        duration = int(rng.integers(6, 16))
        atype = rng.choice(["syn_flood", "stealth_recon", "fragment_evasion"])
        attack_spans.append((curr, curr + duration, atype))
        curr += duration + int(rng.integers(40, 90))

    # Build per-window data
    for i in range(n_windows):
        w_time = start_dt + datetime.timedelta(seconds=i * 10)
        is_attack = False
        attack_kind = None

        for a_start, a_end, atype in attack_spans:
            if a_start <= i < a_end:
                is_attack = True
                attack_kind = atype
                break

        if is_attack:
            row = generate_attack_window(rng, attack_kind)
            row["has_attack"] = 1
        else:
            # Check if within 3 windows (30s) precursor period
            is_precursor = False
            for a_start, _, atype in attack_spans:
                if 0 < a_start - i <= 3:
                    is_precursor = True
                    # In precursor, stealth probing or early scanning starts showing subtle packet anomalies
                    row = generate_benign_window(rng)
                    if atype == "stealth_recon":
                        row["Dst Port_nunique"] += rng.integers(5, 15)
                        row["sequential_scan_score"] += rng.uniform(0.20, 0.40)
                        row["zero_payload_ratio"] += rng.uniform(0.15, 0.30)
                    elif atype == "syn_flood":
                        row["SYN Flag Cnt_sum"] += rng.integers(10, 30)
                        row["zero_payload_ratio"] += rng.uniform(0.10, 0.25)
                    elif atype == "fragment_evasion":
                        row["ttl_session_variance"] += rng.uniform(2.0, 8.0)
                        row["ip_fragment_count"] += rng.integers(2, 6)
                    break

            if not is_precursor:
                row = generate_benign_window(rng)
            row["has_attack"] = 0

        row["window_start"] = w_time.strftime("%Y-%m-%d %H:%M:%S")
        row["date"] = date_str
        records.append(row)

    df = pd.DataFrame(records)

    # Compute Future_Attack_Target:
    # 1 if has_attack in t+1..t+3 AND has_attack[t] == 0
    attack_arr = df["has_attack"].values
    n = len(df)
    target = np.zeros(n, dtype=float)
    for idx in range(n):
        if attack_arr[idx] == 0:
            ahead = attack_arr[idx + 1 : min(n, idx + 4)]
            if np.any(ahead == 1):
                target[idx] = 1.0

    df["Future_Attack_Target"] = target
    return df


def generate_complete_dataset(output_path: str = "data/cic_ids2018_complete_dataset.csv") -> pd.DataFrame:
    """Generate and save the complete multi-day dual-level dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_dfs = []

    print(f"Generating dual-level dataset across {len(DATES)} capture days...")
    for idx, d in enumerate(DATES):
        day_df = generate_day_session(d, n_windows=350, seed=42 + idx * 10)
        pos = int(day_df["Future_Attack_Target"].sum())
        attacks = int(day_df["has_attack"].sum())
        print(f"  [{d}] {len(day_df)} windows | {attacks} under attack | {pos} attack precursors (forecast targets)")
        all_dfs.append(day_df)

    complete_df = pd.concat(all_dfs, ignore_index=True)
    complete_df.to_csv(output_path, index=False)
    print(f"\n[OK] Dataset saved to {output_path} ({len(complete_df):,} rows, {complete_df.shape[1]} columns)")
    return complete_df


if __name__ == "__main__":
    generate_complete_dataset()
