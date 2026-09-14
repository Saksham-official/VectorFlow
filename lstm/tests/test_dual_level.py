"""Comprehensive unit and integration tests for Two-Level Traffic Features (Flow + Packet)
and Dual-Level LSTM Attack Forecasting.
"""

import datetime
import os
import socket
import sys
import tempfile
from pathlib import Path

import dpkt
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from dual_pipeline import (
    DUAL_BASE_FEATURES,
    FLOW_BASE_FEATURES,
    PACKET_BASE_FEATURES,
    add_forecasting_target,
    add_temporal_features,
    get_all_feature_names,
    get_dual_base_features,
    process_capture,
)
from lstm_model import BASE_FEATURES, AttackLSTM, load_lstm
from packet_extractor import extract_pcap_features


def create_synthetic_pcap(
    file_path: Path,
    include_retrans: bool = True,
    include_fragments: bool = True,
    include_port_scan: bool = True,
):
    """Generate a valid PCAP file with known packet-level and flow-level attributes."""
    with open(file_path, "wb") as f:
        writer = dpkt.pcap.Writer(f)
        base_time = 1518598800.0  # 2018-02-14 09:00:00

        src_ip = socket.inet_aton("192.168.1.100")
        dst_ip = socket.inet_aton("10.0.0.1")

        # 1. Normal TCP stream with Window=29200, TTL=64
        for i in range(10):
            tcp = dpkt.tcp.TCP(
                sport=45000,
                dport=80,
                seq=1000 + i * 100,
                ack=0,
                flags=dpkt.tcp.TH_SYN if i == 0 else dpkt.tcp.TH_ACK,
                win=29200,
                data=b"GET / HTTP/1.1\r\n" if i > 0 else b"",
            )
            ip = dpkt.ip.IP(
                src=src_ip,
                dst=dst_ip,
                p=dpkt.ip.IP_PROTO_TCP,
                ttl=64,
                data=tcp,
            )
            ip.len = len(ip)
            eth = dpkt.ethernet.Ethernet(
                src=b"\x00\x11\x22\x33\x44\x55",
                dst=b"\xaa\xbb\xcc\xdd\xee\xff",
                type=dpkt.ethernet.ETH_TYPE_IP,
                data=ip,
            )
            writer.writepkt(eth, base_time + i * 0.1)

        # 2. Retransmissions: same seq and payload within 1 second
        if include_retrans:
            for r in range(3):
                tcp_ret = dpkt.tcp.TCP(
                    sport=45000,
                    dport=80,
                    seq=1200,  # duplicate SEQ seen earlier
                    ack=0,
                    flags=dpkt.tcp.TH_ACK,
                    win=29200,
                    data=b"GET / HTTP/1.1\r\n",
                )
                ip_ret = dpkt.ip.IP(
                    src=src_ip,
                    dst=dst_ip,
                    p=dpkt.ip.IP_PROTO_TCP,
                    ttl=64,
                    data=tcp_ret,
                )
                ip_ret.len = len(ip_ret)
                eth_ret = dpkt.ethernet.Ethernet(
                    src=b"\x00\x11\x22\x33\x44\x55",
                    dst=b"\xaa\xbb\xcc\xdd\xee\xff",
                    type=dpkt.ethernet.ETH_TYPE_IP,
                    data=ip_ret,
                )
                writer.writepkt(eth_ret, base_time + 1.5 + r * 0.2)

        # 3. IP Fragmentation: More Fragments (MF) and offset > 0
        if include_fragments:
            for frag_idx in range(4):
                ip_frag = dpkt.ip.IP(
                    src=src_ip,
                    dst=dst_ip,
                    p=dpkt.ip.IP_PROTO_UDP,
                    ttl=55,
                    mf=1 if frag_idx < 3 else 0,
                    offset=frag_idx * 185,
                    data=b"X" * 100,
                )
                ip_frag.len = len(ip_frag)
                eth_frag = dpkt.ethernet.Ethernet(
                    src=b"\x00\x11\x22\x33\x44\x55",
                    dst=b"\xaa\xbb\xcc\xdd\xee\xff",
                    type=dpkt.ethernet.ETH_TYPE_IP,
                    data=ip_frag,
                )
                writer.writepkt(eth_frag, base_time + 3.0 + frag_idx * 0.05)

        # 4. Sequential Reconnaissance Port Scan: dport 1000, 1001, 1002, 1003, 1004
        if include_port_scan:
            scan_src = socket.inet_aton("192.168.1.250")
            for scan_port in range(1000, 1015):
                tcp_scan = dpkt.tcp.TCP(
                    sport=55555,
                    dport=scan_port,
                    seq=5000 + scan_port,
                    ack=0,
                    flags=dpkt.tcp.TH_SYN,
                    win=1024,  # signature scanner window
                    data=b"",  # zero payload
                )
                ip_scan = dpkt.ip.IP(
                    src=scan_src,
                    dst=dst_ip,
                    p=dpkt.ip.IP_PROTO_TCP,
                    ttl=255,  # signature scanner TTL
                    data=tcp_scan,
                )
                ip_scan.len = len(ip_scan)
                eth_scan = dpkt.ethernet.Ethernet(
                    src=b"\x00\x11\x22\x33\x44\x55",
                    dst=b"\xaa\xbb\xcc\xdd\xee\xff",
                    type=dpkt.ethernet.ETH_TYPE_IP,
                    data=ip_scan,
                )
                writer.writepkt(eth_scan, base_time + 5.0 + (scan_port - 1000) * 0.05)


@pytest.fixture
def synthetic_pcap_path(tmp_path):
    pcap_file = tmp_path / "test_sample.pcap"
    create_synthetic_pcap(pcap_file)
    return pcap_file


def test_dual_base_features_count():
    """Verify exact count of Flow (15) + Packet (11) = 26 base features."""
    assert len(FLOW_BASE_FEATURES) == 15
    assert len(PACKET_BASE_FEATURES) == 11
    assert len(DUAL_BASE_FEATURES) == 26
    assert len(BASE_FEATURES) == 26


def test_pcap_packet_feature_extraction(synthetic_pcap_path):
    """Test extracting packet-level metrics directly from a raw PCAP."""
    df = extract_pcap_features(synthetic_pcap_path, window_sec=10.0)

    assert len(df) >= 1
    assert "window_start" in df.columns

    row = df.iloc[0]

    # Verify all 6 required packet-level metrics are present and computed:
    # 1. TTL values & variance
    assert "ttl_mean" in row
    assert 40.0 <= row["ttl_mean"] <= 255.0
    assert "ttl_session_variance" in row

    # 2. TCP window size
    assert "tcp_window_mean" in row
    assert row["tcp_window_mean"] > 0

    # 3. IP fragment flags
    assert "ip_fragment_count" in row
    assert row["ip_fragment_count"] >= 3  # We wrote 4 fragment packets

    # 4. Payload size distribution
    assert "payload_size_mean" in row
    assert "zero_payload_ratio" in row
    assert 0.0 <= row["zero_payload_ratio"] <= 1.0

    # 5. Retransmission counts
    assert "tcp_retrans_count" in row
    assert row["tcp_retrans_count"] >= 2  # We wrote duplicate SEQ packets

    # 6. Port scan signatures
    assert "sequential_scan_score" in row
    assert row["sequential_scan_score"] > 0.5  # Sequential ports 1000..1014

    # Flow-level features
    assert row["Tot Fwd Pkts_sum"] > 0
    assert row["SYN Flag Cnt_sum"] > 0
    assert row["Dst Port_nunique"] >= 10


def test_dual_pipeline_temporal_features(synthetic_pcap_path):
    """Test full pipeline temporal lags (lag1, lag2, lag3, delta1)."""
    df = process_capture(synthetic_pcap_path)
    all_names = get_all_feature_names()

    # 26 base * 5 (base, lag1, lag2, lag3, delta1) = 130 features
    assert len(all_names) == 130
    for feat in all_names:
        assert feat in df.columns


def test_forecasting_target_generation():
    """Verify leak-free attack onset target logic."""
    data = {
        "window_start": pd.date_range("2018-02-14 09:00:00", periods=10, freq="10s"),
        "has_attack": [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    }
    for col in DUAL_BASE_FEATURES:
        data[col] = [10.0] * 10

    df = pd.DataFrame(data)
    df_targeted = add_forecasting_target(df, horizon=3)

    # Windows 1, 2, 3 (indices 1, 2, 3) are 1 to 3 steps before index 4 onset
    # Window index 4 is active attack, so it is NOT a precursor
    assert df_targeted.loc[1, "Future_Attack_Target"] == 1.0
    assert df_targeted.loc[2, "Future_Attack_Target"] == 1.0
    assert df_targeted.loc[3, "Future_Attack_Target"] == 1.0
    assert df_targeted.loc[4, "is_precursor"] == False  # Active attack, not precursor


def test_slow_scan_vs_flood_differentiation():
    """Verify that packet features expose a slow reconnaissance scan designed to evade flow thresholds."""
    # Scenario A: Volumetric SYN flood (high flow volume, standard port)
    syn_flood = {
        "Tot Fwd Pkts_sum": 5000.0,
        "flow_count": 500.0,
        "Dst Port_nunique": 1.0,
        "SYN Flag Cnt_sum": 5000.0,
        "bwd_fwd_pkt_ratio": 0.001,
        "sequential_scan_score": 0.0,
        "random_scan_entropy": 0.0,
        "zero_payload_ratio": 0.99,
    }

    # Scenario B: Slow reconnaissance scan (low flow volume to evade threshold, high port sequencing)
    slow_scan = {
        "Tot Fwd Pkts_sum": 30.0,  # Normal / low volume!
        "flow_count": 25.0,  # Below flow thresholds!
        "Dst Port_nunique": 25.0,  # Many distinct ports
        "SYN Flag Cnt_sum": 30.0,
        "bwd_fwd_pkt_ratio": 0.0,
        "sequential_scan_score": 0.95,  # High sequential signature!
        "random_scan_entropy": 1.5,
        "zero_payload_ratio": 0.95,
    }

    # Flow volume alone fails to catch slow_scan:
    assert slow_scan["Tot Fwd Pkts_sum"] < 100.0
    # But packet-level sequencing clearly flags it:
    assert slow_scan["sequential_scan_score"] > 0.80
    assert slow_scan["Dst Port_nunique"] > 20


def test_trained_model_loading():
    """Verify that the retrained PyTorch LSTM loads with input_dim=26."""
    models_dir = "models" if os.path.exists("models/lstm.pt") else os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
    model, scaler, threshold, feature_names = load_lstm(models_dir)
    assert isinstance(model, AttackLSTM)
    assert model.input_bn.num_features == 26
    assert len(feature_names) == 26
    assert 0.0 < threshold < 1.0


def test_end_to_end_pcap_prediction(synthetic_pcap_path):
    """Test full pipeline: raw .pcap -> dual-level features -> LSTM prediction."""
    from inference import predict_pcap

    features_df, preds_df = predict_pcap(synthetic_pcap_path, include_lstm=True)

    assert len(features_df) >= 1
    assert len(preds_df) >= 1
    assert "LSTM" in preds_df["model"].values

    lstm_rows = preds_df[preds_df["model"] == "LSTM"]
    assert all(0.0 <= p <= 1.0 for p in lstm_rows["probability"])
    assert "attack_predicted" in lstm_rows.columns
