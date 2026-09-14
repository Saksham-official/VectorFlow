"""High-performance streaming PCAP/PCAPNG feature extractor using dpkt.

Extracts both Flow-level (NetFlow/IPFIX) and Packet-level (PCAP-derived) metrics
and synchronizes them into uniform 10-second time windows.
"""

from __future__ import annotations

import collections
import datetime
import math
import os
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dpkt
import numpy as np
import pandas as pd

# 15 Flow-level base features + 11 Packet-level base features = 26 base features
FLOW_BASE_FEATURES = [
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

PACKET_BASE_FEATURES = [
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

DUAL_BASE_FEATURES = FLOW_BASE_FEATURES + PACKET_BASE_FEATURES


class WelfordVariance:
    """Online streaming mean and variance using Welford's algorithm."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, val: float):
        self.count += 1
        delta = val - self.mean
        self.mean += delta / self.count
        delta2 = val - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.M2 / self.count if self.count > 1 else 0.0


def _inet_to_str(inet: bytes) -> str:
    """Convert binary IP address to string."""
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except (ValueError, OSError):
        try:
            return socket.inet_ntop(socket.AF_INET6, inet)
        except Exception:
            return "0.0.0.0"


class WindowAggregator:
    """Maintains packet & flow statistics for one 10-second window."""

    def __init__(self, window_start: datetime.datetime):
        self.window_start = window_start

        # Packet-level trackers
        self.ttl_values: List[int] = []
        self.tcp_windows: List[int] = []
        self.fragment_count: int = 0
        self.payload_sizes: List[int] = []
        self.retrans_count: int = 0

        # Port scan sequencing: src_ip -> list of (timestamp, dst_port)
        self.src_ports_accessed: Dict[str, List[int]] = collections.defaultdict(list)
        self.src_ports_timestamps: Dict[str, List[float]] = collections.defaultdict(list)

        # Flow-level trackers: flow_key -> flow metrics dict
        # flow_key: (src_ip, dst_ip, src_port, dst_port, proto)
        self.flows: Dict[Tuple[str, str, int, int, int], Dict[str, Any]] = {}
        self.dst_ports_set: set[int] = set()

        # Flag counts
        self.syn_count: int = 0
        self.ack_count: int = 0
        self.rst_count: int = 0
        self.fin_count: int = 0
        self.psh_count: int = 0
        self.urg_count: int = 0

    def add_packet(
        self,
        ts: float,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        proto: int,
        ttl: int,
        session_var: float,
        is_frag: bool,
        tcp_win: Optional[int],
        tcp_flags: int,
        payload_len: int,
        is_retrans: bool,
    ):
        # 1. Packet metrics
        self.ttl_values.append(ttl)
        if is_frag:
            self.fragment_count += 1
        if tcp_win is not None:
            self.tcp_windows.append(tcp_win)
        self.payload_sizes.append(payload_len)
        if is_retrans:
            self.retrans_count += 1

        self.src_ports_accessed[src_ip].append(dst_port)
        self.src_ports_timestamps[src_ip].append(ts)
        self.dst_ports_set.add(dst_port)

        # 2. TCP flags
        if proto == 6:  # TCP
            if tcp_flags & dpkt.tcp.TH_SYN:
                self.syn_count += 1
            if tcp_flags & dpkt.tcp.TH_ACK:
                self.ack_count += 1
            if tcp_flags & dpkt.tcp.TH_RST:
                self.rst_count += 1
            if tcp_flags & dpkt.tcp.TH_FIN:
                self.fin_count += 1
            if tcp_flags & dpkt.tcp.TH_PUSH:
                self.psh_count += 1
            if tcp_flags & dpkt.tcp.TH_URG:
                self.urg_count += 1

        # 3. Flow aggregation
        canonical_key = (
            min((src_ip, src_port), (dst_ip, dst_port)),
            max((src_ip, src_port), (dst_ip, dst_port)),
            proto,
        )

        if canonical_key not in self.flows:
            self.flows[canonical_key] = {
                "fwd_endpoint": (src_ip, src_port),
                "start_time": ts,
                "last_time": ts,
                "fwd_pkts": 0,
                "bwd_pkts": 0,
                "fwd_bytes": 0,
                "bwd_bytes": 0,
                "iats": [],
            }

        fl = self.flows[canonical_key]
        is_fwd = (src_ip, src_port) == fl["fwd_endpoint"]
        if ts > fl["last_time"]:
            fl["iats"].append(ts - fl["last_time"])
        fl["last_time"] = ts

        if is_fwd:
            fl["fwd_pkts"] += 1
            fl["fwd_bytes"] += payload_len
        else:
            fl["bwd_pkts"] += 1
            fl["bwd_bytes"] += payload_len

    def compute_features(self, session_variances: List[float]) -> Dict[str, float]:
        n_pkts = len(self.ttl_values)
        if n_pkts == 0:
            return {col: 0.0 for col in DUAL_BASE_FEATURES}

        # Packet-level features
        ttl_mean = float(np.mean(self.ttl_values))
        ttl_var = float(np.mean(session_variances)) if session_variances else 0.0

        win_mean = float(np.mean(self.tcp_windows)) if self.tcp_windows else 0.0
        win_std = float(np.std(self.tcp_windows)) if len(self.tcp_windows) > 1 else 0.0

        p_mean = float(np.mean(self.payload_sizes)) if self.payload_sizes else 0.0
        p_std = float(np.std(self.payload_sizes)) if len(self.payload_sizes) > 1 else 0.0
        zero_p_ratio = float(sum(1 for p in self.payload_sizes if p == 0) / n_pkts)

        # Port scan signatures
        # Sequential scan score: maximum sequential delta ratio among any single source IP
        max_seq_score = 0.0
        port_entropies = []

        for src, ports in self.src_ports_accessed.items():
            if len(ports) >= 2:
                src_seq = sum(1 for k in range(1, len(ports)) if abs(ports[k] - ports[k - 1]) == 1)
                src_score = src_seq / (len(ports) - 1)
                if src_score > max_seq_score:
                    max_seq_score = src_score

                # Port entropy: -sum(p * log2(p))
                counts = collections.Counter(ports)
                total_p = len(ports)
                ent = -sum((c / total_p) * math.log2(c / total_p) for c in counts.values())
                port_entropies.append(ent)

        seq_scan_score = float(max_seq_score)
        rand_scan_entropy = float(max(port_entropies)) if port_entropies else 0.0

        # Flow-level features
        tot_fwd_pkts = sum(fl["fwd_pkts"] for fl in self.flows.values())
        tot_bwd_pkts = sum(fl["bwd_pkts"] for fl in self.flows.values())
        tot_fwd_bytes = sum(fl["fwd_bytes"] for fl in self.flows.values())
        tot_bwd_bytes = sum(fl["bwd_bytes"] for fl in self.flows.values())

        durations = [(fl["last_time"] - fl["start_time"]) * 1000.0 for fl in self.flows.values()]
        flow_dur_mean = float(np.mean(durations)) if durations else 0.0
        flow_dur_std = float(np.std(durations)) if len(durations) > 1 else 0.0

        all_iats = []
        for fl in self.flows.values():
            all_iats.extend(fl["iats"])
        iat_mean = float(np.mean(all_iats) * 1000.0) if all_iats else 0.0

        bwd_fwd_ratio = float(tot_bwd_pkts / (tot_fwd_pkts + 1e-6))

        return {
            # Flow-level
            "Tot Fwd Pkts_sum": float(tot_fwd_pkts),
            "Tot Bwd Pkts_sum": float(tot_bwd_pkts),
            "TotLen Fwd Pkts_sum": float(tot_fwd_bytes),
            "TotLen Bwd Pkts_sum": float(tot_bwd_bytes),
            "Flow Duration_mean": flow_dur_mean,
            "Flow Duration_std": flow_dur_std,
            "Flow IAT Mean_mean": iat_mean,
            "Dst Port_nunique": float(len(self.dst_ports_set)),
            "SYN Flag Cnt_sum": float(self.syn_count),
            "ACK Flag Cnt_sum": float(self.ack_count),
            "RST Flag Cnt_sum": float(self.rst_count),
            "FIN Flag Cnt_sum": float(self.fin_count),
            "PSH Flag Cnt_sum": float(self.psh_count),
            "flow_count": float(len(self.flows)),
            "bwd_fwd_pkt_ratio": bwd_fwd_ratio,
            # Packet-level
            "ttl_mean": ttl_mean,
            "ttl_session_variance": ttl_var,
            "tcp_window_mean": win_mean,
            "tcp_window_std": win_std,
            "ip_fragment_count": float(self.fragment_count),
            "payload_size_mean": p_mean,
            "payload_size_std": p_std,
            "zero_payload_ratio": zero_p_ratio,
            "tcp_retrans_count": float(self.retrans_count),
            "sequential_scan_score": seq_scan_score,
            "random_scan_entropy": rand_scan_entropy,
        }


def extract_pcap_features(
    pcap_path: str | Path,
    window_sec: float = 10.0,
    max_packets: Optional[int] = None,
) -> pd.DataFrame:
    """Read PCAP / PCAPNG and produce 10-second windowed dual-level features."""
    pcap_path = Path(pcap_path)
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    # Open file with dpkt
    with open(pcap_path, "rb") as f:
        try:
            reader = dpkt.pcap.Reader(f)
        except Exception:
            f.seek(0)
            try:
                reader = dpkt.pcapng.Reader(f)
            except Exception as e:
                raise ValueError(f"Could not parse file as PCAP or PCAPNG: {e}")

        # Active session trackers
        # 5-tuple -> WelfordVariance
        session_ttls: Dict[Tuple[str, str, int, int, int], WelfordVariance] = collections.defaultdict(WelfordVariance)

        # Sliding window for retransmissions: (5-tuple, seq) -> last_seen_ts
        retrans_seen: Dict[Tuple[str, str, int, int, int, int], float] = {}

        # Window buckets
        windows: Dict[int, WindowAggregator] = {}
        window_session_vars: Dict[int, List[float]] = collections.defaultdict(list)

        base_epoch = None
        pkt_idx = 0

        for ts, buf in reader:
            pkt_idx += 1
            if max_packets and pkt_idx > max_packets:
                break

            if base_epoch is None:
                base_epoch = ts

            # Decode Link Layer
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                ip = eth.data
            except Exception:
                try:
                    sll = dpkt.sll.SLL(buf)
                    ip = sll.data
                except Exception:
                    try:
                        ip = dpkt.ip.IP(buf)
                    except Exception:
                        continue

            if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                continue

            src_ip = _inet_to_str(ip.src)
            dst_ip = _inet_to_str(ip.dst)
            proto = ip.p if hasattr(ip, "p") else (ip.nxt if hasattr(ip, "nxt") else 0)
            ttl = ip.ttl if hasattr(ip, "ttl") else (ip.hlim if hasattr(ip, "hlim") else 64)

            # Fragmentation
            is_frag = False
            if isinstance(ip, dpkt.ip.IP):
                is_frag = bool(ip.mf or ip.offset > 0)

            # Transport layer
            src_port, dst_port = 0, 0
            tcp_win = None
            tcp_flags = 0
            payload_len = 0
            seq_num = 0

            if isinstance(ip.data, dpkt.tcp.TCP):
                tcp = ip.data
                src_port = tcp.sport
                dst_port = tcp.dport
                tcp_win = tcp.win
                tcp_flags = tcp.flags
                payload_len = len(tcp.data)
                seq_num = tcp.seq
            elif isinstance(ip.data, dpkt.udp.UDP):
                udp = ip.data
                src_port = udp.sport
                dst_port = udp.dport
                payload_len = len(udp.data)
            elif isinstance(ip.data, (bytes, bytearray)):
                payload_len = len(ip.data)

            # Session TTL Variance
            session_key = (
                min((src_ip, src_port), (dst_ip, dst_port)),
                max((src_ip, src_port), (dst_ip, dst_port)),
                proto,
            )
            session_welford = session_ttls[session_key]
            session_welford.update(ttl)
            curr_sess_var = session_welford.variance

            # Retransmission check
            is_retrans = False
            if proto == 6 and payload_len > 0:  # TCP with payload
                retrans_key = (src_ip, dst_ip, src_port, dst_port, proto, seq_num)
                if retrans_key in retrans_seen:
                    if ts - retrans_seen[retrans_key] <= 3.0:
                        is_retrans = True
                retrans_seen[retrans_key] = ts
                if len(retrans_seen) > 200_000:
                    retrans_seen.clear()

            # Assign to 10-second window bucket
            win_idx = int((ts - base_epoch) // window_sec)
            if win_idx not in windows:
                win_dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
                windows[win_idx] = WindowAggregator(win_dt)

            windows[win_idx].add_packet(
                ts=ts,
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                proto=proto,
                ttl=ttl,
                session_var=curr_sess_var,
                is_frag=is_frag,
                tcp_win=tcp_win,
                tcp_flags=tcp_flags,
                payload_len=payload_len,
                is_retrans=is_retrans,
            )
            window_session_vars[win_idx].append(curr_sess_var)

    # Compile windows into DataFrame
    rows = []
    sorted_win_indices = sorted(windows.keys())
    if not sorted_win_indices:
        return pd.DataFrame(columns=["window_start"] + DUAL_BASE_FEATURES)

    for w_idx in sorted_win_indices:
        agg = windows[w_idx]
        sess_vars = window_session_vars[w_idx]
        feats = agg.compute_features(sess_vars)
        feats["window_start"] = agg.window_start.strftime("%Y-%m-%d %H:%M:%S")
        rows.append(feats)

    df = pd.DataFrame(rows)
    cols = ["window_start"] + DUAL_BASE_FEATURES
    return df[cols]
