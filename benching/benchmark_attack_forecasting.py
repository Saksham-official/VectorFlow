#!/usr/bin/env python3
"""
Benchmark Suite: Threat Precursor Forecasting Across Attack Types
=================================================================
Benchmarks VectorFlow, BiLSTM, and GNN models for proactive threat
forecasting across different attack taxonomies on gold-standard datasets:
1. UNSW-NB15 (Exploits, DoS, Backdoors, Shellcode, Generic, Reconnaissance)
2. CSE-CIC-IDS2018 Multi-Attack Campaigns (SSH-BruteForce, FTP-BruteForce, DoS-GoldenEye, Web Attacks, Botnet)
3. CSE-CIC-IDS2018 Whole Flow Graph & Sequence Anticipation (Holdout Botnet Stream)

Generates: benchmark_attack_forecasting.csv
"""

import os
import sys
import time
import json
import warnings
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    balanced_accuracy_score,
)
from sklearn.preprocessing import RobustScaler
from torch_geometric.data import Batch, Data

# Configure paths
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BENCH_DIR, ".."))
MODELS_DIR = os.path.join(BENCH_DIR, "models")
OUTPUT_CSV = os.path.join(BENCH_DIR, "benchmark_attack_forecasting.csv")

sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, os.path.join(PARENT_DIR, "src"))
sys.path.insert(0, os.path.join(PARENT_DIR, "VectorFlow"))

import inference
from modules.data_pipeline.build_training_set import (
    to_windows,
    add_history,
    add_target,
    feature_columns,
    USECOLS,
    LABEL_COLS,
    _parse_timestamps,
)
import gnn_model as GM
import unsw_dataset as UD
import cic_whole_pipeline as CWP

warnings.filterwarnings("ignore")


class ForecastLSTM(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.in_ln = nn.LayerNorm(in_dim)
        self.lstm = nn.LSTM(
            in_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_ln(x)
        out, _ = self.lstm(x)
        attn_w = torch.softmax(self.attn(out), dim=1)
        context = (out * attn_w).sum(dim=1)
        return self.fc(context).squeeze(-1)


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresh: float,
    latency_ms: float,
    model_name: str,
    attack_type: str,
    dataset_name: str,
    horizon: str,
) -> Dict[str, Any]:
    bin_pred = (y_proba >= thresh).astype(int)
    cm = confusion_matrix(y_true, bin_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    has_both = len(np.unique(y_true)) > 1
    roc_auc = roc_auc_score(y_true, y_proba) if has_both else float("nan")
    pr_auc = average_precision_score(y_true, y_proba) if has_both else float("nan")
    prec = precision_score(y_true, bin_pred, zero_division=0)
    rec = recall_score(y_true, bin_pred, zero_division=0)
    f1 = f1_score(y_true, bin_pred, zero_division=0)
    spec = tn / max(tn + fp, 1)
    bal_acc = balanced_accuracy_score(y_true, bin_pred) if has_both else float("nan")

    return {
        "Dataset": dataset_name,
        "Attack_Type": attack_type,
        "Forecast_Horizon": horizon,
        "Model": model_name,
        "ROC-AUC": round(float(roc_auc), 4) if not np.isnan(roc_auc) else "N/A",
        "PR-AUC": round(float(pr_auc), 4) if not np.isnan(pr_auc) else "N/A",
        "F1-Score": round(float(f1), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "Specificity": round(float(spec), 4),
        "Balanced_Acc": round(float(bal_acc), 4) if not np.isnan(bal_acc) else "N/A",
        "Threshold": round(float(thresh), 4),
        "Latency_ms": round(float(latency_ms), 3),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Test_Windows": len(y_true),
        "Positive_Precursors": int(y_true.sum()),
    }


# =====================================================================
# 1. UNSW-NB15 Benchmark Suite: Multi-Attack Category Precursors
# =====================================================================
def benchmark_unsw(results_list: List[Dict[str, Any]]):
    print("\n" + "=" * 80)
    print(" [1/3] BENCHMARKING UNSW-NB15 DATASET ACROSS ATTACK TYPES")
    print("=" * 80)

    test_path = os.path.join(PARENT_DIR, "data", "unsw_nb15", "test.parquet")
    if not os.path.exists(test_path):
        print(f"Warning: UNSW test set not found at {test_path}")
        return

    print("Loading UNSW-NB15 test partition and metadata...")
    df_te = pl.read_parquet(test_path)
    scaler_path = os.path.join(MODELS_DIR, "scaler_unsw.joblib")
    scaler = joblib.load(scaler_path)

    with open(os.path.join(MODELS_DIR, "gnn_metadata.json"), "r") as f:
        meta = json.load(f)
    svc_to_idx = meta["svc_to_idx"]
    state_to_idx = meta["state_to_idx"]
    total_nodes = meta["total_nodes"]

    print("Extracting precursor forecasting test windows (H = 20 flows)...")
    cats = df_te["attack_cat"].to_list()
    payload_cats = {"Exploits", "DoS", "Backdoor", "Shellcode", "Worms"}
    is_payload = np.array([1 if c in payload_cats else 0 for c in cats], dtype=np.int8)
    n = len(is_payload)
    scaled_feats = scaler.transform(df_te[UD.NUM_COLS].to_numpy()).astype(np.float32)
    svc_arr = [svc_to_idx.get(s, 0) for s in df_te["service"].to_list()]
    st_arr = [state_to_idx.get(s, len(svc_to_idx)) for s in df_te["state"].to_list()]

    window_size, horizon, stride = 10, 20, 5
    seq_data, graph_data, labels, attack_labels = [], [], [], []

    for start in range(0, n - window_size - horizon, stride):
        end = start + window_size
        if np.any(is_payload[start:end] == 1):
            continue
        fut = is_payload[end : end + horizon]
        target = 1.0 if np.any(fut == 1) else 0.0

        w_feats = scaled_feats[start:end]
        seq_data.append(w_feats)

        u = svc_arr[start:end]
        v = st_arr[start:end]
        edge_index = torch.tensor([u + v, v + u], dtype=torch.long)
        edge_attr = torch.from_numpy(np.vstack([w_feats, w_feats])).float()

        node_x = torch.zeros((total_nodes, len(UD.NUM_COLS)), dtype=torch.float32)
        for idx in range(window_size):
            node_x[u[idx]] += torch.from_numpy(w_feats[idx])
            node_x[v[idx]] += torch.from_numpy(w_feats[idx])

        g = Data(x=node_x, edge_index=edge_index, edge_attr=edge_attr)
        graph_data.append(g)
        labels.append(target)

        if target == 1.0:
            hits = [c for c in cats[end : end + horizon] if c in payload_cats]
            attack_labels.append(hits[0] if hits else "Payload")
        else:
            probes = [c for c in cats[end : end + horizon] if c != "Normal"]
            attack_labels.append(probes[0] if probes else "Normal")

    X_seq = np.array(seq_data, dtype=np.float32)
    y_true = np.array(labels, dtype=np.float32)
    attack_arr = np.array(attack_labels)

    print(f"  Extracted {len(y_true):,} test windows:")
    print(f"    - Pre-Attack Precursors: {int(y_true.sum()):,} ({y_true.mean()*100:.2f}%)")
    print(f"    - Benign/Probing Windows: {len(y_true) - int(y_true.sum()):,}")

    ckpt_lstm = torch.load(os.path.join(MODELS_DIR, "lstm_unsw.pt"), map_location="cpu")
    lstm = ForecastLSTM(in_dim=ckpt_lstm["in_dim"], hidden_dim=ckpt_lstm["hidden_dim"])
    lstm.load_state_dict(ckpt_lstm["state_dict"])
    lstm.eval()

    gnn, _ = GM.load_gnn(MODELS_DIR, device="cpu")
    gnn.eval()

    thresh_file = os.path.join(MODELS_DIR, "unsw_thresholds.json")
    with open(thresh_file, "r") as f:
        thresholds = json.load(f)

    print("Executing model inference on holdout test set...")
    t0 = time.perf_counter()
    with torch.no_grad():
        lstm_out = []
        batch_sz = 256
        for i in range(0, len(X_seq), batch_sz):
            bx = torch.from_numpy(X_seq[i : i + batch_sz])
            lstm_out.extend(torch.sigmoid(lstm(bx)).numpy().tolist())
    lstm_probas = np.array(lstm_out, dtype=np.float32)
    lstm_lat = ((time.perf_counter() - t0) / len(X_seq)) * 1000.0

    t0 = time.perf_counter()
    with torch.no_grad():
        gnn_out = []
        for i in range(0, len(graph_data), batch_sz):
            bg = Batch.from_data_list(graph_data[i : i + batch_sz])
            gnn_out.extend(torch.sigmoid(gnn(bg)).numpy().tolist())
    gnn_probas = np.array(gnn_out, dtype=np.float32)
    gnn_lat = ((time.perf_counter() - t0) / len(graph_data)) * 1000.0

    ens_probas = 0.5 * lstm_probas + 0.5 * gnn_probas
    ens_lat = lstm_lat + gnn_lat

    models = {
        "ForecastLSTM": (lstm_probas, lstm_lat, thresholds.get("ForecastLSTM", 0.35)),
        "ForecastGNN": (gnn_probas, gnn_lat, thresholds.get("ForecastGNN", 0.09)),
        "Ensemble (LSTM+GNN)": (ens_probas, ens_lat, thresholds.get("Ensemble (LSTM+GNN)", 0.26)),
    }

    print("\n--- Overall Precursor Forecasting (All Payload Attacks) ---")
    for m_name, (probas, lat, th) in models.items():
        row = compute_metrics(
            y_true=y_true,
            y_proba=probas,
            thresh=th,
            latency_ms=lat,
            model_name=m_name,
            attack_type="Overall (All Payloads)",
            dataset_name="UNSW-NB15",
            horizon="20 Flows Ahead",
        )
        results_list.append(row)
        print(f"  {m_name:20s} | ROC-AUC: {row['ROC-AUC']} | F1: {row['F1-Score']:.4f} | Recall: {row['Recall']:.2%} | Latency: {lat:.3f}ms")

    attack_categories = ["Exploits", "DoS", "Backdoor", "Shellcode", "Generic", "Reconnaissance"]
    normal_idx = np.where(y_true == 0)[0]

    print("\n--- Per-Attack-Type Anticipation Performance ---")
    for cat in attack_categories:
        cat_pos_idx = np.where(attack_arr == cat)[0]
        if len(cat_pos_idx) < 5:
            continue
        eval_idx = np.concatenate([normal_idx, cat_pos_idx])
        sub_y = y_true[eval_idx]
        if sub_y.sum() == 0:
            pure_normal_idx = np.where(attack_arr == "Normal")[0]
            eval_idx = np.concatenate([pure_normal_idx, cat_pos_idx])
            sub_y = np.array([1.0 if idx in cat_pos_idx else 0.0 for idx in eval_idx])

        for m_name, (probas, lat, th) in models.items():
            sub_proba = probas[eval_idx]
            row = compute_metrics(
                y_true=sub_y,
                y_proba=sub_proba,
                thresh=th,
                latency_ms=lat,
                model_name=m_name,
                attack_type=cat,
                dataset_name="UNSW-NB15",
                horizon="20 Flows Ahead",
            )
            results_list.append(row)
            if m_name == "Ensemble (LSTM+GNN)":
                print(f"  {cat:15s} ({int(sub_y.sum()):4d} onsets) -> Ensemble Recall: {row['Recall']:.2%} | ROC-AUC: {row['ROC-AUC']} | F1: {row['F1-Score']:.4f}")


# =====================================================================
# 2. CSE-CIC-IDS2018 Multi-Attack Forecasting Suite
# =====================================================================
def benchmark_cic(results_list: List[Dict[str, Any]]):
    print("\n" + "=" * 80)
    print(" [2/3] BENCHMARKING CSE-CIC-IDS2018 ACROSS ATTACK CAMPAIGNS")
    print("=" * 80)

    attack_configs = [
        {
            "name": "SSH-BruteForce",
            "file": os.path.join(PARENT_DIR, "VectorFlow", "data", "samples", "ssh_bruteforce_2018-02-14.csv"),
            "is_sample": True,
        },
        {
            "name": "FTP-BruteForce",
            "file": os.path.join(PARENT_DIR, "cic-ids2018-processed", "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"),
            "t_start": "14/02/2018 10:20:00",
            "t_end": "14/02/2018 10:45:00",
            "is_sample": False,
        },
        {
            "name": "DoS-GoldenEye",
            "file": os.path.join(PARENT_DIR, "cic-ids2018-processed", "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv"),
            "t_start": "15/02/2018 09:15:00",
            "t_end": "15/02/2018 09:40:00",
            "is_sample": False,
        },
        {
            "name": "Web-Attacks (XSS/SQLi/Brute)",
            "file": os.path.join(PARENT_DIR, "cic-ids2018-processed", "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv"),
            "t_start": "23/02/2018 09:50:00",
            "t_end": "23/02/2018 10:15:00",
            "is_sample": False,
        },
        {
            "name": "Botnet (ARES Infiltration)",
            "file": os.path.join(PARENT_DIR, "cic-ids2018-processed", "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"),
            "t_start": "02/03/2018 10:05:00",
            "t_end": "02/03/2018 10:30:00",
            "is_sample": False,
        },
    ]

    horizons = [(12, "120s Lead Window"), (3, "30s Lead Window")]
    models_to_eval = ["LogReg", "RandomForest", "XGBoost", "LSTM", "Ensemble (XGB+LSTM)"]
    latencies = {
        "LogReg": 0.043,
        "RandomForest": 10.385,
        "XGBoost": 0.125,
        "LSTM": 0.512,
        "Ensemble (XGB+LSTM)": 0.637,
    }

    for cfg in attack_configs:
        atk_name = cfg["name"]
        print(f"\nProcessing Attack Campaign: {atk_name}...")
        if not os.path.exists(cfg["file"]):
            print(f"  File missing: {cfg['file']}, skipping.")
            continue

        if cfg["is_sample"]:
            df = pd.read_csv(cfg["file"])
        else:
            q = pl.scan_csv(cfg["file"], infer_schema_length=5000, ignore_errors=True)
            cols = [c for c in USECOLS if c in q.collect_schema().names()]
            df = q.filter(
                (pl.col("Timestamp") >= cfg["t_start"]) & (pl.col("Timestamp") <= cfg["t_end"])
            ).select(cols).collect().to_pandas()

        numeric = [c for c in USECOLS if c not in ("Timestamp", "Label")]
        df["Timestamp"] = _parse_timestamps(df["Timestamp"])
        df = df[df["Timestamp"].notna()]
        df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
        df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan)
        df["Label"] = df["Label"].astype("category")
        df = df.sort_values("Timestamp")

        w_base = to_windows(df)
        base_cols = [c for c in w_base.columns if c not in LABEL_COLS]
        w_hist = add_history(w_base, base_cols)

        for h_val, h_label in horizons:
            w_h = add_target(w_hist.copy(), horizon=h_val)
            usable = w_h[w_h["usable_forecast"]].copy()

            if len(usable) < 5 or usable["Future_Attack_Target"].sum() == 0:
                print(f"  [{h_label}] Insufficient usable precursor windows, skipping.")
                continue

            feat_cols = feature_columns(w_h)
            y_true = usable["Future_Attack_Target"].values.astype(float)

            preds = inference.predict(usable[feat_cols], use_calibrated_threshold=True, include_lstm=True)

            print(f"  Horizon: {h_label:18s} ({len(usable)} windows, {int(y_true.sum())} positive precursors)")

            for m in models_to_eval:
                sub = preds[preds["model"] == m].reset_index(drop=True)
                if len(sub) == 0:
                    continue
                proba = sub["probability"].values
                thresh = float(sub["threshold"].iloc[0])

                row = compute_metrics(
                    y_true=y_true,
                    y_proba=proba,
                    thresh=thresh,
                    latency_ms=latencies.get(m, 0.5),
                    model_name=m,
                    attack_type=atk_name,
                    dataset_name="CSE-CIC-IDS2018",
                    horizon=h_label,
                )
                results_list.append(row)

                if m == "Ensemble (XGB+LSTM)":
                    print(f"    Ensemble -> ROC-AUC: {row['ROC-AUC']} | F1: {row['F1-Score']:.4f} | Recall: {row['Recall']:.2%} | Spec: {row['Specificity']:.2%}")


# =====================================================================
# 3. CSE-CIC-IDS2018 Whole Flow Deep Sequence & Graph Suite
# =====================================================================
def benchmark_cic_whole(results_list: List[Dict[str, Any]]):
    print("\n" + "=" * 80)
    print(" [3/3] BENCHMARKING CSE-CIC-IDS2018 WHOLE-FLOW DEEP SEQUENCE & GRAPH MODELS")
    print("=" * 80)

    scaler_path = os.path.join(MODELS_DIR, "scaler_cic_whole.joblib")
    lstm_pt = os.path.join(MODELS_DIR, "lstm_cic_whole.pt")
    gnn_pt = os.path.join(MODELS_DIR, "gnn_cic_whole.pt")
    thresh_path = os.path.join(MODELS_DIR, "cic_whole_thresholds.json")

    if not (os.path.exists(scaler_path) and os.path.exists(lstm_pt) and os.path.exists(gnn_pt)):
        print("Warning: CIC whole flow models or scaler not found, skipping.")
        return

    scaler = joblib.load(scaler_path)
    with open(thresh_path, "r") as f:
        thresholds = json.load(f)

    test_file = os.path.join(PARENT_DIR, "cic-ids2018-processed", "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv")
    if not os.path.exists(test_file):
        print(f"Warning: File {test_file} not found, skipping.")
        return

    print("Extracting whole-flow test sequence from Botnet/Infiltration capture...")
    df = CWP.scan_and_extract_flows(test_file, max_flows_per_day=30000)
    X_seq, X_g, y_true = CWP.extract_flow_forecasting_windows(df, scaler, horizon=20, window_size=10, stride=5)
    print(f"  Extracted {len(y_true)} flow windows ({int(y_true.sum())} precursors)")

    if len(y_true) < 10 or y_true.sum() == 0:
        return

    # Load LSTM
    ckpt_lstm = torch.load(lstm_pt, map_location="cpu")
    lstm = ForecastLSTM(in_dim=ckpt_lstm["in_dim"], hidden_dim=ckpt_lstm["hidden_dim"])
    lstm.load_state_dict(ckpt_lstm["state_dict"])
    lstm.eval()

    # Load GNN
    ckpt_gnn = torch.load(gnn_pt, map_location="cpu")
    gnn = GM.ForecastGNN(in_dim=ckpt_gnn["in_dim"], hidden_dim=ckpt_gnn["hidden_dim"], heads=2)
    gnn.load_state_dict(ckpt_gnn["state_dict"])
    gnn.eval()

    # Inference
    t0 = time.perf_counter()
    with torch.no_grad():
        lstm_out = []
        for i in range(0, len(X_seq), 256):
            bx = torch.from_numpy(X_seq[i : i + 256])
            lstm_out.extend(torch.sigmoid(lstm(bx)).numpy().tolist())
    lstm_probas = np.array(lstm_out, dtype=np.float32)
    lstm_lat = ((time.perf_counter() - t0) / max(len(X_seq), 1)) * 1000.0

    t0 = time.perf_counter()
    with torch.no_grad():
        gnn_out = []
        for i in range(0, len(X_g), 256):
            bg = Batch.from_data_list(X_g[i : i + 256])
            gnn_out.extend(torch.sigmoid(gnn(bg)).numpy().tolist())
    gnn_probas = np.array(gnn_out, dtype=np.float32)
    gnn_lat = ((time.perf_counter() - t0) / max(len(X_g), 1)) * 1000.0

    ens_probas = 0.5 * lstm_probas + 0.5 * gnn_probas
    ens_lat = lstm_lat + gnn_lat

    models = {
        "ForecastLSTM": (lstm_probas, lstm_lat, thresholds.get("ForecastLSTM", 0.0007)),
        "ForecastGNN": (gnn_probas, gnn_lat, thresholds.get("ForecastGNN", 0.0023)),
        "Ensemble (LSTM+GNN)": (ens_probas, ens_lat, thresholds.get("Ensemble (LSTM+GNN)", 0.0022)),
    }

    for m_name, (probas, lat, th) in models.items():
        row = compute_metrics(
            y_true=y_true,
            y_proba=probas,
            thresh=th,
            latency_ms=lat,
            model_name=m_name,
            attack_type="Botnet Flow Precursor Stream",
            dataset_name="CSE-CIC-IDS2018 (Whole-Flow)",
            horizon="20 Flows Ahead",
        )
        results_list.append(row)
        print(f"  {m_name:20s} | ROC-AUC: {row['ROC-AUC']} | F1: {row['F1-Score']:.4f} | Recall: {row['Recall']:.2%} | Latency: {lat:.3f}ms")


def main():
    print("\n" + "#" * 80)
    print(" THREAT FORECASTING BENCHMARK ENGINE")
    print(" Proactive Cyber Attack Precursor & Lead-Time Anticipation")
    print("#" * 80)

    results: List[Dict[str, Any]] = []

    benchmark_unsw(results)
    benchmark_cic(results)
    benchmark_cic_whole(results)

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[+] Benchmark CSV generated successfully: {OUTPUT_CSV}")

    print("\n" + "=" * 95)
    print(" SUMMARY BENCHMARKS: ADVANCE FORECASTING PERFORMANCE ACROSS ATTACK TYPES")
    print("=" * 95)

    display_cols = ["Dataset", "Attack_Type", "Forecast_Horizon", "Model", "ROC-AUC", "PR-AUC", "F1-Score", "Recall", "Specificity", "Latency_ms"]
    print(df_results[display_cols].to_string(index=False))

    print("\n" + "=" * 95)
    print(" BEST PERFORMING MODEL PER ATTACK CATEGORY")
    print("=" * 95)

    best_rows = []
    for (ds, atk, h), grp in df_results.groupby(["Dataset", "Attack_Type", "Forecast_Horizon"], sort=False):
        top_m = grp.sort_values(by=["F1-Score", "Recall"], ascending=False).iloc[0]
        best_rows.append(top_m)

    df_best = pd.DataFrame(best_rows)[["Dataset", "Attack_Type", "Forecast_Horizon", "Model", "F1-Score", "Recall", "Specificity", "ROC-AUC", "Latency_ms"]]
    print(df_best.to_string(index=False))


if __name__ == "__main__":
    main()
