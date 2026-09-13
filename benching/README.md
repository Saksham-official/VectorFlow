# VectorFlow Benchmark Suite

> **Multi-model threat precursor forecasting evaluation** across gold-standard network intrusion datasets.

This directory contains the full benchmark and evaluation engine for VectorFlow — comparing the
BiLSTM (ForecastLSTM), GNN (ForecastGNN), XGBoost, Random Forest, and Logistic Regression
models for **proactive attack onset prediction**.

---

## What It Benchmarks

| Dataset | Attack Types | Evaluation Mode |
|---|---|---|
| **UNSW-NB15** | Exploits, DoS, Backdoor, Shellcode, Generic, Reconnaissance | 20-flow-ahead precursor forecasting |
| **CSE-CIC-IDS2018** | SSH-BruteForce, FTP-BruteForce, DoS-GoldenEye, Web Attacks (XSS/SQLi), Botnet | 30s & 120s advance warning |
| **CSE-CIC-IDS2018 Whole-Flow** | Botnet ARES Infiltration stream | 20-flow-ahead graph + sequence anticipation |

---

## Model Architecture (ForecastLSTM)

The primary VectorFlow intelligence engine is a **2-layer Bidirectional LSTM with Attention Pooling**:

```
Input (15 features, seq_len=10)
  └─ LayerNorm
  └─ BiLSTM (hidden=128, 2 layers, dropout=0.2)
  └─ Attention Pooling (context vector)
  └─ LayerNorm → Linear(256→32) → GELU → Dropout → Linear(32→1)
  └─ Sigmoid → attack probability
```

**Input features:** `Tot Fwd/Bwd Pkts`, `TotLen Fwd/Bwd Pkts`, `Flow Duration (mean/std)`,
`Flow IAT Mean`, `Dst Port (nunique)`, `SYN/ACK/RST/FIN/PSH Flag Cnt`, `flow_count`, `bwd_fwd_pkt_ratio`

---

## Key Results

### UNSW-NB15 Overall (20 Flows Ahead, 12,143 test windows)

| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision | FPR | Latency |
|---|---|---|---|---|---|---|---|
| **ForecastLSTM** | 0.9726 | 0.8296 | 0.8606 | **93.15%** | 79.97% | 4.96% | **0.116 ms** |
| ForecastGNN | 0.9726 | 0.8379 | 0.8629 | 92.63% | 80.76% | 4.69% | 0.157 ms |
| Ensemble (LSTM+GNN) | **0.9745** | **0.8422** | **0.8662** | 93.43% | 80.73% | 4.74% | 0.273 ms |
| XGBoost | 1.000* | 1.000* | 1.000* | 100%* | 100%* | 0%* | 0.125 ms |
| Random Forest | 1.000* | 1.000* | 1.000* | 100%* | 100%* | 0%* | 10.39 ms |

*XGBoost/RF achieve perfect scores on the 120s SSH-BruteForce window; performance varies by attack type.

### Per-Attack-Type Recall (ForecastLSTM, UNSW-NB15)

| Attack | Precursors | Recall | ROC-AUC |
|---|---|---|---|
| Reconnaissance | 13 | **100.0%** | 0.9966 |
| Generic | 364 | **98.35%** | 0.9964 |
| Exploits | 1,508 | **95.69%** | 0.9746 |
| DoS | 423 | **93.85%** | 0.9740 |
| Shellcode | 70 | 85.71% | 0.9677 |
| Backdoor | 120 | 62.50% | 0.9441 |

### Inference Performance

| Metric | ForecastLSTM | ForecastGNN | XGBoost |
|---|---|---|---|
| Latency | **0.116 ms** | 0.157 ms | 0.125 ms |
| Throughput | **8,620 win/s** | 6,369 win/s | 8,000 win/s |
| Disk | 0.59 MB | 0.07 MB | 1.25 MB |
| RAM | ~32 MB | ~38 MB | ~15 MB |

---

## Directory Layout

```
benching/
├── benchmark_attack_forecasting.py   # Full multi-model benchmark engine (CLI)
├── app.py                            # Streamlit interactive dashboard
├── benching.py                       # CLI entrypoint
├── requirements.txt                  # Benchmark-specific dependencies
├── run.sh                            # Shell script to run full suite
├── data/                             # Benchmark result CSVs and sample data
│   ├── benchmark_attack_forecasting.csv   # Full results (75 rows, 5 models × datasets)
│   ├── metrics.csv                        # Unified threat pipeline metrics
│   ├── unsw_benchmark_results.csv
│   ├── cic_whole_benchmark_results.csv
│   ├── ssh_bruteforce_benchmarks_h30.csv
│   ├── ssh_bruteforce_benchmarks_h120.csv
│   ├── unified_threat_benchmarks.csv
│   ├── unified_threat_timeline.csv
│   ├── ssh_bruteforce_window_timeline.csv
│   └── ssh_bruteforce_2018-02-14.csv      # Demo sample (75 windows)
├── models/                           # Model artifacts (gitignored binaries)
│   ├── .gitkeep
│   ├── lstm_metadata.json            # Architecture config
│   ├── lstm_history.json             # Training loss curves
│   ├── lstm_threshold.json           # Decision threshold (0.3367)
│   ├── gnn_metadata.json
│   ├── thresholds.json
│   ├── unsw_thresholds.json
│   └── cic_whole_thresholds.json
└── plots/                            # 20 evaluation visualizations (PNG)
    ├── benchmark_summary_dashboard.png
    ├── benchmark_roc_curves.png
    ├── benchmark_pr_curves.png
    ├── benchmark_confusion_matrices.png
    ├── benchmark_metrics_barchart.png
    ├── lstm_training_curves.png
    ├── feature_importance.png
    ├── unified_threat_timeline.png
    ├── ssh_bruteforce_timeline.png
    └── ...
```

> **Note:** Binary model files (`.pt`, `.joblib`, `.ubj`) are excluded by `.gitignore`.
> Place them in `benching/models/` after download. The main LSTM artifact lives at
> `lstm/artifacts/lstm_model.pt` (VectorFlow production model).

---

## Setup

Install benchmark dependencies (separate from root `requirements.txt`):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r benching/requirements.txt
```

Requires the root VectorFlow environment + `torch-geometric` and `polars`.

---

## Run

### Interactive Streamlit Dashboard

```bash
streamlit run benching/app.py
```

Opens the **VectorFlow Threat Engine Hub** at `http://localhost:8501` — interactive model
comparison, ROC/PR curves, confusion matrices, attack timeline replays.

### CLI Full Benchmark

```bash
python benching/benching.py
# or
bash benching/run.sh
```

Runs all three benchmark suites in sequence and outputs `benching/benchmark_attack_forecasting.csv`.

---

## Demo

Use `benching/data/ssh_bruteforce_2018-02-14.csv` (75 ten-second windows) in the Streamlit
dashboard for a live demonstration:

- **Windows 1–26:** Benign — risk stays low
- **Window 27 (02:01:50):** SSH brute-force starts — risk climbs above 65%, stage moves to *Initial Access → Lateral Movement*
- **120s advance warning:** XGBoost/Ensemble detect attack onset 2 minutes ahead
- **BiLSTM:** Detects temporal anomaly patterns across the 10-window causal context

---

## Relation to Main VectorFlow System

```
[This benchmark suite]                [Production VectorFlow pipeline]
benchmark_attack_forecasting.py  ←→  modules/forecasting/
app.py (Streamlit)               ←→  dashboard/ (Flask)
ForecastLSTM (BiLSTM class)      ←→  lstm/src/model.py (CausalLSTM)
benching/models/lstm_*.pt        ←→  lstm/artifacts/lstm_model.pt
```

The benchmark ForecastLSTM and the production CausalLSTM share the same BiLSTM + attention
architecture. The benchmark suite is used to validate and compare model performance before
deploying updates to `lstm/artifacts/`.
