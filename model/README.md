# VectorFlow — `model/` Package

Flow-level BiLSTM forecasting model for proactive attack-onset prediction.
This package provides the trained model weights and inference pipeline consumed
by `modules/forecasting/` in the VectorFlow backend.

---

## Directory layout

```
model/
├── artifacts/
│   ├── lstm_model.pt        ← Trained BiLSTM+Attention checkpoint (2.2 MB)
│   ├── scaler.joblib        ← Fitted RobustScaler (15 features)
│   └── model_metadata.json  ← Architecture config, threshold, benchmark results
└── src/
    ├── __init__.py          ← Public API: get_inference_pipeline, ForecastLSTM
    ├── model.py             ← ForecastLSTM nn.Module definition
    └── inference.py         ← InferencePipeline + get_inference_pipeline()
```

---

## Architecture

```
Input (15 flow features, seq_len=10 windows)
  └─ LayerNorm
  └─ BiLSTM (hidden=128, 2-layer, bidirectional, dropout=0.2)
  └─ Soft Attention Pooling  →  context vector (256-dim)
  └─ LayerNorm → Linear(256→32) → GELU → Dropout(0.2) → Linear(32→1)
  └─ Sigmoid  →  attack onset probability ∈ [0, 1]
```

| Parameter | Value |
|---|---|
| Input features | 15 aggregated flow statistics |
| Sequence length | 10 × 10-second windows (100s of history) |
| Hidden dim | 128 per direction (256 total context) |
| Decision threshold | 0.3367 (calibrated on validation set) |
| Forecast horizon | 30–120 seconds ahead |

---

## Input Features

The model consumes 15 per-window aggregated statistics:

| Feature | Description |
|---|---|
| `Tot Fwd Pkts_sum` | Total forward packets in window |
| `Tot Bwd Pkts_sum` | Total backward packets in window |
| `TotLen Fwd Pkts_sum` | Total bytes sent forward |
| `TotLen Bwd Pkts_sum` | Total bytes sent backward |
| `Flow Duration_mean` | Mean flow duration (μs) |
| `Flow Duration_std` | Std of flow duration |
| `Flow IAT Mean_mean` | Mean inter-arrival time |
| `Dst Port_nunique` | Unique destination ports |
| `SYN Flag Cnt_sum` | SYN flags (connection attempts) |
| `ACK Flag Cnt_sum` | ACK flags |
| `RST Flag Cnt_sum` | RST flags (connection resets) |
| `FIN Flag Cnt_sum` | FIN flags (connection teardowns) |
| `PSH Flag Cnt_sum` | PSH flags (data push) |
| `flow_count` | Number of distinct flows |
| `bwd_fwd_pkt_ratio` | Asymmetry ratio (backward/forward pkts) |

---

## Integration

`modules/forecasting/__init__.py` calls:

```python
from model.src.inference import get_inference_pipeline

pipeline = get_inference_pipeline()           # loads model/artifacts/
result = pipeline.predict_sequence(states)   # list of feature dicts
```

`states` is a list of `NetworkState.features` dicts from the VectorFlow
data pipeline — one dict per 10-second window, oldest first.

`predict_sequence` raises `InsufficientHistoryError` until `sequence_length`
(10) windows have been collected, which the forecasting module handles gracefully.

---

## Benchmark Results

Evaluated on **12,143 holdout windows** from UNSW-NB15 + CSE-CIC-IDS2018:

| Metric | Score |
|---|---|
| ROC-AUC | **0.9726** |
| PR-AUC | **0.8296** |
| F1-Score | **0.8606** |
| Recall | **93.15%** |
| Precision | **79.97%** |
| Balanced Accuracy | **94.09%** |
| False Positive Rate | 4.96% |
| Inference latency (CPU) | **0.116 ms** |
| Throughput | **8,620 windows/sec** |

See `benching/` for the full multi-model comparison suite.

---

## Artifacts

Binary artifacts are tracked via `.gitignore` negation rules:
```
!model/artifacts/*.pt
!model/artifacts/*.joblib
```

To regenerate artifacts from scratch, run the training pipeline in `benching/`.
