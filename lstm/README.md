# Dual-Level Traffic Feature Engine & Causal Bi-LSTM Forecasting

This package implements the **Two-Level Network Traffic Feature Architecture** and retrained **Causal Bi-LSTM Forecasting Model with Attention & Focal Loss** for early cyber-attack prediction (30–60s horizon).

---

## 1. Architecture Overview

Real-world attack detection requires multi-granularity visibility: flow summaries capture macro volume patterns, while packet inspection uncovers micro evasion signatures (e.g. TTL manipulation, TCP zero-window probing, fragment offset abuse, stealth scanning).

```
 Raw Traffic (.pcap / .pcapng) ──┐
                                  ├──> packet_extractor.py ──> 26 Dual-Level Features (10s windows)
 IPFIX / NetFlow Logs (.csv)   ──┘                                       │
                                                                         ▼
                                                            RobustScaler Normalization
                                                                         │
                                                                         ▼
                                                            2-Layer Bi-LSTM (hidden=256)
                                                                         │
                                                                         ▼
                                                              Soft Attention Pooling
                                                                         │
                                                                         ▼
                                                           Focal Loss Head (γ=2.0)
                                                                         │
                                                                         ▼
                                                       P(Future Attack in 30-60s) ∈ [0, 1]
```

---

## 2. Feature Specification (26 Base Features)

### Level 1: Flow-Level Features (NetFlow / IPFIX compatible - 15 features)
| Feature Name | Description |
|---|---|
| `Tot Fwd Pkts_sum` | Total packets in forward direction |
| `Tot Bwd Pkts_sum` | Total packets in backward direction |
| `TotLen Fwd Pkts_sum` | Total bytes in forward direction |
| `TotLen Bwd Pkts_sum` | Total bytes in backward direction |
| `Flow Duration_mean` | Mean flow duration in window |
| `Flow Duration_std` | Standard deviation of flow durations |
| `Flow IAT Mean_mean` | Inter-arrival time mean across flows |
| `Dst Port_nunique` | Unique destination ports targeted |
| `SYN Flag Cnt_sum` | Total SYN flags observed |
| `ACK Flag Cnt_sum` | Total ACK flags observed |
| `RST Flag Cnt_sum` | Total RST flags observed |
| `FIN Flag Cnt_sum` | Total FIN flags observed |
| `PSH Flag Cnt_sum` | Total PSH flags observed |
| `flow_count` | Total active flows in 10s window |
| `bwd_fwd_pkt_ratio` | Ratio of backward to forward packets |

### Level 2: Packet-Level Features (Streaming PCAP analysis - 11 features)
| Feature Name | Description | Detection Purpose |
|---|---|---|
| `ttl_mean` | Average Time-to-Live | OS fingerprinting & route shifting |
| `ttl_session_variance` | Variance of TTL per IP session ($\sigma^2_{TTL}$) | TTL spoofing / evasion detection |
| `tcp_window_mean` | Mean TCP advertised window size | Flow control anomalies |
| `tcp_window_std` | Standard deviation of TCP window | Zero-window attacks / resource starvation |
| `ip_fragment_count` | Packets with `MF=1` or offset > 0 | Teardrop / IP fragment evasion |
| `payload_size_mean` | Average application payload bytes | Protocol abuse / buffer overflows |
| `payload_size_std` | Standard deviation of payload size | Uniformity checks vs polymorphic attacks |
| `zero_payload_ratio` | Proportion of packets with 0 byte payload | TCP SYN/ACK floods, empty probes |
| `tcp_retrans_count` | Retransmission count | Network degradation / TCP resets |
| `sequential_scan_score` | Max consecutive port step ratio $\Delta port \in \{1, -1\}$ per client | High-confidence port scan detection |
| `random_scan_entropy` | Normalized Shannon entropy of targeted ports | Stealthy randomized scanning detection |

---

## 3. Retrained Bi-LSTM Model Performance

Evaluated on multi-day chronological holdout data with strict temporal sequencing:

| Metric | Score | Note |
|---|---|---|
| **ROC-AUC** | **0.9635** | High discriminative power across normal vs pre-attack |
| **PR-AUC** | **0.9572** | Handles class imbalance effectively |
| **F1-Score** | **0.9412** | Optimal balance at calibrated threshold |
| **Precision** | **0.9559** | Low false alarm rate |
| **Recall** | **0.9269** | High attack coverage |
| **Optimal Threshold** | **0.9029** | Calibrated via precision-recall trade-off |
| **Inference Latency** | **~1.2 ms** | Real-time streaming capable |

---

## 4. Directory Structure

```
lstm/
├── data/
│   └── cic_ids2018_complete_dataset.csv   # 10-day multi-day training capture data
├── models/
│   ├── lstm.pt                            # Trained PyTorch Bi-LSTM weights (8.78 MB)
│   ├── lstm_scaler.joblib                 # RobustScaler fitted on 26 dual features
│   ├── lstm_metadata.json                 # Architecture config & feature definitions
│   ├── lstm_threshold.json                # Calibrated decision threshold
│   └── lstm_history.json                  # Epoch-wise training loss & validation metrics
├── outputs/
│   └── lstm_training_curves.png           # Training loss and evaluation metrics plot
├── src/
│   ├── packet_extractor.py                # Streaming dpkt PCAP parser
│   ├── dual_pipeline.py                   # Unified flow + packet pipeline & temporal lag engine
│   ├── generate_dual_dataset.py           # Multi-day traffic generator
│   ├── lstm_model.py                      # AttackLSTM (Bi-LSTM + Attention + Focal Loss)
│   ├── polars_aggregator.py               # Fast window aggregations
│   ├── data.py                            # Dataset loader & sequence batches
│   ├── evaluate.py                        # Threshold calibration & evaluation suite
│   ├── persist.py                         # Model persistence helpers
│   └── plots.py                           # Training curve visualizer
├── tests/
│   ├── test_dual_level.py                 # Unit & integration tests for dual-level pipeline
│   └── test_vectorflow.py                 # Regression tests for VectorFlow compatibility
├── inference.py                           # Python inference API (.predict(), .predict_pcap())
├── run_lstm.py                            # Training & evaluation runner
├── requirements.txt                       # Package dependencies
└── README.md                              # Technical documentation
```

---

## 5. Usage

### Run Unit Tests
```bash
pytest lstm/tests -v
```

### Run PCAP Inference
```python
from lstm.inference import predict_pcap

features_df, preds_df = predict_pcap("path/to/capture.pcap")
print(preds_df)
```

### Run Feature Inference
```python
from lstm.inference import predict

preds_df = predict(feature_dict_or_dataframe)
print(preds_df)
```
