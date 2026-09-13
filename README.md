# SIH26153 — AI-based Network Attack Forecasting from Network Traffic Data

Fully offline SIH prototype that forecasts whether a new network attack is
likely to start in the next 30–60 seconds. The finalized causal PyTorch
forecasting model is the primary intelligence engine; a local hash-linked
ledger records compact alert evidence without storing raw traffic.

```
Network traffic (PCAP/CSV)
  → feature extraction / preprocessing   (Data Pipeline)
  → time-based network states
  → six-window causal forecasting model
  → 30–60 second attack-onset probability
  → attack intelligence → alert → evidence ledger → dashboard
```

## Repository layout

| Path         | What it is |
|--------------|------------|
| `contracts/` | Pydantic data models shared by every module — the team's interface contract |
| `modules/`   | Team modules: `data_pipeline/`, `forecasting/`, `attack_intelligence/` |
| `backend/`   | FastAPI application: session/replay orchestration, JSON API at `/api/*` |
| `dashboard/` | Flask web dashboard for judges (calls the backend over localhost) |
| `model/`     | Canonical training/inference implementation and local artifacts |
| `data/`      | Sample input files (`samples/`) and runtime uploads (`uploads/`, untracked) |
| `tests/`     | pytest suite |
| `docs/`      | Architecture, implementation plan, integration and testing docs |

## Setup (offline, reproducible)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a machine without internet, build a wheel cache once where you have it
(about 52 MB) and install from that:

```bash
pip download -r requirements.txt -d wheels/
pip install --no-index --find-links wheels -r requirements.txt
```

## Run

Two processes, two terminals:

```bash
# 1. Backend API  → http://127.0.0.1:8000/api/health
uvicorn backend.app.main:app --port 8000

# 2. Dashboard    → http://127.0.0.1:5000/
python dashboard/app.py
```

Configuration lives in `config.yaml`. No internet access is required at any
point.

## Demo

Pick `ssh_bruteforce_2018-02-14.csv` on the home page. It contains 75
ten-second windows replayed at two seconds per window. The first five windows
show `Collecting historical context...`; after six real windows the model is
called with the most recent six causal states. The bundled sample has no IP
fields, so empty flagged-flow and top-talker sections are honest.

- Windows 1–26 are benign. Risk stays low with one brief bump around window
  13; the stage sits at Reconnaissance or Initial Access.
- Window 27 (02:01:50) is where the SSH brute force starts. Flows roughly
  triple, risk climbs above 65% and stays there, and the stage moves through
  Initial Access and Lateral Movement to Exfiltration.
- Flagged flows, protocol breakdown and top talkers stay empty: CIC-IDS-2018
  day files carry no IPs or ports, so per-flow records cannot be rebuilt.

`config.yaml` selects `modules.forecasting.implementation: real`. Real mode
loads `model/artifacts/{model.pt,scaler.joblib,model_metadata.json}` and
never pads a short sequence or silently changes its scaler/threshold. Change
the setting to `mock` only for development on a machine without PyTorch or
artifacts; API results then identify themselves as `model_mode: demo`.

Uploading a CSV that is not a CIC-IDS-2018 flow export puts the session in
the error state with the missing columns named. PCAP must be converted with
CICFlowMeter first.

## Blockchain evidence

When risk crosses the configured threshold, one deduplicated alert is created.
The ledger hashes alert ID, timestamp, probability, predicted stage, model
version, and model-derived top features. It stores only that compact metadata
in a local hash-linked JSON chain; raw PCAPs and payloads stay off-chain.

The dashboard Verify button recalculates the hash and validates every block
link through `/api/evidence/{alert_id}/verify`. If the ledger is unavailable,
forecasting and alerting continue and the status says `unavailable`.

## Tests

```bash
pytest
```

The suite covers warm-up behavior, lazy replay inference, alert deduplication,
deterministic hashing, tamper detection, verification, and the complete
traffic → state → model → intelligence → alert → ledger flow.

## Documentation

Start with `docs/implementation-plan.md` and `docs/architecture.md`.
Module owners: read `docs/integration.md` for the contract your module must
implement.
