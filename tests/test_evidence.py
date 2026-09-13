import json
import time

from fastapi.testclient import TestClient

from backend.app.core import store
from backend.app.core.config import get_config
from backend.app.main import app
from backend.app.services.blockchain_service import LocalSecurityEvidenceLedger, evidence_hash
from contracts import Explanation, FeatureContribution, SecurityAlert


def test_hash_is_deterministic(tmp_path):
    alert = SecurityAlert(
        alert_id="a-1", timestamp="2026-01-01T00:00:00Z", attack_probability=0.8,
        predicted_stage="Initial Access", model_name="ForecastModel", severity="high",
    )
    explanation = Explanation(
        top_features=[FeatureContribution(feature="flow_count", contribution=0.5)],
        summary="flow count increased",
    )
    from backend.app.services.blockchain_service import canonical_payload
    payload = canonical_payload(alert, explanation)
    assert evidence_hash(payload) == evidence_hash(json.loads(json.dumps(payload, default=str)))


def test_local_ledger_record_and_tamper_detection(tmp_path):
    ledger = LocalSecurityEvidenceLedger(tmp_path / "ledger.json")
    alert = SecurityAlert(
        alert_id="a-1", timestamp="2026-01-01T00:00:00Z", attack_probability=0.8,
        predicted_stage="Initial Access", model_name="ForecastModel", severity="high",
    )
    explanation = Explanation(top_features=[], summary="observed")
    record = ledger.record(alert, explanation)
    assert record.transaction_id
    assert ledger.record(alert, explanation).transaction_id == record.transaction_id
    assert ledger.verify("a-1").verified is True
    blocks = json.loads((tmp_path / "ledger.json").read_text())
    blocks[0]["payload"]["attack_probability"] = 0.1
    (tmp_path / "ledger.json").write_text(json.dumps(blocks))
    assert ledger.verify("a-1").verified is False


def test_end_to_end_alert_evidence_and_verification(tmp_path, monkeypatch):
    # Use an isolated ledger while exercising the real backend model path.
    cfg = get_config()
    monkeypatch.setitem(cfg["blockchain"], "ledger_path", str(tmp_path / "ledger.json"))
    store.current = None
    client = TestClient(app)
    assert client.post("/api/session", data={"sample": "ssh_bruteforce_2018-02-14.csv"}).status_code == 200
    store.current.replay_started = time.monotonic() - cfg["replay"]["seconds_per_window"] * 20
    alerts = client.get("/api/alerts")
    assert alerts.status_code == 200
    assert len(alerts.json()) == 1
    alert_id = alerts.json()[0]["alert_id"]
    evidence = client.get("/api/evidence").json()[0]
    assert evidence["alert_id"] == alert_id
    assert evidence["ledger_status"] == "recorded"
    assert client.get(f"/api/evidence/{alert_id}/verify").json()["verified"] is True
    # Re-reading result endpoints must not create another transaction.
    client.get("/api/forecast")
    client.get("/api/alerts")
    assert len(client.get("/api/alerts").json()) == 1
