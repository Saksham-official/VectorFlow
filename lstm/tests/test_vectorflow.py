import os
import sys

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import evaluate as E
import persist as Persist
from inference import FEATURE_COLS, predict
from polars_aggregator import compute_temporal_features


@pytest.fixture
def sample_features_dict():
    return {col: 10.0 for col in FEATURE_COLS}


@pytest.fixture
def sample_features_df(sample_features_dict):
    return pd.DataFrame([sample_features_dict] * 5)


def test_feature_columns_count():
    assert len(FEATURE_COLS) == 75
    assert len(set(FEATURE_COLS)) == 75


def test_inference_single_dict(sample_features_dict):
    preds = predict(sample_features_dict, use_calibrated_threshold=True)
    assert isinstance(preds, pd.DataFrame)
    assert "LSTM" in preds["model"].values
    assert "probability" in preds.columns
    assert "attack_predicted" in preds.columns
    assert all(0.0 <= p <= 1.0 for p in preds["probability"])


def test_inference_batch_dataframe(sample_features_df):
    preds = predict(sample_features_df, use_calibrated_threshold=True)
    assert isinstance(preds, pd.DataFrame)
    assert preds["row"].nunique() == 5
    assert "LSTM" in preds["model"].values



def test_inference_missing_column_raises_error():
    incomplete_input = {"Tot Fwd Pkts_sum": 10.0}
    with pytest.raises(ValueError, match="Input is missing"):
        predict(incomplete_input)


def test_polars_temporal_feature_generation():
    data = {
        "window_start": pl.datetime_range(
            pl.datetime(2018, 2, 14, 1, 0, 0),
            pl.datetime(2018, 2, 14, 1, 3, 0),
            interval="10s",
            eager=True,
        ),
        "Tot Fwd Pkts_sum": [10.0] * 19,
        "Tot Bwd Pkts_sum": [5.0] * 19,
        "TotLen Fwd Pkts_sum": [100.0] * 19,
        "TotLen Bwd Pkts_sum": [50.0] * 19,
        "Flow Duration_mean": [1000.0] * 19,
        "Flow Duration_std": [10.0] * 19,
        "Flow IAT Mean_mean": [50.0] * 19,
        "Dst Port_nunique": [2.0] * 19,
        "SYN Flag Cnt_sum": [1.0] * 19,
        "ACK Flag Cnt_sum": [4.0] * 19,
        "RST Flag Cnt_sum": [0.0] * 19,
        "FIN Flag Cnt_sum": [0.0] * 19,
        "PSH Flag Cnt_sum": [2.0] * 19,
        "flow_count": [8.0] * 19,
        "bwd_fwd_pkt_ratio": [0.5] * 19,
        "has_attack": [0] * 15 + [1, 1, 0, 0],
    }
    base_df = pl.DataFrame(data)
    result_df = compute_temporal_features(base_df)

    assert "Future_Attack_Target" in result_df.columns
    assert "window_start" in result_df.columns
    assert len(result_df.columns) == 77
    assert result_df["Future_Attack_Target"].sum() > 0


def test_threshold_calibration_logic():
    y_true = np.array([0] * 90 + [1] * 10)
    y_proba = np.array([0.05] * 90 + [0.85] * 10)

    best_thresh, best_f1, _prec, recall = E.find_best_threshold(
        y_true, y_proba, min_recall=0.50
    )
    assert 0.0 < best_thresh < 1.0
    assert recall >= 0.50
    assert best_f1 > 0.80


def test_model_persistence_and_loading(tmp_path):
    scaler = StandardScaler()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaler.fit(X)

    model = LogisticRegression().fit(X, [0, 1])
    models = {"LogReg": model}
    thresholds = {"LogReg": 0.25}

    model_dir = str(tmp_path / "models_test")
    Persist.save(models, scaler, model_dir=model_dir, thresholds=thresholds)

    assert os.path.exists(os.path.join(model_dir, "scaler.joblib"))
    assert os.path.exists(os.path.join(model_dir, "logreg.joblib"))
    assert os.path.exists(os.path.join(model_dir, "thresholds.json"))
