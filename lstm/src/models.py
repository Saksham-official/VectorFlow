import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

SEED: int = int(os.getenv("RANDOM_SEED", "42"))
RF_N_ESTIMATORS: int = int(os.getenv("RF_N_ESTIMATORS", "400"))
RF_MIN_SAMPLES_LEAF: int = int(os.getenv("RF_MIN_SAMPLES_LEAF", "4"))
XGB_N_ESTIMATORS: int = int(os.getenv("XGB_N_ESTIMATORS", "400"))
XGB_MAX_DEPTH: int = int(os.getenv("XGB_MAX_DEPTH", "6"))
XGB_LR: float = float(os.getenv("XGB_LEARNING_RATE", "0.05"))
XGB_SUBSAMPLE: float = float(os.getenv("XGB_SUBSAMPLE", "0.8"))
XGB_COLSAMPLE: float = float(os.getenv("XGB_COLSAMPLE_BYTREE", "0.8"))


def build_logreg() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=SEED,
        n_jobs=-1,
    )


def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=None,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def build_xgboost(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LR,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )


def train_all(X_train: np.ndarray, y_train: np.ndarray) -> dict[str, object]:
    bincounts = np.bincount(y_train)
    neg = int(bincounts[0]) if len(bincounts) > 0 else 1
    pos = int(bincounts[1]) if len(bincounts) > 1 else 1
    spw = max(neg / max(pos, 1), 1.0)

    print(
        f"\nTrain class distribution: {neg:,} negative / {pos:,} positive | scale_pos_weight = {spw:.2f}"
    )

    models = {
        "LogReg": build_logreg(),
        "RandomForest": build_random_forest(),
        "XGBoost": build_xgboost(spw),
    }

    print("\nTraining models:")
    for name, model in models.items():
        print(f"  -> Fitting {name}...", end="", flush=True)
        model.fit(X_train, y_train)
        print(" [done]")

    return models
