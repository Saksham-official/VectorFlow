import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, min_recall: float = 0.5
) -> tuple[float, float, float, float]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    best_thresh = 0.5
    best_f1 = 0.0
    best_p = 0.0
    best_r = 0.0

    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if r >= min_recall:
            f1 = (2 * p * r) / (p + r + 1e-12)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
                best_p = p
                best_r = r

    if best_f1 == 0.0 and len(thresholds) > 0:
        f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (
            precisions[:-1] + recalls[:-1] + 1e-12
        )
        idx = int(np.argmax(f1_scores))
        best_thresh = thresholds[idx]
        best_f1 = f1_scores[idx]
        best_p = precisions[idx]
        best_r = recalls[idx]

    return (
        round(float(best_thresh), 4),
        round(float(best_f1), 4),
        round(float(best_p), 4),
        round(float(best_r), 4),
    )


def evaluate_one(
    model,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    split_label: str,
    threshold: float = 0.5,
) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])

    return {
        "model": model_name,
        "split": split_label,
        "threshold": round(float(threshold), 4),
        "PR-AUC": round(float(average_precision_score(y, proba)), 4),
        "ROC-AUC": round(float(roc_auc_score(y, proba)), 4),
        "Precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "TP": int(cm[1, 1]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TN": int(cm[0, 0]),
    }


def evaluate_all(
    models: dict, splits: dict, thresholds: dict | None = None
) -> pd.DataFrame:
    rows = []
    for model_name, model in models.items():
        thresh = thresholds.get(model_name, 0.5) if thresholds else 0.5
        for split_label, (X, y) in splits.items():
            rows.append(
                evaluate_one(model, X, y, model_name, split_label, threshold=thresh)
            )
    return pd.DataFrame(rows)


def print_comparison(results: pd.DataFrame):
    print("\n\n===== Model Performance Comparison =====\n")
    metric_cols = ["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "threshold"]
    pivot = results.pivot_table(
        index="model",
        columns="split",
        values=metric_cols,
        aggfunc="first",
    )
    order = [s for s in ["train", "val", "test"] if s in results["split"].unique()]
    pivot = pivot.reindex(order, axis=1, level="split")
    print(pivot.to_string(float_format="%.4f"))

    print("\n\n===== Confusion Matrices =====\n")
    cols = ["model", "split", "threshold", "TP", "FP", "FN", "TN"]
    print(results[cols].to_string(index=False))
