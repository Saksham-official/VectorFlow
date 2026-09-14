import os
import sys
import warnings

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import data as D
from lstm_model import (
    BASE_FEATURES,
    LSTM_SEQ_LEN,
    USE_ALL_FEATURES,
    create_sequences,
    get_feature_cols,
    predict_proba,
    save_lstm,
    scale_sequences,
    train_lstm,
)

warnings.filterwarnings("ignore")
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data/cic_ids2018_complete_dataset.csv")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def find_best_threshold(y_true, y_proba, min_recall=0.50):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    best_thresh, best_f1, best_p, best_r = 0.5, 0.0, 0.0, 0.0

    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if r >= min_recall:
            f1 = (2 * p * r) / (p + r + 1e-12)
            if f1 > best_f1:
                best_f1, best_thresh, best_p, best_r = f1, t, p, r

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


def evaluate_split(y_true, y_proba, threshold, label):
    pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    pr_auc = average_precision_score(y_true, y_proba)
    roc = roc_auc_score(y_true, y_proba)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)

    print(f"\n  [{label}] threshold={threshold:.4f}")
    print(f"    PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc:.4f}")
    print(f"    Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"    TP={cm[1,1]:5d} | FP={cm[0,1]:5d} | FN={cm[1,0]:5d} | TN={cm[0,0]:5d}")
    return {"PR-AUC": pr_auc, "ROC-AUC": roc, "F1": f1}


def main():
    print("=" * 65)
    print(" VectorFlow -- LSTM Attack Forecasting Training Pipeline")
    print("=" * 65)

    df = D.load(DATA_PATH)
    print(f"Loaded {len(df):,} windows, {df.shape[1]} columns")
    print(f"Time range: {df['window_start'].min()} to {df['window_start'].max()}")

    D.audit(df)
    train_df, val_df, test_df = D.split(df)

    feat_cols = get_feature_cols(df)
    feat_type = f"All {len(feat_cols)} Features" if USE_ALL_FEATURES else f"Base {len(feat_cols)} Physical Features"
    print("\n" + "=" * 65)
    print(f" Creating LSTM Sequences (seq_len={LSTM_SEQ_LEN}, features={len(feat_cols)} [{feat_type}])")
    print("=" * 65)

    X_train_seq, y_train = create_sequences(train_df, seq_len=LSTM_SEQ_LEN, base_cols=feat_cols)
    X_val_seq, y_val = create_sequences(val_df, seq_len=LSTM_SEQ_LEN, base_cols=feat_cols)
    X_test_seq, y_test = create_sequences(test_df, seq_len=LSTM_SEQ_LEN, base_cols=feat_cols)

    print(f"  Train sequences: {X_train_seq.shape} | positives: {y_train.sum():,}")
    print(f"  Val   sequences: {X_val_seq.shape} | positives: {y_val.sum():,}")
    print(f"  Test  sequences: {X_test_seq.shape} | positives: {y_test.sum():,}")

    X_train_s, X_val_s, X_test_s, scaler = scale_sequences(
        X_train_seq, X_val_seq, X_test_seq
    )

    print("\n" + "=" * 65)
    print(" Training Bidirectional LSTM with Attention")
    print("=" * 65)

    model, history = train_lstm(X_train_s, y_train, X_val_s, y_val)

    print("\n" + "=" * 65)
    print(" Threshold Calibration on Validation Set (target: Recall >= 50%)")
    print("=" * 65)

    val_proba = predict_proba(model, X_val_s)
    t, f1, p, r = find_best_threshold(y_val, val_proba, min_recall=0.50)
    print(f"  Optimal threshold: {t:.4f} -> Val F1: {f1:.4f} | Prec: {p:.4f} | Recall: {r:.4f}")

    print("\n" + "=" * 65)
    print(" LSTM Evaluation -- Calibrated Threshold")
    print("=" * 65)

    train_proba = predict_proba(model, X_train_s)
    test_proba = predict_proba(model, X_test_s)

    evaluate_split(y_train, train_proba, t, "Train")
    evaluate_split(y_val, val_proba, t, "Val")
    test_metrics = evaluate_split(y_test, test_proba, t, "Test")

    print("\n" + "=" * 65)
    print(" LSTM Evaluation -- Default 0.50 Threshold")
    print("=" * 65)
    evaluate_split(y_train, train_proba, 0.5, "Train")
    evaluate_split(y_val, val_proba, 0.5, "Val")
    evaluate_split(y_test, test_proba, 0.5, "Test")

    print("\n" + "=" * 65)
    print(" Persisting LSTM Artifacts")
    print("=" * 65)

    save_lstm(model, scaler, t, MODEL_DIR, history, feature_names=feat_cols)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs_range = range(1, len(history["train_loss"]) + 1)
        ax1.plot(epochs_range, history["train_loss"], label="Train Loss", color="tab:blue", lw=2)
        ax1.plot(epochs_range, history["val_loss"], label="Val Loss", color="tab:orange", lw=2)
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Focal Loss", fontsize=11)
        ax1.set_title("LSTM Training & Validation Loss", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        if "val_pr_auc" in history and history["val_pr_auc"]:
            ax2.plot(epochs_range, history["val_pr_auc"], label="Val PR-AUC", color="tab:purple", lw=2)
        if "val_roc_auc" in history and history["val_roc_auc"]:
            ax2.plot(epochs_range, history["val_roc_auc"], label="Val ROC-AUC", color="tab:blue", lw=2, linestyle="--")
        ax2.plot(epochs_range, history["val_f1"], label="Val F1 @0.5", color="tab:green", lw=1.5, alpha=0.7)
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Score", fontsize=11)
        ax2.set_title("LSTM Validation Discriminative Metrics", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "lstm_training_curves.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\n  Saved training curves -> {plot_path}")
    except Exception as e:
        print(f"\n  Warning: Could not save training plot: {e}")

    print("\n" + "=" * 65)
    print(" LSTM Training Pipeline Complete!")
    print(f" Test PR-AUC: {test_metrics['PR-AUC']:.4f} | Test F1: {test_metrics['F1']:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
