import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve


def top_features(model, feature_names: list[str], top_n: int = 15) -> list[tuple]:
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    return [(feature_names[i], importances[i]) for i in idx]


def plot_feature_importance(models: dict, feature_names: list[str], out_path: str):
    tree_models = {
        k: v for k, v in models.items() if hasattr(v, "feature_importances_")
    }
    n = len(tree_models)
    _fig, axes = plt.subplots(1, n, figsize=(9 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, tree_models.items()):
        fi = top_features(model, feature_names)
        names = [f for f, _ in fi]
        vals = [v for _, v in fi]
        ax.barh(names[::-1], vals[::-1], color="steelblue", edgecolor="white")
        ax.set_xlabel("Importance")
        ax.set_title(f"{name} — Top 15 Features")
        ax.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def print_feature_importance(models: dict, feature_names: list[str]):
    for name, model in models.items():
        if not hasattr(model, "feature_importances_"):
            continue
        fi = top_features(model, feature_names)
        print(f"\n--- Top 15 features: {name} ---")
        for rank, (feat, imp) in enumerate(fi, 1):
            print(f"  {rank:2d}. {feat:<45s} {imp:.4f}")


def plot_pr_curves(models: dict, splits: dict, out_path: str):
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]
    _fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4))

    for ax, (split_label, (X, y)) in zip(axes, splits.items()):
        for (model_name, model), color in zip(models.items(), colors):
            proba = model.predict_proba(X)[:, 1]
            prec, rec, _ = precision_recall_curve(y, proba)
            ap = average_precision_score(y, proba)
            ax.plot(rec, prec, label=f"{model_name} (AP={ap:.3f})", color=color)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve — {split_label}")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
