"""
plot_cm_grid.py
───────────────
Generate a side-by-side Confusion Matrix Grid for all 5 methods on Same-GO.
Usage:
    python3 scripts/plot_cm_grid.py --strategy same_go --dataset human
    python3 scripts/plot_cm_grid.py --strategy same_go --dataset yeast
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

METHODS = {
    "SPRINT":       ("sprint", "github_baselines/sprint"),
    "Auto Cov.":    ("ac",     "baselines/auto_covariance"),
    "Conj. Triad":  ("ct",     "baselines/conjoint_triad"),
    "ESM-2+MLP":    ("esm2",   "github_baselines/esm2_mlp"),
    "HybridStack":  ("ours",   ""),   # root results dir
}

CMAP = LinearSegmentedColormap.from_list("wr", ["#ffffff", "#1a6b8a"])

def load_preds(base_dir, dataset_key, method_key):
    candidates = [
        f"{base_dir}/results/github_baselines/{method_key}_{dataset_key}/oof_predictions.csv",
        f"{base_dir}/results/baselines/{dataset_key}/{method_key}/oof_predictions.csv",
        f"{base_dir}/results/{dataset_key}/oof_predictions.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


def plot_cm(ax, y_true, y_pred, title, fontsize=9):
    from sklearn.metrics import confusion_matrix, matthews_corrcoef
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=fontsize, fontweight="bold")
        return
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()
    mcc = matthews_corrcoef(y_true, y_pred)
    prec = tp / (tp + fp + 1e-9) * 100
    spec = tn / (tn + fp + 1e-9) * 100

    im = ax.imshow(cm, cmap=CMAP, aspect="auto")
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = val / total * 100
            ax.text(j, i, f"{val:,}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=7.5,
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black")

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=7)
    ax.set_yticklabels(["True 0", "True 1"], fontsize=7)
    ax.set_title(f"{title}\nMCC={mcc:.2f}  Prec={prec:.1f}%  Spec={spec:.1f}%",
                 fontsize=fontsize - 0.5, fontweight="bold")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="same_go",
                        choices=["same_compartment", "same_go"])
    parser.add_argument("--dataset", default="human", choices=["human", "yeast"])
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    base = "/media/SAS/Van/HybridStackPPI"
    suffix = "_same_go" if args.strategy == "same_go" else ""
    dataset_key = f"{args.dataset}{suffix}"

    method_paths = {
        "SPRINT":      f"{base}/results/github_baselines/sprint_{dataset_key}/oof_predictions.csv",
        "Auto Cov.":   f"{base}/results/baselines/{dataset_key}/auto_covariance/oof_predictions.csv",
        "Conj. Triad": f"{base}/results/baselines/{dataset_key}/conjoint_triad/oof_predictions.csv",
        "ESM-2+MLP":   f"{base}/results/github_baselines/esm2_mlp_{dataset_key}/oof_predictions.csv",
        "HybridStack": f"{base}/results/{dataset_key}/oof_predictions.csv",
    }

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))
    fig.suptitle(
        f"Confusion Matrix Grid — {args.dataset.capitalize()} "
        f"({'Same-GO' if args.strategy == 'same_go' else 'Same-Compartment'})\n"
        "Threshold = OOF optimal",
        fontsize=11, fontweight="bold", y=1.02
    )

    for ax, (mname, mpath) in zip(axes, method_paths.items()):
        if not os.path.exists(mpath):
            ax.text(0.5, 0.5, f"{mname}\n[data missing]",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_title(mname, fontsize=9, fontweight="bold")
            ax.axis("off")
            continue

        df = pd.read_csv(mpath)
        # find label and pred columns
        label_col = next((c for c in df.columns if any(k in c.lower() for k in ['y_true','label','true'])), None)
        pred_col  = next((c for c in df.columns if any(k in c.lower() for k in ['y_pred_optimal','y_pred','pred'])), None)
        if label_col is None or pred_col is None:
            # fallback: use proba column with threshold 0.5
            proba_col = next((c for c in df.columns if 'proba' in c.lower()), None)
            if label_col and proba_col:
                pred_col = proba_col
            else:
                ax.text(0.5, 0.5, f"{mname}\n[cols missing]",
                        ha='center', va='center', transform=ax.transAxes, fontsize=8)
                ax.axis('off'); continue
        y_true = df[label_col].values
        y_pred = df[pred_col].values
        if y_pred.dtype == float:
            y_pred = (y_pred >= 0.5).astype(int)

        plot_cm(ax, y_true, y_pred, mname)

    plt.tight_layout()

    out_dir = args.out_dir or f"/media/SAS/Van/HybridStackPPI/results/stats"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/cm_grid_{args.strategy}_{args.dataset}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
