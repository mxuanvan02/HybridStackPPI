"""
plot_multi_boxplot.py
─────────────────────
Multi-method 5-fold boxplot for MCC and PR-AUC.
Usage:
    python3 scripts/plot_multi_boxplot.py --strategy same_go
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = "/media/SAS/Van/HybridStackPPI/results"

PALETTE = {
    "SPRINT":       "#e74c3c",
    "AC":           "#e67e22",
    "CT":           "#f1c40f",
    "ESM-2+MLP":    "#3498db",
    "HybridStack":  "#27ae60",
}

METHOD_ORDER = ["SPRINT", "AC", "CT", "ESM-2+MLP", "HybridStack"]


def load_folds(path, colname):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    match = [c for c in df.columns if colname.lower() in c.lower()]
    if not match:
        return None
    return (df[match[0]].values * 100).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="same_go")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    suffix = f"_{args.strategy}" if args.strategy != "default" else ""

    paths = {
        "SPRINT":      f"{BASE}/github_baselines/sprint_human{suffix}/fold_metrics.csv",
        "AC":          f"{BASE}/baselines/human{suffix}/auto_covariance/fold_metrics.csv",
        "CT":          f"{BASE}/baselines/human{suffix}/conjoint_triad/fold_metrics.csv",
        "ESM-2+MLP":   f"{BASE}/github_baselines/esm2_mlp_human{suffix}/fold_metrics.csv",
        "HybridStack": f"{BASE}/human{suffix}/fold_metrics.csv",
    }

    metrics = [("MCC", "MCC (%)"), ("PR-AUC", "PR-AUC (%)")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"5-Fold Performance Distribution — Human "
        f"({'Same-GO' if 'go' in args.strategy else 'Same-Compartment'})",
        fontsize=12, fontweight="bold"
    )

    for ax, (col_key, ylabel) in zip(axes, metrics):
        data, labels, colors = [], [], []
        for mname in METHOD_ORDER:
            vals = load_folds(paths[mname], col_key)
            if vals is not None:
                data.append(vals)
                labels.append(mname)
                colors.append(PALETTE.get(mname, "#95a5a6"))

        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # scatter individual folds
        for i, (vals, color) in enumerate(zip(data, colors)):
            jitter = np.random.uniform(-0.12, 0.12, len(vals))
            ax.scatter([i + 1 + j for j in jitter], vals,
                       color=color, alpha=0.9, s=40, zorder=5,
                       edgecolors="white", linewidth=0.5)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=10, fontweight="bold")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)
        # highlight HybridStack
        if "HybridStack" in labels:
            idx = labels.index("HybridStack") + 1
            ax.axvline(idx, color="#27ae60", linestyle=":", alpha=0.5, linewidth=1.5)

    plt.tight_layout()

    out_dir = args.out_dir or f"{BASE}/stats"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/multi_boxplot_{args.strategy}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
