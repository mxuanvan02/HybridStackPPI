"""
statistical_tests.py
────────────────────
Run Wilcoxon signed-rank tests comparing HybridStackPPI vs baselines.
Usage:
    python3 scripts/statistical_tests.py --strategy same_go
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

BASE = "/media/SAS/Van/HybridStackPPI/results"

def load_fold_metric(path, col):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # find column
    matches = [c for c in df.columns if col.lower() in c.lower()]
    if not matches:
        return None
    return df[matches[0]].values

def run_tests(strategy="same_go"):
    suffix = f"_{strategy}" if strategy != "default" else ""

    configs = {
        "Human": {
            "HybridStack": f"{BASE}/human{suffix}/fold_metrics.csv",
            "ESM-2+MLP":   f"{BASE}/github_baselines/esm2_mlp_human{suffix}/fold_metrics.csv",
            "SPRINT":      f"{BASE}/github_baselines/sprint_human{suffix}/fold_metrics.csv",
            "AC":          f"{BASE}/baselines/human{suffix}/auto_covariance/fold_metrics.csv",
            "CT":          f"{BASE}/baselines/human{suffix}/conjoint_triad/fold_metrics.csv",
        },
        "Yeast": {
            "HybridStack": f"{BASE}/yeast{suffix}/fold_metrics.csv",
            "ESM-2+MLP":   f"{BASE}/github_baselines/esm2_mlp_yeast{suffix}/fold_metrics.csv",
            "SPRINT":      f"{BASE}/github_baselines/sprint_yeast{suffix}/fold_metrics.csv",
            "AC":          f"{BASE}/baselines/yeast{suffix}/auto_covariance/fold_metrics.csv",
            "CT":          f"{BASE}/baselines/yeast{suffix}/conjoint_triad/fold_metrics.csv",
        }
    }

    metrics_to_test = ["MCC", "PR-AUC", "ROC-AUC"]
    results = []

    for species, paths in configs.items():
        ours_path = paths["HybridStack"]
        for metric_col in metrics_to_test:
            ours_vals = load_fold_metric(ours_path, metric_col)
            if ours_vals is None:
                continue
            for method, path in paths.items():
                if method == "HybridStack":
                    continue
                other_vals = load_fold_metric(path, metric_col)
                if other_vals is None or len(other_vals) < 3:
                    continue
                # align lengths
                n = min(len(ours_vals), len(other_vals))
                stat, p = stats.wilcoxon(ours_vals[:n], other_vals[:n], alternative='greater')
                results.append({
                    "Species": species,
                    "Metric": metric_col,
                    "Baseline": method,
                    "HybridStack Mean": f"{np.mean(ours_vals)*100:.2f}%",
                    "Baseline Mean": f"{np.mean(other_vals)*100:.2f}%",
                    "Δ": f"{(np.mean(ours_vals) - np.mean(other_vals))*100:+.2f}%",
                    "p-value": p,
                    "Significant (p<0.05)": "✅" if p < 0.05 else "❌"
                })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # Save CSV
    out_dir = f"{BASE}/stats"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f"{out_dir}/wilcoxon_{strategy}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Plot p-value heatmap
    if len(df) > 0:
        _plot_heatmap(df, strategy, out_dir)

    return df

def _plot_heatmap(df, strategy, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    fig.suptitle(f"Wilcoxon Signed-Rank p-values — HybridStackPPI vs Baselines\n({strategy.replace('_', ' ').title()})",
                 fontsize=11, fontweight="bold")

    for ax, species in zip(axes, ["Human", "Yeast"]):
        sdf = df[df["Species"] == species]
        if sdf.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(species)
            continue

        pivot = sdf.pivot_table(values="p-value", index="Baseline", columns="Metric", aggfunc="first")
        mask = pivot.isnull()

        sns.heatmap(pivot, ax=ax, cmap="RdYlGn_r", vmin=0, vmax=0.1,
                    annot=True, fmt=".3f", linewidths=0.5,
                    linecolor="gray", mask=mask,
                    cbar_kws={"label": "p-value"})

        # Highlight significant cells
        for (r, c), val in np.ndenumerate(pivot.values):
            if not np.isnan(val) and val < 0.05:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False,
                                           edgecolor="blue", lw=2))

        ax.set_title(f"{species}", fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out_path = f"{out_dir}/wilcoxon_heatmap_{strategy}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap: {out_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="same_go")
    args = parser.parse_args()
    run_tests(args.strategy)
