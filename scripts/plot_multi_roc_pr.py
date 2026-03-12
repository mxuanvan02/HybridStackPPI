import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-method Plotting")
    parser.add_argument("--dataset", type=str, required=True, choices=["human", "yeast"])
    return parser.parse_args()

def main():
    args = parse_args()
    dataset = args.dataset
    
    methods = {
        "HybridStackPPI (Ours)": f"results/{dataset}/oof_predictions.csv",
        "Vanilla ESM-2 + MLP": f"results/github_baselines/esm2_mlp_{dataset}/oof_predictions.csv",
        "Conjoint Triad (CT)": f"results/baselines/{dataset}/conjoint_triad/oof_predictions.csv",
        "Auto Covariance (AC)": f"results/baselines/{dataset}/auto_covariance/oof_predictions.csv",
        "SPRINT": f"results/github_baselines/sprint_{dataset}/oof_predictions.csv",
    }
    
    colors = {
        "HybridStackPPI (Ours)": "#d62728", # red
        "Vanilla ESM-2 + MLP": "#1f77b4", # blue
        "Conjoint Triad (CT)": "#9467bd", # purple
        "Auto Covariance (AC)": "#ff7f0e", # orange
        "SPRINT": "#2ca02c" # green
    }
    
    linestyles = {
        "HybridStackPPI (Ours)": "-",
        "Vanilla ESM-2 + MLP": "--",
        "Conjoint Triad (CT)": ":",
        "Auto Covariance (AC)": ":",
        "SPRINT": "-."
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    
    # 1. Plot ROC Curve
    plt.figure(figsize=(10, 8))
    
    for name, path in methods.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'y_true' in df.columns and 'y_proba' in df.columns:
                y_true = df['y_true'].values
                y_proba = df['y_proba'].values
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc = auc(fpr, tpr)
                
                # SPRINT needs special handling for ROC if the probabilities are not well calibrated
                plt.plot(fpr, tpr, color=colors[name], linestyle=linestyles[name], lw=2.5 if name == "HybridStackPPI (Ours)" else 2,
                         label=f'{name} (AUC = {roc_auc:.3f})')
            else:
                print(f"Warning: {path} missing 'y_true' or 'y_proba' columns.")
        else:
            print(f"Warning: {path} not found.")
            
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title(f'Multi-method ROC Curve overlay ({dataset.capitalize()})', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    
    output_roc = f"results/multi_roc_{dataset}.png"
    plt.tight_layout()
    plt.savefig(output_roc, dpi=300)
    print(f"✅ Saved ROC overlay to {output_roc}")
    
    # 2. Plot PR Curve
    plt.figure(figsize=(10, 8))
    
    for name, path in methods.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'y_true' in df.columns and 'y_proba' in df.columns:
                y_true = df['y_true'].values
                y_proba = df['y_proba'].values
                precision, recall, _ = precision_recall_curve(y_true, y_proba)
                pr_auc = average_precision_score(y_true, y_proba)
                
                plt.plot(recall, precision, color=colors[name], linestyle=linestyles[name], lw=2.5 if name == "HybridStackPPI (Ours)" else 2,
                         label=f'{name} (AUPRC = {pr_auc:.3f})')
    
    # Calculate baseline PR for dummy classifier
    if os.path.exists(methods["HybridStackPPI (Ours)"]):
        df = pd.read_csv(methods["HybridStackPPI (Ours)"])
        pos_ratio = df["y_true"].mean()
        plt.axhline(y=pos_ratio, color='black', lw=2, linestyle='--', label=f'Random (AUPRC = {pos_ratio:.3f})')

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.title(f'Multi-method PR Curve overlay ({dataset.capitalize()})', fontsize=16, fontweight='bold')
    plt.legend(loc="upper right", fontsize=12)
    
    output_pr = f"results/multi_pr_{dataset}.png"
    plt.tight_layout()
    plt.savefig(output_pr, dpi=300)
    print(f"✅ Saved PR overlay to {output_pr}")


if __name__ == "__main__":
    main()
