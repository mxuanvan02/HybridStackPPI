#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def plot_calibration_curve(y_true, y_proba, dataset_name, save_path):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(prob_pred, prob_true, "s-", label=f"HybridStack-PPI ({dataset_name})", color="#e67e22", lw=2)
    
    plt.ylabel("Fraction of Positives", fontsize=12, fontweight='bold')
    plt.xlabel("Mean Predicted Probability", fontsize=12, fontweight='bold')
    plt.title(f"Calibration Curve: Pred. Prob. vs Actual PPIs\n({dataset_name})", fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_confusion_matrix_paper(y_true, y_pred, dataset_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=['Non-PPI', 'PPI'], 
                yticklabels=['Non-PPI', 'PPI'],
                annot_kws={"size": 16, "fontweight": "bold"})
    
    plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title(f"Confusion Matrix (Normalized)\n{dataset_name} Dataset", fontsize=14, fontweight='bold')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = {
        'Yeast': ('results/Yeast_C3_CV_20251228_011030/all_folds_predictions.csv', 'results/Yeast_C3_CV_20251228_011030/plots'),
        'Human': ('results/Human_C3_CV_20251228_015001/all_folds_predictions.csv', 'results/Human_C3_CV_20251228_015001/plots')
    }
    
    for name, (path, out_dir) in datasets.items():
        if not os.path.exists(path):
            print(f"⚠️ Missing predictions for {name} at {path}")
            continue
            
        print(f"📊 Generating extras for {name} in {out_dir}...")
        os.makedirs(out_dir, exist_ok=True)
        df = pd.read_csv(path)
        
        # 1. Calibration
        plot_calibration_curve(df['y_true'], df['y_proba'], name, os.path.join(out_dir, f"calibration_curve_{name.lower()}.png"))
        
        # 2. Confusion Matrix
        plot_confusion_matrix_paper(df['y_true'], df['y_pred'], name, os.path.join(out_dir, f"confusion_matrix_{name.lower()}.png"))
        
    print(f"✅ Extra visualizations saved in {plots_dir}")

if __name__ == "__main__":
    main()
