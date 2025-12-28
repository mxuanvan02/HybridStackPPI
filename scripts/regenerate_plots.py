#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import plot_cv_roc_pr_curves

def regrow_plots(results_dir, dataset_name):
    print(f"\n🎨 Regenerating plots for {dataset_name} in {results_dir}...")
    preds_file = Path(results_dir) / "all_folds_predictions.csv"
    if not preds_file.exists():
        print(f"❌ Error: {preds_file} not found.")
        return

    df = pd.read_csv(preds_file)
    cv_results = []
    
    # Check if fold_id is 0-indexed or 1-indexed
    folds = sorted(df['fold_id'].unique())
    print(f"   Found folds: {folds}")
    
    for f_id in folds:
        fold_df = df[df['fold_id'] == f_id]
        cv_results.append({
            'y_true': fold_df['y_true'].values,
            'y_proba': fold_df['y_proba'].values
        })
    
    plots_dir = Path(results_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # We call with the prefix for the actual PNG files
    plot_cv_roc_pr_curves(
        cv_results,
        save_dir=str(plots_dir),
        title=f"5-Fold C3 CV ({dataset_name})",
        prefix=f"{dataset_name.lower()}_c3_cv"
    )
    print(f"✅ Success! New plots saved in {plots_dir}")

if __name__ == "__main__":
    # Yeast results from 01:10
    regrow_plots("results/Yeast_C3_CV_20251228_011030", "Yeast")
    
    # Human results from 01:50
    regrow_plots("results/Human_C3_CV_20251228_015001", "Human")
