#!/usr/bin/env python3
"""
HybridStackPPI 5-Fold Cross Validation with Visualizations
============================================================
Chạy 5-fold CV với chiến lược UNSEEN-UNSEEN (protein-based splits).

Author: HybridStackPPI Team
"""

import sys
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix
)
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

PROJECT_ROOT = Path('/media/SAS/Van/HybridStackPPI')
sys.path.insert(0, str(PROJECT_ROOT))

from hybridstack.data_utils import load_data, get_protein_based_splits


def load_cached_features_with_index(h5_path: str):
    """Load pre-computed features from H5 cache file with index."""
    print(f"Loading cached features from: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        print(f"  Available keys: {list(f.keys())}")
        
        if 'X_data' in f:
            X = f['X_data'][:]
        elif 'features' in f:
            X = f['features'][:]
        else:
            raise KeyError(f"No feature data found. Keys: {list(f.keys())}")
        
        if 'y_data' in f:
            y = f['y_data'][:]
        elif 'labels' in f:
            y = f['labels'][:]
        else:
            raise KeyError(f"No label data found. Keys: {list(f.keys())}")
        
        # Try to get index
        if 'X_index' in f:
            X_index = f['X_index'][:]
            if X_index.dtype.kind == 'S':  # bytes
                X_index = [x.decode('utf-8') if isinstance(x, bytes) else x for x in X_index]
        else:
            X_index = list(range(len(X)))
    
    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape: {y.shape}, positives: {sum(y)}, negatives: {len(y) - sum(y)}")
    
    return X, y, X_index


def run_5fold_cv_unseen(fasta_path, pairs_path, cache_path, dataset_name, output_dir):
    """
    Run 5-fold CV with UNSEEN-UNSEEN strategy.
    No protein appears in both train and test sets.
    """
    print(f"\n{'='*70}")
    print(f"5-FOLD CROSS VALIDATION (UNSEEN-UNSEEN): {dataset_name.upper()}")
    print(f"{'='*70}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load pairs data for protein-based splitting
    print("\nLoading pairs data...")
    sequences, pairs_df = load_data(fasta_path, pairs_path)
    print(f"  Total proteins: {len(sequences)}")
    print(f"  Total pairs: {len(pairs_df)}")
    
    # Load cached features
    X, y, X_index = load_cached_features_with_index(str(cache_path))
    
    # Create index mapping
    if isinstance(X_index[0], int):
        # Index is integer-based
        index_to_pos = {idx: i for i, idx in enumerate(X_index)}
    else:
        # Index might be string-based
        index_to_pos = {i: i for i in range(len(X))}
    
    # Get protein-based splits (UNSEEN-UNSEEN)
    print("\nGenerating protein-based splits (UNSEEN-UNSEEN)...")
    splits = get_protein_based_splits(pairs_df, n_splits=5, random_state=42)
    
    # Store results per fold
    fold_metrics = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    fold_roc_data = []
    fold_pr_data = []
    
    for fold, (train_indices, test_indices) in enumerate(splits):
        print(f"\n--- Fold {fold + 1}/5 ---")
        
        # Map pair indices to feature indices
        # The indices from get_protein_based_splits are pair indices in pairs_df
        train_mask = np.isin(range(len(X)), train_indices)
        test_mask = np.isin(range(len(X)), test_indices)
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  WARNING: Empty split! Train={len(X_train)}, Test={len(X_test)}")
            continue
        
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"  Train pos/neg: {sum(y_train)}/{len(y_train)-sum(y_train)}")
        print(f"  Test pos/neg: {sum(y_test)}/{len(y_test)-sum(y_test)}")
        
        # Train model
        model = LGBMClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.05,
            num_leaves=31,
            n_jobs=-1,
            random_state=42 + fold,
            verbose=-1,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, zero_division=0) * 100
        rec = recall_score(y_test, y_pred, zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, zero_division=0) * 100
        spec = recall_score(y_test, y_pred, pos_label=0, zero_division=0) * 100
        mcc = matthews_corrcoef(y_test, y_pred) * 100
        auroc = roc_auc_score(y_test, y_prob) * 100
        auprc = average_precision_score(y_test, y_prob) * 100
        
        fold_metrics.append({
            'fold': fold + 1,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'specificity': spec,
            'f1': f1,
            'mcc': mcc,
            'auroc': auroc,
            'auprc': auprc
        })
        
        print(f"  Acc={acc:.2f}%, F1={f1:.2f}%, MCC={mcc:.2f}%, AUROC={auroc:.2f}%")
        
        # Store for aggregated plots
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)
        
        # ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fold_roc_data.append((fpr, tpr, auroc))
        
        # PR curve data
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        fold_pr_data.append((recall_curve, precision_curve, auprc))
    
    if not fold_metrics:
        print("ERROR: No valid folds!")
        return None, None
    
    # Calculate summary statistics
    df_metrics = pd.DataFrame(fold_metrics)
    summary = {}
    for col in ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'mcc', 'auroc', 'auprc']:
        summary[col] = {
            'mean': df_metrics[col].mean(),
            'std': df_metrics[col].std()
        }
    
    print(f"\n{'='*70}")
    print("SUMMARY (Mean ± Std) - UNSEEN-UNSEEN Strategy")
    print(f"{'='*70}")
    for key, stats in summary.items():
        print(f"  {key.upper():<12}: {stats['mean']:.2f} ± {stats['std']:.2f}")
    
    # ==========================================================================
    # PLOT 1: ROC Curves (all folds + mean)
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    
    for i, (fpr, tpr, auroc) in enumerate(fold_roc_data):
        ax.plot(fpr, tpr, lw=1.5, alpha=0.7, label=f'Fold {i+1} (AUC = {auroc:.2f}%)')
    
    mean_auroc = summary['auroc']['mean']
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 50%)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curves (Unseen-Unseen) - {dataset_name.upper()} BioGRID\nMean AUC = {mean_auroc:.2f}%', fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    roc_path = output_dir / f'roc_curves_unseen_{dataset_name}.png'
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {roc_path}")
    
    # ==========================================================================
    # PLOT 2: Precision-Recall Curves (all folds)
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    
    for i, (rec, prec, auprc) in enumerate(fold_pr_data):
        ax.plot(rec, prec, lw=1.5, alpha=0.7, label=f'Fold {i+1} (AP = {auprc:.2f}%)')
    
    mean_auprc = summary['auprc']['mean']
    ax.axhline(y=0.5, color='k', linestyle='--', lw=2, label='Baseline')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curves (Unseen-Unseen) - {dataset_name.upper()} BioGRID\nMean AP = {mean_auprc:.2f}%', fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    pr_path = output_dir / f'pr_curves_unseen_{dataset_name}.png'
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {pr_path}")
    
    # ==========================================================================
    # PLOT 3: Confusion Matrix (aggregated)
    # ==========================================================================
    cm = confusion_matrix(all_y_true, all_y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-Interacting', 'Interacting'],
                yticklabels=['Non-Interacting', 'Interacting'],
                annot_kws={'size': 14})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix (Unseen-Unseen) - {dataset_name.upper()} BioGRID\n(Aggregated from 5 Folds)', fontsize=14)
    
    cm_path = output_dir / f'confusion_matrix_unseen_{dataset_name}.png'
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {cm_path}")
    
    # ==========================================================================
    # PLOT 4: Metrics Boxplot
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'mcc']
    
    colors = sns.color_palette("husl", len(metrics_to_plot))
    box = ax.boxplot([df_metrics[m] for m in metrics_to_plot], patch_artist=True, labels=[m.upper() for m in metrics_to_plot])
    
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title(f'5-Fold CV Metrics (Unseen-Unseen) - {dataset_name.upper()} BioGRID', fontsize=14)
    min_val = df_metrics[metrics_to_plot].values.min()
    ax.set_ylim([max(0, min_val - 10), 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, m in enumerate(metrics_to_plot):
        mean_val = summary[m]['mean']
        ax.text(i + 1, mean_val + 2, f'{mean_val:.1f}', ha='center', fontsize=9, fontweight='bold')
    
    boxplot_path = output_dir / f'metrics_boxplot_unseen_{dataset_name}.png'
    plt.tight_layout()
    plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {boxplot_path}")
    
    # ==========================================================================
    # PLOT 5: Per-Fold Bar Chart
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    
    x = np.arange(len(fold_metrics))
    width = 0.12
    metrics_bar = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'auroc']
    colors = sns.color_palette("husl", len(metrics_bar))
    
    for i, (metric, color) in enumerate(zip(metrics_bar, colors)):
        values = df_metrics[metric].values
        ax.bar(x + i * width, values, width, label=metric.upper(), color=color, alpha=0.85)
    
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title(f'Per-Fold Performance (Unseen-Unseen) - {dataset_name.upper()} BioGRID', fontsize=14)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(len(fold_metrics))])
    ax.legend(loc='lower right', fontsize=10, ncol=3)
    min_val = df_metrics[metrics_bar].values.min()
    ax.set_ylim([max(0, min_val - 10), 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    bar_path = output_dir / f'per_fold_bars_unseen_{dataset_name}.png'
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {bar_path}")
    
    # ==========================================================================
    # Save results to text file
    # ==========================================================================
    results_file = output_dir / f'5fold_cv_unseen_results_{dataset_name}.txt'
    with open(results_file, 'w') as f:
        f.write(f"HybridStackPPI 5-Fold Cross Validation Results (UNSEEN-UNSEEN)\n")
        f.write(f"Dataset: {dataset_name.upper()} BioGRID\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Split Strategy: UNSEEN-UNSEEN (No protein in both train/test)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Per-Fold Results:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Fold':<6} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'Specificity':<12} {'F1':<12} {'MCC':<12} {'AUROC':<12} {'AUPRC':<12}\n")
        f.write("-" * 100 + "\n")
        for m in fold_metrics:
            f.write(f"{m['fold']:<6} {m['accuracy']:<12.2f} {m['precision']:<12.2f} {m['recall']:<12.2f} "
                    f"{m['specificity']:<12.2f} {m['f1']:<12.2f} {m['mcc']:<12.2f} {m['auroc']:<12.2f} {m['auprc']:<12.2f}\n")
        f.write("-" * 100 + "\n\n")
        
        f.write("Summary (Mean ± Std):\n")
        f.write("-" * 70 + "\n")
        for key, stats in summary.items():
            f.write(f"  {key.upper():<12}: {stats['mean']:.2f} ± {stats['std']:.2f}\n")
        f.write("\n")
        
        f.write("LaTeX Table Row:\n")
        f.write("-" * 70 + "\n")
        latex_row = f"HybridStack-PPI & "
        latex_row += f"${summary['accuracy']['mean']:.2f} \\pm {summary['accuracy']['std']:.2f}$ & "
        latex_row += f"${summary['precision']['mean']:.2f} \\pm {summary['precision']['std']:.2f}$ & "
        latex_row += f"${summary['recall']['mean']:.2f} \\pm {summary['recall']['std']:.2f}$ & "
        latex_row += f"${summary['specificity']['mean']:.2f} \\pm {summary['specificity']['std']:.2f}$ & "
        latex_row += f"${summary['f1']['mean']:.2f} \\pm {summary['f1']['std']:.2f}$ & "
        latex_row += f"${summary['mcc']['mean']:.2f} \\pm {summary['mcc']['std']:.2f}$ \\\\"
        f.write(latex_row + "\n")
    
    print(f"✅ Saved: {results_file}")
    
    return summary, fold_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description='HybridStackPPI 5-Fold CV with UNSEEN-UNSEEN')
    parser.add_argument('--dataset', choices=['yeast', 'human', 'all'], default='all',
                        help='Dataset to run (default: all)')
    parser.add_argument('--output-dir', default='results/plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    output_dir = PROJECT_ROOT / args.output_dir
    
    datasets = []
    if args.dataset in ['yeast', 'all']:
        datasets.append({
            'name': 'yeast',
            'fasta': str(PROJECT_ROOT / 'data/BioGrid/Yeast/yeast_dict.fasta'),
            'pairs': str(PROJECT_ROOT / 'data/BioGrid/Yeast/yeast_pairs.tsv'),
            'cache': str(PROJECT_ROOT / 'cache/yeast_pairs_facebook_esm2_t33_650m_ur50d_concat_v2_features.h5')
        })
    if args.dataset in ['human', 'all']:
        datasets.append({
            'name': 'human',
            'fasta': str(PROJECT_ROOT / 'data/BioGrid/Human/human_dict.fasta'),
            'pairs': str(PROJECT_ROOT / 'data/BioGrid/Human/human_pairs.tsv'),
            'cache': str(PROJECT_ROOT / 'cache/human_human_pairs_facebook_esm2_t33_650m_ur50d_concat_v2_features.h5')
        })
    
    all_results = {}
    
    for ds in datasets:
        print(f"\n{'#'*70}")
        print(f"# DATASET: {ds['name'].upper()}")
        print(f"{'#'*70}")
        
        summary, fold_metrics = run_5fold_cv_unseen(
            ds['fasta'], ds['pairs'], ds['cache'],
            ds['name'], output_dir
        )
        if summary:
            all_results[ds['name']] = summary
    
    # Print combined summary
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("COMBINED SUMMARY (UNSEEN-UNSEEN)")
        print(f"{'='*70}")
        for ds, summary in all_results.items():
            print(f"\n{ds.upper()}:")
            print(f"  Accuracy: {summary['accuracy']['mean']:.2f} ± {summary['accuracy']['std']:.2f}")
            print(f"  F1-Score: {summary['f1']['mean']:.2f} ± {summary['f1']['std']:.2f}")
            print(f"  MCC:      {summary['mcc']['mean']:.2f} ± {summary['mcc']['std']:.2f}")
            print(f"  AUROC:    {summary['auroc']['mean']:.2f} ± {summary['auroc']['std']:.2f}")


if __name__ == '__main__':
    main()
