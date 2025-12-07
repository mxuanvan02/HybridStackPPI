#!/usr/bin/env python3
"""
Experiment A: Sequence Length Robustness Analysis
===================================================
Phân tích hiệu suất model theo độ dài chuỗi protein.

Author: HybridStackPPI Team
"""

import sys
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from Bio import SeqIO

# Add project root to path
PROJECT_ROOT = Path('/media/SAS/Van/HybridStackPPI')
sys.path.insert(0, str(PROJECT_ROOT))


def load_cached_features(h5_path: str):
    """Load pre-computed features from H5 cache file."""
    print(f"Loading cached features from: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        # List available keys
        print(f"Available keys in H5: {list(f.keys())}")
        
        # Load features and labels - handle different key naming conventions
        if 'features' in f:
            X = f['features'][:]
        elif 'X_data' in f:
            X = f['X_data'][:]
        else:
            raise KeyError(f"No feature data found. Available keys: {list(f.keys())}")
        
        if 'labels' in f:
            y = f['labels'][:]
        elif 'y_data' in f:
            y = f['y_data'][:]
        else:
            raise KeyError(f"No label data found. Available keys: {list(f.keys())}")
        
        # Try to get pair IDs if available
        pair_ids = None
        if 'pair_ids' in f:
            pair_ids = f['pair_ids'][:]
        
    print(f"Loaded features shape: {X.shape}")
    print(f"Loaded labels shape: {y.shape}")
    
    return X, y, pair_ids


def load_sequences(fasta_path: str):
    """Load protein sequences from FASTA file."""
    seqs = {}
    for record in SeqIO.parse(fasta_path, 'fasta'):
        seqs[record.id] = str(record.seq)
    return seqs


def load_pairs_with_sequences(pairs_path: str, fasta_path: str):
    """Load protein pairs with their sequences."""
    seqs = load_sequences(fasta_path)
    
    pairs = []
    with open(pairs_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                p1, p2, label = parts[0], parts[1], int(parts[2])
                if p1 in seqs and p2 in seqs:
                    avg_len = (len(seqs[p1]) + len(seqs[p2])) / 2
                    pairs.append({
                        'protein1': p1,
                        'protein2': p2,
                        'label': label,
                        'seq1_len': len(seqs[p1]),
                        'seq2_len': len(seqs[p2]),
                        'avg_len': avg_len
                    })
    
    return pd.DataFrame(pairs)


def bin_by_length(df: pd.DataFrame):
    """Split data into Short/Medium/Long bins based on avg_len."""
    bins = {
        'Short (< 200 aa)': df[df['avg_len'] < 200],
        'Medium (200-500 aa)': df[(df['avg_len'] >= 200) & (df['avg_len'] <= 500)],
        'Long (> 500 aa)': df[df['avg_len'] > 500]
    }
    return bins


def train_and_evaluate_by_length(dataset='human'):
    """
    Train model on full data, then evaluate per length bin.
    """
    print("\n" + "=" * 70)
    print(f"EXPERIMENT A: SEQUENCE LENGTH ROBUSTNESS ANALYSIS ({dataset.upper()})")
    print("=" * 70)
    
    # Define paths based on dataset
    if dataset == 'human':
        cache_path = PROJECT_ROOT / 'cache/human_human_pairs_facebook_esm2_t33_650m_ur50d_concat_v2_features.h5'
        fasta_path = PROJECT_ROOT / 'data/BioGrid/Human/human_dict.fasta'
        pairs_path = PROJECT_ROOT / 'data/BioGrid/Human/human_pairs.tsv'
    else:
        cache_path = PROJECT_ROOT / 'cache/yeast_pairs_facebook_esm2_t33_650m_ur50d_concat_v2_features.h5'
        fasta_path = PROJECT_ROOT / 'data/BioGrid/Yeast/yeast_dict.fasta'
        pairs_path = PROJECT_ROOT / 'data/BioGrid/Yeast/yeast_pairs.tsv'
    
    # Load cached features (X, y)
    X, y, _ = load_cached_features(str(cache_path))
    
    # Load pairs with sequence info
    pairs_df = load_pairs_with_sequences(str(pairs_path), str(fasta_path))
    
    print(f"Total pairs: {len(pairs_df)}")
    print(f"Features shape: {X.shape}")
    
    # Ensure lengths match
    if len(pairs_df) != len(y):
        print(f"Warning: pairs_df ({len(pairs_df)}) != labels ({len(y)})")
        # Truncate to minimum
        min_len = min(len(pairs_df), len(y))
        pairs_df = pairs_df.iloc[:min_len]
        X = X[:min_len]
        y = y[:min_len]
    
    # Add to dataframe
    pairs_df['X_idx'] = range(len(pairs_df))
    
    # Train model on 80% data, evaluate on 20%
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from lightgbm import LGBMClassifier
    
    # Split
    train_idx, test_idx = train_test_split(range(len(X)), test_size=0.2, random_state=42, stratify=y)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_df = pairs_df.iloc[test_idx].copy()
    test_df['true_label'] = y_test
    
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train model
    print("\nTraining LightGBM classifier...")
    model = LGBMClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    test_df['pred_label'] = y_pred
    
    # Overall metrics
    print("\n" + "-" * 70)
    print("OVERALL TEST SET METRICS:")
    print("-" * 70)
    overall_acc = accuracy_score(y_test, y_pred) * 100
    overall_f1 = f1_score(y_test, y_pred) * 100
    overall_mcc = matthews_corrcoef(y_test, y_pred) * 100
    print(f"Accuracy: {overall_acc:.2f}%")
    print(f"F1-Score: {overall_f1:.2f}%")
    print(f"MCC: {overall_mcc:.2f}%")
    
    # Bin by length and evaluate
    print("\n" + "-" * 70)
    print("PER-LENGTH BIN METRICS:")
    print("-" * 70)
    
    bins = bin_by_length(test_df)
    
    results = []
    for bin_name, bin_df in bins.items():
        if len(bin_df) == 0:
            print(f"{bin_name:<25}: No samples")
            results.append({
                'Bin': bin_name,
                'N': 0,
                'Accuracy': None,
                'F1': None,
                'MCC': None
            })
            continue
        
        bin_y_true = bin_df['true_label'].values
        bin_y_pred = bin_df['pred_label'].values
        
        acc = accuracy_score(bin_y_true, bin_y_pred) * 100
        f1 = f1_score(bin_y_true, bin_y_pred, zero_division=0) * 100
        mcc = matthews_corrcoef(bin_y_true, bin_y_pred) * 100
        
        print(f"{bin_name:<25} (n={len(bin_df):>5}): Acc={acc:.2f}%, F1={f1:.2f}%, MCC={mcc:.2f}%")
        
        results.append({
            'Bin': bin_name,
            'N': len(bin_df),
            'Accuracy': acc,
            'F1': f1,
            'MCC': mcc
        })
    
    # Print LaTeX table rows
    print("\n" + "=" * 70)
    print("LaTeX TABLE ROWS:")
    print("=" * 70)
    
    for r in results:
        if r['Accuracy'] is not None:
            print(f"{r['Bin']:<25} & {r['Accuracy']:.2f} & {r['F1']:.2f} & {r['MCC']:.2f} \\\\")
        else:
            print(f"{r['Bin']:<25} & - & - & - \\\\")
    
    print(f"{'\\midrule'}")
    print(f"{'Overall':<25} & {overall_acc:.2f} & {overall_f1:.2f} & {overall_mcc:.2f} \\\\")
    
    # Save results
    results_file = PROJECT_ROOT / f'results/exp_a_length_analysis_{dataset}.txt'
    with open(results_file, 'w') as f:
        f.write(f"Experiment A: Sequence Length Robustness Analysis ({dataset.upper()})\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset: {dataset.upper()} BioGRID\n")
        f.write(f"Total test samples: {len(test_df)}\n\n")
        
        f.write("Per-Length Bin Results:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Bin':<25} {'N':<8} {'Accuracy':<12} {'F1':<12} {'MCC':<12}\n")
        f.write("-" * 70 + "\n")
        
        for r in results:
            if r['Accuracy'] is not None:
                f.write(f"{r['Bin']:<25} {r['N']:<8} {r['Accuracy']:<12.2f} {r['F1']:<12.2f} {r['MCC']:<12.2f}\n")
            else:
                f.write(f"{r['Bin']:<25} {r['N']:<8} {'N/A':<12} {'N/A':<12} {'N/A':<12}\n")
        
        f.write("-" * 70 + "\n")
        f.write(f"{'Overall':<25} {len(test_df):<8} {overall_acc:<12.2f} {overall_f1:<12.2f} {overall_mcc:<12.2f}\n")
        
        f.write("\n\nLaTeX Table Rows:\n")
        f.write("-" * 70 + "\n")
        for r in results:
            if r['Accuracy'] is not None:
                f.write(f"{r['Bin']:<25} & {r['Accuracy']:.2f} & {r['F1']:.2f} & {r['MCC']:.2f} \\\\\n")
        f.write(f"{'\\midrule'}\n")
        f.write(f"{'Overall':<25} & {overall_acc:.2f} & {overall_f1:.2f} & {overall_mcc:.2f} \\\\\n")
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Experiment A: Sequence Length Analysis')
    parser.add_argument('--dataset', choices=['human', 'yeast'], default='human',
                        help='Dataset to analyze (human or yeast)')
    parser.add_argument('--all', action='store_true', help='Run on both datasets')
    
    args = parser.parse_args()
    
    if args.all:
        train_and_evaluate_by_length('yeast')
        print("\n\n")
        train_and_evaluate_by_length('human')
    else:
        train_and_evaluate_by_length(args.dataset)
