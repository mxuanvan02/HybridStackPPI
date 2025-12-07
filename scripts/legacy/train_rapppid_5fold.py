#!/usr/bin/env python3
"""
RAPPPID 5-Fold Cross Validation Training Script
================================================

Train RAPPPID from scratch with proper 5-fold CV on BioGrid datasets.

Usage:
    conda run -n rapppid_env python scripts/train_rapppid_5fold.py --dataset yeast --epochs 20
    conda run -n rapppid_env python scripts/train_rapppid_5fold.py --dataset human --epochs 20
"""

import sys
import os
import argparse
import gzip
import pickle
import json
from pathlib import Path
from datetime import datetime
import subprocess
import numpy as np
from sklearn.model_selection import StratifiedKFold
from Bio import SeqIO

PROJECT_ROOT = Path('/media/SAS/Van/HybridStackPPI')
RAPPPID_DIR = PROJECT_ROOT / 'external_tools/RAPPPID'
RESULTS_DIR = PROJECT_ROOT / 'results'


def get_aa_code(aa):
    """Convert amino acid to integer code (RAPPPID format)."""
    aas = ['PAD', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
           'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'O', 'U']
    wobble_aas = {
        'B': ['D', 'N'],
        'Z': ['Q', 'E'],
        'X': aas[1:21]  # Any standard AA
    }

    if aa in aas:
        return aas.index(aa)
    elif aa in wobble_aas:
        return aas.index(wobble_aas[aa][0])  # Deterministic for reproducibility
    else:
        return 0  # PAD for unknown


def encode_seq(seq):
    """Encode amino acid sequence to list of integers."""
    return [get_aa_code(aa) for aa in seq.upper()]


def load_biogrid_data(dataset='yeast'):
    """Load BioGrid pairs and sequences."""
    data_dir = PROJECT_ROOT / f'data/BioGrid/{dataset.capitalize()}'
    
    pairs_file = data_dir / f'{dataset}_pairs.tsv'
    fasta_file = data_dir / f'{dataset}_dict.fasta'
    
    print(f"Loading sequences from: {fasta_file}")
    seqs = {}
    for record in SeqIO.parse(fasta_file, 'fasta'):
        seqs[record.id] = encode_seq(str(record.seq))
    print(f"Loaded {len(seqs)} sequences")
    
    print(f"Loading pairs from: {pairs_file}")
    pairs = []
    labels = []
    with open(pairs_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                p1, p2, label = parts[0], parts[1], int(parts[2])
                if p1 in seqs and p2 in seqs:
                    pairs.append((p1, p2, label))
                    labels.append(label)
    print(f"Loaded {len(pairs)} valid pairs")
    
    return pairs, labels, seqs


def create_fold_data(pairs, seqs, train_idx, test_idx, output_dir, fold):
    """Create train/val/test pkl.gz files for one fold."""
    fold_dir = output_dir / f'fold_{fold}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Split train into train/val (90/10)
    train_pairs_all = [pairs[i] for i in train_idx]
    np.random.shuffle(train_pairs_all)
    
    n_train = int(len(train_pairs_all) * 0.9)
    train_pairs = train_pairs_all[:n_train]
    val_pairs = train_pairs_all[n_train:]
    test_pairs = [pairs[i] for i in test_idx]
    
    print(f"  Fold {fold}: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
    
    # Save files
    with gzip.open(fold_dir / 'train.pkl.gz', 'wb') as f:
        pickle.dump(train_pairs, f)
    
    with gzip.open(fold_dir / 'val.pkl.gz', 'wb') as f:
        pickle.dump(val_pairs, f)
    
    with gzip.open(fold_dir / 'test.pkl.gz', 'wb') as f:
        pickle.dump(test_pairs, f)
    
    with gzip.open(fold_dir / 'seqs.pkl.gz', 'wb') as f:
        pickle.dump(seqs, f)
    
    return fold_dir


def train_one_fold(fold_dir, fold, epochs, log_dir):
    """Train RAPPPID on one fold."""
    print(f"\n{'='*60}")
    print(f"Training Fold {fold}")
    print(f"{'='*60}")
    
    # Create log directories
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / 'args').mkdir(exist_ok=True)
    (log_dir / 'chkpts').mkdir(exist_ok=True)
    (log_dir / 'charts').mkdir(exist_ok=True)
    (log_dir / 'tb_logs').mkdir(exist_ok=True)
    
    # Get SentencePiece model path
    spm_path = RAPPPID_DIR / 'data/pretrained_weights/1690837077.519848_red-dreamy/spm.model'
    
    log_file = log_dir / f'fold_{fold}_training.log'
    
    # Build command - run from RAPPPID dir with proper module path
    # Use PYTHONPATH to include rapppid subdir
    env = os.environ.copy()
    env['PYTHONPATH'] = str(RAPPPID_DIR / 'rapppid')
    
    cmd = [
        'conda', 'run', '-n', 'rapppid_env', 
        'python', 'rapppid/train.py',
        '--batch_size', '32',
        '--train_path', str(fold_dir / 'train.pkl.gz'),
        '--val_path', str(fold_dir / 'val.pkl.gz'),
        '--test_path', str(fold_dir / 'test.pkl.gz'),
        '--seqs_path', str(fold_dir / 'seqs.pkl.gz'),
        '--trunc_len', '1500',
        '--embedding_size', '64',
        '--num_epochs', str(epochs),
        '--lstm_dropout_rate', '0.3',
        '--classhead_dropout_rate', '0.2',
        '--rnn_num_layers', '2',
        '--classhead_num_layers', '2',
        '--lr', '0.001',
        '--weight_decay', '0.0001',
        '--bi_reduce', 'max',
        '--class_head_name', 'mult',
        '--variational_dropout', 'False',
        '--lr_scaling', 'False',
        '--model_file', str(spm_path),
        '--log_path', str(log_dir),
        '--vocab_size', '250',
        '--embedding_droprate', '0.3',
        '--optimizer_type', 'adam',
        '--seed', str(42 + fold),
    ]
    
    print(f"Log file: {log_file}")
    print(f"Running RAPPPID training...")
    
    with open(log_file, 'w') as f:
        result = subprocess.run(cmd, cwd=RAPPPID_DIR, stdout=f, stderr=subprocess.STDOUT, env=env)
    
    # Parse results from log
    metrics = parse_training_log(log_file)
    
    return result.returncode == 0, metrics


def parse_training_log(log_file):
    """Parse metrics from training log."""
    metrics = {
        'accuracy': None,
        'auroc': None,
        'apr': None
    }
    
    if not log_file.exists():
        return metrics
    
    # Look for test metrics in the log
    with open(log_file) as f:
        content = f.read()
        
    # Try to find metrics patterns
    import re
    
    # Look for test_acc, test_auroc, test_apr
    acc_match = re.search(r"'test_acc':\s*([\d.]+)", content)
    auroc_match = re.search(r"'test_auroc':\s*([\d.]+)", content)
    apr_match = re.search(r"'test_apr':\s*([\d.]+)", content)
    
    if acc_match:
        metrics['accuracy'] = float(acc_match.group(1))
    if auroc_match:
        metrics['auroc'] = float(auroc_match.group(1))
    if apr_match:
        metrics['apr'] = float(apr_match.group(1))
    
    return metrics


def run_5fold_cv(dataset, epochs, n_folds=5):
    """Run 5-fold cross validation."""
    print(f"\n{'='*60}")
    print(f"RAPPPID 5-Fold Cross Validation")
    print(f"Dataset: {dataset.upper()}")
    print(f"Epochs: {epochs}")
    print(f"{'='*60}")
    
    # Load data
    pairs, labels, seqs = load_biogrid_data(dataset)
    
    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = RESULTS_DIR / f'rapppid_{dataset}_5fold_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(pairs, labels)):
        print(f"\n--- Preparing Fold {fold} ---")
        
        # Create fold data
        fold_dir = create_fold_data(pairs, seqs, train_idx, test_idx, output_dir, fold)
        
        # Train
        log_dir = output_dir / f'fold_{fold}_logs'
        success, metrics = train_one_fold(fold_dir, fold, epochs, log_dir)
        
        if success and metrics['accuracy']:
            all_metrics.append(metrics)
            print(f"Fold {fold} Results: Acc={metrics['accuracy']:.4f}, AUROC={metrics['auroc']:.4f}")
        else:
            print(f"Fold {fold} training may have had issues, check logs")
    
    # Calculate averages
    if all_metrics:
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in all_metrics if m['accuracy']]),
            'accuracy_std': np.std([m['accuracy'] for m in all_metrics if m['accuracy']]),
            'auroc': np.mean([m['auroc'] for m in all_metrics if m['auroc']]),
            'auroc_std': np.std([m['auroc'] for m in all_metrics if m['auroc']]),
            'apr': np.mean([m['apr'] for m in all_metrics if m['apr']]),
            'apr_std': np.std([m['apr'] for m in all_metrics if m['apr']]),
        }
        
        print(f"\n{'='*60}")
        print(f"RAPPPID {dataset.upper()} 5-Fold CV Final Results")
        print(f"{'='*60}")
        print(f"Accuracy: {avg_metrics['accuracy']:.4f} ± {avg_metrics['accuracy_std']:.4f}")
        print(f"AUROC:    {avg_metrics['auroc']:.4f} ± {avg_metrics['auroc_std']:.4f}")
        print(f"APR:      {avg_metrics['apr']:.4f} ± {avg_metrics['apr_std']:.4f}")
        
        # Save summary
        summary = {
            'dataset': dataset,
            'n_folds': n_folds,
            'epochs': epochs,
            'total_pairs': len(pairs),
            'metrics': avg_metrics,
            'fold_metrics': all_metrics
        }
        
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save to main results file
        results_file = RESULTS_DIR / f'rapppid_{dataset}_5fold_results.txt'
        with open(results_file, 'w') as f:
            f.write(f"RAPPPID {dataset.upper()} 5-Fold CV Results\n")
            f.write(f"{'='*40}\n")
            f.write(f"Total pairs: {len(pairs)}\n")
            f.write(f"Epochs per fold: {epochs}\n\n")
            f.write(f"Metrics (mean ± std):\n")
            f.write(f"  Accuracy: {avg_metrics['accuracy']:.4f} ± {avg_metrics['accuracy_std']:.4f}\n")
            f.write(f"  AUROC:    {avg_metrics['auroc']:.4f} ± {avg_metrics['auroc_std']:.4f}\n")
            f.write(f"  APR:      {avg_metrics['apr']:.4f} ± {avg_metrics['apr_std']:.4f}\n")
        
        print(f"\nResults saved to: {results_file}")
        
        return avg_metrics
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Train RAPPPID with 5-fold CV')
    parser.add_argument('--dataset', choices=['yeast', 'human'], required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--folds', type=int, default=5)
    
    args = parser.parse_args()
    
    np.random.seed(42)
    
    run_5fold_cv(args.dataset, args.epochs, args.folds)


if __name__ == '__main__':
    main()
