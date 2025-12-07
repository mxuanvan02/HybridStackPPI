#!/usr/bin/env python3
"""
Reproduce Results Script
=========================
Reproduces the benchmark results from the paper.

This script runs 5-fold cross-validation on the Human BioGRID dataset
using protein-level splits to prevent data leakage.

Usage:
    python reproduce_results.py
    python reproduce_results.py --dataset yeast
    python reproduce_results.py --n-splits 10
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # For deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce HybridStack-PPI benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--dataset",
        choices=["human", "yeast", "both"],
        default="human",
        help="Dataset to evaluate (default: human)"
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)"
    )
    parser.add_argument(
        "--pairing",
        choices=["concat", "avgdiff"],
        default="concat",
        help="Pairing strategy (default: concat)"
    )
    parser.add_argument(
        "--esm-model",
        default="facebook/esm2_t33_650M_UR50D",
        help="ESM-2 model name"
    )
    parser.add_argument(
        "--h5-cache",
        default="cache/esm2_embeddings.h5",
        help="Path to ESM embedding cache"
    )
    parser.add_argument(
        "--cache-version",
        default="v3",
        help="Cache version tag"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (default: -1 = all cores)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seed FIRST for reproducibility
    set_seed(args.seed)
    
    print("=" * 70)
    print("HybridStack-PPI: Reproducing Paper Results")
    print("=" * 70)
    print(f"\nRandom Seed: {args.seed}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Import after setting seed
    from scripts.run import run_experiment, create_stacking_pipeline_for_notebook
    
    # Dataset configurations
    datasets = []
    if args.dataset in ["human", "both"]:
        datasets.append({
            "name": "Human BioGRID",
            "fasta": str(PROJECT_ROOT / "data/BioGrid/Human/human_dict.fasta"),
            "pairs": str(PROJECT_ROOT / "data/BioGrid/Human/human_pairs.tsv"),
        })
    if args.dataset in ["yeast", "both"]:
        datasets.append({
            "name": "Yeast BioGRID",
            "fasta": str(PROJECT_ROOT / "data/BioGrid/Yeast/yeast_dict.fasta"),
            "pairs": str(PROJECT_ROOT / "data/BioGrid/Yeast/yeast_pairs.tsv"),
        })
    
    all_results = {}
    
    for ds in datasets:
        print("\n" + "=" * 70)
        print(f"Dataset: {ds['name']}")
        print("=" * 70)
        print(f"FASTA: {ds['fasta']}")
        print(f"Pairs: {ds['pairs']}")
        print(f"CV Folds: {args.n_splits}")
        print(f"Pairing Strategy: {args.pairing}")
        
        # Check files exist
        if not os.path.exists(ds['fasta']):
            print(f"ERROR: FASTA file not found: {ds['fasta']}")
            continue
        if not os.path.exists(ds['pairs']):
            print(f"ERROR: Pairs file not found: {ds['pairs']}")
            continue
        
        # Create model factory
        def model_factory(n_jobs=-1):
            return create_stacking_pipeline_for_notebook(
                pairing_strategy=args.pairing,
                n_jobs=n_jobs,
                h5_cache_path=args.h5_cache,
                esm_model_name=args.esm_model,
            )
        
        # Run experiment
        start_time = time.time()
        
        try:
            metrics = run_experiment(
                fasta_path=ds['fasta'],
                pairs_path=ds['pairs'],
                h5_cache_path=args.h5_cache,
                model_factory=model_factory,
                pairing_strategy=args.pairing,
                n_splits=args.n_splits,
                esm_model_name=args.esm_model,
                n_jobs=args.n_jobs,
                cache_version=args.cache_version,
            )
            
            elapsed = time.time() - start_time
            all_results[ds['name']] = metrics
            
            print(f"\n✅ Completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print final summary
    if all_results:
        print("\n" + "=" * 70)
        print("FINAL RESULTS SUMMARY")
        print("=" * 70)
        
        for dataset_name, metrics in all_results.items():
            print(f"\n📊 {dataset_name}:")
            print("-" * 40)
            
            # Map metric names to paper format
            metric_mapping = {
                "Accuracy": "Accuracy",
                "F1 Score": "F1",
                "MCC": "MCC",
                "ROC-AUC": "AUC-ROC",
                "PR-AUC": "AUC-PR",
                "Precision": "Precision",
                "Recall (Sensitivity)": "Recall",
                "Specificity": "Specificity",
            }
            
            for metric_name, display_name in metric_mapping.items():
                if metric_name in metrics:
                    value = metrics[metric_name]
                    if isinstance(value, float):
                        print(f"  {display_name:<20}: {value*100:.2f}%")
        
        # LaTeX table row
        print("\n" + "-" * 70)
        print("LaTeX Table Row:")
        print("-" * 70)
        
        for dataset_name, metrics in all_results.items():
            latex_row = "HybridStack-PPI"
            for key in ["Accuracy", "Precision", "Recall (Sensitivity)", "Specificity", "F1 Score", "MCC"]:
                if key in metrics:
                    latex_row += f" & {metrics[key]*100:.2f}"
            latex_row += " \\\\"
            print(latex_row)
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
