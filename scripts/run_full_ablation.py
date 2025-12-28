#!/usr/bin/env python3
"""
HybridStack-PPI Full Ablation Study (Cache-Optimized)
======================================================
Runs ablation study using PRE-COMPUTED feature caches for speed.
Uses centralized config for all paths.
"""
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Suppress warnings
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import centralized config
from scripts.config import (
    DATASETS, 
    ESM_CACHE_PATH, 
    CV_N_SPLITS,
    get_dataset_config,
    validate_cache_files,
)

from src.data_utils import load_feature_matrix_h5
from src.builders import (
    create_stacking_pipeline,
    create_interp_only_stacking_pipeline,
    create_embed_only_stacking_pipeline,
    define_stacking_columns,
)
from src.feature_engine import EmbeddingComputer, FeatureEngine
from src.metrics import display_full_metrics, generate_latex_table
from src.logger import PipelineLogger
from scripts.run_cv import parse_clstr_to_mapping, get_c3_splits, verify_split_integrity


def run_ablation_with_cache(
    dataset_name: str,
    n_splits: int = 5,
    n_jobs: int = -1,
    clstr_path: str = None,
    output_dir: Path = None,
):
    """
    Run ablation study using pre-computed feature cache.
    
    Variants:
    1. Hybrid (Proposed) - Both branches
    2. Interp-Only - Biological features only
    3. Embed-Only - ESM-2 embeddings only
    
    Args:
        dataset_name: 'yeast' or 'human'
        n_splits: Number of CV folds
        n_jobs: Parallel jobs
        output_dir: Output directory
    """
    logger = PipelineLogger()
    config = get_dataset_config(dataset_name)
    
    # Auto-Routing Protocol
    clstr_path = clstr_path if clstr_path else config['clstr']
    
    # Validate cache
    if not validate_cache_files(dataset_name):
        raise FileNotFoundError(f"Missing cache files for {dataset_name}")
    
    logger.header(f"ABLATION STUDY: {config['full_name']} (Cache Mode)")
    print(f"  Feature Cache: {config['feature_cache']}")
    print(f"  Cluster File: {clstr_path}")
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / f"results/{config['name']}_Ablation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load feature cache
    logger.phase("Loading Feature Cache")
    X_df, y_s = load_feature_matrix_h5(config['feature_cache'])
    print(f"  ✅ Loaded X={X_df.shape}, y={len(y_s)}")
    
    # Step 2: Load cluster mapping
    logger.phase("Loading Cluster Mapping")
    protein_to_cluster = parse_clstr_to_mapping(clstr_path)
    print(f"  ✅ {len(protein_to_cluster):,} proteins mapped")
    
    # Step 3: Load pairs
    logger.phase("Loading Pairs")
    pairs_df = pd.read_csv(config['pairs'], sep='\t', header=None,
                           names=['protein1', 'protein2', 'label'])
    print(f"  ✅ {len(pairs_df):,} pairs")
    
    # Step 4: Generate C3 splits
    logger.phase("Generating C3 Splits")
    splits, valid_df, stats = get_c3_splits(pairs_df, protein_to_cluster, n_splits)
    
    # Step 5: Define column sets
    logger.phase("Defining Feature Columns")
    all_cols = list(X_df.columns)
    # Split by ESM pattern - embedding columns contain 'ESM' in name
    embed_cols = [c for c in all_cols if 'ESM' in c]
    interp_cols = [c for c in all_cols if 'ESM' not in c]
    
    print(f"  Interpretable: {len(interp_cols)} features")
    print(f"  Embedding: {len(embed_cols)} features")
    
    # Define ablation variants (5 Variants - Hybrid/Proposed is handled by run_cv.py)
    variants = {
        'Interp-Only (Stack)': lambda: create_interp_only_stacking_pipeline(interp_cols, n_jobs, use_selector=True),
        'Embed-Only (Stack)': lambda: create_embed_only_stacking_pipeline(embed_cols, n_jobs, use_selector=True),
        'Interp-LR (Baseline)': lambda: create_interp_lr_pipeline(interp_cols, n_jobs),
        'Embed-LR (Baseline)': lambda: create_embed_lr_pipeline(embed_cols, n_jobs),
        'Global-LR (Baseline)': lambda: create_esm_global_lr_pipeline([c for c in embed_cols if 'Global' in c], n_jobs),
    }
    
    # Step 6: Run ablation
    all_results = []
    
    for variant_name, model_factory in variants.items():
        logger.header(f"VARIANT: {variant_name}")
        
        variant_metrics = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            import time
            fold_start = time.time()
            
            print(f"\n  📁 Fold {fold_idx + 1}/{n_splits}")
            
            # Verify no leakage
            verification = verify_split_integrity(
                train_indices, val_indices, pairs_df, protein_to_cluster,
                fold_id=fold_idx + 1, logger=logger
            )
            
            if not verification['is_valid']:
                raise RuntimeError(f"Data leakage in fold {fold_idx + 1}!")
            
            X_train = X_df.iloc[train_indices]
            X_val = X_df.iloc[val_indices]
            y_train = y_s.iloc[train_indices]
            y_val = y_s.iloc[val_indices]
            
            # Build and train
            model = model_factory()
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]
            
            metrics = display_full_metrics(y_val, y_pred, y_proba, title=f"{variant_name} Fold {fold_idx + 1}")
            variant_metrics.append(metrics)
            
            fold_time = time.time() - fold_start
            print(f"  ⏱️ Fold complete in {fold_time:.1f}s")
            
            # Cleanup
            del model, X_train, X_val
            import gc
            gc.collect()
        
        # Aggregate variant results
        metrics_df = pd.DataFrame(variant_metrics)
        mean_metrics = metrics_df.mean()
        std_metrics = metrics_df.std()
        
        result_row = {
            'Variant': variant_name,
            'Accuracy': f"{mean_metrics['Accuracy']*100:.2f} ± {std_metrics['Accuracy']*100:.2f}",
            'ROC-AUC': f"{mean_metrics['ROC-AUC']*100:.2f} ± {std_metrics['ROC-AUC']*100:.2f}",
            'PR-AUC': f"{mean_metrics['PR-AUC']*100:.2f} ± {std_metrics['PR-AUC']*100:.2f}",
            'MCC': f"{mean_metrics['MCC']*100:.2f} ± {std_metrics['MCC']*100:.2f}",
        }
        all_results.append(result_row)
        
        print(f"\n  📊 {variant_name} Summary:")
        print(f"     Accuracy: {mean_metrics['Accuracy']*100:.2f}% ± {std_metrics['Accuracy']*100:.2f}%")
        print(f"     ROC-AUC:  {mean_metrics['ROC-AUC']*100:.2f}% ± {std_metrics['ROC-AUC']*100:.2f}%")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    csv_path = output_dir / "ablation_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")
    
    # Print summary table
    logger.header("ABLATION STUDY SUMMARY")
    print(results_df.to_string(index=False))
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="HybridStack-PPI Ablation Study (Cache-Optimized)"
    )
    parser.add_argument("--dataset", choices=["human", "yeast", "both"], default="both")
    parser.add_argument("--n-splits", type=int, default=CV_N_SPLITS)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--clstr-path", default=None, help="Path to CD-HIT .clstr file (Auto-routed if not provided)")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("  HybridStack-PPI Ablation Study (Cache-Optimized)")
    print("  Mode: Auto-routing active for Cluster Verification")
    print("=" * 80)
    
    datasets_to_run = []
    if args.dataset in ["yeast", "both"]:
        datasets_to_run.append("yeast")
    if args.dataset in ["human", "both"]:
        datasets_to_run.append("human")
    
    for dataset_name in datasets_to_run:
        run_ablation_with_cache(
            dataset_name=dataset_name,
            n_splits=args.n_splits,
            n_jobs=args.n_jobs,
            clstr_path=args.clstr_path  # Pass the (potential) user override
        )


if __name__ == "__main__":
    main()
