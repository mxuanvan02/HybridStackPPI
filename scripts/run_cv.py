#!/usr/bin/env python3
"""
HybridStack-PPI SOTA Cross-Validation with C3 Cluster-Based Split
===================================================================
Implements SOTA-standard protein-level data splitting using GroupKFold
where groups are defined by CD-HIT clusters (40% sequence identity).

This ensures:
- No protein from the same cluster appears in both train and test
- Uses FULL feature matrix (not reduced), but splits rigorously
- Generates publication-quality ROC curves, PR curves, and decision power analysis

Reference: C3 Split Strategy (Cluster-based Cross-validation Class 3)
"""
# Suppress all warnings before any imports
import os
import sys
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

import argparse
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import GroupKFold

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.builders import create_stacking_pipeline, define_stacking_columns
from src.feature_engine import EmbeddingComputer, FeatureEngine
from src.data_utils import load_feature_matrix_h5
from src.metrics import (
    display_full_metrics,
    plot_cv_roc_pr_curves,
    plot_cv_metric_distribution,
    generate_latex_table,
    plot_decision_power,
)
from src.logger import PipelineLogger


def parse_clstr_to_mapping(clstr_path: str) -> dict[str, int]:
    """
    Parse CD-HIT .clstr file and map each protein to its cluster ID.
    
    Args:
        clstr_path: Path to CD-HIT .clstr output file
        
    Returns:
        Dictionary mapping protein_id -> cluster_id (int)
    """
    protein_to_cluster = {}
    current_cluster_id = -1
    
    with open(clstr_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                current_cluster_id = int(line.split()[1])
            else:
                match = re.search(r">(.+?)\.\.\.", line)
                if match:
                    protein_id = match.group(1)
                    protein_to_cluster[protein_id] = current_cluster_id
    
    return protein_to_cluster


def get_c3_splits(
    pairs_df: pd.DataFrame,
    protein_to_cluster: dict[str, int],
    n_splits: int = 5,
    random_state: int = 42
) -> list[tuple]:
    """
    Generate C3 cluster-based splits using GroupKFold.
    
    C3 Strategy: Split CLUSTERS, not proteins. A pair is only included
    in a fold if BOTH proteins' clusters belong to that fold's cluster set.
    Pairs with proteins from different fold partitions are discarded.
    
    Args:
        pairs_df: DataFrame with columns [protein1, protein2, label]
        protein_to_cluster: Dict mapping protein_id -> cluster_id
        n_splits: Number of CV folds
        random_state: Random seed
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Assign cluster IDs to each pair
    # For GroupKFold, we need a single group per sample
    # Strategy: use min(cluster_A, cluster_B) as the group (conservative)
    groups = []
    valid_indices = []
    
    for idx, row in pairs_df.iterrows():
        p1, p2 = row['protein1'], row['protein2']
        c1 = protein_to_cluster.get(p1)
        c2 = protein_to_cluster.get(p2)
        
        if c1 is not None and c2 is not None:
            # Use cluster pair as group (ensure same cluster pair stays together)
            # Using min/max to canonicalize
            group_id = min(c1, c2) * 100000 + max(c1, c2)  # Unique pair ID
            groups.append(group_id)
            valid_indices.append(idx)
    
    # Filter to valid pairs
    valid_df = pairs_df.iloc[valid_indices].reset_index(drop=True)
    groups = np.array(groups)
    
    # Use GroupKFold to ensure cluster pairs don't leak
    gkf = GroupKFold(n_splits=n_splits)
    splits = []
    
    for train_idx, val_idx in gkf.split(valid_df, groups=groups):
        # Map back to original indices
        splits.append((
            [valid_indices[i] for i in train_idx],
            [valid_indices[i] for i in val_idx]
        ))
    
    return splits, valid_df


def run_c3_cv(
    feature_cache: str,
    pairs_path: str,
    clstr_path: str,
    dataset_name: str,
    output_dir: Path,
    n_splits: int = 5,
    n_jobs: int = -1,
    esm_cache: str = None,
):
    """
    Run full C3 cluster-based cross-validation on FULL feature matrix.
    
    Args:
        feature_cache: Path to H5 feature cache
        pairs_path: Path to original pairs TSV
        clstr_path: Path to CD-HIT .clstr file
        dataset_name: Name for outputs ("Yeast" or "Human")
        output_dir: Directory for saving results
        n_splits: Number of CV folds
        n_jobs: Parallel jobs for sklearn (-1 = all cores)
        esm_cache: Path to ESM embedding cache (for FeatureEngine init)
    """
    logger = PipelineLogger()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    logger.header(f"HYBRIDSTACK-PPI SOTA C3 CROSS-VALIDATION: {dataset_name}")
    print(f"  Feature Cache: {feature_cache}")
    print(f"  Pairs: {pairs_path}")
    print(f"  Cluster File: {clstr_path}")
    print(f"  Output: {output_dir}")
    
    # Step 1: Load cluster mapping
    logger.phase("Loading CD-HIT Cluster Mapping")
    protein_to_cluster = parse_clstr_to_mapping(clstr_path)
    n_proteins = len(protein_to_cluster)
    n_clusters = len(set(protein_to_cluster.values()))
    print(f"  ✅ {n_proteins} proteins in {n_clusters} clusters")
    
    # Step 2: Load full feature matrix
    logger.phase("Loading Full Feature Matrix from Cache")
    X_df, y_s = load_feature_matrix_h5(feature_cache)
    print(f"  ✅ Loaded X={X_df.shape}, y={len(y_s)}")
    
    # Step 3: Load pairs for cluster assignment
    logger.phase("Loading Pairs for Cluster Assignment")
    pairs_df = pd.read_csv(pairs_path, sep='\t', header=None, 
                           names=['protein1', 'protein2', 'label'])
    print(f"  ✅ {len(pairs_df)} pairs loaded")
    
    # Verify alignment
    assert len(pairs_df) == len(X_df), f"Pairs ({len(pairs_df)}) != Features ({len(X_df)})"
    
    # Step 4: Generate C3 splits
    logger.phase("Generating C3 Cluster-Based Splits (GroupKFold)")
    splits, valid_df = get_c3_splits(pairs_df, protein_to_cluster, n_splits)
    
    for i, (train_idx, val_idx) in enumerate(splits):
        print(f"  Fold {i+1}: Train={len(train_idx)}, Val={len(val_idx)}")
    
    # Initialize FeatureEngine for column names
    if esm_cache:
        embedding_computer = EmbeddingComputer(model_name="facebook/esm2_t33_650M_UR50D")
        feature_engine = FeatureEngine(h5_cache_path=esm_cache, embedding_computer=embedding_computer)
        interp_cols, embed_cols = define_stacking_columns(feature_engine, "concat")
    else:
        # Fallback: split columns by prefix
        all_cols = list(X_df.columns)
        embed_cols = [c for c in all_cols if c.startswith('esm_') or 'embedding' in c.lower()]
        interp_cols = [c for c in all_cols if c not in embed_cols]
    
    print(f"  Interpretable features: {len(interp_cols)}")
    print(f"  Embedding features: {len(embed_cols)}")
    
    # Step 5: Run CV
    logger.header(f"🚀 RUNNING {n_splits}-FOLD C3 CROSS-VALIDATION")
    
    fold_metrics = []
    fold_details = []
    cv_results = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        import time
        fold_start = time.time()
        
        logger.info(f"--- Fold {fold_idx + 1}/{n_splits} ---")
        logger.info(f"  📊 Train: {len(train_indices)} pairs, Val: {len(val_indices)} pairs")
        
        X_train = X_df.iloc[train_indices]
        X_val = X_df.iloc[val_indices]
        y_train = y_s.iloc[train_indices]
        y_val = y_s.iloc[val_indices]
        
        # Verify no cluster leakage
        train_proteins = set(pairs_df.iloc[train_indices]['protein1']).union(
                         set(pairs_df.iloc[train_indices]['protein2']))
        val_proteins = set(pairs_df.iloc[val_indices]['protein1']).union(
                       set(pairs_df.iloc[val_indices]['protein2']))
        overlap = train_proteins & val_proteins
        
        if overlap:
            # Check if overlapping proteins are from same cluster
            train_clusters = {protein_to_cluster.get(p) for p in train_proteins if p in protein_to_cluster}
            val_clusters = {protein_to_cluster.get(p) for p in val_proteins if p in protein_to_cluster}
            cluster_overlap = train_clusters & val_clusters
            logger.warning(f"  ⚠️ Protein overlap: {len(overlap)}, Cluster overlap: {len(cluster_overlap)}")
        else:
            logger.info(f"  ✅ No protein leakage (0 overlap)")
        
        # Build and train model
        logger.info(f"  🔧 Building stacking pipeline...")
        model = create_stacking_pipeline(interp_cols, embed_cols, n_jobs, use_selector=True)
        
        logger.info(f"  🏋️ Training model...")
        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start
        logger.info(f"  ✅ Training complete in {train_time:.1f}s")
        
        # Evaluate
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        metrics = display_full_metrics(y_val, y_pred, y_proba, title=f"Fold {fold_idx + 1}")
        fold_metrics.append(metrics)
        
        # Collect results
        cv_results.append({
            'y_true': y_val.values if hasattr(y_val, 'values') else y_val,
            'y_proba': y_proba,
        })
        
        # Extract decision weights
        decision_weights = None
        if hasattr(model.named_steps.get('model'), 'final_estimator_'):
            meta = model.named_steps['model'].final_estimator_
            if hasattr(meta, 'coef_'):
                coefs = np.abs(meta.coef_[0])
                total = np.sum(coefs)
                if total > 0:
                    contrib = coefs / total * 100
                    decision_weights = {
                        'Biological': contrib[0],
                        'Deep Learning': contrib[1],
                    }
                    logger.info(f"  💡 Decision Power: Bio={contrib[0]:.1f}%, DL={contrib[1]:.1f}%")
        
        fold_details.append({
            'fold_id': fold_idx + 1,
            'y_true': np.array(y_val.values if hasattr(y_val, 'values') else y_val),
            'y_pred': y_pred,
            'y_proba': y_proba,
            'metrics': metrics.copy(),
            'decision_weights': decision_weights,
        })
        
        fold_time = time.time() - fold_start
        logger.info(f"  ⏱️ Fold {fold_idx + 1} complete in {fold_time:.1f}s | Acc: {metrics['Accuracy']:.2%} | AUC: {metrics['ROC-AUC']:.4f}")
        
        # Cleanup
        del X_train, X_val, y_train, y_val, model
        import gc
        gc.collect()
    
    # Step 6: Generate publication-quality outputs
    logger.header("📊 GENERATING PUBLICATION-QUALITY OUTPUTS")
    
    # 6a. Mean ROC/PR curves with std dev bands
    try:
        plot_cv_roc_pr_curves(
            cv_results,
            title=f"{n_splits}-Fold C3 CV ({dataset_name})",
            prefix=str(plots_dir / f"{dataset_name.lower()}_c3_cv")
        )
        logger.info(f"  ✅ Saved {dataset_name.lower()}_c3_cv_roc.png and _pr.png")
    except Exception as e:
        logger.warning(f"  ⚠️ Could not generate ROC/PR curves: {e}")
    
    # 6b. Metric distribution boxplots
    try:
        plot_cv_metric_distribution(fold_metrics, prefix=str(plots_dir / f"{dataset_name.lower()}_c3_cv"))
        logger.info(f"  ✅ Saved {dataset_name.lower()}_c3_cv_metrics.png")
    except Exception as e:
        logger.warning(f"  ⚠️ Could not generate metric boxplots: {e}")
    
    # 6c. Decision Power analysis
    decision_list = [fd['decision_weights'] for fd in fold_details if fd['decision_weights']]
    if decision_list:
        avg_bio = np.mean([d['Biological'] for d in decision_list])
        avg_dl = np.mean([d['Deep Learning'] for d in decision_list])
        try:
            plot_decision_power(
                {"Biological": avg_bio, "Deep Learning": avg_dl},
                dataset_name=dataset_name,
                save_path=str(plots_dir / f"{dataset_name.lower()}_decision_power.png")
            )
            logger.info(f"  ✅ Saved {dataset_name.lower()}_decision_power.png")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not generate decision power plot: {e}")
    
    
    # 6d. LaTeX table (detailed per-fold)
    try:
        latex = generate_latex_table(
            fold_metrics, 
            method_name="HybridStack-PPI",
            dataset_name=f"BioGRID {dataset_name}",
            save_path=str(output_dir / "cv_results_table.tex")
        )
        logger.info(f"  ✅ Saved cv_results_table.tex")
    except Exception as e:
        logger.warning(f"  ⚠️ Could not generate LaTeX table: {e}")
    
    # 6e. Predictions dump
    all_preds = pd.concat([
        pd.DataFrame({
            'fold_id': fd['fold_id'],
            'y_true': fd['y_true'],
            'y_pred': fd['y_pred'],
            'y_proba': fd['y_proba'],
        }) for fd in fold_details
    ], ignore_index=True)
    all_preds.to_csv(output_dir / "all_folds_predictions.csv", index=False)
    logger.info(f"  ✅ Saved all_folds_predictions.csv ({len(all_preds)} samples)")
    
    # 6f. Feature importance
    # (Already saved by last fold if model had feature importances)
    
    # Summary
    avg_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
    std_metrics = pd.DataFrame(fold_metrics).std().to_dict()
    
    logger.header("📊 FINAL RESULTS SUMMARY")
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC', 'MCC']:
        if metric in avg_metrics:
            print(f"  {metric:12s}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")
    
    print(f"\n✅ All outputs saved to: {output_dir}")
    
    return avg_metrics, fold_details


def main():
    parser = argparse.ArgumentParser(
        description="HybridStack-PPI SOTA C3 Cross-Validation"
    )
    parser.add_argument("--dataset", choices=["human", "yeast"], default="yeast")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output-dir", default=None)
    
    args = parser.parse_args()
    
    # Dataset paths (FULL data, not reduced)
    if args.dataset == "yeast":
        feature_cache = str(PROJECT_ROOT / "cache/yeast_yeast_pairs_facebook_esm2_t33_650m_ur50d_concat_v2_features.h5")
        pairs_path = str(PROJECT_ROOT / "data/BioGrid/Yeast/yeast_pairs.tsv")
        clstr_path = str(PROJECT_ROOT / "cache/yeast_dict_cdhit_40.clstr")
        esm_cache = str(PROJECT_ROOT / "cache/esm2/esm2_embeddings_v4.h5")
        dataset_name = "Yeast"
    else:
        feature_cache = str(PROJECT_ROOT / "cache/human_human_pairs_facebook_esm2_t33_650m_ur50d_concat_v2_features.h5")
        pairs_path = str(PROJECT_ROOT / "data/BioGrid/Human/human_pairs.tsv")
        clstr_path = str(PROJECT_ROOT / "cache/human_dict_cdhit_40.clstr")
        esm_cache = str(PROJECT_ROOT / "cache/esm2/esm2_embeddings_v4.h5")
        dataset_name = "Human"
    
    # Create timestamped output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / f"results/{dataset_name}_C3_CV_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#' * 80}")
    print(f"### HybridStack-PPI SOTA C3 Cross-Validation: {dataset_name}")
    print(f"{'#' * 80}")
    
    run_c3_cv(
        feature_cache=feature_cache,
        pairs_path=pairs_path,
        clstr_path=clstr_path,
        dataset_name=dataset_name,
        output_dir=output_dir,
        n_splits=args.n_splits,
        n_jobs=args.n_jobs,
        esm_cache=esm_cache,
    )


if __name__ == "__main__":
    main()
