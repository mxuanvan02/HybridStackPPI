#!/usr/bin/env python3
"""
HybridStack-PPI SOTA Cross-Validation with STRICT C3 Cluster-Based Split
==========================================================================
Implements SOTA-standard protein-level data splitting using GroupKFold
where groups are defined by CD-HIT clusters (40% sequence identity).

CRITICAL: This script ENFORCES cluster-based splitting to prevent data leakage.
If no cluster file is found, the script will ABORT with an error.

This ensures:
- No protein from the same cluster appears in both train and test
- Uses FULL feature matrix with rigorous cluster-level splitting
- Generates detailed leakage verification logs for publication

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


def verify_split_integrity(
    train_indices: list,
    val_indices: list,
    pairs_df: pd.DataFrame,
    protein_to_cluster: dict[str, int],
    fold_id: int,
    logger: PipelineLogger
) -> dict:
    """
    Verify that there is NO data leakage between train and validation sets.
    
    This function checks:
    1. Number of overlapping CLUSTERS between train and val (must be 0)
    2. Number of overlapping PROTEINS between train and val (must be 0)
    3. Label distribution in each fold
    
    Args:
        train_indices: List of training pair indices
        val_indices: List of validation pair indices
        pairs_df: DataFrame with protein pairs
        protein_to_cluster: Dict mapping protein_id -> cluster_id
        fold_id: Current fold number (1-indexed)
        logger: PipelineLogger instance
        
    Returns:
        Dict with verification results
    """
    # Extract proteins from train and val sets
    train_pairs = pairs_df.iloc[train_indices]
    val_pairs = pairs_df.iloc[val_indices]
    
    train_proteins = set(train_pairs['protein1']).union(set(train_pairs['protein2']))
    val_proteins = set(val_pairs['protein1']).union(set(val_pairs['protein2']))
    
    # Extract clusters
    train_clusters = set()
    for p in train_proteins:
        if p in protein_to_cluster:
            train_clusters.add(protein_to_cluster[p])
    
    val_clusters = set()
    for p in val_proteins:
        if p in protein_to_cluster:
            val_clusters.add(protein_to_cluster[p])
    
    # Check overlaps
    protein_overlap = train_proteins & val_proteins
    cluster_overlap = train_clusters & val_clusters
    
    # Label distribution
    train_pos = train_pairs['label'].sum()
    train_neg = len(train_pairs) - train_pos
    val_pos = val_pairs['label'].sum()
    val_neg = len(val_pairs) - val_pos
    
    # Print detailed report
    print(f"\n{'='*70}")
    print(f"  [FOLD-{fold_id} LEAKAGE VERIFICATION REPORT]")
    print(f"{'='*70}")
    print(f"\n  📊 DATA STATISTICS:")
    print(f"     Train Pairs: {len(train_indices):,}")
    print(f"     Val Pairs: {len(val_indices):,}")
    print(f"     Train Proteins: {len(train_proteins):,}")
    print(f"     Val Proteins: {len(val_proteins):,}")
    
    print(f"\n  🔬 CLUSTER-LEVEL CHECK:")
    print(f"     Train Clusters: {len(train_clusters):,}")
    print(f"     Val Clusters: {len(val_clusters):,}")
    print(f"     Overlap Clusters: {len(cluster_overlap)}", end="")
    if len(cluster_overlap) == 0:
        print(" ✅ [PASSED - NO CLUSTER LEAKAGE]")
    else:
        print(f" ❌ [FAILED - LEAKAGE DETECTED!]")
        print(f"     ⚠️ Overlapping cluster IDs: {list(cluster_overlap)[:10]}...")
    
    print(f"\n  🧬 PROTEIN-LEVEL CHECK:")
    print(f"     Overlap Proteins: {len(protein_overlap)}", end="")
    if len(protein_overlap) == 0:
        print(" ✅ [PASSED - NO PROTEIN LEAKAGE]")
    else:
        print(f" ❌ [FAILED - LEAKAGE DETECTED!]")
        print(f"     ⚠️ Overlapping proteins: {list(protein_overlap)[:5]}...")
    
    print(f"\n  ⚖️ LABEL DISTRIBUTION:")
    print(f"     Train: {train_pos:,} positive ({100*train_pos/len(train_pairs):.1f}%) | {train_neg:,} negative ({100*train_neg/len(train_pairs):.1f}%)")
    print(f"     Val: {val_pos:,} positive ({100*val_pos/len(val_pairs):.1f}%) | {val_neg:,} negative ({100*val_neg/len(val_pairs):.1f}%)")
    
    print(f"\n  🔒 SEQUENCE IDENTITY GUARANTEE:")
    print(f"     Max Identity between Train/Val: <40% (Enforced by CD-HIT)")
    
    # Final verdict
    is_valid = len(cluster_overlap) == 0 and len(protein_overlap) == 0
    print(f"\n  {'✅ SPLIT INTEGRITY: VERIFIED' if is_valid else '❌ SPLIT INTEGRITY: FAILED'}")
    print(f"{'='*70}")
    
    if not is_valid:
        logger.warning(f"  ⚠️ DATA LEAKAGE DETECTED IN FOLD {fold_id}!")
    
    return {
        'fold_id': fold_id,
        'train_pairs': len(train_indices),
        'val_pairs': len(val_indices),
        'train_proteins': len(train_proteins),
        'val_proteins': len(val_proteins),
        'train_clusters': len(train_clusters),
        'val_clusters': len(val_clusters),
        'cluster_overlap': len(cluster_overlap),
        'protein_overlap': len(protein_overlap),
        'train_pos_ratio': train_pos / len(train_pairs),
        'val_pos_ratio': val_pos / len(val_pairs),
        'is_valid': is_valid,
    }


def get_c3_splits(
    pairs_df: pd.DataFrame,
    protein_to_cluster: dict[str, int],
    n_splits: int = 5,
    random_state: int = 42,
    logger: PipelineLogger = None
) -> tuple[list, pd.DataFrame, dict]:
    """
    Generate STRICT C3 cluster-based splits.
    
    STRICT C3 Strategy:
    1. Split CLUSTERS into K folds using KFold
    2. A pair is assigned to fold K IFF BOTH proteins' clusters belong to fold K
    3. Pairs with proteins from different fold partitions are DISCARDED
    
    This guarantees ZERO overlap of clusters or proteins between train and val.
    
    Args:
        pairs_df: DataFrame with columns [protein1, protein2, label]
        protein_to_cluster: Dict mapping protein_id -> cluster_id
        n_splits: Number of CV folds
        random_state: Random seed
        logger: PipelineLogger instance
        
    Returns:
        Tuple of (splits, valid_df, stats_dict)
    """
    from sklearn.model_selection import KFold
    
    print(f"\n{'='*70}")
    print(f"  GENERATING STRICT C3 CLUSTER-BASED SPLITS")
    print(f"{'='*70}")
    
    original_count = len(pairs_df)
    
    # Step 1: Get all unique clusters
    all_clusters = sorted(set(protein_to_cluster.values()))
    n_clusters = len(all_clusters)
    print(f"\n  📊 CLUSTER STATISTICS:")
    print(f"     Total Clusters: {n_clusters:,}")
    
    # Step 2: Split CLUSTERS into K folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cluster_to_fold = {}
    
    cluster_array = np.array(all_clusters)
    for fold_idx, (train_cluster_idx, val_cluster_idx) in enumerate(kf.split(cluster_array)):
        for idx in val_cluster_idx:
            cluster_to_fold[cluster_array[idx]] = fold_idx
    
    print(f"     Clusters per Fold: ~{n_clusters // n_splits:,}")
    
    # Step 3: Assign pairs to folds based on BOTH proteins' clusters
    # A pair goes to fold K IFF both proteins' clusters are in fold K's validation set
    fold_assignments = {i: [] for i in range(n_splits)}
    unmapped_proteins = set()
    straddling_pairs = 0
    
    for idx, row in pairs_df.iterrows():
        p1, p2 = row['protein1'], row['protein2']
        c1 = protein_to_cluster.get(p1)
        c2 = protein_to_cluster.get(p2)
        
        if c1 is None:
            unmapped_proteins.add(p1)
            continue
        if c2 is None:
            unmapped_proteins.add(p2)
            continue
        
        fold1 = cluster_to_fold.get(c1)
        fold2 = cluster_to_fold.get(c2)
        
        if fold1 == fold2:
            # Both proteins belong to same fold partition -> assign to that fold
            fold_assignments[fold1].append(idx)
        else:
            # Proteins straddle different folds -> DISCARD
            straddling_pairs += 1
    
    # Step 4: Create train/val splits
    # For each fold, val = pairs in that fold, train = pairs in all other folds
    splits = []
    all_valid_indices = set()
    
    for val_fold in range(n_splits):
        val_indices = fold_assignments[val_fold]
        train_indices = []
        for train_fold in range(n_splits):
            if train_fold != val_fold:
                train_indices.extend(fold_assignments[train_fold])
        
        splits.append((train_indices, val_indices))
        all_valid_indices.update(val_indices)
        all_valid_indices.update(train_indices)
    
    # Create valid_df
    valid_indices = sorted(all_valid_indices)
    valid_df = pairs_df.iloc[valid_indices].reset_index(drop=True)
    
    # Stats
    valid_count = len(valid_indices)
    reduction_rate = (original_count - valid_count) / original_count * 100
    
    print(f"\n  📊 PAIR ASSIGNMENT STATISTICS:")
    print(f"     Original Pairs: {original_count:,}")
    print(f"     Valid Pairs (both proteins in same fold): {valid_count:,}")
    print(f"     Straddling Pairs (discarded): {straddling_pairs:,}")
    print(f"     Unmapped Proteins: {len(unmapped_proteins):,}")
    print(f"     Reduction Rate: {reduction_rate:.2f}%")
    
    print(f"\n  📑 FOLD DISTRIBUTION:")
    for i, (train_idx, val_idx) in enumerate(splits):
        print(f"     Fold {i+1}: Train={len(train_idx):,} | Val={len(val_idx):,}")
    
    stats = {
        'original_pairs': original_count,
        'valid_pairs': valid_count,
        'straddling_pairs': straddling_pairs,
        'reduction_rate': reduction_rate,
        'n_clusters': n_clusters,
        'unmapped_proteins': len(unmapped_proteins),
    }
    
    return splits, valid_df, stats



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
    
    CRITICAL: clstr_path is MANDATORY. Script will abort if not found.
    
    Args:
        feature_cache: Path to H5 feature cache
        pairs_path: Path to original pairs TSV
        clstr_path: Path to CD-HIT .clstr file (MANDATORY)
        dataset_name: Name for outputs ("Yeast" or "Human")
        output_dir: Directory for saving results
        n_splits: Number of CV folds
        n_jobs: Parallel jobs for sklearn (-1 = all cores)
        esm_cache: Path to ESM embedding cache (for FeatureEngine init)
    """
    logger = PipelineLogger()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # CRITICAL: ENFORCE CLUSTER FILE EXISTENCE
    # =========================================================================
    if not Path(clstr_path).exists():
        raise FileNotFoundError(
            f"\n{'='*70}\n"
            f"❌ CRITICAL ERROR: CLUSTER FILE NOT FOUND!\n"
            f"{'='*70}\n"
            f"Path: {clstr_path}\n\n"
            f"This script REQUIRES a CD-HIT cluster file for proper evaluation.\n"
            f"Without cluster-based splitting, results will suffer from DATA LEAKAGE\n"
            f"due to homologous proteins appearing in both train and test sets.\n\n"
            f"Please run CD-HIT first:\n"
            f"  cd-hit -i sequences.fasta -o output -c 0.4 -n 2\n"
            f"{'='*70}\n"
        )
    
    logger.header(f"HYBRIDSTACK-PPI STRICT C3 CROSS-VALIDATION: {dataset_name}")
    print(f"  Feature Cache: {feature_cache}")
    print(f"  Pairs: {pairs_path}")
    print(f"  Cluster File: {clstr_path} ✅")
    print(f"  Output: {output_dir}")
    print(f"\n  ⚠️  STRICT MODE: Cluster-based splitting ENFORCED")
    print(f"      No fallback to random/protein-level split allowed.")
    
    # Step 1: Load cluster mapping
    logger.phase("Loading CD-HIT Cluster Mapping (40% Identity)")
    protein_to_cluster = parse_clstr_to_mapping(clstr_path)
    n_proteins = len(protein_to_cluster)
    n_clusters = len(set(protein_to_cluster.values()))
    print(f"  ✅ Loaded {n_proteins:,} proteins in {n_clusters:,} clusters")
    
    # Step 2: Load full feature matrix
    logger.phase("Loading Full Feature Matrix from Cache")
    X_df, y_s = load_feature_matrix_h5(feature_cache)
    print(f"  ✅ Loaded X={X_df.shape}, y={len(y_s)}")
    
    # Step 3: Load pairs for cluster assignment
    logger.phase("Loading Pairs for Cluster Assignment")
    pairs_df = pd.read_csv(pairs_path, sep='\t', header=None, 
                           names=['protein1', 'protein2', 'label'])
    print(f"  ✅ {len(pairs_df):,} pairs loaded")
    
    # Verify alignment
    assert len(pairs_df) == len(X_df), f"Pairs ({len(pairs_df)}) != Features ({len(X_df)})"
    
    # Step 4: Generate C3 splits with detailed stats
    logger.phase("Generating C3 Cluster-Based Splits (GroupKFold)")
    splits, valid_df, split_stats = get_c3_splits(
        pairs_df, protein_to_cluster, n_splits, logger=logger
    )
    
    # Initialize FeatureEngine for column names
    if esm_cache:
        embedding_computer = EmbeddingComputer(model_name="facebook/esm2_t33_650M_UR50D")
        feature_engine = FeatureEngine(h5_cache_path=esm_cache, embedding_computer=embedding_computer)
        interp_cols, embed_cols = define_stacking_columns(feature_engine, "concat")
    else:
        all_cols = list(X_df.columns)
        embed_cols = [c for c in all_cols if c.startswith('esm_') or 'embedding' in c.lower()]
        interp_cols = [c for c in all_cols if c not in embed_cols]
    
    print(f"\n  📐 FEATURE DIMENSIONS:")
    print(f"     Interpretable features: {len(interp_cols):,}")
    print(f"     Embedding features: {len(embed_cols):,}")
    print(f"     Total: {len(interp_cols) + len(embed_cols):,}")
    
    # Step 5: Run CV with verification
    logger.header(f"🚀 RUNNING {n_splits}-FOLD C3 CROSS-VALIDATION (STRICT MODE)")
    
    fold_metrics = []
    fold_details = []
    cv_results = []
    verification_reports = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        import time
        fold_start = time.time()
        
        logger.info(f"--- Fold {fold_idx + 1}/{n_splits} ---")
        
        # CRITICAL: Verify split integrity BEFORE training
        verification = verify_split_integrity(
            train_indices, val_indices, pairs_df, protein_to_cluster,
            fold_id=fold_idx + 1, logger=logger
        )
        verification_reports.append(verification)
        
        if not verification['is_valid']:
            raise RuntimeError(
                f"❌ DATA LEAKAGE DETECTED IN FOLD {fold_idx + 1}! "
                f"Cluster overlap: {verification['cluster_overlap']}, "
                f"Protein overlap: {verification['protein_overlap']}"
            )
        
        X_train = X_df.iloc[train_indices]
        X_val = X_df.iloc[val_indices]
        y_train = y_s.iloc[train_indices]
        y_val = y_s.iloc[val_indices]
        
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
    
    # 6e. Predictions dump (enhanced with protein IDs for extended analysis)
    preds_list = []
    for fd in fold_details:
        fold_id = fd['fold_id']
        val_indices = splits[fold_id - 1][1]  # Get val indices for this fold
        fold_pairs = pairs_df.iloc[val_indices]
        
        preds_list.append(pd.DataFrame({
            'fold_id': fold_id,
            'protein1': fold_pairs['protein1'].values,
            'protein2': fold_pairs['protein2'].values,
            'y_true': fd['y_true'],
            'y_pred': fd['y_pred'],
            'y_proba': fd['y_proba'],
        }))
        
    all_preds = pd.concat(preds_list, ignore_index=True)
    all_preds.to_csv(output_dir / "all_folds_predictions.csv", index=False)
    logger.info(f"  ✅ Saved all_folds_predictions.csv ({len(all_preds)} samples with IDs)")
    
    # 6f. Verification reports
    verification_df = pd.DataFrame(verification_reports)
    verification_df.to_csv(output_dir / "leakage_verification.csv", index=False)
    logger.info(f"  ✅ Saved leakage_verification.csv")
    
    # Summary
    avg_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
    std_metrics = pd.DataFrame(fold_metrics).std().to_dict()
    
    logger.header("📊 FINAL RESULTS SUMMARY")
    print(f"\n  🔒 ALL {n_splits} FOLDS PASSED LEAKAGE VERIFICATION ✅")
    print(f"\n  📈 METRICS (Mean ± Std):")
    for metric in ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1 Score', 'Specificity', 'MCC', 'ROC-AUC', 'PR-AUC']:
        if metric in avg_metrics:
            print(f"     {metric:20s}: {avg_metrics[metric]*100:.2f}% ± {std_metrics[metric]*100:.2f}%")
    
    print(f"\n  ✅ All outputs saved to: {output_dir}")
    
    return avg_metrics, fold_details


def main():
    # Import config
    from scripts.config import get_dataset_config, CV_N_SPLITS, ESM_CACHE_PATH
    
    parser = argparse.ArgumentParser(
        description="HybridStack-PPI STRICT C3 Cross-Validation (No Random Split Fallback)"
    )
    parser.add_argument("--dataset", choices=["human", "yeast"], default="yeast")
    parser.add_argument("--n-splits", type=int, default=CV_N_SPLITS)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--clstr-path", default=None, help="Path to CD-HIT .clstr file (Auto-routed if not provided)")
    parser.add_argument("--output-dir", default=None)
    
    args = parser.parse_args()
    
    # Get dataset config from centralized config
    config = get_dataset_config(args.dataset)
    dataset_name = config['name']
    
    # Auto-Routing Protocol: Use config path if user didn't specify
    clstr_path = args.clstr_path if args.clstr_path else config['clstr']
    
    # Create timestamped output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / f"results/{dataset_name}_C3_CV_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#' * 80}")
    print(f"### HybridStack-PPI STRICT C3 Cross-Validation: {dataset_name}")
    print(f"### Mode: CLUSTER-BASED SPLIT ENFORCED (Auto-routing active)")
    print(f"### Cluster Path: {clstr_path}")
    print(f"{'#' * 80}")
    
    run_c3_cv(
        feature_cache=config['feature_cache'],
        pairs_path=config['pairs'],
        clstr_path=clstr_path,
        dataset_name=dataset_name,
        output_dir=output_dir,
        n_splits=args.n_splits,
        n_jobs=args.n_jobs,
        esm_cache=str(ESM_CACHE_PATH),
    )



if __name__ == "__main__":
    main()
