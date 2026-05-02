#!/usr/bin/env python3
"""
Case Study Mining with Branch-Level Analysis
=============================================
Script này chạy CV đúng cách (C3 split) và lưu xác suất từ TỪNG BRANCH:
- interp_branch_proba: Xác suất từ nhánh Interpretable (Bio features + Motifs)
- embed_branch_proba: Xác suất từ nhánh Embedding (ESM-2 features)
- hybrid_proba: Xác suất cuối cùng từ meta-learner

Sau đó tìm các Case Study theo tiêu chí:
- embed_branch_proba < threshold (Deep sai hoặc không tự tin)
- hybrid_proba > threshold (Hybrid đúng và tự tin)
- Label = 1

Author: HybridStackPPI Team
"""

import os
import sys
import warnings
import time
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import re
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hybridstack.builders import create_stacking_pipeline, define_stacking_columns
from hybridstack.data_utils import load_feature_matrix_h5
from hybridstack.feature_engine import EmbeddingComputer, FeatureEngine


def parse_clstr_to_mapping(clstr_path: str) -> dict[str, int]:
    """Parse CD-HIT .clstr file."""
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
                    protein_to_cluster[match.group(1)] = current_cluster_id
    return protein_to_cluster


def get_c3_splits(pairs_df, protein_to_cluster, n_splits=5, random_state=42):
    """Generate STRICT C3 cluster-based splits."""
    all_clusters = sorted(set(protein_to_cluster.values()))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cluster_to_fold = {}
    
    cluster_array = np.array(all_clusters)
    for fold_idx, (_, val_cluster_idx) in enumerate(kf.split(cluster_array)):
        for idx in val_cluster_idx:
            cluster_to_fold[cluster_array[idx]] = fold_idx
    
    fold_assignments = {i: [] for i in range(n_splits)}
    
    for idx, row in pairs_df.iterrows():
        p1, p2 = row['protein1'], row['protein2']
        c1 = protein_to_cluster.get(p1)
        c2 = protein_to_cluster.get(p2)
        
        if c1 is None or c2 is None:
            continue
        
        fold1 = cluster_to_fold.get(c1)
        fold2 = cluster_to_fold.get(c2)
        
        if fold1 == fold2:
            fold_assignments[fold1].append(idx)
    
    splits = []
    for val_fold in range(n_splits):
        val_indices = fold_assignments[val_fold]
        train_indices = []
        for train_fold in range(n_splits):
            if train_fold != val_fold:
                train_indices.extend(fold_assignments[train_fold])
        splits.append((train_indices, val_indices))
    
    return splits


def extract_branch_probabilities(model, X_val):
    """
    Extract probabilities from each sub-estimator in StackingClassifier.
    
    Returns:
        Tuple of (interp_proba, embed_proba, hybrid_proba)
    """
    stacking = model.named_steps.get('ensemble') or model.named_steps.get('model')
    
    if stacking is None:
        raise ValueError("Cannot find StackingClassifier in pipeline")
    
    # Get base estimators
    estimators = stacking.estimators_
    
    if len(estimators) < 2:
        raise ValueError(f"Expected 2 estimators, got {len(estimators)}")
    
    interp_estimator = estimators[0]  # interp_branch
    embed_estimator = estimators[1]   # embed_branch
    
    # Get predictions from each branch
    interp_proba = interp_estimator.predict_proba(X_val)[:, 1]
    embed_proba = embed_estimator.predict_proba(X_val)[:, 1]
    
    # Get final hybrid prediction
    hybrid_proba = model.predict_proba(X_val)[:, 1]
    
    return interp_proba, embed_proba, hybrid_proba


def run_case_study_mining(
    dataset: str = "yeast",
    n_splits: int = 5,
    n_jobs: int = -1,
    embed_threshold: float = 0.5,
    hybrid_threshold: float = 0.5,
    n_samples: int = 10,
):
    """Run CV with branch-level probability extraction for case study mining."""
    
    from scripts.config import get_dataset_config, ESM_CACHE_PATH
    
    config = get_dataset_config(dataset)
    dataset_name = config['name']
    
    print(f"\n{'='*70}")
    print(f"CASE STUDY MINING WITH C3 SPLIT - {dataset_name}")
    print(f"{'='*70}")
    
    # Load data
    print("\n[1/5] Loading data...")
    X_df, y_s = load_feature_matrix_h5(config['feature_cache'])
    pairs_df = pd.read_csv(config['pairs'], sep='\t', header=None,
                           names=['protein1', 'protein2', 'label'])
    
    # Align if needed
    if len(pairs_df) != len(X_df):
        min_len = min(len(pairs_df), len(X_df))
        pairs_df = pairs_df.iloc[:min_len]
        X_df = X_df.iloc[:min_len]
        y_s = y_s.iloc[:min_len]
    
    print(f"   Pairs: {len(pairs_df):,}, Features: {X_df.shape}")
    
    # Load cluster mapping
    print("\n[2/5] Loading cluster mapping...")
    protein_to_cluster = parse_clstr_to_mapping(config['clstr'])
    print(f"   Proteins: {len(protein_to_cluster):,}")
    
    # Generate C3 splits
    print("\n[3/5] Generating C3 splits...")
    splits = get_c3_splits(pairs_df, protein_to_cluster, n_splits)
    
    # Get column definitions
    embedding_computer = EmbeddingComputer(model_name="facebook/esm2_t33_650M_UR50D")
    feature_engine = FeatureEngine(h5_cache_path=str(ESM_CACHE_PATH), 
                                   embedding_computer=embedding_computer)
    interp_cols, embed_cols = define_stacking_columns(feature_engine, "concat")
    
    print(f"   Interp cols: {len(interp_cols):,}, Embed cols: {len(embed_cols):,}")
    
    # Run CV with branch extraction
    print("\n[4/5] Running CV with branch-level extraction...")
    
    all_results = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        print(f"\n   Fold {fold_idx + 1}/{n_splits}...")
        
        start_time = time.time()
        
        X_train = X_df.iloc[train_indices]
        X_val = X_df.iloc[val_indices]
        y_train = y_s.iloc[train_indices]
        y_val = y_s.iloc[val_indices]
        
        # Build and train model
        model = create_stacking_pipeline(interp_cols, embed_cols, n_jobs, use_selector=True)
        model.fit(X_train, y_train)
        
        # Extract branch probabilities
        interp_proba, embed_proba, hybrid_proba = extract_branch_probabilities(model, X_val)
        
        # Get validation pairs info
        val_pairs = pairs_df.iloc[val_indices].reset_index(drop=True)
        
        # Collect results
        fold_results = pd.DataFrame({
            'fold_id': fold_idx + 1,
            'protein1': val_pairs['protein1'].values,
            'protein2': val_pairs['protein2'].values,
            'y_true': y_val.values if hasattr(y_val, 'values') else y_val,
            'interp_proba': interp_proba,
            'embed_proba': embed_proba,
            'hybrid_proba': hybrid_proba,
        })
        all_results.append(fold_results)
        
        # Quick accuracy check
        acc = np.mean((hybrid_proba > 0.5) == y_val.values)
        elapsed = time.time() - start_time
        print(f"   Fold {fold_idx + 1}: Acc={acc:.4f}, Time={elapsed:.1f}s")
        
        # Cleanup
        del model, X_train, X_val
        import gc
        gc.collect()
    
    # Combine results
    all_df = pd.concat(all_results, ignore_index=True)
    
    # Save detailed predictions
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{dataset_name}_branch_predictions.csv"
    all_df.to_csv(output_path, index=False)
    print(f"\n[5/5] Saved branch-level predictions: {output_path}")
    
    # === CASE STUDY MINING ===
    print(f"\n{'='*70}")
    print("CASE STUDY ANALYSIS")
    print(f"Criteria: Embed < {embed_threshold}, Hybrid > {hybrid_threshold}, Label = 1")
    print(f"{'='*70}")
    
    # Find cases where embed branch was wrong/uncertain but hybrid was correct
    case_studies = all_df[
        (all_df['y_true'] == 1) &
        (all_df['embed_proba'] < embed_threshold) &
        (all_df['hybrid_proba'] > hybrid_threshold)
    ].copy()
    
    case_studies['interp_boost'] = case_studies['hybrid_proba'] - case_studies['embed_proba']
    case_studies = case_studies.sort_values('interp_boost', ascending=False)
    
    print(f"\nFound {len(case_studies)} case study candidates")
    
    if len(case_studies) > 0:
        print(f"\n{'='*70}")
        print("TOP CASE STUDIES")
        print(f"{'='*70}")
        
        for i, row in case_studies.head(n_samples).iterrows():
            print(f"\n--- #{case_studies.index.get_loc(i) + 1} ---")
            print(f"  Proteins: {row['protein1']} <-> {row['protein2']}")
            print(f"  Interp Branch Prob: {row['interp_proba']:.4f}")
            print(f"  Embed Branch Prob: {row['embed_proba']:.4f} (< {embed_threshold})")
            print(f"  Hybrid Prob: {row['hybrid_proba']:.4f} (> {hybrid_threshold})")
            print(f"  Boost from Interp: +{row['interp_boost']:.4f}")
        
    else:
        print("\nNo candidates found. Try adjusting thresholds.")
        print("This could indicate that ESM-2 embeddings are very strong on this dataset.")
    
    # Statistical summary
    print(f"\n{'='*70}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*70}")
    
    positive_samples = all_df[all_df['y_true'] == 1]
    print(f"Mean Interp Prob (positives): {positive_samples['interp_proba'].mean():.4f}")
    print(f"Mean Embed Prob (positives): {positive_samples['embed_proba'].mean():.4f}")
    print(f"Mean Hybrid Prob (positives): {positive_samples['hybrid_proba'].mean():.4f}")
    
    return all_df, case_studies


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["yeast", "human"], default="yeast")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--embed-threshold", type=float, default=0.5)
    parser.add_argument("--hybrid-threshold", type=float, default=0.5)
    parser.add_argument("--n-samples", type=int, default=10)
    
    args = parser.parse_args()
    
    run_case_study_mining(
        dataset=args.dataset,
        n_splits=args.n_splits,
        n_jobs=args.n_jobs,
        embed_threshold=args.embed_threshold,
        hybrid_threshold=args.hybrid_threshold,
        n_samples=args.n_samples,
    )


if __name__ == "__main__":
    main()
