#!/usr/bin/env python3
"""
ELM Motif Coverage Analysis
============================
Tính tỷ lệ protein có motif được detect từ ELM database.

Formula: Coverage = |{proteins with I ≠ ∅}| / |Total proteins|

Where I = set of detected ELM motifs for a protein.

Author: HybridStackPPI Team
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_feature_matrix_h5


def calculate_motif_coverage(dataset: str = "yeast"):
    """
    Calculate the percentage of proteins with non-empty ELM motif sets.
    
    Args:
        dataset: "yeast" or "human"
    """
    from scripts.config import get_dataset_config
    
    config = get_dataset_config(dataset)
    dataset_name = config['name']
    
    print(f"\n{'='*70}")
    print(f"ELM MOTIF COVERAGE ANALYSIS - {dataset_name}")
    print(f"{'='*70}")
    
    # Load feature matrix
    print("\n[1/3] Loading feature matrix...")
    X_df, y_s = load_feature_matrix_h5(config['feature_cache'])
    print(f"   Features: {X_df.shape}")
    
    # Load pairs to get protein IDs
    print("\n[2/3] Loading protein pairs...")
    pairs_df = pd.read_csv(config['pairs'], sep='\t', header=None,
                           names=['protein1', 'protein2', 'label'])
    
    # Align if needed
    if len(pairs_df) > len(X_df):
        pairs_df = pairs_df.iloc[:len(X_df)]
    
    print(f"   Pairs: {len(pairs_df):,}")
    
    # Get unique proteins
    all_proteins = set(pairs_df['protein1']).union(set(pairs_df['protein2']))
    n_total_proteins = len(all_proteins)
    print(f"   Unique proteins: {n_total_proteins:,}")
    
    # Identify motif columns
    motif_keywords = ["LIG_", "MOD_", "DOC_", "DEG_", "CLV_", "TRG_"]
    feature_names = list(X_df.columns)
    
    # Motif columns (P1_Motif_XXX and P2_Motif_XXX)
    motif_cols = [c for c in feature_names if any(kw in c.upper() for kw in motif_keywords)]
    p1_motif_cols = [c for c in motif_cols if c.startswith("P1_")]
    p2_motif_cols = [c for c in motif_cols if c.startswith("P2_")]
    
    print(f"\n[3/3] Analyzing motif features...")
    print(f"   Total motif columns: {len(motif_cols)}")
    print(f"   P1 motif columns: {len(p1_motif_cols)}")
    print(f"   P2 motif columns: {len(p2_motif_cols)}")
    
    # === PAIR-LEVEL ANALYSIS ===
    print(f"\n{'='*70}")
    print("PAIR-LEVEL STATISTICS")
    print(f"{'='*70}")
    
    # For each pair, check if ANY motif feature is non-zero
    X_motif = X_df[motif_cols].values
    pairs_with_motif = np.sum(np.any(X_motif != 0, axis=1))
    pairs_total = len(X_df)
    pair_coverage = pairs_with_motif / pairs_total * 100
    
    print(f"   Pairs with at least one motif: {pairs_with_motif:,} / {pairs_total:,}")
    print(f"   Pair-level coverage: {pair_coverage:.2f}%")
    
    # === PROTEIN-LEVEL ANALYSIS ===
    print(f"\n{'='*70}")
    print("PROTEIN-LEVEL STATISTICS")
    print(f"{'='*70}")
    
    # Check P1 motifs
    X_p1_motif = X_df[p1_motif_cols].values
    p1_has_motif = np.any(X_p1_motif != 0, axis=1)
    
    # Check P2 motifs
    X_p2_motif = X_df[p2_motif_cols].values
    p2_has_motif = np.any(X_p2_motif != 0, axis=1)
    
    # Create protein -> has_motif mapping
    protein_has_motif = {}
    
    for idx in range(len(pairs_df)):
        p1 = pairs_df.iloc[idx]['protein1']
        p2 = pairs_df.iloc[idx]['protein2']
        
        # Update P1
        if p1 not in protein_has_motif:
            protein_has_motif[p1] = False
        if p1_has_motif[idx]:
            protein_has_motif[p1] = True
        
        # Update P2
        if p2 not in protein_has_motif:
            protein_has_motif[p2] = False
        if p2_has_motif[idx]:
            protein_has_motif[p2] = True
    
    proteins_with_motif = sum(1 for v in protein_has_motif.values() if v)
    proteins_total = len(protein_has_motif)
    protein_coverage = proteins_with_motif / proteins_total * 100
    
    print(f"   Proteins with I ≠ ∅: {proteins_with_motif:,} / {proteins_total:,}")
    print(f"   Protein-level coverage: {protein_coverage:.2f}%")
    
    # === MOTIF TYPE ANALYSIS ===
    print(f"\n{'='*70}")
    print("MOTIF TYPE BREAKDOWN")
    print(f"{'='*70}")
    
    motif_type_counts = {}
    for col in motif_cols:
        for kw in motif_keywords:
            if kw in col.upper():
                if kw not in motif_type_counts:
                    motif_type_counts[kw] = 0
                # Count non-zero occurrences
                col_idx = feature_names.index(col)
                motif_type_counts[kw] += np.sum(X_df[col].values != 0)
                break
    
    for motif_type, count in sorted(motif_type_counts.items(), key=lambda x: -x[1]):
        pct = count / (len(X_df) * 2) * 100  # *2 for P1 and P2
        print(f"   {motif_type}: {count:,} occurrences ({pct:.2f}%)")
    
    
    return {
        'dataset': dataset_name,
        'proteins_total': proteins_total,
        'proteins_with_motif': proteins_with_motif,
        'protein_coverage': protein_coverage,
        'pairs_total': pairs_total,
        'pairs_with_motif': pairs_with_motif,
        'pair_coverage': pair_coverage,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["yeast", "human", "both"], default="both")
    args = parser.parse_args()
    
    if args.dataset == "both":
        results = []
        for ds in ["yeast", "human"]:
            try:
                r = calculate_motif_coverage(ds)
                results.append(r)
            except Exception as e:
                print(f"\n[ERROR] {ds}: {e}")
        
        if len(results) == 2:
            print(f"\n{'='*70}")
            print("COMBINED SUMMARY")
            print(f"{'='*70}")
            for r in results:
                print(f"   {r['dataset']}: {r['protein_coverage']:.2f}% protein coverage, {r['pair_coverage']:.2f}% pair coverage")
    else:
        calculate_motif_coverage(args.dataset)


if __name__ == "__main__":
    main()
