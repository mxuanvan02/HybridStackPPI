#!/usr/bin/env python3
"""
Feature Cache Migration for CD-HIT Reduced Datasets
====================================================
Migrates features from the original cache to a new cache for the cleaned dataset.

Key Logic:
- Clean pairs use REPRESENTATIVE IDs
- Original cache uses ORIGINAL IDs
- We need to map clean pairs back to find corresponding original pairs
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_data, canonicalize_pairs, save_feature_matrix_h5


def load_cluster_mapping(clstr_path: str) -> tuple[dict, dict]:
    """
    Load cluster mapping from .clstr file.
    
    Returns:
        member_to_rep: {member_id: representative_id}
        rep_to_members: {representative_id: [list of member_ids including itself]}
    """
    import re
    
    clusters = []
    current_members = []
    representative = None
    
    with open(clstr_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                if current_members:
                    clusters.append((current_members, representative))
                current_members = []
                representative = None
            else:
                match = re.search(r">(.+?)\.\.\.", line)
                if match:
                    protein_id = match.group(1)
                    current_members.append(protein_id)
                    if line.endswith("*"):
                        representative = protein_id
        if current_members:
            clusters.append((current_members, representative))
    
    member_to_rep = {}
    rep_to_members = {}
    
    for members, rep in clusters:
        if rep is None:
            rep = members[0]
        rep_to_members[rep] = members
        for m in members:
            member_to_rep[m] = rep
    
    return member_to_rep, rep_to_members


def make_canonical_pair(p1: str, p2: str) -> tuple[str, str]:
    """Ensure consistent ordering for pair comparison."""
    return tuple(sorted([p1, p2]))


def migrate_features(
    original_pairs_path: str,
    clean_pairs_path: str,
    clstr_path: str,
    src_h5: str,
    target_h5: str
):
    """
    Migrate features from original cache to clean cache.
    
    Strategy:
    1. Load original pairs and create index mapping (canonical_pair -> row_index)
    2. Load cluster mapping (member_to_rep)
    3. For each clean pair (R_A, R_B):
       - Find ALL original pairs (X, Y) where map(X)=R_A and map(Y)=R_B
       - Take the first match and use its features
    """
    print(f"\n{'='*70}")
    print(f"Feature Cache Migration")
    print(f"{'='*70}")
    print(f"  Source: {src_h5}")
    print(f"  Target: {target_h5}")
    
    # Step 1: Load original pairs and create index
    print(f"\n  Loading original pairs from {original_pairs_path}...")
    original_df = pd.read_csv(original_pairs_path, sep='\t', header=None, 
                               names=['protein1', 'protein2', 'label'])
    
    # Create canonical pair -> index mapping
    original_pair_to_idx = {}
    for idx, row in original_df.iterrows():
        canonical = make_canonical_pair(row['protein1'], row['protein2'])
        if canonical not in original_pair_to_idx:
            original_pair_to_idx[canonical] = idx
    
    print(f"    Original pairs: {len(original_df)}")
    print(f"    Unique canonical pairs: {len(original_pair_to_idx)}")
    
    # Step 2: Load cluster mapping
    print(f"\n  Loading cluster mapping from {clstr_path}...")
    member_to_rep, rep_to_members = load_cluster_mapping(clstr_path)
    print(f"    Total proteins mapped: {len(member_to_rep)}")
    print(f"    Representatives: {len(rep_to_members)}")
    
    # Step 3: Load clean pairs
    print(f"\n  Loading clean pairs from {clean_pairs_path}...")
    clean_df = pd.read_csv(clean_pairs_path, sep='\t', header=None,
                           names=['protein1', 'protein2', 'label'])
    print(f"    Clean pairs: {len(clean_df)}")
    
    # Step 4: Find matching indices
    print(f"\n  Matching clean pairs to original cache indices...")
    indices = []
    labels = []
    missing = []
    
    for _, row in clean_df.iterrows():
        r_a, r_b = row['protein1'], row['protein2']
        label = row['label']
        
        # Get all members that map to these representatives
        members_a = rep_to_members.get(r_a, [r_a])
        members_b = rep_to_members.get(r_b, [r_b])
        
        # Try to find ANY combination in original pairs
        found = False
        for m_a in members_a:
            if found:
                break
            for m_b in members_b:
                canonical = make_canonical_pair(m_a, m_b)
                if canonical in original_pair_to_idx:
                    indices.append(original_pair_to_idx[canonical])
                    labels.append(label)
                    found = True
                    break
        
        if not found:
            missing.append((r_a, r_b))
    
    print(f"    Matched: {len(indices)}")
    print(f"    Missing: {len(missing)}")
    
    if len(indices) == 0:
        print("  ❌ No matches found. Aborting.")
        return
    
    # Step 5: Extract features from source cache
    print(f"\n  Extracting features from source cache...")
    with h5py.File(src_h5, "r") as hf:
        X_data = hf["X_data"][:]
        X_cols = [col.decode("utf-8") for col in hf["X_cols"][:]]
        print(f"    Source shape: {X_data.shape}")
    
    # Filter to matched indices
    X_clean = X_data[indices]
    y_clean = np.array(labels)
    
    print(f"    Extracted shape: {X_clean.shape}")
    
    # Step 6: Save to target cache
    X_df = pd.DataFrame(X_clean, columns=X_cols)
    y_series = pd.Series(y_clean, name="label")
    
    save_feature_matrix_h5(X_df, y_series, target_h5)
    print(f"\n  ✅ Successfully migrated {len(X_df)} feature rows to {target_h5}")
    
    # Report missing pairs
    if missing:
        print(f"\n  ⚠️ {len(missing)} pairs could not be matched (new combinations after mapping)")


if __name__ == "__main__":
    # Yeast Migration
    migrate_features(
        original_pairs_path="data/BioGrid/Yeast/yeast_pairs.tsv",
        clean_pairs_path="data/BioGrid/Yeast/CDHIT_Reduced/yeast_clean_pairs.tsv",
        clstr_path="cache/yeast_dict_cdhit_40.clstr",
        src_h5="cache/yeast_yeast_pairs_facebook_esm2_t33_650m_ur50d_concat_v2_features.h5",
        target_h5="cache/yeast_clean_pairs_cluster_v5_features.h5"
    )
    
    # Human Migration
    migrate_features(
        original_pairs_path="data/BioGrid/Human/human_pairs.tsv",
        clean_pairs_path="data/BioGrid/Human/CDHIT_Reduced/human_clean_pairs.tsv",
        clstr_path="cache/human_dict_cdhit_40.clstr",
        src_h5="cache/human_human_pairs_facebook_esm2_t33_650m_ur50d_concat_v2_features.h5",
        target_h5="cache/human_clean_pairs_cluster_v5_features.h5"
    )
