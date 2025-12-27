#!/usr/bin/env python3
"""
C3 Cluster-Based Split Strategy for PPI Datasets
=================================================
SOTA standard for preventing data leakage in protein interaction prediction.

Strategy:
1. Map every protein to its cluster ID
2. Split CLUSTERS (not pairs) into Train/Val/Test
3. Assign pairs based on both proteins' clusters
4. Discard pairs that straddle multiple sets

Author: Generated for HybridStackPPI project
"""

import re
import argparse
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd


def parse_clstr_to_mapping(clstr_path: str) -> dict[str, int]:
    """
    Parse CD-HIT .clstr file and map each protein to its cluster ID.
    
    Returns:
        protein_to_cluster: {protein_id: cluster_id}
    """
    print(f"\n{'='*60}")
    print(f"Step 1: Parsing Cluster File")
    print(f"{'='*60}")
    print(f"  Input: {clstr_path}")
    
    protein_to_cluster = {}
    current_cluster_id = -1
    
    with open(clstr_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                # Extract cluster ID
                current_cluster_id = int(line.split()[1])
            else:
                # Parse member: "0\t123aa, >ProteinID... *"
                match = re.search(r">(.+?)\.\.\.", line)
                if match:
                    protein_id = match.group(1)
                    protein_to_cluster[protein_id] = current_cluster_id
    
    num_proteins = len(protein_to_cluster)
    num_clusters = len(set(protein_to_cluster.values()))
    
    print(f"  ✅ Parsed {num_proteins} proteins in {num_clusters} clusters")
    
    return protein_to_cluster


def split_clusters(
    cluster_ids: set,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42
) -> tuple[set, set, set]:
    """
    Split cluster IDs into Train/Val/Test sets.
    
    Returns:
        (train_clusters, val_clusters, test_clusters)
    """
    print(f"\n{'='*60}")
    print(f"Step 2: Splitting Clusters")
    print(f"{'='*60}")
    print(f"  Ratios: Train={train_ratio}, Val={val_ratio}, Test={test_ratio}")
    
    random.seed(random_state)
    cluster_list = list(cluster_ids)
    random.shuffle(cluster_list)
    
    n = len(cluster_list)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_clusters = set(cluster_list[:train_end])
    val_clusters = set(cluster_list[train_end:val_end])
    test_clusters = set(cluster_list[val_end:])
    
    print(f"  ✅ Train clusters: {len(train_clusters)}")
    print(f"  ✅ Val clusters: {len(val_clusters)}")
    print(f"  ✅ Test clusters: {len(test_clusters)}")
    
    # Verify no overlap
    assert len(train_clusters & val_clusters) == 0, "Train/Val overlap!"
    assert len(train_clusters & test_clusters) == 0, "Train/Test overlap!"
    assert len(val_clusters & test_clusters) == 0, "Val/Test overlap!"
    print(f"  ✅ Verified: No cluster overlap between sets")
    
    return train_clusters, val_clusters, test_clusters


def assign_pairs_to_sets(
    pairs_df: pd.DataFrame,
    protein_to_cluster: dict,
    train_clusters: set,
    val_clusters: set,
    test_clusters: set
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    Assign pairs to Train/Val/Test based on cluster membership.
    
    Rules:
    - Pair goes to Train IFF both proteins' clusters are in Train_Clusters
    - Pair goes to Val IFF both proteins' clusters are in Val_Clusters
    - Pair goes to Test IFF both proteins' clusters are in Test_Clusters
    - Otherwise: DISCARD (straddling pairs)
    
    Returns:
        (train_df, val_df, test_df, num_discarded)
    """
    print(f"\n{'='*60}")
    print(f"Step 3: Assigning Pairs to Sets")
    print(f"{'='*60}")
    print(f"  Total pairs: {len(pairs_df)}")
    
    train_pairs = []
    val_pairs = []
    test_pairs = []
    discarded = 0
    unmapped = 0
    
    for idx, row in pairs_df.iterrows():
        p_a, p_b, label = row['protein_A'], row['protein_B'], row['label']
        
        # Get cluster IDs
        cluster_a = protein_to_cluster.get(p_a)
        cluster_b = protein_to_cluster.get(p_b)
        
        # Skip if any protein is not in cluster map
        if cluster_a is None or cluster_b is None:
            unmapped += 1
            continue
        
        # Determine set assignment
        a_in_train = cluster_a in train_clusters
        b_in_train = cluster_b in train_clusters
        a_in_val = cluster_a in val_clusters
        b_in_val = cluster_b in val_clusters
        a_in_test = cluster_a in test_clusters
        b_in_test = cluster_b in test_clusters
        
        if a_in_train and b_in_train:
            train_pairs.append(row)
        elif a_in_val and b_in_val:
            val_pairs.append(row)
        elif a_in_test and b_in_test:
            test_pairs.append(row)
        else:
            # Straddling pair - discard
            discarded += 1
    
    train_df = pd.DataFrame(train_pairs)
    val_df = pd.DataFrame(val_pairs)
    test_df = pd.DataFrame(test_pairs)
    
    print(f"  ✅ Train pairs: {len(train_df)}")
    print(f"  ✅ Val pairs: {len(val_df)}")
    print(f"  ✅ Test pairs: {len(test_df)}")
    print(f"  ⚠️ Discarded (straddling): {discarded}")
    if unmapped > 0:
        print(f"  ⚠️ Unmapped proteins: {unmapped}")
    
    return train_df, val_df, test_df, discarded


def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
):
    """Verify zero protein overlap between Train and Test sets."""
    print(f"\n{'='*60}")
    print(f"Step 4: Verifying Zero Leakage")
    print(f"{'='*60}")
    
    def get_proteins(df):
        if df.empty:
            return set()
        return set(df['protein_A']).union(set(df['protein_B']))
    
    train_proteins = get_proteins(train_df)
    val_proteins = get_proteins(val_df)
    test_proteins = get_proteins(test_df)
    
    train_test_overlap = train_proteins & test_proteins
    train_val_overlap = train_proteins & val_proteins
    val_test_overlap = val_proteins & test_proteins
    
    print(f"  Train proteins: {len(train_proteins)}")
    print(f"  Val proteins: {len(val_proteins)}")
    print(f"  Test proteins: {len(test_proteins)}")
    print(f"  Train ∩ Test: {len(train_test_overlap)} proteins")
    print(f"  Train ∩ Val: {len(train_val_overlap)} proteins")
    print(f"  Val ∩ Test: {len(val_test_overlap)} proteins")
    
    if len(train_test_overlap) == 0 and len(train_val_overlap) == 0 and len(val_test_overlap) == 0:
        print(f"  ✅ VERIFIED: Zero leakage! All sets are completely disjoint.")
    else:
        print(f"  ❌ WARNING: Leakage detected!")
        if train_test_overlap:
            print(f"     Train∩Test examples: {list(train_test_overlap)[:5]}")


def main():
    parser = argparse.ArgumentParser(
        description="C3 Cluster-Based Split for PPI Datasets (SOTA Standard)"
    )
    parser.add_argument("--pairs", required=True, help="Input pairs.tsv")
    parser.add_argument("--clstr", required=True, help="CD-HIT output .clstr file")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  C3 CLUSTER-BASED SPLIT STRATEGY")
    print("  SOTA Standard for Zero-Leakage PPI Datasets")
    print("=" * 70)
    
    # Step 1: Parse cluster file
    protein_to_cluster = parse_clstr_to_mapping(args.clstr)
    all_cluster_ids = set(protein_to_cluster.values())
    
    # Step 2: Split clusters
    train_clusters, val_clusters, test_clusters = split_clusters(
        all_cluster_ids,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed
    )
    
    # Load pairs
    pairs_df = pd.read_csv(args.pairs, sep='\t', header=None, 
                           names=['protein_A', 'protein_B', 'label'])
    
    # Step 3: Assign pairs
    train_df, val_df, test_df, discarded = assign_pairs_to_sets(
        pairs_df, protein_to_cluster,
        train_clusters, val_clusters, test_clusters
    )
    
    # Step 4: Verify no leakage
    verify_no_leakage(train_df, val_df, test_df)
    
    # Save outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    
    # Summary
    total = len(pairs_df)
    used = len(train_df) + len(val_df) + len(test_df)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Original pairs: {total}")
    print(f"  Train: {len(train_df)} ({100*len(train_df)/total:.1f}%)")
    print(f"  Val: {len(val_df)} ({100*len(val_df)/total:.1f}%)")
    print(f"  Test: {len(test_df)} ({100*len(test_df)/total:.1f}%)")
    print(f"  Discarded (straddling): {discarded} ({100*discarded/total:.1f}%)")
    print(f"  Total used: {used} ({100*used/total:.1f}%)")
    print(f"\n  Output directory: {out_dir}")
    print(f"\n✅ C3 Split completed successfully!")


if __name__ == "__main__":
    main()
