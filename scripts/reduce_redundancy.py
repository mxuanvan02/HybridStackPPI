#!/usr/bin/env python3
"""
CD-HIT Redundancy Reduction Pipeline for PPI Datasets
======================================================
Implements the "Entity Consistency" rule:
- Members are redundant; map them to their Representative.
- Keep only Representative sequences.
- Map interactions to Representative IDs, then deduplicate.

Author: Generated for HybridStackPPI project
"""

import re
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd


def parse_fasta(fasta_path: str):
    """Simple FASTA parser (no Biopython dependency)."""
    with open(fasta_path, 'r') as f:
        seq_id = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id is not None:
                    yield seq_id, "".join(seq_lines)
                seq_id = line[1:].split()[0]  # Take first word after >
                seq_lines = []
            else:
                seq_lines.append(line)
        if seq_id is not None:
            yield seq_id, "".join(seq_lines)


def parse_clstr_file(clstr_path: str) -> dict[str, str]:
    """
    Parse CD-HIT .clstr file and create mapping dictionary.
    
    Returns:
        mapping_dict: {protein_id: representative_id}
        - Representatives map to themselves
        - Members map to their representative
    """
    print(f"\n{'='*60}")
    print(f"Step A: Parsing Cluster File")
    print(f"{'='*60}")
    print(f"  Input: {clstr_path}")
    
    clusters = []
    current_members = []
    representative = None
    
    with open(clstr_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                # Save previous cluster
                if current_members:
                    clusters.append((current_members, representative))
                current_members = []
                representative = None
            else:
                # Parse member line: "0\t123aa, >ProteinID... *"
                match = re.search(r">(.+?)\.\.\.", line)
                if match:
                    protein_id = match.group(1)
                    current_members.append(protein_id)
                    if line.endswith("*"):
                        representative = protein_id
        
        # Don't forget the last cluster
        if current_members:
            clusters.append((current_members, representative))
    
    # Build mapping dictionary
    mapping_dict = {}
    for members, rep in clusters:
        if rep is None:
            # Fallback: use first member as representative (shouldn't happen with CD-HIT)
            rep = members[0]
        for member in members:
            mapping_dict[member] = rep
    
    # Statistics
    num_clusters = len(clusters)
    num_representatives = len(set(mapping_dict.values()))
    num_members = len(mapping_dict) - num_representatives
    
    print(f"  ✅ Parsed {num_clusters} clusters")
    print(f"     - Representatives: {num_representatives}")
    print(f"     - Members (redundant): {num_members}")
    print(f"     - Total proteins: {len(mapping_dict)}")
    
    return mapping_dict


def reduce_sequences(
    fasta_path: str,
    output_path: str,
    mapping_dict: dict[str, str]
) -> int:
    """
    Step B: Sequence Reduction (Strict Filtering)
    
    Keep only sequences whose ID is a Representative ID.
    Do NOT rename IDs. Do NOT swap sequences.
    """
    print(f"\n{'='*60}")
    print(f"Step B: Sequence Reduction")
    print(f"{'='*60}")
    print(f"  Input: {fasta_path}")
    print(f"  Output: {output_path}")
    
    # Get set of representative IDs
    representative_ids = set(mapping_dict.values())
    
    # Read and filter
    total_count = 0
    kept_count = 0
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as out_handle:
        for seq_id, sequence in parse_fasta(fasta_path):
            total_count += 1
            if seq_id in representative_ids:
                # Write as-is (no renaming, no sequence swapping)
                out_handle.write(f">{seq_id}\n{sequence}\n")
                kept_count += 1
    
    discarded_count = total_count - kept_count
    
    print(f"  ✅ Sequences processed:")
    print(f"     - Before: {total_count}")
    print(f"     - After:  {kept_count}")
    print(f"     - Discarded (redundant members): {discarded_count}")
    
    return kept_count


def reduce_interactions(
    pairs_path: str,
    output_path: str,
    mapping_dict: dict[str, str]
) -> int:
    """
    Step C: Interaction Mapping & Deduplication
    
    1. Map both protein columns to their representatives
    2. Canonicalize pairs (sorted order for symmetry)
    3. Drop duplicates
    """
    print(f"\n{'='*60}")
    print(f"Step C: Interaction Reduction")
    print(f"{'='*60}")
    print(f"  Input: {pairs_path}")
    print(f"  Output: {output_path}")
    
    # Load pairs
    df = pd.read_csv(pairs_path, sep='\t', header=None, names=['protein_A', 'protein_B', 'label'])
    original_count = len(df)
    
    # Step 1: Map to representatives (vectorized)
    df['protein_A'] = df['protein_A'].map(mapping_dict).fillna(df['protein_A'])
    df['protein_B'] = df['protein_B'].map(mapping_dict).fillna(df['protein_B'])
    
    # Step 2: Canonicalize (ensure A <= B for symmetry)
    # This ensures (X, Y) and (Y, X) are treated as the same pair
    mask = df['protein_A'] > df['protein_B']
    df.loc[mask, ['protein_A', 'protein_B']] = df.loc[mask, ['protein_B', 'protein_A']].values
    
    # Step 3: Drop duplicates (keep first occurrence)
    df_dedup = df.drop_duplicates(subset=['protein_A', 'protein_B'], keep='first')
    
    # Reset index and save
    df_dedup = df_dedup.reset_index(drop=True)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_dedup.to_csv(output_path, sep='\t', header=False, index=False)
    
    final_count = len(df_dedup)
    duplicates_removed = original_count - final_count
    
    # Label statistics
    pos_count = (df_dedup['label'] == 1).sum()
    neg_count = (df_dedup['label'] == 0).sum()
    
    print(f"  ✅ Interactions processed:")
    print(f"     - Before: {original_count}")
    print(f"     - After:  {final_count}")
    print(f"     - Duplicates removed: {duplicates_removed}")
    print(f"     - Positive pairs: {pos_count}")
    print(f"     - Negative pairs: {neg_count}")
    
    return final_count


def main():
    parser = argparse.ArgumentParser(
        description="CD-HIT Redundancy Reduction Pipeline for PPI Datasets"
    )
    parser.add_argument("--fasta", required=True, help="Input sequences.fasta")
    parser.add_argument("--pairs", required=True, help="Input pairs.tsv")
    parser.add_argument("--clstr", required=True, help="CD-HIT output .clstr file")
    parser.add_argument("--out-dir", required=True, help="Output directory for cleaned files")
    parser.add_argument("--prefix", default="cleaned", help="Prefix for output files")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  CD-HIT REDUNDANCY REDUCTION PIPELINE")
    print("  Entity Consistency Rule: Members → Representative")
    print("=" * 70)
    
    # Step A: Parse cluster file
    mapping_dict = parse_clstr_file(args.clstr)
    
    # Prepare output paths
    out_dir = Path(args.out_dir)
    out_fasta = out_dir / f"{args.prefix}_sequences.fasta"
    out_pairs = out_dir / f"{args.prefix}_pairs.tsv"
    
    # Step B: Reduce sequences
    num_seqs = reduce_sequences(args.fasta, str(out_fasta), mapping_dict)
    
    # Step C: Reduce interactions
    num_pairs = reduce_interactions(args.pairs, str(out_pairs), mapping_dict)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Output directory: {out_dir}")
    print(f"  Cleaned sequences: {out_fasta.name} ({num_seqs} records)")
    print(f"  Cleaned pairs: {out_pairs.name} ({num_pairs} interactions)")
    print(f"\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
