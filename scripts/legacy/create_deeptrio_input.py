#!/usr/bin/env python3
"""Create test FASTA files for DeepTrio prediction from pairs file."""

import argparse
from Bio import SeqIO

def create_pair_fastas(pairs_file, fasta_file, output_p1, output_p2, n_pairs=100):
    """Create two FASTA files for protein1 and protein2 from pairs."""
    # Load sequences
    seq_dict = {}
    for record in SeqIO.parse(fasta_file, 'fasta'):
        seq_dict[record.id] = str(record.seq)
    
    print(f"Loaded {len(seq_dict)} sequences from {fasta_file}")
    
    # Read pairs
    pairs = []
    with open(pairs_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_pairs:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                p1, p2, label = parts[0], parts[1], parts[2]
                if p1 in seq_dict and p2 in seq_dict:
                    pairs.append((p1, p2, label))
    
    print(f"Read {len(pairs)} valid pairs")
    
    # Write FASTA files
    with open(output_p1, 'w') as f1, open(output_p2, 'w') as f2:
        for p1, p2, label in pairs:
            f1.write(f">{p1}\n{seq_dict[p1]}\n")
            f2.write(f">{p2}\n{seq_dict[p2]}\n")
    
    print(f"Created {output_p1} and {output_p2} with {len(pairs)} pairs")
    return len(pairs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--output_p1", required=True)
    parser.add_argument("--output_p2", required=True)
    parser.add_argument("--n_pairs", type=int, default=100)
    args = parser.parse_args()
    
    create_pair_fastas(args.pairs, args.fasta, args.output_p1, args.output_p2, args.n_pairs)
