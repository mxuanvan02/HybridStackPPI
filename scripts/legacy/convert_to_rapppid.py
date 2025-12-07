#!/usr/bin/env python3
"""
Convert BioGrid data to RAPPPID format (gzipped pickle files).

RAPPPID expects:
- train.pkl.gz: [(protein_id_a, protein_id_b, label), ...]
- val.pkl.gz: [(protein_id_a, protein_id_b, label), ...]
- test.pkl.gz: [(protein_id_a, protein_id_b, label), ...]
- seqs.pkl.gz: {protein_id: [encoded_amino_acids], ...}
"""

import gzip
import pickle
import argparse
from pathlib import Path
from random import randint, shuffle
from Bio import SeqIO


def get_aa_code(aa):
    """Convert amino acid to integer code (RAPPPID format)."""
    aas = ['PAD', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'O', 'U']
    wobble_aas = {
        'B': ['D', 'N'],
        'Z': ['Q', 'E'],
        'X': ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    }

    if aa in aas:
        return aas.index(aa)
    elif aa in ['B', 'Z', 'X']:
        idx = randint(0, len(wobble_aas[aa])-1)
        return aas.index(wobble_aas[aa][idx])
    else:
        return 0  # PAD for unknown


def encode_seq(seq):
    """Encode amino acid sequence to list of integers."""
    return [get_aa_code(aa) for aa in seq.upper()]


def convert_biogrid_to_rapppid(pairs_file, fasta_file, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Convert BioGrid data to RAPPPID format.
    
    Args:
        pairs_file: TSV with protein_id1, protein_id2, label
        fasta_file: FASTA with protein sequences
        output_dir: Directory to save pkl.gz files
        train_ratio: Fraction for training
        val_ratio: Fraction for validation (rest is test)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sequences
    print(f"Loading sequences from {fasta_file}...")
    seqs = {}
    for record in SeqIO.parse(fasta_file, 'fasta'):
        seqs[record.id] = encode_seq(str(record.seq))
    print(f"Loaded {len(seqs)} sequences")
    
    # Load pairs
    print(f"Loading pairs from {pairs_file}...")
    pairs = []
    with open(pairs_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                p1, p2, label = parts[0], parts[1], int(parts[2])
                if p1 in seqs and p2 in seqs:
                    pairs.append((p1, p2, label))
    print(f"Loaded {len(pairs)} valid pairs")
    
    # Shuffle and split
    shuffle(pairs)
    n_total = len(pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]
    
    print(f"Split: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
    
    # Save files
    print("Saving gzipped pickle files...")
    
    with gzip.open(output_dir / 'train.pkl.gz', 'wb') as f:
        pickle.dump(train_pairs, f)
    
    with gzip.open(output_dir / 'val.pkl.gz', 'wb') as f:
        pickle.dump(val_pairs, f)
    
    with gzip.open(output_dir / 'test.pkl.gz', 'wb') as f:
        pickle.dump(test_pairs, f)
    
    with gzip.open(output_dir / 'seqs.pkl.gz', 'wb') as f:
        pickle.dump(seqs, f)
    
    print(f"Saved to {output_dir}")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert BioGrid to RAPPPID format')
    parser.add_argument('--pairs', required=True, help='Input pairs TSV file')
    parser.add_argument('--fasta', required=True, help='Input FASTA file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    
    args = parser.parse_args()
    convert_biogrid_to_rapppid(args.pairs, args.fasta, args.output, args.train_ratio, args.val_ratio)
