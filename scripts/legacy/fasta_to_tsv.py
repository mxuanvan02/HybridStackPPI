#!/usr/bin/env python3
"""Convert FASTA to TSV format for DeepTrio."""

import argparse
from Bio import SeqIO

def fasta_to_tsv(fasta_path, output_path):
    """Convert FASTA file to TSV (protein_id [Tab] sequence)."""
    with open(output_path, 'w') as out:
        for record in SeqIO.parse(fasta_path, 'fasta'):
            protein_id = record.id
            sequence = str(record.seq)
            out.write(f"{protein_id}\t{sequence}\n")
    print(f"Converted {fasta_path} -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FASTA to TSV for DeepTrio")
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument("--output", required=True, help="Output TSV file")
    args = parser.parse_args()
    
    fasta_to_tsv(args.fasta, args.output)
