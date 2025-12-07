#!/usr/bin/env python3
"""
Convert BioGrid data to PIPR format and run 5-fold CV training.

PIPR needs:
1. protein.dictionary.tsv: ID\tSEQUENCE
2. protein.actions.tsv: ID1\tID2\tLABEL (with header to skip)
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from Bio import SeqIO

PROJECT_ROOT = Path('/media/SAS/Van/HybridStackPPI')
PIPR_DIR = PROJECT_ROOT / 'external_tools/PIPR'
RESULTS_DIR = PROJECT_ROOT / 'results'


def convert_biogrid_to_pipr(dataset='yeast'):
    """Convert BioGrid FASTA and pairs to PIPR format."""
    data_dir = PROJECT_ROOT / f'data/BioGrid/{dataset.capitalize()}'
    output_dir = PIPR_DIR / f'{dataset}_biogrid'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert FASTA to dictionary.tsv
    fasta_file = data_dir / f'{dataset}_dict.fasta'
    dict_file = output_dir / 'protein.dictionary.tsv'
    
    print(f"Converting {fasta_file} to {dict_file}...")
    with open(dict_file, 'w') as f:
        for record in SeqIO.parse(fasta_file, 'fasta'):
            f.write(f"{record.id}\t{str(record.seq)}\n")
    
    # Convert pairs.tsv to actions.tsv (PIPR expects header)
    pairs_file = data_dir / f'{dataset}_pairs.tsv'
    actions_file = output_dir / 'protein.actions.tsv'
    
    print(f"Converting {pairs_file} to {actions_file}...")
    with open(pairs_file) as f_in, open(actions_file, 'w') as f_out:
        # Write header (PIPR skips first line)
        f_out.write("protein1\tprotein2\tlabel\n")
        for line in f_in:
            f_out.write(line)
    
    print(f"Data saved to {output_dir}")
    return output_dir


def run_pipr_biogrid(dataset='yeast', epochs=20):
    """Run PIPR on BioGrid data."""
    print(f"\n{'='*60}")
    print(f"PIPR 5-Fold CV on {dataset.upper()} BioGrid Dataset")
    print(f"{'='*60}")
    
    # Convert data
    data_dir = convert_biogrid_to_pipr(dataset)
    
    # Count pairs
    actions_file = data_dir / 'protein.actions.tsv'
    with open(actions_file) as f:
        n_pairs = sum(1 for _ in f) - 1  # Subtract header
    print(f"Total pairs: {n_pairs}")
    
    # Modify PIPR rcnn.py to use our dictionary
    rcnn_file = PIPR_DIR / 'binary/model/lasagna/rcnn.py'
    
    # Create a patched version pointing to our data
    patched_rcnn = PIPR_DIR / 'binary/model/lasagna/rcnn_biogrid.py'
    
    with open(rcnn_file) as f:
        content = f.read()
    
    # Replace dictionary path
    old_dict = "id2seq_file = '../../../yeast/preprocessed/protein.dictionary.tsv'"
    new_dict = f"id2seq_file = '../../../{dataset}_biogrid/protein.dictionary.tsv'"
    content = content.replace(old_dict, new_dict)
    
    with open(patched_rcnn, 'w') as f:
        f.write(content)
    
    print(f"Created patched PIPR script: {patched_rcnn}")
    
    # Run PIPR
    result_file = RESULTS_DIR / f'pipr_{dataset}_biogrid_5fold.txt'
    
    cmd = [
        'conda', 'run', '-n', 'pipr_env', 'python', 'rcnn_biogrid.py',
        f'../../../{dataset}_biogrid/protein.actions.tsv', '2',
        str(result_file), '0', '25', str(epochs)
    ]
    
    print(f"\nRunning PIPR with {epochs} epochs...")
    print(f"Result file: {result_file}")
    print(f"Command: {' '.join(cmd)}")
    
    cwd = PIPR_DIR / 'binary/model/lasagna'
    
    result = subprocess.run(cmd, cwd=cwd)
    
    # Display results
    if result_file.exists():
        print(f"\n{'='*60}")
        print(f"PIPR {dataset.upper()} BioGrid Results")
        print(f"{'='*60}")
        with open(result_file) as f:
            print(f.read())
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run PIPR on BioGrid data')
    parser.add_argument('--dataset', choices=['yeast', 'human'], required=True)
    parser.add_argument('--epochs', type=int, default=20)
    
    args = parser.parse_args()
    
    run_pipr_biogrid(args.dataset, args.epochs)


if __name__ == '__main__':
    main()
