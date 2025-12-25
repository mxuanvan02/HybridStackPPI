#!/usr/bin/env python3
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run import run_ablation_study

def main():
    parser = argparse.ArgumentParser(description="Full HybridStack-PPI Ablation Study")
    parser.add_argument("--dataset", choices=["human", "yeast", "both"], default="yeast")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--esm-model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--h5-cache", default=str(PROJECT_ROOT / "cache/esm2_embeddings.h5"))
    
    args = parser.parse_args()
    
    datasets = []
    if args.dataset in ["yeast", "both"]:
        datasets.append({
            "name": "Yeast DIP",
            "fasta": str(PROJECT_ROOT / "data/data/yeast/sequences.fasta"),
            "pairs": str(PROJECT_ROOT / "data/data/yeast/pairs.tsv")
        })
    if args.dataset in ["human", "both"]:
        # Assuming similar path for human if it exists
        human_fasta = PROJECT_ROOT / "data/data/human/sequences.fasta"
        human_pairs = PROJECT_ROOT / "data/data/human/pairs.tsv"
        if human_fasta.exists():
            datasets.append({
                "name": "Human Biogrid",
                "fasta": str(human_fasta),
                "pairs": str(human_pairs)
            })

    for ds in datasets:
        print(f"\n" + "#" * 100)
        print(f"### FULL ABLATION STUDY: {ds['name']}")
        print("#" * 100)
        
        run_ablation_study(
            fasta_path=ds['fasta'],
            pairs_path=ds['pairs'],
            h5_cache_path=args.h5_cache,
            esm_model_name=args.esm_model,
            n_splits=args.n_splits,
            n_jobs=args.n_jobs
        )

if __name__ == "__main__":
    main()
