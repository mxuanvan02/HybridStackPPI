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
    parser.add_argument("--dataset", choices=["human", "yeast", "both"], default="both")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--esm-model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--h5-cache", default=str(PROJECT_ROOT / "cache/esm2/esm2_embeddings_v4.h5"))
    
    args = parser.parse_args()
    
    datasets = []
    if args.dataset in ["yeast", "both"]:
        datasets.append({
            "name": "Yeast BioGrid (CD-HIT 40%)",
            "fasta": str(PROJECT_ROOT / "data/BioGrid/Yeast/yeast_dict.fasta"),
            "pairs": str(PROJECT_ROOT / "data/BioGrid/Yeast/yeast_pairs.tsv"),
            "cluster": str(PROJECT_ROOT / "cache/yeast_cluster_map.csv")
        })
    if args.dataset in ["human", "both"]:
        datasets.append({
            "name": "Human BioGrid (CD-HIT 40%)",
            "fasta": str(PROJECT_ROOT / "data/BioGrid/Human/human_dict.fasta"),
            "pairs": str(PROJECT_ROOT / "data/BioGrid/Human/human_pairs.tsv"),
            "cluster": str(PROJECT_ROOT / "cache/human_cluster_map.csv")
        })

    for ds in datasets:
        print(f"\n" + "#" * 100)
        print(f"### FULL ABLATION STUDY: {ds['name']}")
        print("#" * 100)
        
        results_df = run_ablation_study(
            fasta_path=ds['fasta'],
            pairs_path=ds['pairs'],
            h5_cache_path=args.h5_cache,
            esm_model_name=args.esm_model,
            n_splits=args.n_splits,
            n_jobs=args.n_jobs,
            cluster_path=ds['cluster']
        )
        
        # Save results to CSV
        os.makedirs("results", exist_ok=True)
        safe_name = ds['name'].lower().replace(" ", "_")
        csv_path = f"results/ablation_{safe_name}.csv"
        results_df.to_csv(csv_path)
        print(f"\n✅ Results for {ds['name']} saved to {csv_path}")

if __name__ == "__main__":
    main()
