#!/usr/bin/env python3
"""Run 5-fold CV for the main HybridStack-PPI pipeline."""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run import run_experiment
from src.builders import create_stacking_pipeline, define_stacking_columns
from src.feature_engine import EmbeddingComputer, FeatureEngine

def main():
    parser = argparse.ArgumentParser(description="HybridStack-PPI 5-Fold Cross-Validation")
    parser.add_argument("--dataset", choices=["human", "yeast"], default="yeast")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--esm-model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--h5-cache", default=str(PROJECT_ROOT / "cache/esm2/esm2_embeddings_v4.h5"))
    parser.add_argument("--cluster-path", default=None, help="Path to CD-HIT cluster file (.clstr or mapping)")
    
    args = parser.parse_args()
    
    # Dataset paths
    if args.dataset == "yeast":
        fasta_path = str(PROJECT_ROOT / "data/BioGrid/Yeast/CDHIT_Reduced/yeast_clean.fasta")
        pairs_path = str(PROJECT_ROOT / "data/BioGrid/Yeast/CDHIT_Reduced/yeast_clean_pairs.tsv")
        name = "Yeast BioGrid (CD-HIT 40%)"
    else:
        fasta_path = str(PROJECT_ROOT / "data/BioGrid/Human/CDHIT_Reduced/human_clean.fasta")
        pairs_path = str(PROJECT_ROOT / "data/BioGrid/Human/CDHIT_Reduced/human_clean_pairs.tsv")
        name = "Human BioGrid (CD-HIT 40%)"
    
    print(f"\n{'#' * 80}")
    print(f"### HybridStack-PPI 5-Fold CV: {name}")
    print(f"{'#' * 80}")
    
    # Initialize FeatureEngine to get column names
    print("\n📦 Initializing FeatureEngine...")
    embedding_computer = EmbeddingComputer(model_name=args.esm_model)
    feature_engine = FeatureEngine(h5_cache_path=args.h5_cache, embedding_computer=embedding_computer)
    interp_cols, embed_cols = define_stacking_columns(feature_engine, "concat")
    
    print(f"  Interpretable features: {len(interp_cols)}")
    print(f"  Embedding features: {len(embed_cols)}")
    
    # Main Hybrid pipeline with selector
    model_factory = lambda n_jobs=-1: create_stacking_pipeline(
        interp_cols, embed_cols, n_jobs, use_selector=True
    )
    
    print("\n🚀 Running 5-Fold Cross-Validation...")
    results = run_experiment(
        fasta_path=fasta_path,
        pairs_path=pairs_path,
        h5_cache_path=args.h5_cache,
        model_factory=model_factory,
        n_splits=args.n_splits,
        n_jobs=args.n_jobs,
        esm_model_name=args.esm_model,
        pairing_strategy="concat",
        cluster_path=args.cluster_path,
    )
    
    print(f"\n✅ {name} 5-Fold CV Complete!")
    print(f"   Accuracy: {results.get('Accuracy', 0):.4f}")
    print(f"   ROC-AUC:  {results.get('ROC-AUC', 0):.4f}")
    print(f"   PR-AUC:   {results.get('PR-AUC', 0):.4f}")

if __name__ == "__main__":
    main()
