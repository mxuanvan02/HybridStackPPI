#!/usr/bin/env python3
"""
Reproduce Results Script
=========================
Reproduces the benchmark results from the paper.

This script runs 5-fold cross-validation on the Human BioGRID dataset
using protein-level splits to prevent data leakage.

Usage:
    python reproduce_results.py
    python reproduce_results.py --dataset yeast
    python reproduce_results.py --n-splits 10
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # For deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce HybridStack-PPI benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--dataset",
        choices=["human", "yeast", "both"],
        default="human",
        help="Dataset to evaluate (default: human)"
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)"
    )
    parser.add_argument(
        "--pairing",
        choices=["concat", "avgdiff", "symmetric", "hadamard_abs"],
        default="hadamard_abs",
        help="Pairing strategy (default: hadamard_abs). 'symmetric'/'hadamard_abs': Hadamard(P1*P2) + |P1-P2|"
    )
    parser.add_argument(
        "--esm-model",
        default="facebook/esm2_t33_650M_UR50D",
        help="ESM-2 model name"
    )
    parser.add_argument(
        "--h5-cache",
        default="cache/esm2/esm2_embeddings_v4.h5",
        help="Path to ESM embedding cache"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (default: -1 = all cores)"
    )
    parser.add_argument(
        "--cache-version",
        type=str,
        default="v3",
        help="Version of the cached pair features to load/save (default: v3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=["protein", "cluster"],
        default="protein",
        help="CV split strategy (protein=no overlap, cluster=CD-HIT family disjoint)"
    )
    parser.add_argument(
        "--cluster-path",
        type=str,
        default=None,
        help="Path to CD-HIT cluster map (e.g., data/BioGrid/Human/human_clusters.json). Required if split-strategy is 'cluster'."
    )
    parser.add_argument(
        "--strategy",
        choices=["default", "random", "same_compartment", "diff_compartment", "same_go", "negatome"],
        default="same_compartment",
        help="Negative sampling strategy to evaluate (default: same_compartment hard negatives)"
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study summary (cached A/B variants + C1/C2/C3 pipeline tests) instead of main experiment"
    )
    parser.add_argument(
        "--ablation-no-cache",
        action="store_true",
        help="Disable loading cached A/B ablations; only newly run experiments will be reported."
    )
    parser.add_argument(
        "--ablation-rerun-c-series",
        action="store_true",
        help="Force rerun C1/C2/C3 ablations even when cached fold_metrics.csv exists."
    )

    args = parser.parse_args()
    
    # Map strategy to file suffix
    strategy_suffix = {
        "random": "_random",
        "same_compartment": "_same_compartment",
        "diff_compartment": "_diff_compartment",
        "same_go": "_same_go",
        "negatome": "_negatome_balanced",
    }
    
    # Set random seed FIRST for reproducibility
    set_seed(args.seed)
    
    # Ensure cache directory exists before any file operations
    cache_dir = os.path.dirname(args.h5_cache)
    if cache_dir:  # Only create if there's a directory component
        os.makedirs(cache_dir, exist_ok=True)
    
    print("=" * 70)
    print("HybridStack-PPI: Reproducing Paper Results")
    print("=" * 70)
    print(f"\nRandom Seed: {args.seed}")
    print(f"Strategy: {args.strategy.upper()}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Import after setting seed
    from scripts.run import run_experiment, run_ablation_study, create_stacking_pipeline_for_notebook
    
    # Dataset configurations
    datasets = []
    suffix = strategy_suffix.get(args.strategy, "")
    
    if args.dataset in ["human", "both"]:
        datasets.append({
            "name": f"Human BioGRID ({args.strategy})",
            "fasta": str(PROJECT_ROOT / "data/BioGrid/Human/human_dict.fasta"),
            "pairs": str(PROJECT_ROOT / f"data/BioGrid/Human/human_pairs{suffix}.tsv"),
        })
    if args.dataset in ["yeast", "both"]:
        datasets.append({
            "name": f"Yeast BioGRID ({args.strategy})",
            "fasta": str(PROJECT_ROOT / "data/BioGrid/Yeast/yeast_dict.fasta"),
            "pairs": str(PROJECT_ROOT / f"data/BioGrid/Yeast/yeast_pairs{suffix}.tsv"),
        })
    
    all_results = {}
    
    # --- ABLATION MODE ---
    if args.ablation:
        print("\n🔬 ABLATION STUDY MODE")
        for ds in datasets:
            dataset_name_lower = ds['name'].split(" ")[0].lower()
            output_suffix = f"_{args.strategy}" if args.strategy != "default" else ""
            output_dir = os.path.join("results", f"{dataset_name_lower}{output_suffix}")
            print(f"\n  Running ablation for {ds['name']}...")
            run_ablation_study(
                fasta_path=ds['fasta'],
                pairs_path=ds['pairs'],
                h5_cache_path=args.h5_cache,
                esm_model_name=args.esm_model,
                pairing_strategy=args.pairing,
                n_splits=args.n_splits,
                n_jobs=args.n_jobs,
                output_dir=output_dir,
                cache_version=args.cache_version,
                reuse_cached_ablations=not args.ablation_no_cache,
                rerun_c_series=args.ablation_rerun_c_series,
            )
        print("\n✅ Ablation study complete!")
        return

    for ds in datasets:
        print("\n" + "=" * 70)
        print(f"Dataset: {ds['name']}")
        print("=" * 70)
        print(f"FASTA: {ds['fasta']}")
        print(f"Pairs: {ds['pairs']}")
        print(f"CV Folds: {args.n_splits}")
        print(f"Pairing Strategy: {args.pairing}")
        
        # Check files exist
        if not os.path.exists(ds['fasta']):
            print(f"ERROR: FASTA file not found: {ds['fasta']}")
            continue
        if not os.path.exists(ds['pairs']):
            print(f"ERROR: Pairs file not found: {ds['pairs']}")
            continue
        
        # [Optimization] Define columns once to avoid redundant FeatureEngine init in folds
        from scripts.run import define_stacking_columns, create_stacking_pipeline
        from hybridstack.feature_engine import FeatureEngine, EmbeddingComputer
        
        print("  Initializing feature names for pipeline...")
        embedding_computer = EmbeddingComputer(model_name=args.esm_model)
        feature_engine = FeatureEngine(h5_cache_path=args.h5_cache, embedding_computer=embedding_computer)
        interp_cols, embed_cols = define_stacking_columns(feature_engine, pairing_strategy=args.pairing)
        
        # Create model factory
        def model_factory(n_jobs=-1, feature_names=None):
            # Reduced cv_n_jobs from -1 to 1 to avoid OOM spikes on base-learner forks.
            # parallelism is handled by LightGBM internal threads.
            return create_stacking_pipeline(
                interp_cols=interp_cols,
                embed_cols=embed_cols,
                n_jobs=n_jobs,
                use_selector=True,
                cv_n_jobs=1,
                feature_names=feature_names,
                meta_learner_type="lr"
            )
        
        # Run experiment
        start_time = time.time()
        
        try:
            dataset_name_lower = ds['name'].split(" ")[0].lower()
            output_suffix = f"_{args.strategy}" if args.strategy != "default" else ""
            output_dir = os.path.join("results", f"{dataset_name_lower}{output_suffix}")
            metrics = run_experiment(
                fasta_path=ds['fasta'],
                pairs_path=ds['pairs'],
                h5_cache_path=args.h5_cache,
                model_factory=model_factory,
                pairing_strategy=args.pairing,
                n_splits=args.n_splits,
                esm_model_name=args.esm_model,
                n_jobs=args.n_jobs,
                cluster_path=args.cluster_path if args.split_strategy == "cluster" else None,
                cache_version=args.cache_version,
                output_dir=output_dir,
            )
            
            elapsed = time.time() - start_time
            all_results[ds['name']] = metrics
            
            print(f"\n✅ Completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print final summary
    if all_results:
        print("\n" + "=" * 70)
        print("FINAL RESULTS SUMMARY")
        print("=" * 70)
        
        for dataset_name, metrics in all_results.items():
            print(f"\n📊 {dataset_name}:")
            print("-" * 40)
            
            # Map metric names to paper format
            metric_mapping = {
                "Accuracy": "Accuracy",
                "F1 Score": "F1",
                "MCC": "MCC",
                "ROC-AUC": "AUC-ROC",
                "PR-AUC": "AUC-PR",
                "Precision": "Precision",
                "Recall (Sensitivity)": "Recall",
                "Specificity": "Specificity",
            }
            
            for metric_name, display_name in metric_mapping.items():
                if metric_name in metrics:
                    value = metrics[metric_name]
                    if isinstance(value, float):
                        print(f"  {display_name:<20}: {value*100:.2f}%")
        
        # LaTeX table row
        print("\n" + "-" * 70)
        print("LaTeX Table Row:")
        print("-" * 70)
        
        latex_lines = []
        for dataset_name, metrics in all_results.items():
            latex_row = "HybridStack-PPI"
            for key in ["Accuracy", "Precision", "Recall (Sensitivity)", "Specificity", "F1 Score", "MCC"]:
                if key in metrics:
                    latex_row += f" & {metrics[key]*100:.2f}"
            latex_row += " \\\\"
            print(latex_row)
            latex_lines.append(f"% {dataset_name}\n{latex_row}")
            
        # Thực hiện lưu kết quả tĩnh ra file cứng
        try:
            import json
            results_dir = PROJECT_ROOT / "results"
            results_dir.mkdir(exist_ok=True, parents=True)
            
            # Thêm suffix từ strategy để tránh ghi đè nếu chạy chiến lược khác
            suffix = f"_{args.strategy}" if args.strategy != "default" else ""
            
            # Save raw metrics to JSON
            json_path = results_dir / f"reproduce_metrics{suffix}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=4)
                
            # Save LaTeX rows to txt
            latex_path = results_dir / f"latex_row{suffix}.txt"
            with open(latex_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(latex_lines) + "\n")
                
            # Save DataFrame to CSV (Pivot để Rows là Model hoặc Fold, Cols là Metrics)
            df = pd.DataFrame(all_results).T
            csv_path = results_dir / f"reproduce_metrics{suffix}.csv"
            df.to_csv(csv_path)
            
            print("\n💾 TỰ ĐỘNG THU THẬP VÀ LƯU KẾT QUẢ VÀO:")
            print(f"  - {json_path}")
            print(f"  - {csv_path}")
            print(f"  - {latex_path}")
        except Exception as e:
            print(f"\n⚠️ Lỗi khi lưu kết quả ra đĩa cứng: {e}")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
