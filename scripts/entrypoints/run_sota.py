#!/usr/bin/env python
"""
run_sota.py — External SOTA Baselines for HybridStackPPI.

This script coordinates the execution of various SOTA comparison methods:
- SPRINT          : Classical ultrafast sequence-based (C++)
- D-SCRIPT (TT)   : Topsy-Turvy, Pre-trained Deep Learning (PyTorch Zero-shot)
- ESM-2 + MLP     : Vanilla PLM + Feedforward (Proxy DL Reference, 2024-2025)

Usage:
    python scripts/run_sota.py --methods sprint dscript esm2_mlp proteinprompt raftppi --dataset both --strategy same_compartment
    python scripts/run_sota.py --methods raftppi --dataset human --strategy same_compartment
"""
import argparse
import os
import sys
import time
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hybridstack.data_utils import load_data, canonicalize_pairs
from hybridstack.logger import PipelineLogger
from scripts.run import set_seed

# =====================================================================
# EXTERNAL BASELINE MODULES
# =====================================================================

from scripts.baselines.sota_sprint import run_sprint_baseline
from scripts.baselines.sota_dscript import run_dscript_baseline
from scripts.baselines.sota_esm2_mlp import run_esm2_mlp_baseline
from scripts.baselines.sota_proteinprompt import run_proteinprompt_baseline
from scripts.baselines.sota_raftppi import run_raftppi_baseline

from hybridstack.metrics import (
    plot_cv_roc_pr_curves,
    plot_oof_confusion_matrix,
    plot_f1_threshold_curve,
    plot_cv_metric_distribution
)

# =====================================================================
# REGISTRY
# =====================================================================

BASELINE_REGISTRY = {
    "sprint": {
        "display_name": "SPRINT (Ultrafast Sequence-based)",
        "runner_fn": run_sprint_baseline
    },
    "dscript": {
        "display_name": "D-SCRIPT/Topsy-Turvy (Zero-shot Inference)",
        "runner_fn": run_dscript_baseline
    },
    "esm2_mlp": {
        "display_name": "Vanilla ESM-2 (650M) + MLP (Simple DL Reference)",
        "runner_fn": run_esm2_mlp_baseline
    },
    "proteinprompt": {
        "display_name": "ProteinPrompt (AC210 + RF)",
        "runner_fn": run_proteinprompt_baseline,
    },
    "raftppi": {
        "display_name": "RaftPPI (Official Checkpoint, ESM2-8M)",
        "runner_fn": run_raftppi_baseline,
    },
}


def plot_baseline_results(output_dir, logger):
    """Generate standardized plots from baseline OOF predictions."""
    oof_path = os.path.join(output_dir, "oof_predictions.csv")
    metrics_path = os.path.join(output_dir, "fold_metrics.csv")
    
    if not os.path.exists(oof_path):
        return
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    logger.info(f"Generating diagnostic plots in {plots_dir}...")
    
    oof_df = pd.read_csv(oof_path)
    
    # ROC/PR Curves
    plot_cv_roc_pr_curves(oof_df, plots_dir)
    
    # Threshold curve
    plot_f1_threshold_curve(oof_df, plots_dir)
    
    # Confusion Matrix (at 0.5 default)
    plot_oof_confusion_matrix(oof_df, plots_dir, threshold=0.5)
    
    # Metric distribution if fold metrics exist
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        plot_cv_metric_distribution(metrics_df, plots_dir)

# =====================================================================
# MAIN ROUTINE
# =====================================================================

def main():
    print("DEBUG: Entering main()", flush=True)
    parser = argparse.ArgumentParser(description="External SOTA Baselines for HybridStackPPI")
    parser.add_argument("--methods", nargs='+', choices=["sprint", "dscript", "esm2_mlp", "proteinprompt", "raftppi", "all"],
                        default=["all"], help="Which SOTA methods to run")
    parser.add_argument("--dataset", choices=["human", "yeast", "both"], default="both")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--strategy", choices=["default", "same_compartment", "same_go"], default="same_compartment",
                        help="Negative sampling strategy to use for pairs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plots-only", action="store_true", help="Only generate plots from existing results")
    args = parser.parse_args()

    set_seed(args.seed)
    logger = PipelineLogger()

    # Determine which methods to run
    if "all" in args.methods:
        methods_to_run = list(BASELINE_REGISTRY.keys())
    else:
        methods_to_run = args.methods

    suffix = {
        "same_compartment": "_same_compartment",
        "same_go": "_same_go",
    }.get(args.strategy, "")

    # Configure datasets
    datasets_cfg = []
    if args.dataset in ["human", "both"]:
        datasets_cfg.append({
            "name": "human",
            "fasta": os.path.join(PROJECT_ROOT, "data/BioGrid/Human/human_dict.fasta"),
            "pairs": os.path.join(PROJECT_ROOT, f"data/BioGrid/Human/human_pairs{suffix}.tsv"),
        })
    if args.dataset in ["yeast", "both"]:
        datasets_cfg.append({
            "name": "yeast",
            "fasta": os.path.join(PROJECT_ROOT, "data/BioGrid/Yeast/yeast_dict.fasta"),
            "pairs": os.path.join(PROJECT_ROOT, f"data/BioGrid/Yeast/yeast_pairs{suffix}.tsv"),
        })

    logger.header("EXTERNAL SOTA BASELINES EXECUTION ENGINE")
    logger.info(f"Methods selected: {', '.join(methods_to_run)}")
    logger.info(f"Dataset(s): {args.dataset}")
    logger.info(f"Strategy: {args.strategy}\n")

    for ds in datasets_cfg:
        logger.header(f"DATASET {ds['name'].upper()}")
        
        # Load datasets
        try:
            sequences, pairs_df = load_data(ds["fasta"], ds["pairs"])
            pairs_df = canonicalize_pairs(pairs_df, dataset_name=ds["name"], logger=logger)
        except Exception as e:
            logger.error(f"Failed to load dataset {ds['name']}: {e}")
            continue

        for method in methods_to_run:
            cfg = BASELINE_REGISTRY[method]
            display = cfg["display_name"]
            runner_fn = cfg["runner_fn"]

            print(f"\n{'=' * 70}")
            print(f"Executing: {display}")
            print(f"{'=' * 70}")

            output_suffix = {"same_compartment": "_same_compartment", "same_go": "_same_go"}.get(args.strategy, "")
            output_dir = os.path.join(PROJECT_ROOT, "results", "github_baselines", f"{method}_{ds['name']}{output_suffix}")
            os.makedirs(output_dir, exist_ok=True)

            if not args.plots_only:
                t0 = time.time()
                try:
                    runner_kwargs = {
                        "dataset_name": ds["name"],
                        "sequences": sequences,
                        "pairs_df": pairs_df,
                        "n_splits": args.n_splits,
                        "output_dir": output_dir,
                        "logger": logger,
                    }
                    if method == "esm2_mlp":
                        runner_kwargs["epochs"] = 40
                    result = runner_fn(**runner_kwargs)
                except Exception as e:
                    print(f"Error during {method} execution: {e}")
                
                elapsed = time.time() - t0
                logger.info(f"  [{method.upper()}] Finished in {elapsed:.2f} seconds.")
            
            # Post-execution Plotting
            try:
                plot_baseline_results(output_dir, logger)
            except Exception as e:
                logger.warning(f"Failed to generate plots for {method}: {e}")

    print("\n" + "=" * 70)
    print("All configured baselines have finished execution.")
    print("=" * 70)

if __name__ == "__main__":
    main()
