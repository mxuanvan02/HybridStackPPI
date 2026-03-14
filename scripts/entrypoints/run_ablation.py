#!/usr/bin/env python3
"""
Ablation Study: HybridStackPPI Component Analysis
===================================================
Runs 4 ablation experiments on both Human and Yeast datasets:

  A1: Interp-Only  (Handcraft + Motif, NO ESM-2 embedding)
  A2: Embed-Only   (ESM-2 Global + Local embedding, NO Handcraft/Motif)
  A3: Full Stacking with CONCAT pairing  (for Symmetric vs Concat comparison)
  A4: Full Stacking with HADAMARD_ABS pairing (reference = main results)

Usage:
    python scripts/run_ablation.py --dataset both
    python scripts/run_ablation.py --dataset human
"""

import argparse
import gc
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="HybridStackPPI Ablation Study")
    parser.add_argument("--dataset", choices=["human", "yeast", "both"], default="both")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--h5-cache", default="cache/esm2/esm2_embeddings_v4.h5")
    parser.add_argument("--esm-model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--cache-version", type=str, default="v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategy",
        choices=["default", "same_compartment", "same_go"],
        default="same_compartment",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.h5_cache) or ".", exist_ok=True)

    from scripts.run import run_experiment
    from hybridstack.feature_engine import EmbeddingComputer, FeatureEngine
    from hybridstack.builders import (
        create_interp_only_pipeline,
        create_embed_only_pipeline,
        create_stacking_pipeline,
        define_stacking_columns,
    )

    # Initialize FeatureEngine once to get column names
    embedding_computer = EmbeddingComputer(model_name=args.esm_model)
    feature_engine = FeatureEngine(h5_cache_path=args.h5_cache, embedding_computer=embedding_computer)

    # Get column names for both pairing strategies
    interp_cols_sym, embed_cols_sym = define_stacking_columns(feature_engine, "hadamard_abs")
    interp_cols_cat, embed_cols_cat = define_stacking_columns(feature_engine, "concat")

    # Strategy suffix
    suffix_map = {"same_compartment": "_same_compartment", "same_go": "_same_go"}
    suffix = suffix_map.get(args.strategy, "")

    # Dataset configs
    datasets = []
    if args.dataset in ["human", "both"]:
        datasets.append({
            "name": "human",
            "fasta": str(PROJECT_ROOT / "data/BioGrid/Human/human_dict.fasta"),
            "pairs": str(PROJECT_ROOT / f"data/BioGrid/Human/human_pairs{suffix}.tsv"),
        })
    if args.dataset in ["yeast", "both"]:
        datasets.append({
            "name": "yeast",
            "fasta": str(PROJECT_ROOT / "data/BioGrid/Yeast/yeast_dict.fasta"),
            "pairs": str(PROJECT_ROOT / f"data/BioGrid/Yeast/yeast_pairs{suffix}.tsv"),
        })

    # ---- Define Ablation Experiments ----
    ablations = [
        {
            "id": "A1_InterpOnly",
            "label": "A1: Interp-Only (Handcraft+Motif)",
            "pairing": "hadamard_abs",
            "factory": lambda n_jobs=-1, feature_names=None, ic=interp_cols_sym: create_interp_only_pipeline(
                ic, n_jobs, use_selector=True, feature_names=feature_names
            ),
        },
        {
            "id": "A2_EmbedOnly",
            "label": "A2: Embed-Only (ESM-2+Local)",
            "pairing": "hadamard_abs",
            "factory": lambda n_jobs=-1, feature_names=None, ec=embed_cols_sym: create_embed_only_pipeline(
                ec, n_jobs, use_selector=True, feature_names=feature_names
            ),
        },
        {
            "id": "A3_FullStacking_Concat",
            "label": "A3: Full Stacking (Concat Pairing)",
            "pairing": "concat",
            "factory": lambda n_jobs=-1, feature_names=None, ic=interp_cols_cat, ec=embed_cols_cat: create_stacking_pipeline(
                interp_cols=ic, embed_cols=ec, n_jobs=n_jobs,
                use_selector=True, cv_n_jobs=-1, feature_names=feature_names
            ),
        },
    ]

    all_summary_rows = []

    for ds in datasets:
        print("\n" + "=" * 80)
        print(f"  ABLATION STUDY — Dataset: {ds['name'].upper()}")
        print("=" * 80)

        for abl in ablations:
            print("\n" + "-" * 70)
            print(f"  {abl['label']}  |  Pairing: {abl['pairing']}  |  Dataset: {ds['name'].upper()}")
            print("-" * 70)

            ds_out_name = f"{ds['name']}{suffix}"
            output_dir = os.path.join("results", ds_out_name, "ablation", abl["id"])
            os.makedirs(output_dir, exist_ok=True)

            set_seed(args.seed)
            t0 = time.time()
            try:
                metrics = run_experiment(
                    fasta_path=ds["fasta"],
                    pairs_path=ds["pairs"],
                    h5_cache_path=args.h5_cache,
                    model_factory=abl["factory"],
                    pairing_strategy=abl["pairing"],
                    n_splits=args.n_splits,
                    esm_model_name=args.esm_model,
                    n_jobs=args.n_jobs,
                    cache_version=args.cache_version,
                    output_dir=output_dir,
                )
                elapsed = time.time() - t0
                print(f"\n  ✅ {abl['label']} on {ds['name'].upper()} completed in {elapsed:.0f}s ({elapsed/60:.1f}min)")

                row = {"Dataset": ds["name"].upper(), "Ablation": abl["label"], "Time (s)": f"{elapsed:.0f}"}
                row.update(metrics)
                all_summary_rows.append(row)

            except Exception as e:
                import traceback
                elapsed = time.time() - t0
                print(f"\n  ❌ {abl['label']} on {ds['name'].upper()} FAILED after {elapsed:.0f}s: {e}")
                traceback.print_exc()
                all_summary_rows.append({
                    "Dataset": ds["name"].upper(),
                    "Ablation": abl["label"],
                    "Error": str(e),
                })

            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- Final Summary Table ----
    print("\n\n" + "=" * 100)
    print("  📊 ABLATION STUDY — CONSOLIDATED SUMMARY")
    print("=" * 100)

    df = pd.DataFrame(all_summary_rows)
    cols_order = ["Dataset", "Ablation", "Accuracy", "Precision", "Recall (Sensitivity)",
                  "F1 Score", "MCC", "ROC-AUC", "PR-AUC", "Time (s)"]
    cols_to_show = [c for c in cols_order if c in df.columns]
    
    # Format metrics as percentages
    metric_cols = ["Accuracy", "Precision", "Recall (Sensitivity)", "F1 Score", "MCC", "ROC-AUC", "PR-AUC"]
    for c in metric_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: f"{x*100:.2f}" if isinstance(x, float) else x)

    print(df[cols_to_show].to_string(index=False))

    # Save to CSV
    out_csv = os.path.join("results", "ablation", "ablation_summary.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n💾 Saved summary to {out_csv}")

    # ---- LaTeX Output ----
    print("\n📄 LaTeX Table Rows:")
    print("-" * 80)
    for _, row in df.iterrows():
        parts = [str(row.get("Ablation", ""))]
        for c in ["Accuracy", "F1 Score", "MCC", "ROC-AUC", "PR-AUC"]:
            parts.append(str(row.get(c, "—")))
        print("  " + " & ".join(parts) + " \\\\")
    print("-" * 80)

    print("\n" + "=" * 100)
    print("  ✅ ABLATION STUDY COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
