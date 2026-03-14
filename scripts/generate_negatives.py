#!/usr/bin/env python3
"""
generate_negatives.py
=====================
CLI script – generate negative PPI pairs and write new dataset files.

This script is the entry point for the negative-sampling pipeline.
It calls ``hybridstack.negative_sampler.NegativeSampler`` and handles:
  - command-line argument parsing
  - reading input files
  - writing output files
  - logging a run summary

Usage examples
--------------
# Random negatives for the Human dataset (same count as positives):
python scripts/generate_negatives.py --dataset human --strategy random

# Hard negatives: same subcellular compartment, not in BioGRID:
python scripts/generate_negatives.py \
    --dataset human \
    --strategy same_compartment \
    --annotation-cache data/annotations/uniprot_subcellular_human.tsv

# Generate all strategies in sequence:
python scripts/generate_negatives.py --dataset human --strategy all

Output files
------------
For --dataset human, --strategy same_compartment:
    data/BioGrid/Human/human_pairs_same_compartment.tsv

For --strategy all:
    data/BioGrid/Human/human_pairs_random.tsv
    data/BioGrid/Human/human_pairs_same_compartment.tsv

Each output file is tab-separated (no header): protein1, protein2, label
where label ∈ {0, 1}  (both positives AND generated negatives are included).
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Allow running from project root without installation.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hybridstack.negative_sampler import NegativeSampler
from hybridstack.logger import PipelineLogger

# ─── Dataset configuration ────────────────────────────────────────────────────

_DATASET_CONFIG = {
    "human": {
        "fasta":      "data/BioGrid/Human/human_dict.fasta",
        "pairs":      "data/BioGrid/Human/human_pairs.tsv",
        "ann_subcel": "data/annotations/uniprot_subcellular_human.tsv",
        "ann_go":     "data/annotations/uniprot_go_human.tsv",
        "negatome":   "data/annotations/negatome_human.tsv",
        "out_dir":    "data/BioGrid/Human",
        "prefix":     "human_pairs",
    },
    "yeast": {
        "fasta":      "data/BioGrid/Yeast/yeast_dict.fasta",
        "pairs":      "data/BioGrid/Yeast/yeast_pairs.tsv",
        "ann_subcel": "data/annotations/uniprot_subcellular_yeast.tsv",
        "ann_go":     "data/annotations/uniprot_go_yeast.tsv",
        "negatome":   "data/annotations/negatome_yeast.tsv",
        "out_dir":    "data/BioGrid/Yeast",
        "prefix":     "yeast_pairs",
    },
}

# Default order when --strategy all is requested.
# 'random' first (fast, baseline); then the hard 'same_compartment' strategy.
_ALL_STRATEGIES = ("random", "same_compartment", "same_go", "negatome")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_positives(pairs_path: str, logger: PipelineLogger) -> pd.DataFrame:
    """
    Load the original pairs file and return ONLY the positive interactions.

    Original negatives (mutant-based, label == 0) are intentionally dropped;
    they will be replaced by the new strategy-based negatives.

    Input file format (TSV, no header):
        protein1  <TAB>  protein2  <TAB>  label

    Returns
    -------
    pd.DataFrame  columns = ['protein1', 'protein2', 'label']
    """
    df = pd.read_csv(
        pairs_path,
        sep="\t",
        header=None,
        names=["protein1", "protein2", "label"],
    )
    total = len(df)
    positives = df[df["label"] == 1].copy().reset_index(drop=True)
    dropped = total - len(positives)
    logger.info(
        f"Pairs file: {total:,} total → {len(positives):,} positives kept, "
        f"{dropped:,} original negatives dropped."
    )
    return positives


def _write_pairs(combined_df: pd.DataFrame, out_path: str, logger: PipelineLogger) -> None:
    """
    Write a combined positive + negative DataFrame to a TSV file.

    Output columns: protein1, protein2, label  (no header, tab-separated).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined_df.to_csv(out_path, sep="\t", header=False, index=False)
    n_pos = (combined_df["label"] == 1).sum()
    n_neg = (combined_df["label"] == 0).sum()
    logger.info(
        f"Written: {out_path}\n"
        f"  Positives : {n_pos:,}\n"
        f"  Negatives : {n_neg:,}\n"
        f"  Total     : {len(combined_df):,}  "
        f"(ratio neg/pos = {n_neg / max(n_pos, 1):.2f})"
    )


def _run_strategy(
    strategy: str,
    positives_df: pd.DataFrame,
    protein_ids: list,
    n_negatives: int,
    ann_cache: str,
    negatome_file: str,
    random_state: int,
    logger: PipelineLogger,
) -> pd.DataFrame:
    """
    Instantiate, fit, and sample from NegativeSampler for one strategy.

    Returns a shuffled combined DataFrame (positives + negatives).
    """
    logger.phase(f"Strategy: {strategy.upper()}")

    sampler = NegativeSampler(
        strategy=strategy,
        n_negatives=n_negatives,
        random_state=random_state,
        annotation_cache_path=ann_cache if ann_cache and os.path.exists(ann_cache) else None,
        negatome_path=negatome_file if negatome_file and os.path.exists(negatome_file) else None,
    )
    sampler.fit(positives_df, protein_ids)
    negatives_df = sampler.sample()

    combined = pd.concat([positives_df, negatives_df], ignore_index=True)
    return combined.sample(frac=1, random_state=random_state).reset_index(drop=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate biologically-informed negative PPI pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dataset",
        choices=list(_DATASET_CONFIG.keys()),
        default="human",
        help="Which organism dataset to process (default: human).",
    )
    parser.add_argument(
        "--strategy",
        choices=[*_ALL_STRATEGIES, "all"],
        default="same_compartment",
        help=(
            "Sampling strategy. "
            "random (easy) → same_compartment (hard, gold standard). "
            "'all' runs every strategy (default: same_compartment)."
        ),
    )
    parser.add_argument(
        "--n-negatives",
        type=int,
        default=None,
        help=(
            "Number of negative pairs to generate.  "
            "Defaults to the number of positive pairs in the dataset."
        ),
    )
    parser.add_argument(
        "--annotation-cache",
        type=str,
        default=None,
        help=(
            "Path to pre-downloaded annotation TSV "
            "(protein_id <TAB> annotation_value).  When omitted, the script "
            "queries UniProt REST API live.  "
            "Required for same_compartment strategy."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    logger = PipelineLogger()

    cfg = _DATASET_CONFIG[args.dataset]

    # ── Resolve absolute paths ─────────────────────────────────────────────────
    fasta_path = str(PROJECT_ROOT / cfg["fasta"])
    pairs_path = str(PROJECT_ROOT / cfg["pairs"])
    out_dir    = str(PROJECT_ROOT / cfg["out_dir"])
    prefix     = cfg["prefix"]

    logger.phase(f"HybridStack-PPI → Generate Negatives  [{args.dataset.upper()}]")
    logger.info(f"FASTA  : {fasta_path}")
    logger.info(f"Pairs  : {pairs_path}")
    logger.info(f"Output : {out_dir}/")
    logger.info(f"Strategy: {args.strategy}  |  seed={args.seed}")

    # ── Input validation ──────────────────────────────────────────────────────
    for path, label in [(fasta_path, "FASTA"), (pairs_path, "pairs")]:
        if not os.path.exists(path):
            logger.error(f"{label} file not found: {path}")
            sys.exit(1)

    # ── Load inputs ───────────────────────────────────────────────────────────
    logger.phase("Loading inputs")
    protein_ids = NegativeSampler.load_protein_ids_from_fasta(fasta_path)
    logger.info(f"Protein IDs from FASTA: {len(protein_ids):,}")

    positives_df = _load_positives(pairs_path, logger)
    n_negatives  = args.n_negatives or len(positives_df)

    # ── Determine which strategies to run ─────────────────────────────────────
    strategies = list(_ALL_STRATEGIES) if args.strategy == "all" else [args.strategy]

    # ── Run each strategy ─────────────────────────────────────────────────────
    for strategy in strategies:
        # Resolve annotation cache: explicit flag > dataset default.
        if args.annotation_cache:
            ann_cache = args.annotation_cache
        elif strategy == "same_compartment":
            ann_cache = str(PROJECT_ROOT / cfg["ann_subcel"])
        else:
            ann_cache = ""  # unused for random

        if strategy == "negatome":
            negatome_file = str(PROJECT_ROOT / cfg["negatome"])
        else:
            negatome_file = ""

        combined_df = _run_strategy(
            strategy=strategy,
            positives_df=positives_df,
            protein_ids=protein_ids,
            n_negatives=n_negatives,
            ann_cache=ann_cache,
            negatome_file=negatome_file,
            random_state=args.seed,
            logger=logger,
        )

        out_path = os.path.join(out_dir, f"{prefix}_{strategy}.tsv")
        _write_pairs(combined_df, out_path, logger)

    logger.phase("Done")


if __name__ == "__main__":
    main()
