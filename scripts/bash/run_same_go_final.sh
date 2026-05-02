#!/bin/bash
# Master script to run all Same-GO experiments

set -e

PROJECT_ROOT="/media/SAS/Van/HybridStackPPI"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ppis_env

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "--- [1/3] Running Feature Extraction (fast_extract_5.py) ---"
python3 fast_extract_5.py

echo "--- [2/3] Running Within-Species Cross-Validation (Human) ---"
python3 scripts/entrypoints/reproduce_results.py \
    --dataset human \
    --strategy same_go \
    --pairing hadamard_abs \
    --h5-cache cache/esm2/esm2_embeddings_v4.h5 \
    --n-jobs 4

echo "--- [2/3] Running Within-Species Cross-Validation (Yeast) ---"
python3 scripts/entrypoints/reproduce_results.py \
    --dataset yeast \
    --strategy same_go \
    --pairing hadamard_abs \
    --h5-cache cache/esm2/esm2_embeddings_v4.h5 \
    --n-jobs 4

echo "--- [3/3] Running Cross-Species Evaluation (Human -> Yeast) ---"
python3 scripts/entrypoints/cross_species_eval.py \
    --train-dataset human \
    --test-dataset yeast \
    --strategy same_go \
    --pairing hadamard_abs \
    --h5-cache cache/esm2/esm2_embeddings_v4.h5

echo "--- [3/3] Running Cross-Species Evaluation (Yeast -> Human) ---"
python3 scripts/entrypoints/cross_species_eval.py \
    --train-dataset yeast \
    --test-dataset human \
    --strategy same_go \
    --pairing hadamard_abs \
    --h5-cache cache/esm2/esm2_embeddings_v4.h5

echo "✅ ALL SAME-GO EXPERIMENTS COMPLETED!"
