#!/bin/bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
PYTHON="/media/ssd/conda/envs/ppis_env/bin/python3"

echo "--- Cross-Species: Human -> Yeast ---"
$PYTHON scripts/entrypoints/cross_species_eval.py --train-dataset human --test-dataset yeast --strategy same_go --pairing hadamard_abs --h5-cache cache/esm2/esm2_embeddings_v4.h5 --n-jobs 4

echo "--- Cross-Species: Yeast -> Human ---"
$PYTHON scripts/entrypoints/cross_species_eval.py --train-dataset yeast --test-dataset human --strategy same_go --pairing hadamard_abs --h5-cache cache/esm2/esm2_embeddings_v4.h5 --n-jobs 4

echo "DONE!"
