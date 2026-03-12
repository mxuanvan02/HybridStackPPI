# HybridStackPPI

HybridStackPPI is a reproducible experiment workspace for protein-protein interaction (PPI) prediction.
This README is focused on **how to run and maintain the repo**, not on paper narrative.

## What this repo contains

- Core model and feature pipeline in `hybridstack/`
- Reproducible experiment entrypoints in `scripts/entrypoints/`
- Baseline method implementations in `scripts/baselines/`
- Dataset inputs in `data/BioGrid/`
- Runtime outputs in `results/`, `logs/`, and `cache/` (ignored by git)

## Environment

- Python 3.9+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Required data layout

Make sure these files exist:

- `data/BioGrid/Human/human_dict.fasta`
- `data/BioGrid/Human/human_pairs.tsv`
- `data/BioGrid/Human/human_pairs_same_compartment.tsv`
- `data/BioGrid/Human/human_pairs_same_go.tsv`
- `data/BioGrid/Yeast/yeast_dict.fasta`
- `data/BioGrid/Yeast/yeast_pairs.tsv`
- `data/BioGrid/Yeast/yeast_pairs_same_compartment.tsv`
- `data/BioGrid/Yeast/yeast_pairs_same_go.tsv`

## Main run commands

### HybridStackPPI (5-fold CV)

```bash
python scripts/entrypoints/reproduce_results.py --dataset human --n-splits 5 --pairing hadamard_abs --strategy same_compartment
python scripts/entrypoints/reproduce_results.py --dataset yeast --n-splits 5 --pairing hadamard_abs --strategy same_compartment
```

### same_go stress test

```bash
python scripts/entrypoints/reproduce_results.py --dataset both --n-splits 5 --pairing hadamard_abs --strategy same_go
```

### Classical baselines (CT, AC)

```bash
python scripts/entrypoints/run_baselines.py --dataset both --strategy same_compartment
python scripts/entrypoints/run_baselines.py --dataset both --strategy same_go
```

### External SOTA baselines

```bash
python scripts/entrypoints/run_sota.py --methods sprint dscript esm2_mlp proteinprompt raftppi --dataset both --strategy same_compartment
```

Notes:
- `proteinprompt` requires Docker image `proteinprompt`.
- `raftppi` requires local/cache files for `facebook/esm2_t6_8M_UR50D`.

### Ablation

```bash
python scripts/entrypoints/run_ablation.py --dataset both --n-splits 5 --strategy same_compartment
```

## Script structure

See `scripts/README.md` for script organization details.

## Backward compatibility

Legacy paths in `scripts/` (for example `scripts/reproduce_results.py`) are kept as wrappers and still work.

## Repository hygiene

- Heavy/generated artifacts are ignored in `.gitignore`:
  - `cache/`, `logs/`, `results/`
  - external model repos (for example `D-SCRIPT/`, `RaftPPI/`, `proteinPrompt/`)
  - large derived datasets (for example `data/BioGrid/*/CDHIT_Reduced/`)
- Keep commits focused on source code and documentation.
