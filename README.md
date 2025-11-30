# HybridStackPPI

HybridStackPPI is a sequence-only protein-protein interaction (PPI) pipeline that blends handcrafted physicochemical/motif features with ESM2 embeddings and a LightGBM-based stacking model. This repository packages the core code plus two BioGrid datasets (Human and Yeast) so experiments can be reproduced quickly.

## What’s inside
- `pipelines/`: feature extraction (handcrafted + motif + ESM2), selectors, model builders, utilities.
- `experiments/`: experiment runner with leakage-safe protein/cluster splits, ablations, metrics/plots.
- `run_experiments.ipynb`: notebook to train/evaluate on the bundled datasets.
- `data/BioGrid/Human`: `human_dict.fasta`, `human_pairs.tsv`, `human_dict.tsv`.
- `data/BioGrid/Yeast`: `yeast_dict.fasta`, `yeast_pairs.tsv`, `yeast_dict.tsv`.

## Quickstart
1. **Install deps (CUDA-capable PyTorch recommended):**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the notebook:** open `run_experiments.ipynb` and execute cells. Defaults:
   - Train: BioGrid Human (`data/BioGrid/Human`)
   - Independent test: BioGrid Yeast (`data/BioGrid/Yeast`)
   - 5-fold protein-level CV to avoid leakage.
3. **Feature cache:** embeddings/features are stored in `cache/` (`H5_CACHE_FILE`), and cache version is set to `v3` so you can safely regenerate after code changes.

## Notes to avoid inflated metrics
- Pairs are canonicalized (`protein1`/`protein2` sorted) and duplicates are dropped before splitting/feature building. Conflicting duplicate labels are logged and the first occurrence is kept.
- Use protein-level or cluster-level CV (`n_splits>1`) instead of random train/test splits for reporting.
- If you previously created caches with older code, delete `cache/*` to regenerate with the dedup logic.

## Citation
If you use this code or datasets, please cite the HybridStackPPI work (add your preferred citation here).
