# Scripts Layout (Standardized)

This folder is organized by role to keep experiment entrypoints easy to find.

## 1) `scripts/entrypoints/` (what you run)
- `reproduce_results.py` — main HybridStackPPI 5-fold benchmark.
- `run_ablation.py` — ablation runs.
- `run_baselines.py` — classical baselines (CT, AC).
- `run_sota.py` — external SOTA baselines orchestrator.

## 2) `scripts/baselines/` (method implementations)
- `sota_sprint.py`
- `sota_dscript.py`
- `sota_esm2_mlp.py`
- `sota_proteinprompt.py`
- `sota_raftppi.py`

## 3) Backward compatibility
Legacy paths are kept:
- `scripts/reproduce_results.py`
- `scripts/run_ablation.py`
- `scripts/run_baselines.py`
- `scripts/run_sota.py`
- `scripts/sota_*.py`

These files are thin wrappers so old commands in report/README still work.
