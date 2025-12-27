# HybridStack-PPI

A hybrid stacking ensemble framework for protein-protein interaction prediction, combining interpretable bio-features with ESM-2 embeddings.

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/mxuanvan02/HybridStackPPI.git
cd HybridStackPPI
pip install -r requirements.txt
```

### Run 5-Fold Cross-Validation (CD-HIT 40% Reduced)

The pipeline now uses redundancy-reduced datasets by default (CD-HIT 40% identity cutoff) to ensure no data leakage.

```bash
# Run on Yeast dataset (CD-HIT 40%)
python scripts/run_cv.py --dataset yeast --n-splits 5 --n-jobs -1

# Run on Human dataset (CD-HIT 40%)
python scripts/run_cv.py --dataset human --n-splits 5 --n-jobs -1
```

### Run Full Ablation Study

```bash
# Run ablation on both datasets using clean cluster-level data
python scripts/run_full_ablation.py --dataset both --n-splits 5 --n-jobs -1
```

### Precompute Features

```bash
python scripts/precompute_all.py
```

## 📊 Ablation Variants

| # | Features | Pipeline | Description |
|---|----------|----------|-------------|
| 1 | Interpretable | LR | Baseline (bio-features only) |
| 2 | Interpretable | Selector + Stacking → LR | Full pipeline |
| 3 | Embedding | LR | Baseline (ESM-2 embeddings) |
| 4 | Embedding | Selector + Stacking → LR | Full pipeline |
| 5 | ESM2-Global | LR | Baseline (global ESM-2 only) |
| 6 | ESM2-Global | Selector + Stacking → LR | Full pipeline |
| 7 | Hybrid | Stacking → LR | Combined interpretable + embedding |

## 📁 Project Structure

```
HybridStackPPI/
├── src/                    # Core source code
│   ├── builders.py         # Pipeline builders
│   ├── data_utils.py       # Data loading/processing
│   ├── feature_engine.py   # Feature extraction (ESM-2, bio-features)
│   ├── selectors.py        # 3-stage feature selection
│   ├── metrics.py          # Evaluation metrics
│   └── logger.py           # Logging utilities
├── scripts/                # Runnable scripts
│   ├── run_full_ablation.py  # Main ablation study
│   ├── run.py              # Core experiment runner
│   └── precompute_all.py   # Precompute features
├── data/                   # Dataset files
│   └── BioGrid/            # Original BioGrid (Human, Yeast)
│       └── CDHIT_Reduced/  # CD-HIT 40% Cleaned Versions
├── cache/                  # Feature matrices (local only)
└── results/                # Output results and plots
```

## 🔬 Pipeline Overview

HybridStack-PPI uses a dual-branch stacking architecture with **Logistic Regression** as meta-learner. We employ a three-stage **Cumulative Feature Selector** to ensure a balance between performance and interpretability.

| Branch | Features | Selection Strategy |
|--------|----------|-------------------|
| **ESM-2 Branch** | ESM-2 650M embeddings | Strict (q=0.92, corr=0.85, var=0.01) |
| **Interp Branch** | Physicochemical + SLiM motifs | Relaxed (q=0.97, corr=0.95, var=0.0) |

## 🔬 Features

- **Interpretable Bio-features**: AAC, DPC, CTD, PAAC, Moran autocorrelation
- **ESM-2 Embeddings**: Global (mean-pooling) + Local (motif-specific)
- **3-Stage Feature Selection**: Variance → Importance → Correlation
- **Stacking Ensemble**: Multi-branch architecture with LR meta-learner

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{hybridstackppi2024,
  title={HybridStack-PPI: A Hybrid Stacking Ensemble for Protein-Protein Interaction Prediction},
  author={...},
  year={2024}
}
```

## 📄 License

MIT License
