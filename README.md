# HybridStack-PPI

> **A Biologically-Informed Hybrid Stacking Framework for Protein-Protein Interaction Prediction**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Abstract

**HybridStack-PPI** is a novel **hybrid stacking ensemble** designed for high-accuracy and interpretable PPI prediction. It synergistically combines specialized biological knowledge with evolutionary representations from a 650M-parameter protein language model (ESM-2), processed through a dual-branch architecture that quantifies the contribution of each modality.

To ensure scientific rigor, the framework enforces a **Strict C3 Cluster-based Cross-Validation** protocol using CD-HIT (40% identity), eliminating homology-based data leakage and providing a true measure of generalization.

---

## Key Features

- 🧬 **Hybrid Architecture**: Dual-branch stacking: 2,020 interpretable features + 5,120 ESM-2 embeddings.
- 🛡️ **Zero-Leakage (C3)**: Strict cluster-based splitting prevents homologous sequence contamination.
- 📊 **Decision Power Analysis**: Quantifies the weighting of biological vs. deep learning branches.
- 🎨 **Publication Figures**: Automatic generation of ROC/PR curves (with confidence bands), Calibration curves, t-SNE, and Detailed LaTeX tables.
- ⚡ **Auto-Routing Pipeline**: Automatically resolves correct cluster and data paths for simplified execution.

---

## Architecture

The framework employs a **Meta-Learner (Logistic Regression)** that optimizes weights for two **Base-Learners (LightGBM)**, trained independently on:
1. **Biological Branch**: AAC, DPC, CTD, PAAC, Moran Autocorrelation, and ELM Motifs.
2. **Deep Learning Branch**: ESM-2 Global mean-pool + Local motif-aware max-pool embeddings.

---

## Methodology: C3 Cluster-based Split

Standard random splitting in PPI prediction often leads to "homolog leakage," where similar sequences appear in both training and testing. HybridStack-PPI solves this via the **C3 protocol**:
1. **Clustering**: Group all proteins using CD-HIT at 40% sequence identity.
2. **Cluster Grouping**: Divide clusters into 5 folds.
3. **Filtering**: Only pairs where *both* proteins belong to the same fold's clusters are used for validation, ensuring zero homology overlap with training.

---

## Benchmark Results (5-Fold C3 CV)

| Dataset | Accuracy | Precision | Sensitivity | F1-Score | MCC | ROC-AUC | PR-AUC |
|---------|----------|-----------|-------------|----------|-----|---------|--------|
| **Yeast** | 99.27% | 99.94% | 98.73% | 99.33% | 98.54% | 99.82% | 99.89% |
| **Human** | 99.32% | 99.99% | 98.74% | 99.36% | 98.65% | 99.82% | 99.88% |

*Note: Results achieved using ESM-2 650M and standardized C3 splits.*

---

## Usage

### 1. Installation
```bash
git clone https://github.com/mxuanvan02/HybridStackPPI.git
cd HybridStackPPI
conda create -n hybridstack python=3.10 -y && conda activate hybridstack
pip install -r requirements.txt
```

### 2. Run Single Cross-Validation
The pipeline automatically routes to the correct cluster files in `data/BioGrid/`.
```bash
python scripts/run_cv.py --dataset yeast --n-splits 5 --n-jobs 1
```

### 3. Run Full Ablation Study
Analyzes sub-components (Interp-Only, Embed-Only, Logistic Baselines).
```bash
python scripts/run_full_ablation.py --dataset both --n-splits 5 --n-jobs 1
```

### 4. Analysis & Visualizations
```bash
# Regenerate standard ROC/PR curves
python scripts/regenerate_plots.py

# Generate Calibration, Confusion Matrix, and Sequence Length Analysis
python scripts/visualize_paper_extras.py
python scripts/analyze_sequence_length.py

# Feature Space Visualization
python scripts/visualize_tsne.py
```

---

## Project Structure

```
HybridStackPPI/
├── scripts/              # High-level experiment and analysis runners
│   ├── run_cv.py         # Main Cross-Validation (Hybrid model)
│   ├── run_full_ablation.py # Multi-variant comparison study
│   ├── system_check.py   # Integrity and leakage verification suite
│   └── visualize_*.py    # Publication-ready figure generators
├── src/                  # Core library
│   ├── builders.py       # Stacking pipeline building logic
│   ├── feature_engine.py # Feature extraction (Biological + ESM-2)
│   ├── metrics.py        # Charting and LaTeX utilities
│   └── selectors.py      # 3-stage feature selection
├── data/                 # Raw BioGrid FASTA/Pairs (CD-HIT data excluded)
├── legacy/               # Maintenance and migration scripts (Git ignored)
└── results/              # Organized outputs per dataset experiment
```

---

## License
MIT License.
