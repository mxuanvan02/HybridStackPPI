# HybridStack-PPI: A Biologically-Informed Hybrid Stacking Framework

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Bioinformatics](https://img.shields.io/badge/Task-PPI_Prediction-purple.svg)]()

> **Abstract:** Recent Protein Language Models (PLMs) like ESM-2 have revolutionized PPI prediction but often lack interpretability. **HybridStack-PPI** bridges this gap by systematically integrating deep semantic embeddings with explicit evolutionary motifs (SLiMs). Validated on Human and Yeast datasets under strict **homology-based partitioning (CD-HIT 40% identity cutoff)**, our framework ensures robust evaluation by preventing data leakage between training and validation sets.

<p align="center">
  <img src="docs/HybridStackPPI_pipeline.png" alt="HybridStack-PPI Architecture" width="800">
  <br>
  <em>Figure 1: The dual-branch architecture of HybridStack-PPI.</em>
</p>

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/mxuanvan02/HybridStackPPI.git
cd HybridStackPPI
pip install -r requirements.txt
# Ensure cd-hit is installed: sudo apt-get install cd-hit
```

### 2. Run Experiments

```bash
# Run full ablation study with CD-HIT clustering
python scripts/run_full_ablation.py --dataset both --n-splits 5
```

## 📋 Pipeline Overview

**HybridStack-PPI** uses a dual-branch stacking architecture with **Logistic Regression** as meta-learner. We employ a three-stage **Cumulative Feature Selector** to ensure a balance between performance and interpretability.

| Branch | Features | Selection Strategy |
|--------|----------|-------------------|
| **ESM-2 Branch** | ESM-2 650M embeddings | Strict (q=0.90, corr=0.90) |
| **Interp Branch** | Physicochemical + SLiM motifs | Relaxed (q=0.97, corr=0.95) |

## 🛡️ Evaluation Strategy (Strict Anti-Leakage)

To prevent homology-based data leakage, we implement:
1. **Sequence-level Deduplication**: Unifying redundant protein IDs sharing identical sequences.
2. **Homology Clustering (CD-HIT)**: Clustering sequences at a **40% identity threshold**.
3. **Cluster-based CV**: Ensuring that no proteins from the same cluster (homologs) are shared between training and validation folds.

## 📊 Datasets (BioGrid v4.4)

| Dataset | Proteins | Total Pairs | Unique Clusters (40%) |
|---------|----------|--------------|-----------------------|
| Human | 42,205 | 62,328 | ~25,000 |
| Yeast | 17,435 | 26,924 | 17,178 |

## 📂 Project Structure

```text
HybridStackPPI/
├── hybridstack/              # Core Python package
│   ├── __init__.py
│   ├── feature_engine.py     # Feature extraction (ESM-2 + ELM motifs)
│   ├── builders.py           # Model pipeline builders
│   ├── selectors.py          # Feature selection logic
│   ├── metrics.py            # Evaluation metrics & visualization
│   ├── data_utils.py         # Data loading & preprocessing
│   └── logger.py             # Logging utilities
├── scripts/                  # Experiments & Utility scripts
│   ├── run.py                # Main experiment runner
│   ├── predict.py            # Inference script
│   └── reproduce_results.py  # Reproduce paper results
├── data/                     # Datasets
│   └── BioGrid/              # Human & Yeast PPI datasets
├── docs/                     # Documentation & figures
├── requirements.txt
└── README.md
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
