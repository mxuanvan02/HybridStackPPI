# HybridStack-PPI: A Biologically-Informed Hybrid Stacking Framework

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Bioinformatics](https://img.shields.io/badge/Task-PPI_Prediction-purple.svg)]()

> **Abstract:** Recent Protein Language Models (PLMs) like ESM-2 have revolutionized PPI prediction but often lack interpretability. **HybridStack-PPI** bridges this gap by systematically integrating deep semantic embeddings with explicit evolutionary motifs (SLiMs). Validated on Human and Yeast datasets under strict protein-level splitting, our framework achieves **99.45% accuracy** while maintaining biological transparency.

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
```

### 2. Run Experiments

```bash
# 5-Fold Cross-Validation on Human dataset
python scripts/run.py --dataset human

# 5-Fold Cross-Validation on Yeast dataset  
python scripts/run.py --dataset yeast

# Ablation Study
python scripts/run.py --dataset human --ablation
```

## 📋 Pipeline Overview

**HybridStack-PPI** uses a dual-branch stacking architecture with **Logistic Regression** as meta-learner:

| Branch | Features | Base Learners |
|--------|----------|---------------|
| **ESM-2 Branch** | ESM-2 650M embeddings (2560-dim) | LightGBM |
| **Bio Branch** | Physicochemical + SLiM motifs | LightGBM |

## 📊 Datasets

| Dataset | Proteins | Interactions | Source |
|---------|----------|--------------|--------|
| Human | 6,754 | 37,480 | BioGRID |
| Yeast | 2,433 | 11,188 | BioGRID |

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
