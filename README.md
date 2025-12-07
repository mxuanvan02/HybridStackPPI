# HybridStack-PPI: A Biologically-Informed Hybrid Stacking Framework

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This repository is the official implementation of the paper: **"A Biologically-Informed Hybrid Stacking Framework for Protein-Protein Interaction Prediction"**.

## 🚀 Key Features

- **Biologically-Informed:** Explicitly utilizes SLiMs (Short Linear Motifs) from ELM database combined with deep learning.
- **High Accuracy:** Achieves **99.45%** accuracy on Human BioGRID dataset via rigorous Protein-level split.
- **Hybrid Architecture:** A dual-branch system merging ESM-2 embeddings with physicochemical priors.
- **Reproducible:** Deterministic results with fixed random seeds and protein-level cross-validation.

## 🛠️ Installation

```bash
git clone https://github.com/mxuanvan02/HybridStackPPI.git
cd HybridStackPPI
pip install -r requirements.txt
```

## 📊 Reproducing Results

To reproduce the benchmark results reported in the paper (Table 3 & Table 4):

1. **Prepare Data:** Ensure `data/BioGrid` contains the processed `.tsv` files.

2. **Run Evaluation:**
```bash
python scripts/reproduce_results.py
```

## 🧪 Usage (Prediction)

To predict the interaction probability between two arbitrary protein sequences:

```bash
python scripts/predict.py \
  --seq1 "MEEPQSDPSVEPPLSQETFSDLWKLLP..." \
  --seq2 "MCNTNMSVPTDGAVTTSQIPASEQET..."
```

## 📂 Project Structure

```
HybridStackPPI/
├── hybridstack/          # Core Python package
│   ├── __init__.py
│   ├── feature_engine.py # Feature extraction (ESM-2 + Bio)
│   ├── builders.py       # Model pipeline builders
│   ├── selectors.py      # Feature selection
│   ├── metrics.py        # Evaluation metrics
│   ├── data_utils.py     # Data loading utilities
│   └── logger.py         # Logging utilities
├── scripts/              # Training and evaluation scripts
│   ├── run.py            # Main experiment runner
│   ├── predict.py        # Inference script
│   └── reproduce_results.py
├── data/                 # Processed datasets
│   └── BioGrid/
│       ├── Human/
│       └── Yeast/
├── models/               # Trained weights
│   └── saved/
├── notebooks/            # Demo notebooks
├── requirements.txt
└── README.md
```

## 📜 Citation

If you use this code, please cite our paper:

```bibtex
@article{mai2025hybridstack,
  title={A Biologically-Informed Hybrid Stacking Framework for Protein-Protein Interaction Prediction},
  author={Mai, Xuan Van and et al.},
  journal={Computer Science and Information Systems},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
