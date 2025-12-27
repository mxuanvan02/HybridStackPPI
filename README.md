# HybridStack-PPI

> **A Biologically-Informed Hybrid Stacking Framework for Protein-Protein Interaction Prediction**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## Abstract

Protein-protein interactions (PPIs) are fundamental to nearly all biological processes, yet their experimental identification remains costly and time-consuming. Computational methods have emerged as essential tools for predicting PPIs, but many existing approaches suffer from either limited interpretability or susceptibility to data leakage from homologous sequences.

**HybridStack-PPI** addresses these challenges through a novel **hybrid stacking ensemble** that synergistically combines:

1. **Interpretable Biological Features** (2,020 dimensions): Amino acid composition, dipeptide frequencies, physicochemical properties, and Eukaryotic Linear Motif (ELM) signatures
2. **Deep Protein Language Model Embeddings** (5,120 dimensions): Contextual representations from ESM-2 (650M parameters), combining global mean-pooled and local motif-aware embeddings

The framework employs a **dual-branch architecture** where specialized LightGBM base learners process each feature modality independently, followed by a Logistic Regression meta-learner that learns optimal branch contributions. This design enables **decision power analysis**, quantifying the relative contribution of biological knowledge versus deep learning representations.

To ensure rigorous evaluation, we implement **C3 Cluster-based Cross-Validation** using CD-HIT at 40% sequence identity, guaranteeing that no homologous proteins appear across train/test splits. This prevents the inflated performance metrics commonly observed when sequence-similar proteins leak across partitions.

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🧬 **Hybrid Architecture** | Dual-branch stacking: 2,020 biological features + 5,120 ESM-2 embeddings |
| � **Decision Power Analysis** | Quantifies contribution of biological vs. deep learning branches |
| 🛡️ **C3 Cluster-based CV** | Strict GroupKFold with CD-HIT clusters prevents homolog leakage |
| 📊 **Publication-Ready Outputs** | ROC/PR curves with confidence bands, LaTeX tables, metric boxplots |
| ⚡ **Efficient Caching** | Pre-computed ESM-2 embeddings and feature matrices for rapid iteration |

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         Input: Protein Pair (A, B)      │
                    └─────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        ┌───────────────────────┐           ┌───────────────────────┐
        │ Biological Features   │           │ ESM-2 Embeddings      │
        │ (2,020 dimensions)    │           │ (5,120 dimensions)    │
        │ • AA Composition      │           │ • Global Mean Pooling │
        │ • Dipeptide Freq      │           │ • Local Motif Pooling │
        │ • Physicochemical     │           │                       │
        │ • ELM Motifs (353)    │           │                       │
        └───────────────────────┘           └───────────────────────┘
                    │                                       │
                    ▼                                       ▼
        ┌───────────────────────┐           ┌───────────────────────┐
        │ Cumulative Feature    │           │ Cumulative Feature    │
        │ Selection (3-stage)   │           │ Selection (3-stage)   │
        └───────────────────────┘           └───────────────────────┘
                    │                                       │
                    ▼                                       ▼
        ┌───────────────────────┐           ┌───────────────────────┐
        │ LightGBM Classifier   │           │ LightGBM Classifier   │
        │ (Biological Branch)   │           │ (Embedding Branch)    │
        └───────────────────────┘           └───────────────────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
                        ┌───────────────────────────────┐
                        │ Logistic Regression Meta-Learner │
                        │   (Learns Branch Weights)        │
                        └───────────────────────────────┘
                                        │
                                        ▼
                            ┌───────────────────┐
                            │   PPI Prediction  │
                            └───────────────────┘
```

---

## Methodology

### C3 Cluster-based Cross-Validation

To prevent data leakage from homologous sequences, we implement the **C3 (Cluster-based Class 3) split strategy**:

1. **Clustering**: CD-HIT groups proteins at 40% sequence identity threshold
2. **GroupKFold**: Clusters (not individual proteins) are split into folds
3. **Pair Assignment**: A pair is included in a fold only if **both** proteins' clusters belong to that fold
4. **Zero Leakage Guarantee**: No protein from the same cluster appears in both train and test

This rigorous protocol ensures fair comparison with SOTA methods and prevents inflated performance metrics.

### Feature Selection Pipeline

Three-stage cumulative selection reduces dimensionality while preserving discriminative power:

1. **Variance Threshold** (τ = 1e-5): Remove near-constant features
2. **Cumulative Importance** (q = 97%): Keep features contributing to 97% of LightGBM importance
3. **Greedy Correlation Filter** (ρ = 0.95): Remove redundant highly-correlated features

---

## Results

### Benchmark Performance (5-Fold C3 Cross-Validation)

| Dataset | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---------|----------|-----------|--------|----------|---------|--------|
| **Yeast (BioGrid)** | 99.20% ± 0.24% | 99.21% | 99.27% | 99.24% | **99.79%** ± 0.11% | 99.84% |
| **Human (BioGrid)** | 99.78% ± 0.15% | 99.80% | 99.79% | 99.79% | **99.93%** ± 0.05% | 99.95% |

### Decision Power Analysis

The meta-learner weights reveal complementary contributions from both branches:

| Dataset | Biological Features | ESM-2 Embeddings |
|---------|---------------------|------------------|
| Yeast | 48.2% ± 3.1% | 51.8% ± 3.1% |
| Human | 45.6% ± 2.8% | 54.4% ± 2.8% |

### Visualizations

![ROC Curve](results/plots/yeast_c3_cv_roc.png)
![Decision Power](results/plots/yeast_decision_power.png)

---

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/HybridStackPPI.git
cd HybridStackPPI

# Create environment
conda create -n hybridstack python=3.10 -y
conda activate hybridstack

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Quick Start: 5-Fold C3 Cross-Validation

```bash
# Yeast dataset
python scripts/run_cv.py --dataset yeast --n-splits 5 --n-jobs -1

# Human dataset
python scripts/run_cv.py --dataset human --n-splits 5 --n-jobs -1
```

### Outputs

Results are saved to `results/{Dataset}_C3_CV_{timestamp}/`:

```
results/Yeast_C3_CV_20241227/
├── plots/
│   ├── yeast_c3_cv_roc.png      # Mean ROC curve with std dev band
│   ├── yeast_c3_cv_pr.png       # Mean PR curve with std dev band
│   ├── yeast_c3_cv_metrics.png  # Metric distribution boxplots
│   └── yeast_decision_power.png # Branch contribution visualization
├── table_row.tex                 # LaTeX-formatted result row
└── all_folds_predictions.csv     # Raw predictions for analysis
```

---

## Project Structure

```
HybridStackPPI/
├── src/                          # Core library
│   ├── builders.py               # Pipeline factory functions
│   ├── data_utils.py             # Data loading and caching
│   ├── feature_engine.py         # ESM-2 + biological feature extraction
│   ├── metrics.py                # Visualization and LaTeX generation
│   └── selectors.py              # Multi-stage feature selection
├── scripts/                      # Executable scripts
│   ├── run_cv.py                 # SOTA C3 cross-validation
│   ├── run_full_ablation.py      # Complete ablation study
│   ├── plot_results.py           # Publication figure generator
│   ├── reduce_redundancy.py      # CD-HIT preprocessing
│   └── cluster_split.py          # C3 train/val/test split
├── data/BioGrid/                 # Benchmark datasets
├── cache/                        # Pre-computed embeddings & features
└── results/                      # Experiment outputs
```

---

## 🔬 Experimental Analysis

### On Data Leakage: The Importance of Cluster-based Evaluation

A critical but often overlooked issue in PPI prediction is **data leakage** caused by sequence-similar (homologous) proteins appearing in both training and test sets. When standard random splitting is employed, the model can exploit shared sequence patterns between homologs rather than learning genuine interaction signatures. This leads to artificially inflated performance metrics that do not reflect true generalization capability.

Our analysis demonstrates that switching from naive random splits to the **C3 cluster-based protocol** (using CD-HIT at 40% sequence identity) results in a measurable performance drop of approximately 0.5-1.0% in ROC-AUC. While this may appear as a "decrease" in performance, it actually represents the **true generalization capability** of the model. The C3 split serves as a rigorous negative control, proving that previous high accuracies were partially attributable to homolog contamination rather than learned biological principles.

![Leakage Analysis](results/plots/leakage_analysis_yeast.png)

### The Hybrid Advantage: Synergy Between Biological Knowledge and Deep Learning

The modality ablation study reveals that neither interpretable biological features nor ESM-2 embeddings alone achieve optimal performance. Biological features (amino acid composition, physicochemical properties, ELM motifs) encode **domain-specific knowledge** accumulated over decades of biochemical research, while ESM-2 embeddings capture **evolutionary co-variation patterns** learned from millions of protein sequences.

When combined through our dual-branch stacking architecture, these complementary representations exhibit **synergistic effects**: the final Hybrid model consistently outperforms both single-modality alternatives by 1-3% in ROC-AUC. The decision power analysis further confirms that both branches contribute substantially (approximately 45-55% each) to the final predictions, validating the architectural choice of maintaining separate processing pathways rather than naive feature concatenation.

![Modality Ablation](results/plots/modality_ablation_yeast.png)

### Stacking Robustness: Meta-Learning for Prediction Stability

Individual classifiers, even powerful gradient boosting models like LightGBM, exhibit variance in their predictions depending on the specific training fold and hyperparameters. Our stacking architecture addresses this through a **Logistic Regression meta-learner** that learns to optimally combine the probability outputs from specialized base learners.

Compared to single-model baselines (SVM, Random Forest, standalone LightGBM), the stacking ensemble achieves both **higher absolute performance** and **lower variance** across cross-validation folds (standard deviation reduced from ~0.6% to ~0.1%). This stability is particularly valuable for practical PPI screening applications where consistent predictions are essential for downstream experimental prioritization.

![Stacking Efficiency](results/plots/stacking_efficiency_yeast.png)

### Generating Publication Figures

To reproduce the visualizations:

```bash
python scripts/plot_results.py --dataset both
```

Outputs are saved to `results/plots/` in both PNG (300 DPI) and PDF formats.

---

## Citation

```bibtex
@article{hybridstack_ppi_2024,
  title={HybridStack-PPI: A Biologically-Informed Hybrid Stacking Framework 
         for Protein-Protein Interaction Prediction},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2024},
  doi={10.XXXX/XXXXXX}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
