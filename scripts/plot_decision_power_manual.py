#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import plot_decision_power

def main():
    # 1. Human Dataset
    human_dir = "results/Human_C3_CV_20251228_015001/plots"
    human_contributions = {
        'Biological Branch': 23.9,
        'Deep Learning Branch': 76.1
    }
    plot_decision_power(human_contributions, dataset_name="HUMAN", save_dir=human_dir)
    
    # 2. Yeast Dataset
    yeast_dir = "results/Yeast_C3_CV_20251228_011030/plots"
    yeast_contributions = {
        'Biological Branch': 26.5,
        'Deep Learning Branch': 73.5
    }
    plot_decision_power(yeast_contributions, dataset_name="YEAST", save_dir=yeast_dir)

if __name__ == "__main__":
    main()
