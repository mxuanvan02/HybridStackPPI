import sys
import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import joblib

PROJECT_ROOT = Path('/media/SAS/Van/HybridStackPPI')
sys.path.insert(0, str(PROJECT_ROOT))

def analyze_meta_learner_same_go(dataset='human'):
    print(f"Analyzing Meta-Learner for {dataset} Same-GO...")
    model_path = PROJECT_ROOT / f'results/{dataset}_same_go/models/model_fold1.joblib'
    output_dir = PROJECT_ROOT / f'results/{dataset}_same_go/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    model = joblib.load(model_path)
    meta_learner = model.final_estimator_
    coefs = meta_learner.coef_[0]
    if len(coefs) >= 4:
        # predict_proba mode: [bio_p0, bio_p1, deep_p0, deep_p1]
        bio_weight = coefs[1]
        deep_weight = coefs[3]
    else:
        # Case where it only gives class 1 probs: [bio_p1, deep_p1]
        bio_weight = coefs[0]
        deep_weight = coefs[1]
    
    total_abs = abs(bio_weight) + abs(deep_weight)
    bio_contrib = (abs(bio_weight) / total_abs) * 100
    deep_contrib = (abs(deep_weight) / total_abs) * 100
    
    print(f"Bio: {bio_contrib:.2f}%, Deep: {deep_contrib:.2f}%")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2ecc71', '#3498db']
    wedges, texts, autotexts = ax.pie(
        [bio_contrib, deep_contrib],
        labels=['Biological Branch', 'Deep Learning Branch'],
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0),
        textprops={'fontsize': 12}
    )
    ax.set_title(f'Decision Power Distribution - {dataset.upper()} (Same-GO)', fontsize=14)
    
    plt.savefig(output_dir / 'meta_learner_analysis.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'meta_learner_analysis.png', bbox_inches='tight')
    plt.close()
    
    # Also copy to local ACCESS folder
    access_dir = PROJECT_ROOT / 'IEEE_Access/ACCESS_latex_template_20240429/figures_new'
    os.makedirs(access_dir, exist_ok=True)
    import shutil
    shutil.copy(output_dir / 'meta_learner_analysis.pdf', access_dir / f'{dataset}_same_go_meta.pdf')

if __name__ == "__main__":
    analyze_meta_learner_same_go('human')
    analyze_meta_learner_same_go('yeast')
