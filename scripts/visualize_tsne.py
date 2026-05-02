#!/usr/bin/env python3
import sys
import os
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
H5_PATH = "cache/yeast_yeast_pairs_facebook_esm2_t33_650m_ur50d_concat_v2_features.h5"
SAMPLE_SIZE = 1500  # Reasonable size for fast t-SNE

def plot_tsne():
    print(f"🧬 Loading feature matrix for t-SNE (Sample: {SAMPLE_SIZE})...")
    if not os.path.exists(H5_PATH):
        print(f"❌ Error: {H5_PATH} not found.")
        return

    with h5py.File(H5_PATH, 'r') as h5f:
        X = h5f['X_data'][:]
        y = h5f['y_data'][:]
        
    print(f"   Matrix loaded. Total: {X.shape}. Sampling...")
    
    # Stratified sample
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    
    sample_pos = np.random.choice(pos_idx, SAMPLE_SIZE // 2, replace=False)
    sample_neg = np.random.choice(neg_idx, SAMPLE_SIZE // 2, replace=False)
    indices = np.concatenate([sample_pos, sample_neg])
    
    X_sample = X[indices]
    y_sample = y[indices]
    
    print(f"🚀 Running t-SNE projection (Perplexity=30)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    X_embedded = tsne.fit_transform(X_sample)
    
    # Plotting
    plt.figure(figsize=(10, 8), dpi=300)
    sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1],
        hue=y_sample, palette=['#e74c3c', '#3498db'],
        alpha=0.6, s=60, edgecolor='w', linewidth=0.5
    )
    
    plt.title("t-SNE Visualization of Hybrid Feature Space\n(Yeast Dataset - High-Dimensional to 2D)", 
              fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontweight='bold')
    plt.ylabel("t-SNE Dimension 2", fontweight='bold')
    plt.legend(title="PPI Presence", labels=["Negative (Non-PPI)", "Positive (PPI)"], loc="best")
    plt.grid(True, alpha=0.2)
    
    out_path = "results/plots/tsne_feature_space.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"✅ t-SNE plot saved to {out_path}")

if __name__ == "__main__":
    plot_tsne()
