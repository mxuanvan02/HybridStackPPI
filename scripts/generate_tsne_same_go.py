import os
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path

PROJECT_ROOT = Path('/media/SAS/Van/HybridStackPPI')
sys.path.insert(0, str(PROJECT_ROOT))

def generate_tsne_same_go(dataset='human'):
    print(f"Generating t-SNE for {dataset} Same-GO...")
    h5_path = PROJECT_ROOT / f'cache/{dataset}_{dataset}_pairs_same_go_facebook_esm2_t33_650m_ur50d_hadamard_abs_v3_features.h5'
    output_dir = PROJECT_ROOT / f'results/{dataset}_same_go/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(h5_path, 'r') as hf:
        X = hf['X_data'][:]
        y = hf['y_data'][:]
    
    # Sample for performance if needed
    num_samples = 2000
    if len(y) > num_samples:
        idx = np.random.choice(len(y), num_samples, replace=False)
        X = X[idx]
        y = y[idx]
    
    print(f"Fitting t-SNE on {len(y)} samples...")
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    X_embedded = tsne.fit_transform(X)
    
    df = pd.DataFrame(X_embedded, columns=['t-SNE 1', 't-SNE 2'])
    df['Label'] = ['Interaction' if val == 1 else 'Same-GO Negative' for val in y]
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='t-SNE 1', y='t-SNE 2', hue='Label', alpha=0.6, palette={'Interaction': '#3498db', 'Same-GO Negative': '#e74c3c'})
    plt.title(f't-SNE Visualization of Feature Space - {dataset.upper()} (Same-GO)')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'tsne_feature_space.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'tsne_feature_space.png', bbox_inches='tight')
    plt.close()
    
    # Copy to ACCESS
    access_dir = PROJECT_ROOT / 'IEEE_Access/ACCESS_latex_template_20240429/figures_new'
    import shutil
    shutil.copy(output_dir / 'tsne_feature_space.pdf', access_dir / f'{dataset}_same_go_tsne.pdf')
    print("✅ Done!")

if __name__ == "__main__":
    generate_tsne_same_go('human')
    generate_tsne_same_go('yeast')
