#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
RESULTS_PATHS = {
    'Yeast': 'results/Yeast_C3_CV_20251228_011030/all_folds_predictions.csv',
    'Human': 'results/Human_C3_CV_20251228_015001/all_folds_predictions.csv'
}

def load_fasta(fasta_path):
    sequences = {}
    with open(fasta_path, "r") as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                header = line[1:].split()[0]
                sequences[header] = ""
            elif header:
                sequences[header] += line
    return {k: len(v) for k, v in sequences.items()}

def analyze_length_impact(dataset_name, preds_path, fasta_path, out_dir):
    print(f"\n📏 Analyzing {dataset_name} sequence length impact...")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(preds_path)
    # Check if protein IDs are present (they might not be in old files)
    if 'protein1' not in df.columns:
        print(f"⚠️ Warning: Protein IDs missing in {preds_path}. Attempting to reconstruct IDs...")
        from scripts.config import get_dataset_config
        from scripts.run_cv import parse_clstr_to_mapping, get_c3_splits
        
        config = get_dataset_config(dataset_name.lower())
        pairs_df = pd.read_csv(config['pairs'], sep='\t', header=None, names=['protein1', 'protein2', 'label'])
        clstr_map = parse_clstr_to_mapping(config['clstr'])
        splits, _, _ = get_c3_splits(pairs_df, clstr_map, n_splits=5, random_state=42)
        
        id_list = []
        for fold_idx, (_, val_idx) in enumerate(splits, 1):
            fold_pairs = pairs_df.iloc[val_idx][['protein1', 'protein2']]
            df_fold = df[df['fold_id'] == fold_idx]
            if len(df_fold) == len(fold_pairs):
                fold_pairs['fold_id'] = fold_idx
                id_list.append(fold_pairs)
        
        if id_list:
            ids_df = pd.concat(id_list, ignore_index=True)
            df = df.sort_values('fold_id').reset_index(drop=True)
            ids_df = ids_df.sort_values('fold_id').reset_index(drop=True)
            df['protein1'] = ids_df['protein1']
            df['protein2'] = ids_df['protein2']
            print("   ✅ IDs reconstructed successfully.")
        else:
            print("   ❌ Failed to reconstruct IDs. Analysis aborted.")
            return

    # 2. Get Lengths
    lengths = load_fasta(fasta_path)
    df['len1'] = df['protein1'].map(lengths)
    df['len2'] = df['protein2'].map(lengths)
    df['avg_len'] = (df['len1'] + df['len2']) / 2
    
    # Define bins
    bins = [0, 200, 400, 600, 800, 1000, 10000]
    labels = ['<200', '200-400', '400-600', '600-800', '800-1000', '>1000']
    df['len_bin'] = pd.cut(df['avg_len'], bins=bins, labels=labels)
    
    # Calculate performance per bin
    bin_stats = []
    for label in labels:
        bin_df = df[df['len_bin'] == label]
        if len(bin_df) > 5:
            acc = accuracy_score(bin_df['y_true'], bin_df['y_pred'])
            f1 = f1_score(bin_df['y_true'], bin_df['y_pred'])
            count = len(bin_df)
            bin_stats.append({'Length Range': label, 'Accuracy': acc, 'F1-Score': f1, 'Count': count})
            
    stats_df = pd.DataFrame(bin_stats)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    melt_df = stats_df.melt(id_vars='Length Range', value_vars=['Accuracy', 'F1-Score'], var_name='Metric', value_name='Score')
    sns.lineplot(data=melt_df, x='Length Range', y='Score', hue='Metric', marker='o', lw=3)
    plt.title(f'Performance vs. Protein Sequence Length ({dataset_name})', fontweight='bold')
    plt.ylim(0.8, 1.05)
    plt.grid(True, alpha=0.3)
    
    for i, row in stats_df.iterrows():
        plt.text(i, row['Accuracy'] + 0.01, f"n={row['Count']}", ha='center', fontsize=9)
        
    plt.savefig(out_dir / f"sequence_length_impact_{dataset_name.lower()}.png", bbox_inches='tight')
    plt.close()
    print(f"✅ Analysis complete. Plot saved to {out_dir}")

if __name__ == "__main__":
    from scripts.config import DATASETS
    
    # Yeast
    analyze_length_impact(
        "Yeast", 
        RESULTS_PATHS['Yeast'], 
        DATASETS['yeast']['fasta'],
        "results/Yeast_C3_CV_20251228_011030/plots"
    )
    
    # Human
    analyze_length_impact(
        "Human", 
        RESULTS_PATHS['Human'], 
        DATASETS['human']['fasta'],
        "results/Human_C3_CV_20251228_015001/plots"
    )
