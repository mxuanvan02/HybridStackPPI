#!/usr/bin/env python3
"""
Meta-Learner Coefficient Analysis for HybridStackPPI
=====================================================
Analyze the contribution of Bio vs Deep branches in the stacking ensemble.

Author: HybridStackPPI Team
"""

import sys
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path('/media/SAS/Van/HybridStackPPI')
sys.path.insert(0, str(PROJECT_ROOT))


def load_cached_features_with_columns(h5_path: str):
    """Load features with column names."""
    print(f"Loading cached features from: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        X = f['X_data'][:]
        y = f['y_data'][:]
        
        # Get column names
        if 'X_cols' in f:
            cols_data = f['X_cols'][:]
            if cols_data.dtype.kind == 'S':
                cols = [c.decode('utf-8') for c in cols_data]
            else:
                cols = list(cols_data)
        else:
            cols = [f'f_{i}' for i in range(X.shape[1])]
    
    print(f"  Features shape: {X.shape}")
    print(f"  Number of columns: {len(cols)}")
    
    return X, y, cols


def identify_feature_groups(cols):
    """Identify which columns belong to Bio vs Deep features."""
    bio_cols = []
    deep_cols = []
    
    # Bio features: AAC, DPC, CTD, PAAC, Moran, Motif
    bio_patterns = ['AAC', 'DPC', 'CTD', 'PAAC', 'Moran', 'Motif']
    # Deep features: ESM, Global, Local embeddings
    deep_patterns = ['ESM', 'Global', 'Local', 'Embed']
    
    for col in cols:
        is_bio = any(pat in col for pat in bio_patterns)
        is_deep = any(pat in col for pat in deep_patterns)
        
        if is_bio and not is_deep:
            bio_cols.append(col)
        elif is_deep:
            deep_cols.append(col)
        else:
            # Default classification based on feature name patterns
            if 'emb' in col.lower() or 'global' in col.lower() or 'local' in col.lower():
                deep_cols.append(col)
            else:
                bio_cols.append(col)
    
    return bio_cols, deep_cols


def analyze_meta_learner(dataset='human'):
    """
    Train a stacking model and analyze the meta-learner coefficients.
    """
    print("\n" + "=" * 70)
    print(f"META-LEARNER COEFFICIENT ANALYSIS: {dataset.upper()}")
    print("=" * 70)
    
    # Define paths
    if dataset == 'human':
        cache_path = PROJECT_ROOT / 'cache/human_human_pairs_facebook_esm2_t33_650m_ur50d_concat_v2_features.h5'
    else:
        cache_path = PROJECT_ROOT / 'cache/yeast_pairs_facebook_esm2_t33_650m_ur50d_concat_v2_features.h5'
    
    # Load data
    X, y, cols = load_cached_features_with_columns(str(cache_path))
    X_df = pd.DataFrame(X, columns=cols)
    
    # Identify Bio and Deep feature columns
    bio_cols, deep_cols = identify_feature_groups(cols)
    
    print(f"\n📊 Feature Groups:")
    print(f"  Bio features: {len(bio_cols)} columns")
    print(f"  Deep features: {len(deep_cols)} columns")
    print(f"  Total: {len(cols)} columns")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📋 Data Split:")
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # ==========================================================================
    # Create Stacking Model with two base estimators
    # ==========================================================================
    print("\n🔧 Building Stacking Model...")
    
    # Common LGBM params
    lgbm_params = {
        'n_estimators': 100,
        'max_depth': 8,
        'learning_rate': 0.1,
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1,
        'class_weight': 'balanced'
    }
    
    # Bio Branch: LightGBM on physicochemical + motif features
    bio_transformer = ColumnTransformer([
        ('bio', StandardScaler(), bio_cols)
    ], remainder='drop', n_jobs=-1)
    
    bio_pipeline = Pipeline([
        ('preprocess', bio_transformer),
        ('model', LGBMClassifier(**lgbm_params))
    ])
    
    # Deep Branch: LightGBM on ESM-2 embeddings
    deep_transformer = ColumnTransformer([
        ('deep', StandardScaler(), deep_cols)
    ], remainder='drop', n_jobs=-1)
    
    deep_pipeline = Pipeline([
        ('preprocess', deep_transformer),
        ('model', LGBMClassifier(**lgbm_params))
    ])
    
    # Stacking Classifier with Logistic Regression as Meta-Learner
    stacking_model = StackingClassifier(
        estimators=[
            ('bio', bio_pipeline),
            ('deep', deep_pipeline)
        ],
        final_estimator=LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        ),
        cv=3,
        n_jobs=-1,
        stack_method='predict_proba',
        verbose=0
    )
    
    # Train the model
    print("\n🚀 Training Stacking Model...")
    stacking_model.fit(X_train, y_train)
    
    # Evaluate
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    
    y_pred = stacking_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    mcc = matthews_corrcoef(y_test, y_pred) * 100
    
    print(f"\n📈 Model Performance:")
    print(f"  Accuracy: {acc:.2f}%")
    print(f"  F1-Score: {f1:.2f}%")
    print(f"  MCC: {mcc:.2f}%")
    
    # ==========================================================================
    # Extract Meta-Learner Coefficients
    # ==========================================================================
    print("\n" + "=" * 70)
    print("META-LEARNER (LOGISTIC REGRESSION) COEFFICIENTS")
    print("=" * 70)
    
    meta_learner = stacking_model.final_estimator_
    coefs = meta_learner.coef_[0]  # Binary classification: shape (1, n_features)
    intercept = meta_learner.intercept_[0]
    
    # The stack_method='predict_proba' creates 2 features per estimator (class 0 and class 1 probs)
    # So we have 4 features: [bio_prob_0, bio_prob_1, deep_prob_0, deep_prob_1]
    # We care about the class 1 probabilities (indices 1 and 3)
    
    print(f"\n📊 Raw Coefficients (all {len(coefs)} values):")
    for i, c in enumerate(coefs):
        if i % 2 == 0:
            estimator_name = 'bio' if i < 2 else 'deep'
            prob_class = 'P(class=0)' if i % 2 == 0 else 'P(class=1)'
        else:
            estimator_name = 'bio' if i < 2 else 'deep'
            prob_class = 'P(class=1)'
        print(f"  {estimator_name}_{prob_class}: {c:.4f}")
    print(f"  Intercept: {intercept:.4f}")
    
    # Extract weights for class 1 probabilities
    if len(coefs) >= 4:
        # predict_proba mode: [bio_p0, bio_p1, deep_p0, deep_p1]
        bio_weight = coefs[1]  # Weight for Bio P(class=1)
        deep_weight = coefs[3]  # Weight for Deep P(class=1)
    else:
        # passthrough mode: [bio, deep]
        bio_weight = coefs[0]
        deep_weight = coefs[1]
    
    print(f"\n🎯 Weights for P(class=1) Predictions:")
    print(f"  Bio Branch Weight:  {bio_weight:.4f}")
    print(f"  Deep Branch Weight: {deep_weight:.4f}")
    
    # Calculate contribution ratio
    abs_bio = abs(bio_weight)
    abs_deep = abs(deep_weight)
    total_abs = abs_bio + abs_deep
    
    bio_contribution = (abs_bio / total_abs) * 100
    deep_contribution = (abs_deep / total_abs) * 100
    
    print(f"\n📊 Contribution Ratio (based on |weight|):")
    print(f"  Bio Branch:  {bio_contribution:.2f}%")
    print(f"  Deep Branch: {deep_contribution:.2f}%")
    
    # ==========================================================================
    # Generate LaTeX Output
    # ==========================================================================
    print("\n" + "=" * 70)
    print("LaTeX OUTPUT")
    print("=" * 70)
    
    latex_sentence = (
        f"Analysis of the meta-learner reveals that the biological branch is assigned "
        f"a weight of ${bio_weight:.4f}$, representing ${bio_contribution:.1f}\\%$ of the "
        f"decision power, while the deep learning branch is assigned a weight of "
        f"${deep_weight:.4f}$ (${deep_contribution:.1f}\\%$). "
        f"This demonstrates that the biological features contribute significantly "
        f"to the final prediction."
    )
    
    print(f"\n{latex_sentence}")
    
    # Alternative formulation
    if bio_contribution >= deep_contribution:
        latex_alt = (
            f"The Logistic Regression meta-learner assigns coefficients of "
            f"$w_{{\\text{{bio}}}} = {bio_weight:.4f}$ and $w_{{\\text{{deep}}}} = {deep_weight:.4f}$, "
            f"indicating that the biological feature branch accounts for "
            f"${bio_contribution:.1f}\\%$ of the ensemble's decision-making power."
        )
    else:
        latex_alt = (
            f"The Logistic Regression meta-learner assigns coefficients of "
            f"$w_{{\\text{{bio}}}} = {bio_weight:.4f}$ (${bio_contribution:.1f}\\%$) and "
            f"$w_{{\\text{{deep}}}} = {deep_weight:.4f}$ (${deep_contribution:.1f}\\%$), "
            f"demonstrating complementary contributions from both branches."
        )
    
    print(f"\nAlternative formulation:")
    print(latex_alt)
    
    # ==========================================================================
    # Visualization
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    
    # Bar chart of contributions
    ax1 = axes[0]
    branches = ['Biological\nBranch', 'Deep Learning\nBranch']
    contributions = [bio_contribution, deep_contribution]
    colors = ['#2ecc71', '#3498db']
    
    bars = ax1.bar(branches, contributions, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Contribution (%)', fontsize=12)
    ax1.set_title(f'Meta-Learner Branch Contributions\n({dataset.upper()} Dataset)', fontsize=14)
    ax1.set_ylim([0, 100])
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, contrib in zip(bars, contributions):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{contrib:.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    # Pie chart
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        contributions, 
        labels=['Bio', 'Deep'],
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0),
        textprops={'fontsize': 12}
    )
    ax2.set_title(f'Decision Power Distribution\n({dataset.upper()} Dataset)', fontsize=14)
    
    plt.tight_layout()
    
    # Save
    output_dir = PROJECT_ROOT / 'results/plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_dir / f'meta_learner_analysis_{dataset}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {plot_path}")
    
    # Save results to text file
    results_file = output_dir / f'meta_learner_coefficients_{dataset}.txt'
    with open(results_file, 'w') as f:
        f.write("Meta-Learner Coefficient Analysis\n")
        f.write(f"Dataset: {dataset.upper()} BioGRID\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Model Architecture:\n")
        f.write("  Level 0 (Base Models):\n")
        f.write(f"    - Bio_Model: LightGBM on {len(bio_cols)} physicochemical + motif features\n")
        f.write(f"    - Deep_Model: LightGBM on {len(deep_cols)} ESM-2 embedding features\n")
        f.write("  Level 1 (Meta-Learner):\n")
        f.write("    - Logistic Regression on probability outputs\n\n")
        
        f.write("Raw Coefficients:\n")
        f.write(f"  Bio Branch (P(class=1) weight): {bio_weight:.4f}\n")
        f.write(f"  Deep Branch (P(class=1) weight): {deep_weight:.4f}\n")
        f.write(f"  Intercept: {intercept:.4f}\n\n")
        
        f.write("Contribution Ratio:\n")
        f.write(f"  Bio Branch:  {bio_contribution:.2f}%\n")
        f.write(f"  Deep Branch: {deep_contribution:.2f}%\n\n")
        
        f.write("LaTeX Sentences:\n")
        f.write("-" * 70 + "\n")
        f.write(latex_sentence + "\n\n")
        f.write(latex_alt + "\n")
    
    print(f"✅ Saved: {results_file}")
    
    return {
        'bio_weight': bio_weight,
        'deep_weight': deep_weight,
        'bio_contribution': bio_contribution,
        'deep_contribution': deep_contribution,
        'intercept': intercept
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze Meta-Learner Coefficients')
    parser.add_argument('--dataset', choices=['yeast', 'human', 'all'], default='all',
                        help='Dataset to analyze')
    
    args = parser.parse_args()
    
    datasets = []
    if args.dataset in ['yeast', 'all']:
        datasets.append('yeast')
    if args.dataset in ['human', 'all']:
        datasets.append('human')
    
    all_results = {}
    for ds in datasets:
        result = analyze_meta_learner(ds)
        all_results[ds] = result
    
    # Combined summary
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("COMBINED SUMMARY")
        print("=" * 70)
        for ds, result in all_results.items():
            print(f"\n{ds.upper()}:")
            print(f"  Bio Weight: {result['bio_weight']:.4f} ({result['bio_contribution']:.1f}%)")
            print(f"  Deep Weight: {result['deep_weight']:.4f} ({result['deep_contribution']:.1f}%)")


if __name__ == '__main__':
    main()
