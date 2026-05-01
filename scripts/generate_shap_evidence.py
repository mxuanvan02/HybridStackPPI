"""
SHAP Evidence Generator - Publication Quality
HybridStackPPI — Same-GO Dataset

Key fix: Feed DataFrame với tên sinh học (không phải numpy) vào ColumnTransformer
để CumulativeFeatureSelector giữ lại tên thực (Hadamard_AAC_E, Motif_LIG_SH2...)

Output (4 figures):
  figures/shap_beeswarm.png      — Top 20 features SHAP beeswarm
  figures/shap_category_bar.png  — Grouped bar: AAC / DPC / CTD / Motif / ESM2
  figures/shap_motif_detail.png  — Focus: Motif-only SHAP ranks
  figures/shap_summary.txt       — Text report for paper verification
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import joblib
import pandas as pd
import numpy as np
import h5py
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hybridstack.logger import PipelineLogger
from hybridstack.feature_engine import FeatureEngine
from hybridstack.builders import define_stacking_columns

# ══════════════════════════════════════════════════════════════════════════════
# Colour palette for categories
# ══════════════════════════════════════════════════════════════════════════════
CATEGORY_COLORS = {
    'Motif':   '#E05252',   # Đỏ — most important to highlight
    'CTD':     '#F5A623',   # Cam — physicochemical CTD
    'PAAC':    '#F5D623',   # Vàng — PseAAC
    'Moran':   '#7ED321',   # Xanh lá — autocorrelation
    'DPC':     '#4A90D9',   # Xanh dương — dipeptide
    'AAC':     '#9B59B6',   # Tím — amino-acid composition
    'Global_ESM': '#1ABC9C', # Cyan — global embedding
    'Local_ESM': '#16A085', # Xanh đậm — local motif embedding
    'Other':   '#95A5A6',
}


def classify_feature(name: str) -> str:
    """Classify a feature name into a biological category."""
    n = name.lower()
    if 'motif' in n:         return 'Motif'
    if 'ctd' in n:           return 'CTD'
    if 'paac' in n:          return 'PAAC'
    if 'moran' in n:         return 'Moran'
    if 'dpc' in n:           return 'DPC'
    if 'aac' in n:           return 'AAC'
    if 'global_esm' in n:    return 'Global_ESM'
    if 'local' in n or 'local_motif' in n: return 'Local_ESM'
    return 'Other'


def get_biological_names(total_features: int) -> list[str] | None:
    """
    Load FeatureEngine (ELM motifs from local cache) to get ordered biological names.
    Matches 'hadamard_abs' pairing strategy.
    """
    class MockEmbComp:
        embedding_dim = 1280

    h5_dummy = 'cache/human_human_pairs_same_go_facebook_esm2_t33_650m_ur50d_hadamard_abs_v3_features.h5'
    try:
        eng = FeatureEngine(h5_dummy, MockEmbComp())
        interp_cols, embed_cols = define_stacking_columns(eng, pairing_strategy='hadamard_abs')
        all_names = interp_cols + embed_cols
        print(f"  FeatureEngine names: {len(all_names)}  |  H5 total cols: {total_features}")
        return all_names
    except Exception as e:
        print(f"  [WARN] FeatureEngine naming failed: {e}")
        return None


def compute_shap_with_bio_names(model_path, h5_cache_path, n_samples=300):
    """
    Core computation:
    1. Load H5 → assign biological names
    2. Feed DataFrame (named) into ColumnTransformer so CumulativeFeatureSelector
       stores real names (not f_xxx)
    3. Compute SHAP on interp branch (LightGBM with biological features)
    """
    logger = PipelineLogger()
    logger.phase("Loading feature matrix")
    
    with h5py.File(h5_cache_path, 'r') as hf:
        total_features = hf['X_data'].shape[0], hf['X_data'].shape[1]
        n_rows, n_cols = total_features
        n_samples = min(n_samples, n_rows)
        X_raw = hf['X_data'][-n_samples:].astype(np.float32)
        y_raw = hf['y_data'][-n_samples:]

    print(f"  Loaded {n_samples} samples × {n_cols} features")

    # Gán tên sinh học trực tiếp
    bio_names = get_biological_names(n_cols)
    if bio_names and len(bio_names) == n_cols:
        X_df = pd.DataFrame(X_raw, columns=bio_names)
        print(f"  ✅ Biological names assigned ({len(bio_names)} names)")
    else:
        # Fallback: vẫn dùng generic nhưng log cảnh báo
        print(f"  ⚠️  Name count mismatch ({len(bio_names) if bio_names else 0} vs {n_cols}). Using generic.")
        X_df = pd.DataFrame(X_raw, columns=[f"F_{i}" for i in range(n_cols)])

    logger.phase("Loading stacking model → Interp branch")
    model = joblib.load(model_path)
    interp_pipeline = model.estimators_[0]  # Interp branch
    col_transformer = interp_pipeline.named_steps['preprocessor']
    lgbm_model = interp_pipeline.named_steps['model']

    # CRITICAL FIX: Model was trained with numpy arrays, so CumulativeFeatureSelector
    # stores selected_features_ as ["f_0","f_1",...] (generic).
    # Feed raw numpy → get correct output → THEN map f_N → bio_names[N] post-hoc.
    logger.phase("Transforming via ColumnTransformer (numpy → post-hoc bio name mapping)")
    X_transformed = col_transformer.transform(X_raw)

    _, transformer, input_cols_idx = col_transformer.transformers_[0]

    if hasattr(transformer, 'selected_features_') and transformer.selected_features_:
        f_xxx_names = transformer.selected_features_   # e.g. ['f_1871', 'f_848', ...]
        print(f"  Selector: {len(f_xxx_names)} selected features  |  sample: {f_xxx_names[:3]}")

        # Parse "f_N" → integer N → bio_names[N]
        final_feature_names = []
        for fname in f_xxx_names:
            try:
                idx = int(fname.split('_')[1])
                final_feature_names.append(bio_names[idx] if bio_names and idx < len(bio_names) else fname)
            except (IndexError, ValueError):
                final_feature_names.append(fname)
    else:
        n_out = X_transformed.shape[1] if hasattr(X_transformed, 'shape') else 0
        final_feature_names = [f"F_{i}" for i in range(n_out)]
        print("  ⚠️ No selected_features_ found.")

    # Build final DataFrame
    if isinstance(X_transformed, pd.DataFrame):
        X_final = X_transformed.copy()
        X_final.columns = final_feature_names[:X_final.shape[1]]
    else:
        n_out = X_transformed.shape[1]
        names_out = final_feature_names[:n_out]
        if len(names_out) < n_out:
            names_out += [f"F_{i}" for i in range(len(names_out), n_out)]
        X_final = pd.DataFrame(X_transformed, columns=names_out)

    bio_cols = [c for c in X_final.columns if not c.startswith('f_') and not c.startswith('F_')]
    print(f"  ✅ Transformed: {X_final.shape}  |  Bio-named: {len(bio_cols)}/{X_final.shape[1]}")
    print(f"  Sample bio names: {bio_cols[:4]}")

    logger.phase("Computing SHAP values (TreeExplainer)")
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values_raw = explainer.shap_values(X_final)


    # Xử lý shape SHAP tương thích nhiều version
    if isinstance(shap_values_raw, list) and len(shap_values_raw) > 1:
        sv = shap_values_raw[1]
    elif hasattr(shap_values_raw, 'shape') and len(shap_values_raw.shape) == 3:
        sv = shap_values_raw[:, :, 1]
    elif hasattr(shap_values_raw, 'values') and len(shap_values_raw.values.shape) == 3:
        sv = shap_values_raw.values[:, :, 1]
    else:
        sv = shap_values_raw

    return X_final, sv, y_raw


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — SHAP Beeswarm (Top 20 features)
# ══════════════════════════════════════════════════════════════════════════════
def plot_beeswarm(X_final, sv, out_path):
    """Publication-quality SHAP beeswarm with colour-coded category labels."""
    plt.figure(figsize=(12, 9))
    shap.summary_plot(
        sv, X_final,
        max_display=20,
        show=False,
        plot_type='dot',
        color_bar_label='Feature Value'
    )
    ax = plt.gca()
    ax.set_title('SHAP Feature Importance — HybridStack-PPI Interpretable Branch (Same-GO)',
                 fontsize=13, fontweight='bold', pad=15)
    
    # Colour the y-axis labels by category
    ylabels = [t.get_text() for t in ax.get_yticklabels()]
    for tick, label in zip(ax.get_yticklabels(), ylabels):
        cat = classify_feature(label)
        tick.set_color(CATEGORY_COLORS.get(cat, '#333333'))
        tick.set_fontweight('bold' if cat == 'Motif' else 'normal')

    # Legend patches
    present_cats = list(dict.fromkeys(classify_feature(c) for c in X_final.columns
                        if c in X_final.columns[:20]))
    patches = [Patch(color=CATEGORY_COLORS.get(c, '#ccc'), label=c) for c in present_cats]
    ax.legend(handles=patches, loc='lower right', fontsize=9, title='Feature Category',
              framealpha=0.9, title_fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved beeswarm → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Category-level grouped bar
# ══════════════════════════════════════════════════════════════════════════════
def plot_category_bar(X_final, sv, out_path):
    """Aggregate SHAP by feature category to give the big-picture view."""
    mean_abs = np.abs(sv).mean(axis=0)
    df = pd.DataFrame({'Feature': X_final.columns, 'SHAP': mean_abs})
    df['Category'] = df['Feature'].apply(classify_feature)
    cat_df = df.groupby('Category')['SHAP'].sum().sort_values(ascending=False).reset_index()
    cat_df['Color'] = cat_df['Category'].map(CATEGORY_COLORS)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(cat_df['Category'], cat_df['SHAP'], color=cat_df['Color'],
                  edgecolor='white', linewidth=1.2, zorder=3)
    ax.bar_label(bars, fmt='%.4f', fontsize=10, padding=3)
    ax.set_ylabel('Sum of Mean |SHAP| Value', fontsize=12)
    ax.set_xlabel('Feature Category', fontsize=12)
    ax.set_title('Feature Category Contribution — Interp Branch (Same-GO)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_facecolor('#f9f9f9')
    fig.patch.set_facecolor('white')

    # Annotate the Motif bar specially
    motif_val = cat_df[cat_df['Category'] == 'Motif']['SHAP'].values
    if len(motif_val) > 0:
        ax.annotate('← Biological\n   Motif Signal',
                    xy=(cat_df[cat_df['Category'] == 'Motif'].index[0], motif_val[0]),
                    xytext=(cat_df[cat_df['Category'] == 'Motif'].index[0] + 0.5, motif_val[0] * 1.3),
                    fontsize=9, color='#E05252', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#E05252'))

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved category bar → {out_path}")
    return cat_df


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Motif-only detail bar
# ══════════════════════════════════════════════════════════════════════════════
def plot_motif_detail(X_final, sv, out_path, top_n=15):
    """Show only ELM Motif features ranked by SHAP to prove motif contribution."""
    mean_abs = np.abs(sv).mean(axis=0)
    df = pd.DataFrame({'Feature': X_final.columns, 'SHAP': mean_abs})
    df['Category'] = df['Feature'].apply(classify_feature)
    motif_df = df[df['Category'] == 'Motif'].sort_values('SHAP', ascending=True).tail(top_n)

    if motif_df.empty:
        print("  ⚠️  No Motif features found in selected columns — skipping motif detail plot.")
        return

    # Clean names for readability
    clean_names = [n.replace('Hadamard_', 'Had_').replace('AbsDiff_', 'Diff_')
                   .replace('Motif_', '') for n in motif_df['Feature']]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(clean_names, motif_df['SHAP'], color='#E05252', edgecolor='white',
                   linewidth=0.8, alpha=0.85, zorder=3)
    ax.bar_label(bars, fmt='%.5f', fontsize=9, padding=3)
    ax.set_xlabel('Mean |SHAP| Value', fontsize=12)
    ax.set_title(f'Top {top_n} ELM Motif Features — SHAP Contribution (Same-GO)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, zorder=0)
    ax.set_facecolor('#fff5f5')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved motif detail → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# TEXT SUMMARY for paper
# ══════════════════════════════════════════════════════════════════════════════
def save_text_summary(X_final, sv, cat_df, out_path):
    mean_abs = np.abs(sv).mean(axis=0)
    df = pd.DataFrame({'Feature': X_final.columns, 'SHAP': mean_abs})
    df['Category'] = df['Feature'].apply(classify_feature)
    top20 = df.nlargest(20, 'SHAP')

    has_motif      = 'Motif' in top20['Category'].values
    has_ctd        = 'CTD' in top20['Category'].values
    has_dpc        = 'DPC' in top20['Category'].values
    has_aac        = 'AAC' in top20['Category'].values
    has_esm        = any(c in top20['Category'].values for c in ['Global_ESM', 'Local_ESM'])

    with open(out_path, 'w') as f:
        f.write("SHAP Evidence Report — HybridStack-PPI (Same-GO, Human)\n")
        f.write("="*70 + "\n\n")
        f.write("TOP 20 FEATURES (Biological Names):\n")
        f.write("-"*70 + "\n")
        for i, row in top20.iterrows():
            f.write(f"  {df.index.get_loc(i)+1:2d}. [{row['Category']:10s}] {row['Feature']:<55s} SHAP={row['SHAP']:.5f}\n")
        f.write("\nFEATURE CATEGORY SUMMARY (Total SHAP contribution):\n")
        f.write("-"*70 + "\n")
        for _, row in cat_df.iterrows():
            f.write(f"  {row['Category']:<12s}: {row['SHAP']:.5f}\n")
        f.write("\nBIOLOGICAL EVIDENCE ANALYSIS:\n")
        f.write("-"*70 + "\n")
        f.write(f"  Motif (ELM SLiMs) in Top 20: {'✅ YES' if has_motif else '❌ NO'}\n")
        f.write(f"  CTD / Physicochemical in Top 20: {'✅ YES' if has_ctd else '❌ NO'}\n")
        f.write(f"  DPC (Dipeptide pairs) in Top 20: {'✅ YES' if has_dpc else '❌ NO'}\n")
        f.write(f"  AAC (Amino acid comp.) in Top 20: {'✅ YES' if has_aac else '❌ NO'}\n")
        f.write(f"  ESM-2 Embeddings in Top 20: {'✅ YES' if has_esm else '❌ NO'}\n")
    print(f"  ✅ Saved summary → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    logger = PipelineLogger()
    logger.header("🔬 SHAP EVIDENCE GENERATOR — PUBLICATION QUALITY 🔬")

    out_dir = 'figures'
    os.makedirs(out_dir, exist_ok=True)

    datasets = {
        'Human': {
            'model_path': 'results/human_same_go/models/model_fold1.joblib',
            'h5_cache_path': 'cache/human_human_pairs_same_go_facebook_esm2_t33_650m_ur50d_hadamard_abs_v3_features.h5'
        },
        'Yeast': {
            'model_path': 'results/yeast_same_go/models/model_fold1.joblib',
            'h5_cache_path': 'cache/yeast_yeast_pairs_same_go_facebook_esm2_t33_650m_ur50d_hadamard_abs_v3_features.h5'
        }
    }

    import shutil

    for ds_name, paths in datasets.items():
        logger.header(f"▶ Processing Dataset: {ds_name.upper()} (Same-GO)")
        
        # Compute
        X_final, sv, y = compute_shap_with_bio_names(
            paths['model_path'], 
            paths['h5_cache_path'], 
            n_samples=300
        )

        prefix = ds_name.lower()

        # Plot 1: Beeswarm
        logger.phase(f"[{ds_name}] Generating Plot 1: SHAP Beeswarm")
        plot_beeswarm(X_final, sv, f'{out_dir}/shap_beeswarm_{prefix}.png')

        # Plot 2: Category bar
        logger.phase(f"[{ds_name}] Generating Plot 2: Feature Category Bar")
        cat_df = plot_category_bar(X_final, sv, f'{out_dir}/shap_category_bar_{prefix}.png')

        # Plot 3: Motif detail
        logger.phase(f"[{ds_name}] Generating Plot 3: Motif Detail")
        plot_motif_detail(X_final, sv, f'{out_dir}/shap_motif_detail_{prefix}.png')

        # Text summary
        logger.phase(f"[{ds_name}] Saving Text Summary")
        save_text_summary(X_final, sv, cat_df, f'{out_dir}/shap_summary_{prefix}.txt')

        # Backwards compatibility alias for Human
        if ds_name == 'Human':
            shutil.copy(f'{out_dir}/shap_beeswarm_human.png', f'{out_dir}/shap_beeswarm.png')

    logger.header("✅ ALL SHAP FIGURES GENERATED SUCCESSFULLY")
    print(f"\nOutput files generated in {out_dir}/")


if __name__ == "__main__":
    main()
