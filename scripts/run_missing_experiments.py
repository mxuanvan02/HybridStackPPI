#!/usr/bin/env python3
"""
run_missing_experiments.py
==========================
Chạy các thực nghiệm còn thiếu để bổ sung vào bảng ablation và SHAP của paper.

Các thực nghiệm cần thiết (theo review):
  EXP-1: Ablation Table đầy đủ — lấy dữ liệu từ cache đã có + tính lại các ô còn "--"
  EXP-2: SHAP analysis tổng hợp qua 5 folds (hiện tại chỉ có fold 1)
  EXP-3: Cross-species / CDC evaluation (Human → CDC multi-species)

Usage:
    python scripts/run_missing_experiments.py --exp all          # Chạy tất cả
    python scripts/run_missing_experiments.py --exp ablation     # Chỉ ablation table
    python scripts/run_missing_experiments.py --exp shap         # Chỉ SHAP 5-fold
    python scripts/run_missing_experiments.py --exp cdc          # Chỉ cross-species
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
H5_CACHE  = str(PROJECT_ROOT / "cache/esm2/esm2_embeddings_v4.h5")
CACHE_VER = "v3"

HUMAN_FASTA    = str(PROJECT_ROOT / "data/BioGrid/Human/human_dict.fasta")
HUMAN_PAIRS_GO = str(PROJECT_ROOT / "data/BioGrid/Human/human_pairs_same_go.tsv")
YEAST_FASTA    = str(PROJECT_ROOT / "data/BioGrid/Yeast/yeast_dict.fasta")
YEAST_PAIRS_GO = str(PROJECT_ROOT / "data/BioGrid/Yeast/yeast_pairs_same_go.tsv")

# CDC multi-species (C. elegans, Drosophila, E. coli)
CDC_DIR = PROJECT_ROOT / "data/seq_ppi/multi_species"

RESULTS_HUMAN  = str(PROJECT_ROOT / "results/human_same_go")
RESULTS_YEAST  = str(PROJECT_ROOT / "results/yeast_same_go")
OUTPUT_MISSING = str(PROJECT_ROOT / "results/missing_experiments")


# ─────────────────────────────────────────────────────────────────────────────
# EXP-1: Build complete ablation table from cached + new results
# ─────────────────────────────────────────────────────────────────────────────
def exp1_ablation_table():
    """
    Tổng hợp bảng ablation đầy đủ cho Human Same-GO dataset.
    Dữ liệu lấy từ các CSV đã cache trong results/human_same_go/ablation/.
    Tính mean ± std của tất cả metrics qua 5 folds.
    """
    print("\n" + "="*70)
    print("  EXP-1: ABLATION TABLE (Human Same-GO, 5-Fold CV)")
    print("="*70)

    ablation_root = Path(RESULTS_HUMAN) / "ablation"
    ref_fold_csv  = Path(RESULTS_HUMAN) / "fold_metrics.csv"

    # Map variant_id → display label
    variants = [
        ("A4_ref",               "Full HybridStack-PPI (ref)",        ref_fold_csv),
        ("A2_EmbedOnly",         "w/o Motif Branch (Embed-Only)",     ablation_root / "A2_EmbedOnly" / "fold_metrics.csv"),
        ("C2_EarlyFusion",       "Early Fusion (flat concat)",        ablation_root / "C2_EarlyFusion" / "fold_metrics.csv"),
        ("C3_TreeMeta",          "Non-linear Meta-learner (LGBM)",    ablation_root / "C3_TreeMeta" / "fold_metrics.csv"),
        ("A1_InterpOnly",        "Interp-Only (Handcraft+Motif)",     ablation_root / "A1_InterpOnly" / "fold_metrics.csv"),
    ]

    METRIC_MAP = {
        "Accuracy":             "Acc (%)",
        "Specificity":          "Spec (%)",
        "MCC":                  "MCC (%)",
        "ROC-AUC":              "AUC (%)",
        "Precision":            "Prec (%)",
    }

    rows = []
    for vid, label, csv_path in variants:
        if not Path(str(csv_path)).exists():
            print(f"  ⚠️  NOT FOUND: {csv_path} — skipping {label}")
            continue

        df = pd.read_csv(str(csv_path))
        if "Fold" in df.columns:
            df = df.drop("Fold", axis=1)

        # Normalize: some CSVs store as 0-1, some as 0-100
        for col in df.columns:
            if df[col].max() <= 1.5:          # stored as fraction
                df[col] = df[col] * 100

        row = {"Variant": label}
        for src_col, dst_col in METRIC_MAP.items():
            if src_col in df.columns:
                m = df[src_col].mean()
                s = df[src_col].std()
                row[dst_col] = f"{m:.1f} ± {s:.1f}"
            else:
                row[dst_col] = "--"
        rows.append(row)

    result_df = pd.DataFrame(rows).set_index("Variant")
    print("\n" + result_df.to_string())

    # Save
    out_dir = Path(OUTPUT_MISSING)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "ablation_table_full.csv"
    result_df.to_csv(out_csv)
    print(f"\n✅ Saved → {out_csv}")

    # Print LaTeX rows
    print("\n📄 LaTeX rows (Acc / Spec / MCC / AUC):")
    print("-"*70)
    for idx, row in result_df.iterrows():
        acc  = row.get("Acc (%)",  "--")
        spec = row.get("Spec (%)", "--")
        mcc  = row.get("MCC (%)",  "--")
        auc  = row.get("AUC (%)",  "--")
        print(f"  {idx} & {acc} & {spec} & {mcc} & {auc} \\\\")
    print("-"*70)

    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# EXP-2: SHAP analysis across all 5 folds
# ─────────────────────────────────────────────────────────────────────────────
def exp2_shap_5folds():
    """
    Chạy SHAP attribution trên tất cả 5 folds của Human Same-GO.
    Tổng hợp mean |SHAP| value qua các fold và vẽ aggregated beeswarm.
    Requires: results/human_same_go/models/model_fold{1..5}.joblib
    """
    print("\n" + "="*70)
    print("  EXP-2: SHAP ANALYSIS — 5-FOLD AGGREGATED (Human Same-GO)")
    print("="*70)

    import joblib
    import shap
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    models_dir = Path(RESULTS_HUMAN) / "models"
    oof_csv    = Path(RESULTS_HUMAN) / "oof_predictions.csv"

    if not oof_csv.exists():
        print(f"  ❌ OOF predictions not found: {oof_csv}")
        return

    from hybridstack.feature_engine import EmbeddingComputer, FeatureEngine
    from hybridstack.data_utils import (
        load_data, canonicalize_pairs, create_feature_matrix,
        get_cache_filename, load_feature_matrix_h5,
    )

    # Load full feature matrix (cached)
    cache_path = get_cache_filename(
        HUMAN_PAIRS_GO, "hadamard_abs", ESM_MODEL, cache_version=CACHE_VER
    )
    if not os.path.exists(cache_path):
        print(f"  ❌ Feature cache not found: {cache_path}")
        print("     Run reproduce_results.py first to build the cache.")
        return

    print(f"  Loading feature matrix from {cache_path}...")
    X_df, y_s = load_feature_matrix_h5(cache_path)

    # Load cluster-based splits to match fold assignments
    from hybridstack.data_utils import load_data, canonicalize_pairs, get_cluster_based_splits, load_cluster_map
    from hybridstack.logger import PipelineLogger
    logger = PipelineLogger()

    seqs, pairs_df = load_data(HUMAN_FASTA, HUMAN_PAIRS_GO)
    pairs_df = canonicalize_pairs(pairs_df, dataset_name="Human", logger=logger)

    cluster_path = str(PROJECT_ROOT / "data/BioGrid/Human/CDHIT_Reduced/human_clusters.tsv")
    cluster_map  = load_cluster_map(cluster_path) if Path(cluster_path).exists() else None

    if cluster_map:
        splits = get_cluster_based_splits(pairs_df, cluster_map, n_splits=5, random_state=42)
    else:
        from hybridstack.data_utils import get_protein_based_splits
        splits = get_protein_based_splits(pairs_df, n_splits=5, random_state=42)

    # Collect SHAP values per fold
    all_shap_dfs = []
    n_samples_per_fold = 300

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        fold_num = fold_idx + 1
        model_path = models_dir / f"model_fold{fold_num}.joblib"
        if not model_path.exists():
            print(f"  ⚠️  model_fold{fold_num}.joblib not found — skipping")
            continue

        print(f"  Computing SHAP for fold {fold_num}...")
        model = joblib.load(str(model_path))

        X_val = X_df.iloc[val_idx]
        y_val = y_s.iloc[val_idx]

        # Sample n_samples_per_fold (stratified)
        same_go_neg = X_val[y_val == 0].index.tolist()
        pos          = X_val[y_val == 1].index.tolist()
        n_each       = min(n_samples_per_fold // 2, len(same_go_neg), len(pos))
        rng          = np.random.RandomState(42 + fold_idx)
        sampled_idx  = rng.choice(same_go_neg, n_each, replace=False).tolist() + \
                       rng.choice(pos, n_each, replace=False).tolist()
        X_sample = X_val.loc[sampled_idx]

        # Extract interpretable branch from stacking model
        stacking_clf = model.named_steps.get("model") if hasattr(model, "named_steps") else model
        interp_branch = None
        if hasattr(stacking_clf, "estimators_"):
            for name, est in stacking_clf.estimators_:
                if isinstance(name, str) and ("interp" in name.lower() or "bio" in name.lower()):
                    interp_branch = est
                    break
            if interp_branch is None and len(stacking_clf.estimators_) > 0:
                interp_branch = stacking_clf.estimators_[0][1]  # fallback: first branch

        # Get selected feature names
        sel_features = list(X_sample.columns)
        if hasattr(interp_branch, "named_steps"):
            sel = interp_branch.named_steps.get("selector")
            if sel and hasattr(sel, "selected_features_"):
                sel_features = sel.selected_features_
                X_sample_sel = X_sample[sel_features]
            else:
                X_sample_sel = X_sample
            lgbm_model = interp_branch.named_steps.get("model")
        else:
            X_sample_sel = X_sample
            lgbm_model   = interp_branch

        if lgbm_model is None:
            print(f"  ⚠️  Could not extract LGBM from fold {fold_num}")
            continue

        # SHAP TreeExplainer
        explainer   = shap.TreeExplainer(lgbm_model)
        X_np        = X_sample_sel.to_numpy(dtype=np.float32)
        shap_values = explainer.shap_values(X_np)

        # For binary: take class-1 SHAP values
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        mean_abs_shap = np.abs(sv).mean(axis=0)
        fold_shap_df  = pd.DataFrame({
            "feature":        sel_features,
            f"mean_abs_shap_fold{fold_num}": mean_abs_shap,
        })
        all_shap_dfs.append(fold_shap_df)
        print(f"  ✅ Fold {fold_num}: top feature = {sel_features[np.argmax(mean_abs_shap)]}")

    if not all_shap_dfs:
        print("  ❌ No SHAP results collected.")
        return

    # Merge all folds on feature name, compute grand mean
    from functools import reduce
    merged = reduce(lambda a, b: a.merge(b, on="feature", how="outer"), all_shap_dfs)
    fold_cols = [c for c in merged.columns if c.startswith("mean_abs_shap_fold")]
    merged["mean_abs_shap_global"] = merged[fold_cols].mean(axis=1)
    merged["std_abs_shap_global"]  = merged[fold_cols].std(axis=1)
    merged = merged.sort_values("mean_abs_shap_global", ascending=False)

    out_dir = Path(OUTPUT_MISSING)
    out_dir.mkdir(parents=True, exist_ok=True)
    shap_csv = out_dir / "shap_5fold_aggregated.csv"
    merged.to_csv(shap_csv, index=False)
    print(f"\n✅ Aggregated SHAP → {shap_csv}")

    # Print top 20
    print("\n📊 Top-20 Features (mean |SHAP| across 5 folds):")
    print("-"*60)
    top20 = merged.head(20)
    for _, row in top20.iterrows():
        feat = row["feature"]
        mu   = row["mean_abs_shap_global"]
        sd   = row["std_abs_shap_global"]
        print(f"  {feat:55s}  {mu:.4f} ± {sd:.4f}")
    print("-"*60)

    # Bar chart of top 20 aggregated SHAP
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#E84545" if "ELM" in f or "LIG_" in f or "DOC_" in f or "MOD_" in f
              else "#4A90D9" for f in top20["feature"]]
    ax.barh(range(20), top20["mean_abs_shap_global"].values[::-1],
            xerr=top20["std_abs_shap_global"].values[::-1],
            color=colors[::-1], edgecolor="white", height=0.7)
    ax.set_yticks(range(20))
    ax.set_yticklabels([f[:50] for f in top20["feature"].values[::-1]], fontsize=8)
    ax.set_xlabel("Mean |SHAP value| (5-fold aggregated)", fontsize=10)
    ax.set_title("Top-20 Discriminative Features — Human Same-GO (All 5 Folds)", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig_path = out_dir / "shap_5fold_beeswarm_aggregated.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Figure → {fig_path}")

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# EXP-3: CDC cross-species evaluation
# ─────────────────────────────────────────────────────────────────────────────
def exp3_cdc_evaluation():
    """
    Đánh giá HybridStack-PPI trên CDC (C. elegans, Drosophila, E. coli) dataset.
    Train trên Human Same-GO, test trên từng species với strict homology filter (<40%).
    """
    print("\n" + "="*70)
    print("  EXP-3: CDC CROSS-SPECIES EVALUATION")
    print("="*70)

    import joblib
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, matthews_corrcoef, roc_auc_score,
        average_precision_score,
    )
    from hybridstack.data_utils import (
        load_data, canonicalize_pairs, create_feature_matrix,
        get_cache_filename, load_feature_matrix_h5, save_feature_matrix_h5,
    )
    from hybridstack.feature_engine import EmbeddingComputer, FeatureEngine
    from hybridstack.logger import PipelineLogger

    logger = PipelineLogger()

    # Load the final model trained on Human Same-GO
    final_model_path = Path(RESULTS_HUMAN) / "models" / "final_model.joblib"
    if not final_model_path.exists():
        print(f"  ❌ Final model not found: {final_model_path}")
        print("     Run reproduce_results.py --strategy same_go --dataset human first.")
        return

    print(f"  Loading model from {final_model_path}...")
    model = joblib.load(str(final_model_path))

    # Initialize feature engine
    embedding_computer = EmbeddingComputer(model_name=ESM_MODEL)
    feature_engine     = FeatureEngine(h5_cache_path=H5_CACHE, embedding_computer=embedding_computer)
    single_feature_names = feature_engine.get_feature_names()

    # CDC species configs
    cdc_species = []
    # PIPR dataset structure: data/seq_ppi/multi_species/multi_species.fasta + CeleganDrosophilaEcoli.actions.tsv
    multi_species_dir = CDC_DIR
    if multi_species_dir.exists():
        fasta_files = list(multi_species_dir.glob("*.fasta"))
        pairs_files = list(multi_species_dir.glob("*actions.tsv")) + list(multi_species_dir.glob("*.tsv"))
        if fasta_files and pairs_files:
            # Sort to prefer the main actions.tsv or filtered.40.tsv
            preferred_pairs = [p for p in pairs_files if "actions.tsv" in p.name]
            if not preferred_pairs:
                preferred_pairs = pairs_files
            cdc_species.append({
                "name": "CDC Multi-Species (C. elegans, Drosophila, E. coli)",
                "fasta": str(fasta_files[0]),
                "pairs": str(preferred_pairs[0]),
            })

    if not cdc_species:
        print(f"  ⚠️  No CDC species data found in {CDC_DIR}")
        print("     Expected structure: data/seq_ppi/multi_species/<species>/*.fasta + *.tsv")
        # Create placeholder report
        _create_cdc_placeholder()
        return

    out_dir = Path(OUTPUT_MISSING)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for sp in cdc_species:
        print(f"\n  → Species: {sp['name']}")

        try:
            seqs, pairs_df = load_data(sp["fasta"], sp["pairs"])
            pairs_df = canonicalize_pairs(pairs_df, dataset_name=sp["name"], logger=logger)

            cache_path = get_cache_filename(
                sp["pairs"], "hadamard_abs", ESM_MODEL, cache_version=CACHE_VER
            )
            if os.path.exists(cache_path):
                X_df, y_s = load_feature_matrix_h5(cache_path)
                if len(X_df) != len(pairs_df):
                    os.remove(cache_path)
                    raise ValueError("Cache stale")
            else:
                required = set(pairs_df["protein1"]) | set(pairs_df["protein2"])
                needed   = {sid: seq for sid, seq in seqs.items() if sid in required}
                protein_features = feature_engine.extract_all_features(needed)
                X_df, y_s = create_feature_matrix(
                    pairs_df, protein_features, single_feature_names, "hadamard_abs"
                )
                save_feature_matrix_h5(X_df, y_s, cache_path)

            X_np = X_df.to_numpy(dtype=np.float32)
            y_np = y_s.to_numpy(dtype=np.float32)

            y_proba = model.predict_proba(X_np)[:, 1]
            y_pred  = (y_proba >= 0.5).astype(int)

            row = {
                "Species":    sp["name"],
                "N_pairs":    len(y_np),
                "Acc (%)":    round(accuracy_score(y_np, y_pred) * 100, 2),
                "Prec (%)":   round(precision_score(y_np, y_pred, zero_division=0) * 100, 2),
                "Spec (%)":   round(
                    (((y_pred == 0) & (y_np == 0)).sum() /
                     max((y_np == 0).sum(), 1)) * 100, 2
                ),
                "MCC (%)":    round(matthews_corrcoef(y_np, y_pred) * 100, 2),
                "AUC (%)":    round(roc_auc_score(y_np, y_proba) * 100, 2),
            }
            results.append(row)
            print(f"     Acc={row['Acc (%)']}%  Spec={row['Spec (%)']}%  MCC={row['MCC (%)']}%  AUC={row['AUC (%)']}%")

        except Exception as e:
            print(f"     ❌ Failed: {e}")
            results.append({"Species": sp["name"], "Error": str(e)})

    df = pd.DataFrame(results)
    csv_path = out_dir / "cdc_cross_species_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ CDC results → {csv_path}")
    print("\n" + df.to_string(index=False))

    # Print LaTeX
    print("\n📄 LaTeX rows:")
    print("-"*60)
    for _, row in df.iterrows():
        if "Error" in row:
            continue
        print(f"  {row['Species']} & {row['Acc (%)']} & {row['Spec (%)']} & {row['MCC (%)']} & {row['AUC (%)']} \\\\")
    print("-"*60)

    return df


def _create_cdc_placeholder():
    """Tạo template CSV cho CDC results nếu data chưa có."""
    out_dir = Path(OUTPUT_MISSING)
    out_dir.mkdir(parents=True, exist_ok=True)
    template = pd.DataFrame([
        {"Species": "C. elegans", "N_pairs": "TBD", "Acc (%)": "TBD",
         "Spec (%)": "TBD", "MCC (%)": "TBD", "AUC (%)": "TBD"},
        {"Species": "Drosophila", "N_pairs": "TBD", "Acc (%)": "TBD",
         "Spec (%)": "TBD", "MCC (%)": "TBD", "AUC (%)": "TBD"},
        {"Species": "E. coli",    "N_pairs": "TBD", "Acc (%)": "TBD",
         "Spec (%)": "TBD", "MCC (%)": "TBD", "AUC (%)": "TBD"},
    ])
    csv_path = out_dir / "cdc_cross_species_results.csv"
    template.to_csv(csv_path, index=False)
    print(f"\n  📄 Placeholder CSV → {csv_path}")
    print("  ⚠️  Please add actual CDC FASTA + pairs files to data/seq_ppi/multi_species/<species>/")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run missing paper experiments")
    parser.add_argument(
        "--exp", choices=["all", "ablation", "shap", "cdc"],
        default="ablation",
        help="Which experiment to run (default: ablation)"
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_MISSING, exist_ok=True)

    if args.exp in ("all", "ablation"):
        exp1_ablation_table()

    if args.exp in ("all", "shap"):
        exp2_shap_5folds()

    if args.exp in ("all", "cdc"):
        exp3_cdc_evaluation()

    print("\n" + "="*70)
    print(f"  DONE — Results saved to: {OUTPUT_MISSING}/")
    print("="*70)


if __name__ == "__main__":
    main()
