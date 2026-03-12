#!/usr/bin/env python
"""
run_baselines.py — SOTA Comparison Baselines cho HybridStackPPI.

Mỗi baseline là một PHƯƠNG PHÁP HOÀN CHỈNH (feature extraction + classifier),
khác biệt hoàn toàn về cách trích xuất đặc trưng so với HybridStackPPI.

Phase 1 (Classical Sequence-Based):
  1. Conjoint Triad (CT)   — Shen et al. 2007, PNAS
  2. Auto Covariance (AC)  — Guo et al. 2008, BMC Bioinformatics

Tất cả baseline chạy trên CÙNG dataset, CÙNG fold splits, CÙNG negative strategy
để đảm bảo so sánh công bằng.

Usage:
    python scripts/run_baselines.py --dataset both --strategy same_compartment
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hybridstack.data_utils import (
    canonicalize_pairs,
    get_protein_based_splits,
    load_data,
)
from hybridstack.logger import PipelineLogger
from scripts.run import set_seed


# ══════════════════════════════════════════════════════════════
# FEATURE EXTRACTORS — mỗi hàm là một phương pháp HOÀN CHỈNH
# ══════════════════════════════════════════════════════════════

# ── Conjoint Triad (CT) ──────────────────────────────────────
# Shen et al., "Predicting protein-protein interactions based
# only on sequences information", PNAS 2007.
# 20 amino acids → 7 classes → triad (3-gram) frequency = 343-dim

_CT_CLASS_MAP = {}
_CT_GROUPS = [
    "AGV",       # Class 0: Small hydrophobic
    "ILFP",      # Class 1: Hydrophobic
    "YMTS",      # Class 2: Polar/small hydroxyl
    "HNQW",      # Class 3: Polar/aromatic
    "RK",        # Class 4: Positive charged
    "DE",        # Class 5: Negative charged
    "C",         # Class 6: Cysteine (special)
]
for _cls_id, _group in enumerate(_CT_GROUPS):
    for _aa in _group:
        _CT_CLASS_MAP[_aa] = _cls_id


def _seq_to_ct_classes(seq: str) -> list:
    """Convert amino acid sequence to class indices (0-6)."""
    return [_CT_CLASS_MAP.get(aa.upper(), 0) for aa in seq]


def conjoint_triad_descriptor(seq: str) -> np.ndarray:
    """
    Compute 343-dimensional Conjoint Triad descriptor for a protein sequence.
    Each dimension = frequency of a specific triad (3-gram of 7 classes).
    """
    classes = _seq_to_ct_classes(seq)
    vec = np.zeros(343, dtype=np.float64)
    for i in range(len(classes) - 2):
        idx = classes[i] * 49 + classes[i + 1] * 7 + classes[i + 2]
        vec[idx] += 1
    # Normalize by sequence length
    total = max(len(classes) - 2, 1)
    vec /= total
    return vec


# ── Auto Covariance (AC) ─────────────────────────────────────
# Guo et al., "Using support vector machine combined with auto
# covariance to predict protein-protein interactions from protein
# sequences", Nucleic Acids Research 2008.
# 7 physicochemical properties × lag positions = 7 * lag dimensions

# Normalized Amino Acid Physicochemical Properties
# Source: AAindex (Kawashima & Kanehisa, 2000)
_AA_PROPERTIES = {
    # (Hydrophobicity, Hydrophilicity, Net Charge, Polarity, Polarizability, SASA, Volume)
    "A": ( 0.62, -0.50,  0.0,  0.0,  0.046,  1.18,  0.167),
    "R": (-2.53,  3.00,  1.0,  1.0,  0.291,  2.56,  0.596),
    "N": (-0.78,  0.20,  0.0,  1.0,  0.134,  1.66,  0.315),
    "D": (-0.90,  3.00, -1.0,  1.0,  0.105,  1.59,  0.295),
    "C": ( 0.29, -1.00,  0.0,  0.0,  0.128,  1.40,  0.257),
    "Q": (-0.85,  0.20,  0.0,  1.0,  0.180,  1.93,  0.407),
    "E": (-0.74,  3.00, -1.0,  1.0,  0.151,  1.86,  0.397),
    "G": ( 0.48, -0.50,  0.0,  0.0,  0.000,  0.88,  0.000),
    "H": (-0.40, -0.50,  0.5,  1.0,  0.230,  2.02,  0.457),
    "I": ( 1.38, -1.80,  0.0,  0.0,  0.186,  1.82,  0.394),
    "L": ( 1.06, -1.80,  0.0,  0.0,  0.186,  1.80,  0.394),
    "K": (-1.50,  3.00,  1.0,  1.0,  0.219,  2.26,  0.523),
    "M": ( 0.64, -1.30,  0.0,  0.0,  0.221,  2.04,  0.429),
    "F": ( 1.19, -2.50,  0.0,  0.0,  0.290,  2.18,  0.560),
    "P": ( 0.12, -1.40,  0.0,  0.0,  0.131,  1.47,  0.305),
    "S": (-0.18, -0.04,  0.0,  1.0,  0.062,  1.33,  0.198),
    "T": (-0.05, -0.70,  0.0,  1.0,  0.108,  1.53,  0.268),
    "W": ( 0.81, -3.40,  0.0,  1.0,  0.409,  2.59,  0.688),
    "Y": ( 0.26, -2.30,  0.0,  1.0,  0.298,  2.29,  0.580),
    "V": ( 1.08, -1.50,  0.0,  0.0,  0.140,  1.64,  0.316),
}
_N_PROPS = 7
_DEFAULT_LAG = 30


def _seq_to_property_matrix(seq: str) -> np.ndarray:
    """Convert sequence to (L, 7) matrix of physicochemical properties."""
    default = (0.0,) * _N_PROPS
    return np.array([_AA_PROPERTIES.get(aa.upper(), default) for aa in seq], dtype=np.float64)


def auto_covariance_descriptor(seq: str, lag: int = _DEFAULT_LAG) -> np.ndarray:
    """
    Compute Auto Covariance descriptor: for each of 7 properties, compute
    auto-covariance at lags 1..lag. Output = 7 * lag dimensions.
    """
    props = _seq_to_property_matrix(seq)
    L = len(props)
    if L <= lag:
        lag = max(L - 1, 1)

    # Normalize each property to zero mean
    means = props.mean(axis=0)
    props_centered = props - means

    ac_vec = np.zeros(_N_PROPS * lag, dtype=np.float64)
    for p in range(_N_PROPS):
        col = props_centered[:, p]
        for d in range(1, lag + 1):
            ac_val = np.sum(col[:L - d] * col[d:]) / (L - d)
            ac_vec[p * lag + (d - 1)] = ac_val

    return ac_vec


# ══════════════════════════════════════════════════════════════
# PAIR FEATURE BUILDER
# ══════════════════════════════════════════════════════════════

def build_pair_feature_matrix(
    pairs_df: pd.DataFrame,
    sequences: dict,
    feature_fn: callable,
    method_name: str,
    logger: PipelineLogger,
) -> tuple:
    """
    Build feature matrix for all protein pairs using a given feature extractor.

    Returns:
        X: np.ndarray (n_pairs, feature_dim * 2)
        y: np.ndarray (n_pairs,)
    """
    logger.phase(f"Extracting Features: {method_name}")

    # Pre-compute per-protein features (avoid re-computation)
    unique_proteins = set(pairs_df["protein1"]).union(set(pairs_df["protein2"]))
    protein_features = {}
    skipped = 0

    for pid in unique_proteins:
        seq = sequences.get(pid)
        if seq and len(seq) >= 3:
            protein_features[pid] = feature_fn(seq)
        else:
            skipped += 1

    if skipped > 0:
        logger.warning(f"  Skipped {skipped} proteins (missing/too short sequence)")

    first_feat = next(iter(protein_features.values()))
    feat_dim = len(first_feat)
    logger.info(f"  Per-protein feature dim: {feat_dim}")
    logger.info(f"  Pair feature dim: {feat_dim * 2} (symmetric: Hadamard + AbsDiff)")

    # Build pair matrix
    rows = []
    labels = []
    valid_mask = []

    for _, row in pairs_df.iterrows():
        p1, p2, label = row["protein1"], row["protein2"], row["label"]
        f1 = protein_features.get(p1)
        f2 = protein_features.get(p2)
        if f1 is not None and f2 is not None:
            rows.append(np.concatenate([f1 * f2, np.abs(f1 - f2)]))
            labels.append(label)
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    n_skipped_pairs = len(pairs_df) - len(rows)
    if n_skipped_pairs > 0:
        logger.warning(f"  Skipped {n_skipped_pairs} pairs (missing protein features)")
    logger.info(f"  Final matrix: {X.shape[0]} pairs × {X.shape[1]} features")

    return X, y, valid_mask


# ══════════════════════════════════════════════════════════════
# CV RUNNER
# ══════════════════════════════════════════════════════════════

def run_baseline_cv(
    X: np.ndarray,
    y: np.ndarray,
    pairs_df: pd.DataFrame,
    valid_mask: list,
    model_name: str,
    n_splits: int,
    output_dir: str,
    logger: PipelineLogger,
):
    """Run 5-fold protein-level CV and save all results."""
    logger.phase(f"CV Evaluation: {model_name} ({n_splits}-Fold)")

    # Get splits on FULL pairs_df, then filter by valid_mask
    splits = get_protein_based_splits(pairs_df, n_splits=n_splits, random_state=42)

    # Map original indices to valid-only indices
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    original_to_valid = {orig: valid_idx for valid_idx, orig in enumerate(valid_indices)}

    fold_metrics_list = []
    all_y_true, all_y_proba, all_fold_ids = [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # Filter to valid-only
        train_valid = [original_to_valid[i] for i in train_idx if i in original_to_valid]
        val_valid = [original_to_valid[i] for i in val_idx if i in original_to_valid]

        X_train, X_val = X[train_valid], X[val_valid]
        y_train, y_val = y[train_valid], y[val_valid]

        # Fresh LightGBM for each fold
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LGBMClassifier(
                n_estimators=300, num_leaves=20, max_depth=10,
                learning_rate=0.05, reg_alpha=0.1, reg_lambda=0.1,
                min_child_samples=30, colsample_bytree=0.8,
                random_state=42, class_weight="balanced",
                n_jobs=-1, verbose=-1,
            )),
        ])
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        # Metrics
        tn_fp_fn_tp = np.bincount(y_val.astype(int) * 2 + y_pred, minlength=4)
        tn, fp, fn, tp = tn_fp_fn_tp[0], tn_fp_fn_tp[1], tn_fp_fn_tp[2], tn_fp_fn_tp[3]

        fold_m = {
            "Accuracy": accuracy_score(y_val, y_pred),
            "Precision": precision_score(y_val, y_pred, zero_division=0),
            "Recall (Sensitivity)": recall_score(y_val, y_pred, zero_division=0),
            "F1 Score": f1_score(y_val, y_pred, zero_division=0),
            "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "MCC": matthews_corrcoef(y_val, y_pred),
            "ROC-AUC": roc_auc_score(y_val, y_proba),
            "PR-AUC": average_precision_score(y_val, y_proba),
        }
        fold_metrics_list.append(fold_m)
        all_y_true.append(y_val)
        all_y_proba.append(y_proba)
        all_fold_ids.append(np.full(len(y_val), fold_idx + 1))

        logger.info(f"  Fold {fold_idx+1}: Acc={fold_m['Accuracy']:.4f}  F1={fold_m['F1 Score']:.4f}  AUC={fold_m['ROC-AUC']:.4f}")

    # ── Save fold metrics ──
    os.makedirs(output_dir, exist_ok=True)
    fold_df = pd.DataFrame(fold_metrics_list)
    fold_df.index = [f"Fold {i+1}" for i in range(n_splits)]
    fold_df.to_csv(os.path.join(output_dir, "fold_metrics.csv"), index_label="Fold")

    # ── OOF Dynamic Thresholding ──
    all_y_true = np.concatenate(all_y_true)
    all_y_proba = np.concatenate(all_y_proba)
    precs, recs, threshs = precision_recall_curve(all_y_true, all_y_proba)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1s = 2 * (precs[:-1] * recs[:-1]) / (precs[:-1] + recs[:-1] + 1e-8)
    opt_idx = np.argmax(f1s)
    opt_thresh = threshs[opt_idx]
    y_opt = (all_y_proba >= opt_thresh).astype(int)

    oof_m = {
        "Optimal Threshold": [opt_thresh],
        "Accuracy": [accuracy_score(all_y_true, y_opt)],
        "Precision": [precision_score(all_y_true, y_opt, zero_division=0)],
        "Recall": [recall_score(all_y_true, y_opt, zero_division=0)],
        "F1 Score": [f1_score(all_y_true, y_opt, zero_division=0)],
        "MCC": [matthews_corrcoef(all_y_true, y_opt)],
    }
    pd.DataFrame(oof_m).to_csv(os.path.join(output_dir, "oof_optimal_metrics.csv"), index=False)

    # ── OOF Predictions ──
    oof_pred = pd.DataFrame({
        "fold_id": np.concatenate(all_fold_ids).astype(int),
        "y_true": all_y_true.astype(int),
        "y_proba": all_y_proba,
    })
    oof_pred.to_csv(os.path.join(output_dir, "oof_predictions.csv"), index=False)

    # Print summary
    means = fold_df.mean()
    stds = fold_df.std()
    print(f"\n  📊 {model_name} Summary:")
    for col in fold_df.columns:
        print(f"    {col:<25} {means[col]*100:.2f}% ± {stds[col]*100:.2f}%")
    print(f"    {'OOF Optimal Threshold':<25} {opt_thresh:.4f}")
    print(f"    {'OOF F1 (Optimal)':<25} {oof_m['F1 Score'][0]*100:.2f}%")

    logger.info(f"  Saved results to {output_dir}/")
    return {"means": means.to_dict(), "stds": stds.to_dict(), "oof": oof_m}


# ══════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════

def build_comparison_table(all_results: dict, output_path: str, logger: PipelineLogger):
    """Build a publication-ready comparison CSV + LaTeX rows."""
    rows = []
    metric_cols = ["Accuracy", "Precision", "Recall (Sensitivity)", "F1 Score",
                   "MCC", "ROC-AUC", "PR-AUC"]

    for dataset, methods in all_results.items():
        for method_name, result in methods.items():
            row = {"Dataset": dataset.capitalize(), "Method": method_name}
            for col in metric_cols:
                m = result["means"].get(col, 0.0)
                s = result["stds"].get(col, 0.0)
                row[col] = f"{m*100:.2f} ± {s*100:.2f}"
                row[f"{col}_mean"] = m
                row[f"{col}_std"] = s
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 100)
    print("📊 SOTA COMPARISON TABLE (Mean ± Std, %)")
    print("=" * 100)
    for _, r in df.iterrows():
        print(f"\n  {r['Dataset']} | {r['Method']}")
        for col in metric_cols:
            print(f"    {col:<25} {r[col]}")
    print("=" * 100)

    print("\n📄 LaTeX Table Rows:")
    print("-" * 100)
    for _, r in df.iterrows():
        vals = " & ".join([f"{r[f'{c}_mean']*100:.2f}" for c in metric_cols])
        print(f"  {r['Method']} & {vals} \\\\")
    print("-" * 100)


# ══════════════════════════════════════════════════════════════
# BASELINE REGISTRY
# ══════════════════════════════════════════════════════════════

BASELINES = {
    "conjoint_triad": {
        "display_name": "Conjoint Triad (CT)",
        "reference": "Shen et al. 2007, PNAS",
        "feature_fn": conjoint_triad_descriptor,
    },
    "auto_covariance": {
        "display_name": "Auto Covariance (AC)",
        "reference": "Guo et al. 2008, NAR",
        "feature_fn": auto_covariance_descriptor,
    },
}


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SOTA Comparison Baselines for HybridStackPPI"
    )
    parser.add_argument("--dataset", choices=["human", "yeast", "both"], default="both")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--strategy", choices=["default", "same_compartment", "same_go"],
                        default="same_compartment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    logger = PipelineLogger()

    suffix = {
        "same_compartment": "_same_compartment",
        "same_go": "_same_go",
    }.get(args.strategy, "")

    datasets_cfg = []
    if args.dataset in ["human", "both"]:
        datasets_cfg.append({
            "name": "human",
            "fasta": os.path.join(PROJECT_ROOT, "data/BioGrid/Human/human_dict.fasta"),
            "pairs": os.path.join(PROJECT_ROOT, f"data/BioGrid/Human/human_pairs{suffix}.tsv"),
        })
    if args.dataset in ["yeast", "both"]:
        datasets_cfg.append({
            "name": "yeast",
            "fasta": os.path.join(PROJECT_ROOT, "data/BioGrid/Yeast/yeast_dict.fasta"),
            "pairs": os.path.join(PROJECT_ROOT, f"data/BioGrid/Yeast/yeast_pairs{suffix}.tsv"),
        })

    all_results = {}

    for ds in datasets_cfg:
        logger.header(f"SOTA BASELINES — {ds['name'].upper()}")

        sequences, pairs_df = load_data(ds["fasta"], ds["pairs"])
        pairs_df = canonicalize_pairs(pairs_df, dataset_name=ds["name"], logger=logger)

        dataset_results = {}

        for key, cfg in BASELINES.items():
            display = cfg["display_name"]
            ref = cfg["reference"]
            feat_fn = cfg["feature_fn"]

            print(f"\n{'─' * 70}")
            print(f"  Method: {display}")
            print(f"  Reference: {ref}")
            print(f"{'─' * 70}")

            t0 = time.time()
            X, y, valid_mask = build_pair_feature_matrix(
                pairs_df, sequences, feat_fn, display, logger
            )
            feat_time = time.time() - t0
            logger.info(f"  Feature extraction: {feat_time:.1f}s")

            output_dir = os.path.join("results", "baselines", f"{ds['name']}{suffix}", key)
            t0 = time.time()
            result = run_baseline_cv(
                X, y, pairs_df, valid_mask, display,
                args.n_splits, output_dir, logger,
            )
            cv_time = time.time() - t0
            logger.info(f"  CV evaluation: {cv_time:.1f}s")
            logger.info(f"  Total: {feat_time + cv_time:.1f}s")

            dataset_results[display] = result

        # Load HybridStackPPI main results if available
        main_csv = os.path.join("results", f"{ds['name']}{suffix}", "fold_metrics.csv")
        if os.path.exists(main_csv):
            main_df = pd.read_csv(main_csv, index_col=0)
            dataset_results["HybridStackPPI (Ours)"] = {
                "means": main_df.mean().to_dict(),
                "stds": main_df.std().to_dict(),
            }
            logger.info(f"  Loaded main results from {main_csv}")

        all_results[ds["name"]] = dataset_results

    # Build comparison table
    if all_results:
        table_suffix = {"same_compartment": "_same_compartment", "same_go": "_same_go"}.get(args.strategy, "")
        table_path = os.path.join("results", "baselines", f"comparison_table{table_suffix}.csv")
        build_comparison_table(all_results, table_path, logger)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
