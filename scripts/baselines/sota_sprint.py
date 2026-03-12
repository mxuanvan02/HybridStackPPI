import os
import time
import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score, precision_recall_curve
)

from hybridstack.data_utils import get_protein_based_splits

def run_sprint_baseline(dataset_name, sequences, pairs_df, n_splits, output_dir, logger):
    """
    Executes the SPRINT C++ pipeline for 5-fold cross validation.
    """
    sprint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "SPRINT",
    )
    sprint_bin_compute = os.path.join(sprint_dir, "bin", "compute_HSPs")
    sprint_bin_predict = os.path.join(sprint_dir, "bin", "predict_interactions")

    if not os.path.exists(sprint_bin_compute) or not os.path.exists(sprint_bin_predict):
        print(f"SPRINT binaries not found in {sprint_dir}/bin. Have you run 'make' inside SPRINT?")
        return {"error": "SPRINT binaries missing"}

    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # 1. Write the massive FASTA file for SPRINT
    fasta_path = os.path.join(tmp_dir, f"{dataset_name}_sequences.fasta")
    if not os.path.exists(fasta_path):
        unique_prots = set(pairs_df["protein1"]).union(set(pairs_df["protein2"]))
        logger.info(f"  [SPRINT] Filtering {len(sequences)} sequences to {len(unique_prots)} found in pairs")
        sequences = {pid: seq for pid, seq in sequences.items() if pid in unique_prots}
        logger.info(f"  [SPRINT] Writing {len(sequences)} sequences to {fasta_path}")
        with open(fasta_path, "w") as f:
            for pid, seq in sequences.items():
                f.write(f">{pid}\n{seq}\n")
    
    # 2. Compute HSPs
    hsp_path = os.path.join(tmp_dir, f"{dataset_name}.hsp")
    if not os.path.exists(hsp_path):
        logger.info("  [SPRINT] Computing HSPs (this may take a while...)")
        t0 = time.time()
        cmd = [sprint_bin_compute, "-p", fasta_path, "-h", hsp_path]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=sprint_dir)
        if res.returncode != 0:
            print(f"SPRINT compute_HSPs failed: {res.stderr}")
            return {"error": "HSP computation failed"}
        logger.info(f"  [SPRINT] HSP computed in {time.time() - t0:.1f}s")
    else:
        logger.info("  [SPRINT] HSP file already exists, skipping compute_HSPs.")

    # 3. CV Loop
    from tqdm import tqdm
    splits = get_protein_based_splits(pairs_df, n_splits=n_splits, random_state=42)
    fold_metrics_list = []
    all_y_true, all_y_proba, all_fold_ids = [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(tqdm(splits, desc=f"  [SPRINT] CV Folds", unit="fold", leave=False)):
        # logger.info(f"  [SPRINT] Running Fold {fold_idx+1}/{n_splits}")
        
        train_df = pairs_df.iloc[train_idx]
        val_df = pairs_df.iloc[val_idx]

        # SPRINT training file ONLY takes positive pairs
        train_pos_df = train_df[train_df["label"] == 1]
        
        val_pos_df = val_df[val_df["label"] == 1]
        val_neg_df = val_df[val_df["label"] == 0]

        train_path = os.path.join(tmp_dir, f"fold{fold_idx}_train_pos.txt")
        val_pos_path = os.path.join(tmp_dir, f"fold{fold_idx}_val_pos.txt")
        val_neg_path = os.path.join(tmp_dir, f"fold{fold_idx}_val_neg.txt")
        out_eval_path = os.path.join(tmp_dir, f"fold{fold_idx}_eval.txt")

        # Write interaction formats: "Protein1 Protein2"
        train_pos_df[["protein1", "protein2"]].to_csv(train_path, sep=" ", header=False, index=False)
        val_pos_df[["protein1", "protein2"]].to_csv(val_pos_path, sep=" ", header=False, index=False)
        val_neg_df[["protein1", "protein2"]].to_csv(val_neg_path, sep=" ", header=False, index=False)

        # Run Prediction
        cmd = [
            sprint_bin_predict,
            "-p", fasta_path,
            "-h", hsp_path,
            "-tr", train_path,
            "-pos", val_pos_path,
            "-neg", val_neg_path,
            "-o", out_eval_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=sprint_dir)
        if res.returncode != 0:
            print(f"SPRINT predict_interactions failed on fold {fold_idx}: {res.stderr}")
            return {"error": "Prediction failed"}

        # Parse SPRINT output
        # SPRINT writes scores for POS in out_eval_path.pos and for NEG in out_eval_path.neg
        # Wait, the documentation said it outputs to `output_file`, and in our test we saw `out_eval_path` and `out_eval_path.pos`
        # SPRINT just prints two columns in `out_eval_path`: <score> <label>
        
        y_proba_local = []
        y_true_local = []

        with open(out_eval_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    score = float(parts[0])
                    label = int(parts[1])
                    y_proba_local.append(score)
                    y_true_local.append(label)

        y_proba_local = np.array(y_proba_local)
        y_true_local = np.array(y_true_local)

        # Scale raw SPRINT scores (0 to unbounded) into [0, 1] probabilties purely for thresholding if needed
        # AUC works without scaling. But F1 needs a threshold.
        # We'll use relative thresholding or min/max scaling
        if len(y_proba_local) > 0:
            p_min, p_max = y_proba_local.min(), y_proba_local.max()
            if p_max > p_min:
                y_proba_scaled = (y_proba_local - p_min) / (p_max - p_min)
            else:
                y_proba_scaled = y_proba_local

            # Predict using 0.5 threshold on scaled score (just a heuristic for Fold Acc)
            y_pred = (y_proba_scaled >= 0.5).astype(int)
        else:
            y_proba_scaled = y_proba_local
            y_pred = y_proba_local

        # Metrics
        acc = accuracy_score(y_true_local, y_pred)
        prec = precision_score(y_true_local, y_pred, zero_division=0)
        rec = recall_score(y_true_local, y_pred, zero_division=0)
        f1 = f1_score(y_true_local, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true_local, y_pred)
        try:
            auc = roc_auc_score(y_true_local, y_proba_local)
            prauc = average_precision_score(y_true_local, y_proba_local)
        except ValueError:
            auc, prauc = 0.0, 0.0

        fold_m = {
            "Accuracy": acc, "Precision": prec, "Recall (Sensitivity)": rec,
            "F1 Score": f1, "MCC": mcc, "ROC-AUC": auc, "PR-AUC": prauc,
        }
        fold_metrics_list.append(fold_m)
        all_y_true.append(y_true_local)
        all_y_proba.append(y_proba_scaled)
        all_fold_ids.append(np.full(len(y_true_local), fold_idx + 1))
        
        logger.info(f"    Fold {fold_idx+1}: ACC={acc:.4f} F1={f1:.4f} AUC={auc:.4f}")

    # Aggregation
    fold_df = pd.DataFrame(fold_metrics_list)
    fold_df.index = [f"Fold {i+1}" for i in range(n_splits)]
    fold_df.to_csv(os.path.join(output_dir, "fold_metrics.csv"), index_label="Fold")

    # Optimal Thresholding on OOF
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
    
    oof_pred = pd.DataFrame({
        "fold_id": np.concatenate(all_fold_ids).astype(int),
        "y_true": all_y_true.astype(int),
        "y_proba": all_y_proba,
    })
    oof_pred.to_csv(os.path.join(output_dir, "oof_predictions.csv"), index=False)

    means = fold_df.mean()
    stds = fold_df.std()
    
    return {"means": means.to_dict(), "stds": stds.to_dict(), "oof": zoof_m}
