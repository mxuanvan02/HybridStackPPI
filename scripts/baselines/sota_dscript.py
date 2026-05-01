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

def run_dscript_baseline(dataset_name, sequences, pairs_df, n_splits, output_dir, logger):
    """
    Executes the D-SCRIPT (Topsy-Turvy) Inference baseline.
    D-SCRIPT / Topsy-Turvy are zero-shot out-of-the-box pretrained PyTorch models.
    """
    logger.info("  [D-SCRIPT] Initializing D-SCRIPT / Topsy-Turvy Zero-shot Inference...")
    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # 1. Write the FASTA file for D-SCRIPT Embedding
    fasta_path = os.path.join(tmp_dir, f"{dataset_name}_sequences.fasta")
    if not os.path.exists(fasta_path):
        unique_prots = set(pairs_df["protein1"]).union(set(pairs_df["protein2"]))
        logger.info(f"  [D-SCRIPT] Filtering {len(sequences)} sequences to {len(unique_prots)} found in pairs")
        sequences = {pid: seq for pid, seq in sequences.items() if pid in unique_prots}
        logger.info(f"  [D-SCRIPT] Writing {len(sequences)} sequences to {fasta_path}")
        with open(fasta_path, "w") as f:
            for pid, seq in sequences.items():
                f.write(f">{pid}\n{seq}\n")
    
    # 2. Embed using Bepler-Berger Model via D-SCRIPT
    emb_path = os.path.join(tmp_dir, f"{dataset_name}_embed.h5")
    if not os.path.exists(emb_path):
        logger.info("  [D-SCRIPT] Computing Bepler-Berger language model embeddings (this may take a while on CPU...)")
        t0 = time.time()
        cmd = ["python", "-m", "dscript", "embed", "--seqs", fasta_path, "-o", emb_path, "-d", "cpu"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"D-SCRIPT embedding failed: {res.stderr}")
            return {"error": "D-SCRIPT Embedding failed"}
        logger.info(f"  [D-SCRIPT] Embeddings generated in {time.time() - t0:.1f}s")
    else:
        logger.info("  [D-SCRIPT] Embs file already exists, skipping embedding.")

    # 3. CV Loop for Topsy-Turvy model evaluation
    from tqdm import tqdm
    splits = get_protein_based_splits(pairs_df, n_splits=n_splits, random_state=42)
    fold_metrics_list = []
    all_y_true, all_y_proba, all_fold_ids = [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(tqdm(splits, desc=f"  [D-SCRIPT] CV Folds", unit="fold", leave=False)):
        # logger.info(f"  [D-SCRIPT] Evaluating Fold {fold_idx+1}/{n_splits}")
        t0 = time.time()
        
        # We only need the validation pairs since D-SCRIPT is zero-shot
        val_df = pairs_df.iloc[val_idx]
        val_path = os.path.join(tmp_dir, f"fold{fold_idx}_val.tsv")
        out_pred_path = os.path.join(tmp_dir, f"fold{fold_idx}_pred")

        # Write TSV: pt1, pt2
        val_df[["protein1", "protein2"]].to_csv(val_path, sep="\t", header=False, index=False)

        # Run Prediction using standard topsy turvy human v1
        # dscript predict --pairs [input data] --embeddings [embedding file] --model [model file] --outfile [predictions file]
        cmd = [
            "python", "-m", "dscript", "predict",
            "--pairs", val_path,
            "--embeddings", emb_path,
            "--model", "samsl/topsy_turvy_human_v1",
            "--outfile", out_pred_path,
            "-d", "cpu",
            "--load_proc", "1"
        ]
        
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"D-SCRIPT sequence-predict failed on fold {fold_idx}: {res.stderr}")
            if os.path.exists(out_pred_path + ".tsv"):
                 pass # D-SCRIPT sometimes returns non-zero code but outputs properly if interrupted or warning out
            else:
                 return {"error": "Prediction failed"}
                 
        # Parse output D-SCRIPT predictions (which outputs to {out_pred_path}.tsv)
        pred_tsv = out_pred_path + ".tsv"
        try:
            pred_df = pd.read_csv(pred_tsv, sep="\t", header=None, names=["p1", "p2", "score"])
        except FileNotFoundError:
            print(f"Error: {pred_tsv} not generated. STDOUT: {res.stdout} \nSTDERR: {res.stderr}")
            return {"error": "Missing output file"}
            
        y_proba_local = pred_df["score"].values
        y_true_local = val_df["label"].values
        
        y_pred = (y_proba_local >= 0.5).astype(int)

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
        all_y_proba.append(y_proba_local)
        all_fold_ids.append(np.full(len(y_true_local), fold_idx + 1))
        
        logger.info(f"    Fold {fold_idx+1}: ACC={acc:.4f} F1={f1:.4f} AUC={auc:.4f} [{time.time() - t0:.1f}s]")

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
    
    return {"means": means.to_dict(), "stds": stds.to_dict(), "oof": oof_m}
