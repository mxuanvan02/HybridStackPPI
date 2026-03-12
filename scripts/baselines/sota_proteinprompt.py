import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
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

from hybridstack.data_utils import get_protein_based_splits


def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(tn / max(tn + fp, 1))


def _compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, thr: float = 0.5) -> dict:
    y_pred = (y_proba >= thr).astype(int)
    try:
        auc = roc_auc_score(y_true, y_proba)
        prauc = average_precision_score(y_true, y_proba)
    except ValueError:
        auc, prauc = 0.0, 0.0
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall (Sensitivity)": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "Specificity": _specificity(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "ROC-AUC": auc,
        "PR-AUC": prauc,
    }


def _ensure_docker_ready(image: str):
    if shutil.which("docker") is None:
        raise RuntimeError("docker binary not found. Install Docker to run ProteinPrompt CLI baseline.")
    inspect = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
    )
    if inspect.returncode != 0:
        raise RuntimeError(
            f"Docker image '{image}' not found. Build/pull it first (e.g., from ./proteinPrompt)."
        )


def _write_fasta(path: Path, seq_map: dict):
    with path.open("w", encoding="utf-8") as f:
        for pid, seq in seq_map.items():
            f.write(f">{pid}\n{seq}\n")


def _write_pairs(path: Path, pairs_df: pd.DataFrame):
    pairs_df[["protein1", "protein2"]].to_csv(path, sep="\t", header=False, index=False)


def _run_proteinprompt_cli(project_root: Path, image: str, fasta_abs: Path, pairs_abs: Path, out_dir_abs: Path, out_name: str):
    # Paths inside container are mapped under /work.
    fasta_in = f"/work/{fasta_abs.relative_to(project_root)}"
    pairs_in = f"/work/{pairs_abs.relative_to(project_root)}"
    out_dir_in = f"/work/{out_dir_abs.relative_to(project_root)}"
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{project_root}:/work",
        "-w",
        "/work",
        image,
        "predict",
        "-f",
        fasta_in,
        "-p",
        pairs_in,
        "-d",
        out_dir_in,
        "-o",
        out_name,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"ProteinPrompt docker CLI failed.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )


def _parse_proteinprompt_output(out_file: Path, pairs_df: pd.DataFrame) -> np.ndarray:
    # Expected line format (tab-separated): protein1 protein2 prob0 prob1
    score_map = {}
    if out_file.exists():
        with out_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                p1, p2 = parts[0], parts[1]
                try:
                    prob1 = float(parts[3])
                except ValueError:
                    continue
                score_map[(p1, p2)] = prob1

    # Pair order is canonicalized upstream; keep exact mapping first, then fallback swapped.
    y_proba = []
    for _, row in pairs_df.iterrows():
        p1 = row["protein1"]
        p2 = row["protein2"]
        prob = score_map.get((p1, p2))
        if prob is None:
            prob = score_map.get((p2, p1), 0.0)
        y_proba.append(prob)
    return np.asarray(y_proba, dtype=np.float32)


def run_proteinprompt_baseline(dataset_name, sequences, pairs_df, n_splits, output_dir, logger):
    """
    ProteinPrompt baseline via official Docker CLI (predict mode) per fold.
    """
    os.makedirs(output_dir, exist_ok=True)
    project_root = Path(__file__).resolve().parents[2]
    image = os.environ.get("PROTEINPROMPT_IMAGE", "proteinprompt")
    _ensure_docker_ready(image)

    tmp_dir = Path(output_dir) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    unique_proteins = set(pairs_df["protein1"]).union(set(pairs_df["protein2"]))
    seq_map = {pid: seq for pid, seq in sequences.items() if pid in unique_proteins}
    fasta_abs = tmp_dir / f"{dataset_name}_proteinprompt_input.fasta"
    if not fasta_abs.exists():
        logger.info(f"  [ProteinPrompt] Writing FASTA for {len(seq_map)} proteins ...")
        _write_fasta(fasta_abs, seq_map)

    splits = get_protein_based_splits(pairs_df, n_splits=n_splits, random_state=42)
    fold_metrics = []
    all_y_true = []
    all_y_proba = []
    all_fold = []

    for fold_idx, (_, val_idx) in enumerate(splits, start=1):
        val_df = pairs_df.iloc[val_idx].copy()
        val_pairs_abs = tmp_dir / f"fold{fold_idx}_val_pairs.tsv"
        out_name = f"fold{fold_idx}_pred.tsv"
        out_abs = tmp_dir / out_name
        _write_pairs(val_pairs_abs, val_df)

        _run_proteinprompt_cli(
            project_root=project_root,
            image=image,
            fasta_abs=fasta_abs,
            pairs_abs=val_pairs_abs,
            out_dir_abs=tmp_dir,
            out_name=out_name,
        )

        y_true = val_df["label"].to_numpy(dtype=np.int32)
        y_proba = _parse_proteinprompt_output(out_abs, val_df)
        fm = _compute_metrics(y_true, y_proba, thr=0.5)
        fold_metrics.append(fm)
        all_y_true.append(y_true)
        all_y_proba.append(y_proba)
        all_fold.append(np.full(len(y_true), fold_idx, dtype=int))
        logger.info(
            f"    Fold {fold_idx}: ACC={fm['Accuracy']:.4f} F1={fm['F1 Score']:.4f} "
            f"MCC={fm['MCC']:.4f} AUC={fm['ROC-AUC']:.4f}"
        )

    fold_df = pd.DataFrame(fold_metrics)
    fold_df.index = [f"Fold {i+1}" for i in range(len(fold_metrics))]
    fold_df.to_csv(os.path.join(output_dir, "fold_metrics.csv"), index_label="Fold")

    y_true_all = np.concatenate(all_y_true)
    y_proba_all = np.concatenate(all_y_proba)
    prec, rec, thr = precision_recall_curve(y_true_all, y_proba_all)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1s = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-8)
    opt_i = int(np.argmax(f1s))
    opt_thr = float(thr[opt_i]) if len(thr) else 0.5

    oof = _compute_metrics(y_true_all, y_proba_all, thr=opt_thr)
    oof_out = {
        "Optimal Threshold": [opt_thr],
        "Accuracy": [oof["Accuracy"]],
        "Precision": [oof["Precision"]],
        "Recall": [oof["Recall (Sensitivity)"]],
        "F1 Score": [oof["F1 Score"]],
        "Specificity": [oof["Specificity"]],
        "MCC": [oof["MCC"]],
    }
    pd.DataFrame(oof_out).to_csv(os.path.join(output_dir, "oof_optimal_metrics.csv"), index=False)
    pd.DataFrame(
        {
            "fold_id": np.concatenate(all_fold),
            "y_true": y_true_all,
            "y_proba": y_proba_all,
        }
    ).to_csv(os.path.join(output_dir, "oof_predictions.csv"), index=False)

    return {"means": fold_df.mean().to_dict(), "stds": fold_df.std().to_dict(), "oof": oof_out}
