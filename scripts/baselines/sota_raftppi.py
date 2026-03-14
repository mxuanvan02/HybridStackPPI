import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig
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
from tqdm import tqdm

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


def _load_raft_model(logger):
    root = Path(__file__).resolve().parents[2]
    raft_root = root / "external" / "RaftPPI"
    ckpt_path = raft_root / "checkpoints" / "dscript" / "pytorch_model.bin"
    model_py = raft_root / "src" / "raft" / "model.py"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing RaftPPI checkpoint: {ckpt_path}")
    if not model_py.exists():
        raise FileNotFoundError(f"Missing RaftPPI model file: {model_py}")

    # Dynamically import RaftModel from the cloned repository.
    if "omegaconf" not in sys.modules:
        try:
            import omegaconf  # noqa: F401
        except Exception:
            fake = types.ModuleType("omegaconf")
            fake.DictConfig = dict
            sys.modules["omegaconf"] = fake

    spec = importlib.util.spec_from_file_location("raftppi_model_module", str(model_py))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import RaftPPI model module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    RaftModel = module.RaftModel

    class _Cfg(SimpleNamespace):
        def get(self, key, default=None):
            return getattr(self, key, default)

    cfg = _Cfg(
        offline_mode=False,
        hf_ckpt="facebook/esm2_t6_8M_UR50D",
        seed=1,
        prot_readout="mlp_attn",
        attn_rank=1,
        prot_emb_norm=True,
        res_emb_norm=True,
        sigma=0.5,
        loss_type="Ranking",
        use_sorf=True,
        rff_dim=2048,
        adv_temp=4.0,
    )
    try:
        AutoConfig.from_pretrained(cfg.hf_ckpt, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "RaftPPI requires local/cache files for facebook/esm2_t6_8M_UR50D. "
            "Please pre-download this model (or enable network access) before running raftppi baseline."
        ) from exc
    logger.info("  [RaftPPI] Loading official backbone facebook/esm2_t6_8M_UR50D ...")
    model = RaftModel(cfg=cfg, logger=logger, float_dtype=torch.float32)
    state = torch.load(str(ckpt_path), map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"  [RaftPPI] Missing keys while loading checkpoint (first 5): {missing[:5]}")
    if unexpected:
        logger.warning(f"  [RaftPPI] Unexpected keys while loading checkpoint (first 5): {unexpected[:5]}")
    model.eval()
    return model


@torch.no_grad()
def _predict_pairs(model, tokenizer, pair_batch, token_cache, device):
    r_tok_list = [token_cache[p1] for p1, _ in pair_batch]
    l_tok_list = [token_cache[p2] for _, p2 in pair_batch]

    r_tokens = tokenizer.pad(r_tok_list, padding=True, return_tensors="pt")
    l_tokens = tokenizer.pad(l_tok_list, padding=True, return_tensors="pt")
    r_tokens = {k: v.to(device) for k, v in r_tokens.items()}
    l_tokens = {k: v.to(device) for k, v in l_tokens.items()}
    dummy_labels = torch.zeros(len(pair_batch), dtype=torch.float32, device=device)

    logits, _ = model(r_tokens, l_tokens, dummy_labels, mode="inference")
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    return probs


def run_raftppi_baseline(dataset_name, sequences, pairs_df, n_splits, output_dir, logger):
    """
    RaftPPI baseline with official released checkpoint (dscript variant) and ESM2-8M backbone.
    Evaluated in zero-shot mode on fold validation sets.
    """
    os.makedirs(output_dir, exist_ok=True)

    model = _load_raft_model(logger)
    tokenizer = model.tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"  [RaftPPI] Device: {device}")

    # Pre-tokenize only proteins that appear in current benchmark pairs.
    unique_proteins = set(pairs_df["protein1"]).union(set(pairs_df["protein2"]))
    seq_map = {pid: seq for pid, seq in sequences.items() if pid in unique_proteins}
    token_cache = {}
    for pid, seq in seq_map.items():
        token_cache[pid] = tokenizer(seq, truncation=True, max_length=1024)

    splits = get_protein_based_splits(pairs_df, n_splits=n_splits, random_state=42)
    fold_metrics = []
    all_y_true = []
    all_y_proba = []
    all_fold = []

    batch_size = 32
    for fold_idx, (_, val_idx) in enumerate(splits, start=1):
        val_df = pairs_df.iloc[val_idx]
        pairs = list(zip(val_df["protein1"].tolist(), val_df["protein2"].tolist()))
        y_true = val_df["label"].to_numpy(dtype=np.int32)

        probs = []
        for i in tqdm(range(0, len(pairs), batch_size), desc=f"  [RaftPPI] Fold {fold_idx}", leave=False):
            batch = pairs[i : i + batch_size]
            p = _predict_pairs(model, tokenizer, batch, token_cache, device)
            probs.append(p)
        y_proba = np.concatenate(probs) if probs else np.zeros_like(y_true, dtype=np.float32)

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
