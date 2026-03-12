"""
sota_esm2_mlp.py
================
Proxy SOTA Baseline: Vanilla ESM-2 (Global Embedding) + MLP (Multi-Layer Perceptron)

Scientific Rationale:
    Trong các bài báo PPI hàng đầu 2024-2025 (SENS-PPI, STAMP-PPI, MAGNETO), "PLM + Shallow
    Classifier" được sử dụng như một "Simple DL Reference" chính thức để chứng minh rằng sức
    mạnh đến từ kiến trúc đặc thù, không phải chỉ từ embedding LLM. Nếu HybridStackPPI vượt
    trội mô hình này, nghĩa là: phần giá trị gia tăng đến từ Stacking Meta-learner + Local
    Motif Features + Symmetric Pairing, không phải chỉ do ESM-2 global vector.

Pipeline:
    1. Load pre-computed global ESM-2 (650M, dim=1280) từ cache HDF5 (key: {SEQ_UPPER}_global_v2)
    2. Pairing: Symmetric [emb_P1 ⊙ emb_P2, |emb_P1 - emb_P2|] → shape (2560,)
    3. MLP PyTorch: Linear(2560→512, LN, ReLU, Dropout) → Linear(512→128, ReLU, Dropout) → Linear(128→1)
    4. 5-Fold CV chuẩn (Protein-based splits) + OOF dynamic thresholding

Dependency: h5py, torch, sklearn, numpy, pandas, filelock (all already in environment)
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from filelock import FileLock
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    precision_recall_curve,
)

from hybridstack.data_utils import get_protein_based_splits

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ESM2_DIM = 1280          # ESM-2 650M hidden size
H5_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "cache", "esm2", "esm2_embeddings_v4.h5"
)


# ---------------------------------------------------------------------------
# MLP Architecture
# ---------------------------------------------------------------------------
class PairMLP(nn.Module):
    """
    Vanilla 3-layer MLP for paired protein interaction prediction.
    Input: Symmetric pair vector from two ESM-2 global embeddings → (2 * ESM2_DIM,)
    """
    def __init__(self, input_dim: int = ESM2_DIM * 2, dropout1: float = 0.3, dropout2: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


# ---------------------------------------------------------------------------
# Embedding Loader
# ---------------------------------------------------------------------------
def _load_embeddings_from_cache(sequences: dict, h5_path: str, logger=None) -> dict:
    """
    Load pre-computed global ESM-2 embeddings từ HDF5.

    Key format (theo FeatureEngine.py): {SEQUENCE_UPPER}_global_v2
    Missing keys → zero vector (graceful fallback + log warning).
    """
    emb_dict: dict[str, np.ndarray] = {}
    lock_path = h5_path + ".lock"
    missing = []

    def _log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    with FileLock(lock_path):
        with h5py.File(h5_path, "r") as h5f:
            for pid, seq in sequences.items():
                key = f"{seq.upper()}_global_v2"
                if key in h5f:
                    emb_dict[pid] = h5f[key][:]
                else:
                    emb_dict[pid] = np.zeros(ESM2_DIM, dtype=np.float32)
                    missing.append(pid)

    if missing:
        _log(f"  [ESM2-MLP] ⚠️  {len(missing)} proteins not in cache → zero-padded. "
             f"(First 5: {missing[:5]})")
    _log(f"  [ESM2-MLP] Loaded embeddings for {len(emb_dict) - len(missing)}/{len(emb_dict)} proteins.")
    return emb_dict


def _build_pair_matrix(pairs_df: pd.DataFrame, emb_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Tạo ma trận đặc trưng cặp protein theo chiến lược symmetric pairing.
    X[i] = [emb_P1 ⊙ emb_P2, |emb_P1 - emb_P2|] → shape (N, 2*ESM2_DIM=2560)
    """
    X_list, y_list = [], []
    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df),
                       desc="  [ESM2-MLP] Building pair matrix", unit="pair", leave=False):
        p1, p2, label = row["protein1"], row["protein2"], int(row["label"])
        e1 = emb_dict.get(p1, np.zeros(ESM2_DIM, dtype=np.float32))
        e2 = emb_dict.get(p2, np.zeros(ESM2_DIM, dtype=np.float32))
        X_list.append(np.concatenate([e1 * e2, np.abs(e1 - e2)], axis=0))
        y_list.append(label)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def _train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple[PairMLP, np.ndarray]:
    """Train PairMLP on one fold, return trained model + val probabilities."""
    model = PairMLP(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(
        torch.from_numpy(X_train).to(device),
        torch.from_numpy(y_train).to(device),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model.train()
    pbar = tqdm(range(epochs), desc="    Training", unit="ep", leave=False)
    for epoch in pbar:
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}")

    # Inference on validation set
    model.eval()
    with torch.no_grad():
        X_val_t = torch.from_numpy(X_val).to(device)
        logits = model(X_val_t).cpu().numpy()
        y_proba = torch.sigmoid(torch.tensor(logits)).numpy()

    return model, y_proba


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def run_esm2_mlp_baseline(
    dataset_name: str,
    sequences: dict,
    pairs_df: pd.DataFrame,
    n_splits: int,
    output_dir: str,
    logger,
    epochs: int = 40,
    batch_size: int = 256,
) -> dict:
    """
    Runs the Vanilla ESM-2 + MLP proxy SOTA baseline.

    Args:
        dataset_name: e.g. "human", "yeast"
        sequences: dict {protein_id: sequence_str}
        pairs_df:  DataFrame with columns ["protein1", "protein2", "label"]
        n_splits:  Number of CV folds (usually 5)
        output_dir: Directory to save results
        logger:    PipelineLogger or any object with .info() method
        epochs:    Training epochs per fold (default: 40)
        batch_size: Mini-batch size for training (default: 256)
    Returns:
        dict with "means" and "stds" of fold metrics
    """
    def _log(msg):
        try:
            logger.info(msg)
        except Exception:
            print(msg)

    _log(f"  [ESM2-MLP] === Vanilla ESM-2 (650M) + MLP Baseline === "
         f"[Dataset: {dataset_name}]")
    _log(f"  [ESM2-MLP] Loading pre-computed global embeddings from cache: {H5_CACHE}")

    t0 = time.time()
    unique_prots = set(pairs_df["protein1"]).union(set(pairs_df["protein2"]))
    sequences = {pid: seq for pid, seq in sequences.items() if pid in unique_prots}
    emb_dict = _load_embeddings_from_cache(sequences, H5_CACHE, logger=None)
    _log(f"  [ESM2-MLP] Embeddings loaded in {time.time() - t0:.1f}s")

    _log("  [ESM2-MLP] Building pair feature matrix (strategy: symmetric Hadamard+AbsDiff)...")
    X_all, y_all = _build_pair_matrix(pairs_df, emb_dict)
    _log(f"  [ESM2-MLP] X shape: {X_all.shape}, y distribution: "
         f"{{'pos': int(y_all.sum()), 'neg': int((1 - y_all).sum())}}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"  [ESM2-MLP] Training device: {device.upper()}")

    splits = get_protein_based_splits(pairs_df, n_splits=n_splits, random_state=42)
    fold_metrics_list = []
    all_y_true, all_y_proba, all_fold_ids = [], [], []

    fold_pbar = tqdm(enumerate(splits), total=n_splits,
                     desc="  [ESM2-MLP] CV Folds", unit="fold")
    for fold_idx, (train_idx, val_idx) in fold_pbar:
        t_fold = time.time()
        # _log(f"  [ESM2-MLP] ── Fold {fold_idx + 1}/{n_splits} ──")

        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, y_proba = _train_one_fold(
                X_train, y_train, X_val,
                epochs=epochs, batch_size=batch_size, lr=1e-3, device=device,
            )

        y_pred = (y_proba >= 0.5).astype(int)

        acc  = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec  = recall_score(y_val, y_pred, zero_division=0)
        f1   = f1_score(y_val, y_pred, zero_division=0)
        mcc  = matthews_corrcoef(y_val, y_pred)
        spec = accuracy_score(1 - y_val, 1 - y_pred)
        try:
            auc   = roc_auc_score(y_val, y_proba)
            prauc = average_precision_score(y_val, y_proba)
        except ValueError:
            auc, prauc = 0.0, 0.0

        fold_m = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall (Sensitivity)": rec,
            "F1 Score": f1,
            "Specificity": spec,
            "MCC": mcc,
            "ROC-AUC": auc,
            "PR-AUC": prauc,
        }
        fold_metrics_list.append(fold_m)
        all_y_true.append(y_val)
        all_y_proba.append(y_proba)
        all_fold_ids.append(np.full(len(y_val), fold_idx + 1, dtype=int))

        elapsed = time.time() - t_fold
        fold_pbar.set_postfix(
            f1=f"{f1:.3f}", mcc=f"{mcc:.3f}",
            auc=f"{auc:.3f}", t=f"{elapsed:.0f}s"
        )
        _log(
            f"    Fold {fold_idx + 1}: "
            f"ACC={acc:.4f} | F1={f1:.4f} | MCC={mcc:.4f} | "
            f"ROC-AUC={auc:.4f} | PR-AUC={prauc:.4f}  [{elapsed:.1f}s]"
        )

    # ── Save fold-level metrics ──────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    fold_df = pd.DataFrame(fold_metrics_list)
    fold_df.index = [f"Fold {i + 1}" for i in range(n_splits)]
    fold_df.to_csv(os.path.join(output_dir, "fold_metrics.csv"), index_label="Fold")

    # ── OOF Dynamic Thresholding (F1-optimized) ──────────────────────────────
    oof_y_true  = np.concatenate(all_y_true)
    oof_y_proba = np.concatenate(all_y_proba)
    precs, recs, threshs = precision_recall_curve(oof_y_true, oof_y_proba)
    f1s = 2 * (precs[:-1] * recs[:-1]) / (precs[:-1] + recs[:-1] + 1e-8)
    opt_thresh = threshs[np.argmax(f1s)]
    y_opt = (oof_y_proba >= opt_thresh).astype(int)

    oof_m = {
        "Optimal Threshold": [opt_thresh],
        "Accuracy": [accuracy_score(oof_y_true, y_opt)],
        "Precision": [precision_score(oof_y_true, y_opt, zero_division=0)],
        "Recall": [recall_score(oof_y_true, y_opt, zero_division=0)],
        "F1 Score": [f1_score(oof_y_true, y_opt, zero_division=0)],
        "MCC": [matthews_corrcoef(oof_y_true, y_opt)],
    }
    pd.DataFrame(oof_m).to_csv(
        os.path.join(output_dir, "oof_optimal_metrics.csv"), index=False
    )

    # ── OOF predictions ──────────────────────────────────────────────────────
    pd.DataFrame({
        "fold_id": np.concatenate(all_fold_ids),
        "y_true":  oof_y_true.astype(int),
        "y_proba": oof_y_proba,
    }).to_csv(os.path.join(output_dir, "oof_predictions.csv"), index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    means = fold_df.mean()
    stds  = fold_df.std()
    _log(f"\n  [ESM2-MLP] ✅ 5-Fold CV Complete:")
    _log(f"     ROC-AUC : {means['ROC-AUC']:.4f} ± {stds['ROC-AUC']:.4f}")
    _log(f"     PR-AUC  : {means['PR-AUC']:.4f}  ± {stds['PR-AUC']:.4f}")
    _log(f"     F1-Score: {means['F1 Score']:.4f} ± {stds['F1 Score']:.4f}")
    _log(f"     MCC     : {means['MCC']:.4f}     ± {stds['MCC']:.4f}")

    return {"means": means.to_dict(), "stds": stds.to_dict(), "oof": oof_m}
