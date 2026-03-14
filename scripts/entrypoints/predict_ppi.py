#!/usr/bin/env python3
import argparse
import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from hybridstack.feature_engine import EmbeddingComputer, FeatureEngine
from hybridstack.data_utils import load_data, create_feature_matrix
from hybridstack.logger import PipelineLogger

def evaluate_model(model, X, y_true):
    """Run evaluation on a single test set."""
    X_np = np.ascontiguousarray(X.to_numpy(dtype=np.float32))
    
    y_proba = model.predict_proba(X_np)[:, 1]
    
    # We use 0.5 as default threshold for consistency
    y_pred = (y_proba >= 0.5).astype(int)
    
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1 Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "ROC-AUC": float(roc_auc_score(y_true, y_proba)),
        "PR-AUC": float(average_precision_score(y_true, y_proba))
    }
    
    return metrics, y_proba, y_pred

def main():
    parser = argparse.ArgumentParser(description="Predict using pre-trained HybridStack model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved joblib model")
    parser.add_argument("--fasta-path", type=str, required=True, help="Path to FASTA file")
    parser.add_argument("--pairs-path", type=str, required=True, help="Path to pairs TSV file")
    parser.add_argument("--output-name", type=str, default="evaluation_results", help="Base name for output files")
    parser.add_argument("--pairing", type=str, default="hadamard_abs", help="Pairing strategy")
    parser.add_argument("--h5-cache", type=str, default="cache/esm2/esm2_embeddings_v4.h5")
    parser.add_argument("--esm-model", type=str, default="facebook/esm2_t33_650M_UR50D")
    args = parser.parse_args()

    logger = PipelineLogger()
    logger.header(f"PREDICTION TASK: {args.output_name}")

    if not os.path.exists(args.model_path):
        logger.error(f"Model not found: {args.model_path}")
        sys.exit(1)

    # 1. Load Model
    logger.info(f"Loading model from {args.model_path}...")
    model = joblib.load(args.model_path)

    # 2. Initialize Feature Engine
    embedding_computer = EmbeddingComputer(model_name=args.esm_model)
    feature_engine = FeatureEngine(h5_cache_path=args.h5_cache, embedding_computer=embedding_computer)

    # 3. Load Data
    logger.phase("Loading Data")
    universe, pairs_df = load_data(args.fasta_path, args.pairs_path)
    
    # 4. Feature Extraction (Batch)
    logger.phase("Extracting Features")
    all_needed_ids = sorted(list(set(pairs_df["protein1"]) | set(pairs_df["protein2"])))
    logger.info(f"Checking {len(all_needed_ids):,} unique proteins...")
    
    sequences_to_extract = {pid: universe[pid] for pid in all_needed_ids if pid in universe}
    missing = set(all_needed_ids) - set(sequences_to_extract.keys())
    if missing:
        logger.warning(f"{len(missing)} proteins missing from FASTA!")

    protein_features = feature_engine.extract_all_features(sequences_to_extract)

    # 5. Build Interaction Matrix
    logger.phase("Building Pair Matrix")
    feat_names = feature_engine.get_feature_names()
    X_df, y_s = create_feature_matrix(pairs_df, protein_features, feat_names, pairing_strategy=args.pairing)
    
    # 6. Evaluate
    logger.phase("Inference")
    metrics, y_proba, y_pred = evaluate_model(model, X_df, y_s)

    # 7. Report Results
    print("\n" + "="*40)
    print(f"RESULTS: {args.output_name}")
    print("="*40)
    for m, v in metrics.items():
        print(f"{m:<15}: {v*100:.2f}%")
    
    # 8. Save Results
    out_dir = Path("results/external_eval")
    out_dir.mkdir(exist_ok=True, parents=True)
    
    json_path = out_dir / f"{args.output_name}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    pred_path = out_dir / f"{args.output_name}_predictions.csv"
    pred_df = pd.DataFrame({
        "label": y_s.values,
        "proba": y_proba,
        "pred": y_pred
    })
    pred_df.to_csv(pred_path, index=False)
    
    logger.info(f"Metrics saved to {json_path}")
    logger.info(f"Predictions saved to {pred_path}")

if __name__ == "__main__":
    main()
