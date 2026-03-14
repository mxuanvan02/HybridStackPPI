#!/usr/bin/env python3
import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hybridstack.feature_engine import EmbeddingComputer, FeatureEngine
from hybridstack.data_utils import load_data, canonicalize_pairs, create_feature_matrix
from hybridstack.builders import define_stacking_columns

def main():
    # Configuration
    dataset_name = "human"
    strategy = "same_go" # Target training strategy
    esm_model = "facebook/esm2_t33_650M_UR50D"
    h5_cache = "cache/esm2/esm2_embeddings_v4.h5"
    pairing_strategy = "hadamard_abs"
    
    # Paths
    results_dir = PROJECT_ROOT / "results" / f"{dataset_name}_{strategy}"
    models_dir = results_dir / "models"
    negatome_pairs_path = PROJECT_ROOT / f"data/BioGrid/Human/human_pairs_negatome.tsv"
    fasta_path = PROJECT_ROOT / "data/BioGrid/Human/human_dict.fasta"
    
    if not results_dir.exists():
        print(f"Error: Results dir {results_dir} not found.")
        return

    print(f"============================================================")
    print(f"BENCHMARKING: Model({strategy}) on NEGATOME dataset")
    print(f"============================================================")

    # 1. Load Data
    print(f"Loading data...")
    sequences, pairs_df = load_data(str(fasta_path), str(negatome_pairs_path))
    pairs_df = canonicalize_pairs(pairs_df, dataset_name="NEGATOME", logger=None)
    
    # Filter to get a balanced test set (NEGATOME has few negatives)
    neg_pairs = pairs_df[pairs_df["label"] == 0]
    pos_pairs = pairs_df[pairs_df["label"] == 1]
    
    print(f"Found {len(neg_pairs)} NEGATOME negatives and {len(pos_pairs)} positives.")
    
    # Sample positives to match negatives for a balanced benchmark
    n_test = len(neg_pairs)
    pos_test = pos_pairs.sample(n=n_test, random_state=42)
    test_df = pd.concat([pos_test, neg_pairs]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Created balanced test set with {len(test_df)} pairs.")

    # 2. Extract Features
    print(f"Extracting features for {len(test_df)} pairs...")
    embedding_computer = EmbeddingComputer(model_name=esm_model)
    feature_engine = FeatureEngine(str(PROJECT_ROOT / h5_cache), embedding_computer)
    single_feature_names = feature_engine.get_feature_names()
    
    required_prots = set(test_df["protein1"]).union(set(test_df["protein2"]))
    protein_features = feature_engine.extract_all_features({pid: sequences[pid] for pid in required_prots})
    
    X_test, y_test = create_feature_matrix(test_df, protein_features, single_feature_names, pairing_strategy)
    X_test_np = np.ascontiguousarray(X_test.to_numpy(dtype=np.float32))

    # 3. Evaluate each fold model
    metrics_list = []
    
    for fold in range(1, 6):
        model_path = models_dir / f"model_fold{fold}.joblib"
        if not model_path.exists():
            print(f"Warning: Model for fold {fold} not found at {model_path}")
            continue
            
        print(f"Evaluating Fold {fold} model...")
        model = joblib.load(model_path)
        
        y_proba = model.predict_proba(X_test_np)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        
        m = {
            "Fold": fold,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "MCC": matthews_corrcoef(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
            "PR-AUC": average_precision_score(y_test, y_proba),
            "Recall": recall_score(y_test, y_pred)
        }
        metrics_list.append(m)

    # 4. Summary
    if metrics_list:
        summary_df = pd.DataFrame(metrics_list)
        print("\nResults on Balanced NEGATOME Test Set:")
        print(summary_df.to_string(index=False))
        print("\nMean Metrics:")
        print(summary_df.mean().drop("Fold").to_string())
        
        # Save to file
        out_path = results_dir / "negatome_benchmark_results.csv"
        summary_df.to_csv(out_path, index=False)
        print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    main()
