#!/usr/bin/env python3
import argparse
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description="Cross-species Generalization Evaluation")
    parser.add_argument("--train-dataset", choices=["human", "yeast"], default="human")
    parser.add_argument("--test-dataset", choices=["human", "yeast"], default="yeast")
    parser.add_argument("--strategy", default="same_go")
    parser.add_argument("--pairing", default="hadamard_abs")
    parser.add_argument("--h5-cache", default="cache/esm2/esm2_embeddings_v3.h5")
    parser.add_argument("--esm-model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from scripts.run import define_stacking_columns, create_stacking_pipeline
    from hybridstack.feature_engine import FeatureEngine, EmbeddingComputer
    from hybridstack.data_utils import load_data, create_feature_matrix
    from hybridstack.logger import PipelineLogger

    logger = PipelineLogger()
    logger.header(f"CROSS-SPECIES EVALUATION: {args.train_dataset.upper()} -> {args.test_dataset.upper()} ({args.strategy})")

    # Paths
    dataset_cfg = {
        "human": {
            "fasta": str(PROJECT_ROOT / "data/BioGrid/Human/human_dict.fasta"),
            "pairs": str(PROJECT_ROOT / f"data/BioGrid/Human/human_pairs_{args.strategy}.tsv"),
        },
        "yeast": {
            "fasta": str(PROJECT_ROOT / "data/BioGrid/Yeast/yeast_dict.fasta"),
            "pairs": str(PROJECT_ROOT / f"data/BioGrid/Yeast/yeast_pairs_{args.strategy}.tsv"),
        }
    }

    train_cfg = dataset_cfg[args.train_dataset]
    test_cfg = dataset_cfg[args.test_dataset]

    # Initialize Feature Engine (embedding_computer only needed if extraction is required)
    try:
        embedding_computer = EmbeddingComputer(model_name=args.esm_model)
    except Exception:
        embedding_computer = None
    feature_engine = FeatureEngine(h5_cache_path=args.h5_cache, embedding_computer=embedding_computer)
    feature_names = feature_engine.get_feature_names()

    def _get_matrix(dataset_name, cfg):
        # Construct expected cache path
        # Note: we use the same naming convention as reproduce_results.py
        cache_path = Path("cache") / f"{dataset_name}_{dataset_name}_pairs_{args.strategy}_{args.esm_model.replace('/', '_').lower()}_{args.pairing}_v3_features.h5"
        
        if cache_path.exists():
            logger.phase(f"Loading {dataset_name.upper()} Features from Cache")
            from hybridstack.data_utils import load_feature_matrix_h5
            X_df, y_s = load_feature_matrix_h5(str(cache_path))
            return X_df, y_s
        
        logger.phase(f"Extracting Features for {dataset_name.upper()} (Cache not found)")
        seqs, pairs_df = load_data(cfg["fasta"], cfg["pairs"])
        needed_seqs = {seq_id: seq for seq_id, seq in seqs.items() if seq_id in set(pairs_df["protein1"]).union(set(pairs_df["protein2"]))}
        protein_features = feature_engine.extract_all_features(needed_seqs)
        X_df, y_s = create_feature_matrix(pairs_df, protein_features, feature_names, pairing_strategy=args.pairing)
        return X_df, y_s

    # 1. Load Train Data
    X_train_df, y_train_s = _get_matrix(args.train_dataset, train_cfg)
    
    # 2. Train Model
    logger.phase("Training Model")
    interp_cols, embed_cols = define_stacking_columns(feature_engine, pairing_strategy=args.pairing)
    model = create_stacking_pipeline(
        interp_cols=interp_cols,
        embed_cols=embed_cols,
        n_jobs=args.n_jobs,
        use_selector=True,
        meta_learner_type="lr",
        feature_names=list(X_train_df.columns)
    )
    
    X_train_np = np.ascontiguousarray(X_train_df.to_numpy(dtype=np.float32))
    y_train_np = np.ascontiguousarray(y_train_s.to_numpy(dtype=np.float32))
    model.fit(X_train_np, y_train_np)

    # 3. Load Test Data
    X_test_df, y_test_s = _get_matrix(args.test_dataset, test_cfg)
    X_test_np = np.ascontiguousarray(X_test_df.to_numpy(dtype=np.float32))
    y_test_np = np.ascontiguousarray(y_test_s.to_numpy(dtype=np.float32))

    # 4. Evaluate
    logger.phase("Evaluating")
    y_prob = model.predict_proba(X_test_np)[:, 1]
    
    # Optional: Find optimal threshold on train data
    from sklearn.metrics import precision_recall_curve
    y_train_proba = model.predict_proba(X_train_np)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train_np, y_train_proba)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    logger.info(f"Optimal Threshold on Train Set: {optimal_threshold:.4f}")
    
    y_pred = (y_prob >= optimal_threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test_np, y_pred),
        "Precision": precision_score(y_test_np, y_pred, zero_division=0),
        "Recall": recall_score(y_test_np, y_pred, zero_division=0),
        "F1": f1_score(y_test_np, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test_np, y_pred),
        "AUC-ROC": roc_auc_score(y_test_np, y_prob),
        "AUC-PR": average_precision_score(y_test_np, y_prob)
    }

    print("\n" + "="*40)
    print("CROSS-SPECIES RESULTS")
    print("="*40)
    for m, v in metrics.items():
        if isinstance(v, float):
            print(f"{m:<15}: {v*100:.2f}%")
        else:
            print(f"{m:<15}: {v}")
    
    out_dir = Path("results/cross_species")
    out_dir.mkdir(exist_ok=True, parents=True)
    out_file = out_dir / f"{args.train_dataset}_to_{args.test_dataset}_{args.strategy}.json"
    
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Results saved to {out_file}")

    csv_file = out_dir / f"cross_species_{args.strategy}_summary.csv"
    metrics["Train"] = args.train_dataset
    metrics["Test"] = args.test_dataset
    df_new = pd.DataFrame([metrics])
    
    if csv_file.exists():
        df_existing = pd.read_csv(csv_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset=["Train", "Test"], keep="last", inplace=True)
        df_combined.to_csv(csv_file, index=False)
    else:
        df_new.to_csv(csv_file, index=False)
        
    logger.info(f"Summary CSV updated at {csv_file}")

if __name__ == "__main__":
    main()
