#!/usr/bin/env python3
import argparse
import os
import sys
import time
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
    parser.add_argument("--strategy", choices=["random", "same_compartment", "same_go"], default="same_go")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from scripts.run import define_stacking_columns, create_stacking_pipeline, run_experiment
    from hybridstack.feature_engine import FeatureEngine, EmbeddingComputer
    from hybridstack.data_utils import load_data, create_feature_matrix
    from hybridstack.logger import PipelineLogger

    logger = PipelineLogger()
    logger.header(f"CROSS-SPECIES EVALUATION: {args.train_dataset.upper()} -> {args.test_dataset.upper()} ({args.strategy})")

    # Paths
    dataset_cfg = {
        "human": {
            "fasta": "data/BioGrid/Human/human_dict.fasta",
            "pairs": f"data/BioGrid/Human/human_pairs_{args.strategy}.tsv",
        },
        "yeast": {
            "fasta": "data/BioGrid/Yeast/yeast_dict.fasta",
            "pairs": f"data/BioGrid/Yeast/yeast_pairs_{args.strategy}.tsv",
        }
    }

    train_cfg = dataset_cfg[args.train_dataset]
    test_cfg = dataset_cfg[args.test_dataset]

    # Initialize Feature Engine
    esm_model = "facebook/esm2_t33_650M_UR50D"
    h5_cache = "cache/esm2/esm2_embeddings_v4.h5"
    embedding_computer = EmbeddingComputer(model_name=esm_model)
    feature_engine = FeatureEngine(h5_cache_path=h5_cache, embedding_computer=embedding_computer)

    # 1. Load Train Data
    logger.phase(f"Loading Train Data ({args.train_dataset})")
    train_pairs, train_labels, train_universe = load_data(train_cfg["fasta"], train_cfg["pairs"])
    X_train = create_feature_matrix(train_pairs, train_universe, feature_engine, pairing_strategy="hadamard_abs")
    
    # 2. Train Model
    logger.phase("Training Model")
    interp_cols, embed_cols = define_stacking_columns(feature_engine, pairing_strategy="hadamard_abs")
    model = create_stacking_pipeline(
        interp_cols=interp_cols,
        embed_cols=embed_cols,
        use_selector=True,
        meta_learner_type="lr"
    )
    model.fit(X_train, train_labels)

    # 3. Load Test Data
    logger.phase(f"Loading Test Data ({args.test_dataset})")
    test_pairs, test_labels, test_universe = load_data(test_cfg["fasta"], test_cfg["pairs"])
    X_test = create_feature_matrix(test_pairs, test_universe, feature_engine, pairing_strategy="hadamard_abs")

    # 4. Evaluate
    logger.phase("Evaluating")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(test_labels, y_pred),
        "Precision": precision_score(test_labels, y_pred),
        "Recall": recall_score(test_labels, y_pred),
        "F1": f1_score(test_labels, y_pred),
        "MCC": matthews_corrcoef(test_labels, y_pred),
        "AUC-ROC": roc_auc_score(test_labels, y_prob),
        "AUC-PR": average_precision_score(test_labels, y_prob)
    }

    print("\n" + "="*40)
    print("CROSS-SPECIES RESULTS")
    print("="*40)
    for m, v in metrics.items():
        print(f"{m:<15}: {v*100:.2f}%")
    
    # Output result to file for reporting
    out_dir = Path("results/cross_species")
    out_dir.mkdir(exist_ok=True, parents=True)
    out_file = out_dir / f"{args.train_dataset}_to_{args.test_dataset}_{args.strategy}.json"
    import json
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
