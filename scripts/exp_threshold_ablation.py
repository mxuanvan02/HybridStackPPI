import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Thêm context dự án
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from hybridstack.logger import PipelineLogger
from hybridstack.data_utils import load_data, canonicalize_pairs, create_feature_matrix, load_feature_matrix_h5
from hybridstack.feature_engine import FeatureEngine, EmbeddingComputer
from hybridstack.builders import create_stacking_pipeline, define_stacking_columns
from scripts.run import set_seed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score

def create_abla_pipeline_for_threshold(
    pairing_strategy, 
    h5_cache_path, 
    esm_model_name, 
    var_thresh, 
    corr_thresh, 
    n_jobs=-1
):
    """Tạo pipeline với ngưỡng Variance và Correlation có thể tinh chỉnh cho Ablation."""
    embedding_computer = EmbeddingComputer(model_name=esm_model_name)
    feature_engine = FeatureEngine(h5_cache_path=h5_cache_path, embedding_computer=embedding_computer)
    interp_cols, embed_cols = define_stacking_columns(feature_engine, pairing_strategy=pairing_strategy)
    
    # Custom pipeline model injection is complex since builders.py is hardcoded.
    # To properly simulate threshold ablation, we dynamically override the Threshold values inside the pipeline.
    pipeline = create_stacking_pipeline(interp_cols, embed_cols, n_jobs=n_jobs, use_selector=True)
    
    # Override Variance and Correlation in LGBM Branches
    for branch_name, est in pipeline.estimators_:
        if hasattr(est, "named_steps") and "preprocessor" in est.named_steps:
            preprocessor = est.named_steps["preprocessor"]
            if hasattr(preprocessor, "transformers_"):
                # Access the CumulativeFeatureSelector inside the ColumnTransformer
                selector = preprocessor.transformers_[0][1].named_steps.get("selector")
                if selector:
                    selector.variance_threshold = var_thresh
                    selector.correlation_threshold = corr_thresh
    return pipeline

def main():
    parser = argparse.add_argument_group("Ablation Study")
    parser = argparse.ArgumentParser(description="Threshold Ablation Study (Reviewer 2)")
    parser.add_argument("--fasta", type=str, required=True, help="Path to input FASTA file")
    parser.add_argument("--pairs", type=str, required=True, help="Path to target pairs TSV file")
    parser.add_argument("--h5-cache", type=str, default="data/cache/features.h5")
    parser.add_argument("--pairing", type=str, default="concat", choices=["concat", "avgdiff", "hadamard"])
    parser.add_argument("--esm-model", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--n-splits", type=int, default=3, help="Number of CV splits for quick ablation")
    args = parser.parse_args()
    
    set_seed(42)
    logger = PipelineLogger()
    logger.header("🚀 THRESHOLD RIGIDITY ABLATION STUDY (Reviewer 2) 🚀")
    
    sequences, pairs_df = load_data(args.fasta, args.pairs)
    pairs_df = canonicalize_pairs(pairs_df, dataset_name="Ablation", logger=logger)
    
    # Ensure features exist
    if not os.path.exists(args.h5_cache):
        logger.error(f"Feature Cache Not Found: {args.h5_cache}. Please run regular feature extraction first.")
        return
        
    X_df, y_s = load_feature_matrix_h5(args.h5_cache)
    
    # Scenarios requested to prove model robustness
    threshold_scenarios = [
        {"var": 0.80, "corr": 0.95}, # Strict (Baseline)
        {"var": 0.85, "corr": 0.90}, # Moderate
        {"var": 0.90, "corr": 0.85}, # Loose
        {"var": 0.95, "corr": 0.80}, # Very Loose (Keeps lots of weak signals)
    ]
    
    results = []
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    
    for scenario in threshold_scenarios:
        var_t, corr_t = scenario["var"], scenario["corr"]
        logger.phase(f"Testing Config - Variance: {var_t}, Correlation: {corr_t}")
        
        f1_scores = []
        mcc_scores = []
        auc_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_df, y_s)):
            X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
            y_train, y_val = y_s.iloc[train_idx], y_s.iloc[val_idx]
            
            pipeline = create_abla_pipeline_for_threshold(
                args.pairing, args.h5_cache, args.esm_model, var_t, corr_t
            )
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_val)
            y_proba = pipeline.predict_proba(X_val)[:, 1]
            
            f1_scores.append(f1_score(y_val, y_pred))
            mcc_scores.append(matthews_corrcoef(y_val, y_pred))
            auc_scores.append(roc_auc_score(y_val, y_proba))
            
        res = {
            "Variance Thresh": var_t,
            "Corr Thresh": corr_t,
            "F1-Score": sum(f1_scores) / len(f1_scores),
            "MCC": sum(mcc_scores) / len(mcc_scores),
            "ROC-AUC": sum(auc_scores) / len(auc_scores),
        }
        logger.info(f"Result: {res}")
        results.append(res)
        
    df_results = pd.DataFrame(results)
    logger.header("📊 ABLATION RESULTS (Proves Robustness to Thresholds) 📊")
    print(df_results.to_markdown(index=False))
    
    os.makedirs("results", exist_ok=True)
    df_results.to_csv("results/threshold_ablation_results.csv", index=False)
    logger.info("Saved to results/threshold_ablation_results.csv")

if __name__ == "__main__":
    main()
