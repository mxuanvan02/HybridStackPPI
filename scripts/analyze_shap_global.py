import os
import sys
import argparse
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from hybridstack.logger import PipelineLogger
from hybridstack.data_utils import load_data, canonicalize_pairs, load_feature_matrix_h5
from hybridstack.feature_engine import FeatureEngine, EmbeddingComputer
from hybridstack.builders import create_stacking_pipeline, define_stacking_columns
from scripts.run import set_seed

def explain_global_shap(
    fasta_path: str,
    pairs_path: str,
    h5_cache_path: str,
    esm_model_name: str,
    pairing_strategy: str = "concat",
    n_sample_background: int = 1000,
    n_sample_explain: int = 5000,
):
    logger = PipelineLogger()
    logger.header("🔍 SYSTEMATIC SHAP INTERPRETABILITY ANALYSIS (Reviewer 1 & 2) 🔍")

    set_seed(42)

    if not os.path.exists(h5_cache_path):
        logger.error(f"Cannot find feature matrix cache at {h5_cache_path}")
        return

    logger.phase("1. Loading Cached Feature Data")
    X_df, y_s = load_feature_matrix_h5(h5_cache_path)

    # Lấy mô hình cơ bản (Ở đây ta lấy nhánh Interpretable LGBM để giải thích dễ nhất)
    # Vì GridSearchCV LGBM rất dễ tương thích với SHAP TreeExplainer
    logger.phase("2. Training Baseline Explainable Model")
    embedding_computer = EmbeddingComputer(model_name=esm_model_name)
    feature_engine = FeatureEngine(h5_cache_path=h5_cache_path, embedding_computer=embedding_computer)
    interp_cols, embed_cols = define_stacking_columns(feature_engine, pairing_strategy)
    
    # Huấn luyện mô hình ngay trên toàn bộ dữ liệu (Hoặc một split)
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from lightgbm import LGBMClassifier
    from hybridstack.builders import CumulativeFeatureSelector
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('interp', Pipeline([('selector', CumulativeFeatureSelector())]), interp_cols)
        ]
    )
    
    model = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    
    # Rút gọn dữ liệu cho nhanh
    sample_idx = X_df.sample(min(len(X_df), n_sample_explain)).index
    X_explain = X_df.loc[sample_idx]
    y_explain = y_s.loc[sample_idx]
    
    logger.phase("Fitting Feature Selector on Interpretable Columns...")
    X_trans = preprocessor.fit_transform(X_explain)
    
    # Get physical names of selected features
    selector = preprocessor.named_transformers_['interp'].named_steps['selector']
    saved_feat_names = selector.selected_features_
    
    X_trans_df = pd.DataFrame(X_trans, columns=saved_feat_names)
    
    logger.phase("Training LightGBM Core Model...")
    model.fit(X_trans_df, y_explain)
    
    logger.phase("3. Running SHAP TreeExplainer on Full Test Cohort")
    explainer = shap.TreeExplainer(model)
    
    # Giải thích trên toàn bộ Data để tạo Bằng chứng Thống kê (Systematic Evidence)
    shap_values = explainer.shap_values(X_trans_df)
    
    # Tùy bản LightGBM, shap_values có thể trả về list [negative, positive]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    logger.phase("4. Generating Global Interpretability Plots")
    os.makedirs("results/plots", exist_ok=True)
    
    # Summary Plot (Mức độ quan trọng toàn cầu)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_trans_df, show=False)
    plt.title("Systematic SHAP Global Feature Importance (All Test Samples)")
    plt.tight_layout()
    plt.savefig("results/plots/shap_global_summary_reviewer.png", dpi=300)
    plt.close()
    
    # Tính tóan độ quan trọng theo Absolute Mean SHAP để xuất ra bảng
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "Feature": saved_feat_names,
        "Mean_Abs_SHAP": mean_shap
    }).sort_values(by="Mean_Abs_SHAP", ascending=False)
    
    logger.info("\nTop 15 Most Biologically Important Features (Systematic Evidence):")
    print(shap_df.head(15).to_markdown(index=False))
    shap_df.to_csv("results/shap_global_importance.csv", index=False)
    
    logger.success("✅ Saved Global SHAP summary to results/plots/shap_global_summary_reviewer.png")
    logger.success("✅ Saved importance CSV to results/shap_global_importance.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Systematic SHAP Interpretability Evidence")
    parser.add_argument("--fasta", type=str, required=True, help="FASTA file path")
    parser.add_argument("--pairs", type=str, required=True, help="Pairs TSV file path")
    parser.add_argument("--h5-cache", type=str, default="data/cache/features.h5")
    parser.add_argument("--samples", type=int, default=10000, help="Number of instances to explain")
    args = parser.parse_args()
    
    # Need numpy inside the function scope due to late bindings, adding here globally
    global np
    import numpy as np
    
    explain_global_shap(
        fasta_path=args.fasta,
        pairs_path=args.pairs,
        h5_cache_path=args.h5_cache,
        esm_model_name="facebook/esm2_t33_650M_UR50D",
        n_sample_explain=args.samples
    )
