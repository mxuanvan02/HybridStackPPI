import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
import h5py
import shap
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hybridstack.logger import PipelineLogger

def generate_shap_evidence():
    logger = PipelineLogger()
    logger.header("🔍 GENERATING SHAP EVIDENCE 🔍")
    
    model_path = 'results/human/models/model_fold1.joblib'
    h5_cache_path = 'cache/human_human_pairs_same_compartment_facebook_esm2_t33_650m_ur50d_hadamard_abs_v3_features.h5'
    
    logger.phase("1. Loading Test Samples from HDF5")
    if not os.path.exists(h5_cache_path):
        logger.error(f"Cannot find feature matrix cache at {h5_cache_path}")
        return
        
    with h5py.File(h5_cache_path, 'r') as hf:
        num_samples = min(1000, hf['X_data'].shape[0])
        X_sample_raw = hf['X_data'][-num_samples:]
        X_cols = [col.decode('utf-8') for col in hf['X_cols'][:]]
        
    X_sample_raw_df = pd.DataFrame(X_sample_raw, columns=X_cols)
    logger.info(f"Loaded {num_samples} samples: X_raw_shape={X_sample_raw_df.shape}")

    logger.phase("2. Loading Trained Stacking Model")
    model = joblib.load(model_path)
    
    # Branch 0 is the Interp pipeline (Handcrafted + Motif features)
    interp_pipeline = model.estimators_[0]
    col_transformer = interp_pipeline.named_steps['preprocessor']
    
    # Extract feature selection mask / names
    _, transformer, input_cols = col_transformer.transformers_[0]
    
    if hasattr(transformer, 'selected_features_'):
        selected_feature_names = transformer.selected_features_
        logger.info(f"Extracted {len(selected_feature_names)} selected features.")
    else:
        logger.info("Could not find selected_features_. Using fallback naming.")
        # Fallback to map the indices
        if hasattr(transformer, 'get_support'):
            mask = transformer.get_support()
            selected_feature_names = [X_cols[i] for idx, i in enumerate(input_cols) if mask[idx]]
        else:
            selected_feature_names = [f"Interp_F_{i}" for i in range(len(input_cols))]
            
    logger.phase("3. Transforming raw data through Preprocessor (using mapped input columns)")
    # ColumnTransformer in HybridStackPPI uses integer indices for columns based on builders.py refactor
    # To avoid shape errors, we subset the exact array space it expects or pass Numpy directly
    X_sample_transformed = col_transformer.transform(X_sample_raw)
    
    if isinstance(X_sample_transformed, np.ndarray):
        if X_sample_transformed.shape[1] == len(selected_feature_names):
            X_sample_df = pd.DataFrame(X_sample_transformed, columns=selected_feature_names)
        else:
            X_sample_df = pd.DataFrame(X_sample_transformed) # Fallback without names if mismatched
            logger.info("Shape mismatch in column names alignment. Reverting to integer columns.")
    else:
        # PANDAS DataFrame transform
        values = X_sample_transformed.values if hasattr(X_sample_transformed, 'values') else X_sample_transformed
        if values.shape[1] == len(selected_feature_names):
            X_sample_df = pd.DataFrame(values, columns=selected_feature_names)
        else:
            X_sample_df = pd.DataFrame(values)
            
    logger.info(f"Transformed test sample shape: {X_sample_df.shape}")

    logger.phase("4. Extracting LightGBM Base Learner & Applying SHAP")
    lgbm_model = interp_pipeline.named_steps['model']
    
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(X_sample_df)
    
    # Handle SHAP versions compatibility
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values_pos = shap_values[1]
    elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
        shap_values_pos = shap_values[:, :, 1]
    elif hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3 and shap_values.values.shape[2] == 2:
         shap_values_pos = shap_values.values[:, :, 1]
    else:
        shap_values_pos = shap_values
        
    logger.phase("5. Generating and Saving Plot")
    os.makedirs('figures', exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_pos, X_sample_df, max_display=20, show=False)
    plt.title('SHAP Value Interpretability - HybridStackPPI (Top 20 Features)')
    plt.tight_layout()
    plt.savefig('figures/shap_interpretability_evidence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("✅ Saved SHAP visual evidence to figures/shap_interpretability_evidence.png")
    
    # Text Analysis check
    try:
        mean_abs_shap = np.abs(shap_values_pos).mean(axis=0)
        shap_importance = pd.DataFrame({
            'Feature': X_sample_df.columns,
            'Mean_Abs_SHAP': mean_abs_shap
        }).sort_values(by='Mean_Abs_SHAP', ascending=False)
        
        top_10 = shap_importance.head(10)['Feature'].astype(str).tolist()
        has_motif = any('Motif' in f for f in top_10)
        has_physical = any(term in f for f in top_10 for term in ['Charge', 'Hydrophobicity', 'Mass', 'Polar'])
        
        print("\n" + "="*50)
        print("🌟 TOP 10 FEATURES SUMMARY 🌟")
        print("="*50)
        for i, feature in enumerate(top_10, 1):
            print(f"  {i}. {feature}")
            
        print("\n" + "="*50)
        print("📊 EVIDENCE ANALYSIS 📊")
        print("="*50)
        if has_motif:
             print("✅ Yes, Motif features appear in the top 10. (Biological significance confirmed)")
        else:
             print("❌ No, Motif features do not appear in the top 10.")
             
        if has_physical:
             print("✅ Yes, Physical features (e.g. Charge, Hydrophobicity) appear in the top 10. (Strong interpretable signal)")
        else:
             print("❌ No, Physical features do not appear in the top 10.")
             
        with open("figures/shap_summary.txt", "w") as f:
            f.write("SHAP Top 10 Features:\n")
            f.write("\n".join([f"{i+1}. {x}" for i, x in enumerate(top_10)]))
            f.write("\n\nAnalysis:\n")
            f.write("Motif in Top 10: " + str(has_motif) + "\n")
            f.write("Physical properties in Top 10: " + str(has_physical) + "\n")
    except Exception as e:
        logger.warning(f"Could not generate feature text summary due to missing feature names: {e}")

if __name__ == "__main__":
    generate_shap_evidence()

