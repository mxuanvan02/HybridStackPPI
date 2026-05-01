import os
import sys
import joblib
import pandas as pd
import numpy as np
import h5py
import shap
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def generate_shap_same_go(dataset='human'):
    model_path = f'results/{dataset}_same_go/models/model_fold1.joblib'
    h5_cache_path = f'cache/{dataset}_{dataset}_pairs_same_go_facebook_esm2_t33_650m_ur50d_hadamard_abs_v3_features.h5'
    output_dir = f'results/{dataset}_same_go/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Generating SHAP for {dataset} Same-GO ---")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"Loading data from {h5_cache_path}...")
    with h5py.File(h5_cache_path, 'r') as hf:
        num_samples = min(500, hf['X_data'].shape[0])
        X_sample_raw = hf['X_data'][-num_samples:]
        X_cols = [col.decode('utf-8') for col in hf['X_cols'][:]]
        
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Branch 0 is the Interp pipeline
    interp_pipeline = model.estimators_[0]
    col_transformer = interp_pipeline.named_steps['preprocessor']
    
    # Extract feature names
    _, transformer, input_cols = col_transformer.transformers_[0]
    if hasattr(transformer, 'selected_features_'):
        selected_feature_names = transformer.selected_features_
    else:
        # fallback
        mask = transformer.get_support()
        selected_feature_names = [X_cols[i] for idx, i in enumerate(input_cols) if mask[idx]]
    
    print(f"Transforming {num_samples} samples...")
    X_transformed = col_transformer.transform(X_sample_raw)
    X_df = pd.DataFrame(X_transformed, columns=selected_feature_names)
    
    print("Running SHAP Explainer...")
    lgbm_model = interp_pipeline.named_steps['model']
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(X_df)
    
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values_pos = shap_values[1]
    else:
        shap_values_pos = shap_values

    print(f"Saving plots to {output_dir}...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values_pos, X_df, max_display=20, show=False)
    plt.title(f'SHAP Value Interpretability - HybridStackPPI (Same-GO {dataset.capitalize()})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Copy to ACCESS
    access_dir = 'IEEE_Access/ACCESS_latex_template_20240429/figures_new'
    os.makedirs(access_dir, exist_ok=True)
    import shutil
    shutil.copy(os.path.join(output_dir, 'shap_summary.pdf'), os.path.join(access_dir, f'{dataset}_same_go_shap.pdf'))
    print(f"✅ {dataset.upper()} Done!")

if __name__ == "__main__":
    generate_shap_same_go('human')
    generate_shap_same_go('yeast')
