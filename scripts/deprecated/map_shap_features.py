import joblib
import h5py

model_path = 'results/human_same_go/models/model_fold1.joblib'
h5_cache_path = 'cache/human_human_pairs_same_go_facebook_esm2_t33_650m_ur50d_hadamard_abs_v3_features.h5'

with h5py.File(h5_cache_path, 'r') as hf:
    X_cols = [col.decode('utf-8') for col in hf['X_cols'][:]]

model = joblib.load(model_path)
opt_pipeline = model.estimators_[0]
col_transformer = opt_pipeline.named_steps['preprocessor']
_, transformer, input_cols = col_transformer.transformers_[0]

mask = transformer.get_support()
selected_feature_names = [X_cols[i] for idx, i in enumerate(input_cols) if mask[idx]]

top_indices = [891, 1007, 721, 1848, 971, 1723, 965, 978, 763, 923]

print("Mapping f_ indices to biological features:")
for i in top_indices:
    try:
        print(f"f_{i} -> {selected_feature_names[i]}")
    except IndexError:
        print(f"f_{i} -> Index out of bounds (Max: {len(selected_feature_names)})")
