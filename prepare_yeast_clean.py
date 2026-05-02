import pandas as pd
import numpy as np
import pickle
import h5py
import os
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from hybridstack.feature_engine import FeatureEngine

def create_clean_cache(dataset):
    print(f"--- Preparing Clean Cache for {dataset} ---")
    pairs_path = f"data/BioGrid/{dataset.capitalize()}/{dataset}_pairs_same_go.tsv"
    ckpt_path = f"cache/{dataset}_protein_features_ckpt.pkl"
    
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return

    # 1. Load data
    pairs_df = pd.read_csv(pairs_path, sep="\t", header=None, names=["protein1", "protein2", "label"])
    
    # 2. Canonicalize like run_experiment
    p1 = pairs_df["protein1"].values
    p2 = pairs_df["protein2"].values
    mask = p1 > p2
    p1[mask], p2[mask] = p2[mask], p1[mask]
    pairs_df["protein1"] = p1
    pairs_df["protein2"] = p2
    
    # Drop duplicates
    pairs_clean = pairs_df.drop_duplicates(subset=["protein1", "protein2"])
    print(f"Cleaned {len(pairs_df)} -> {len(pairs_clean)} pairs.")
    
    # 3. Load features
    with open(ckpt_path, "rb") as f:
        protein_features = pickle.load(f)
    
    # 4. Get correct feature names from FeatureEngine
    mock_comp = MockEmbeddingComputer(embedding_dim=1280)
    engine = FeatureEngine(h5_cache_path="cache/esm2/esm2_embeddings_v4.h5", embedding_computer=mock_comp)
    individual_feature_names = engine.get_feature_names()
    
    # Construct pair names correctly (Hadamard_ and AbsDiff_)
    pair_feature_names = [f"Hadamard_{n}" for n in individual_feature_names] + [f"AbsDiff_{n}" for n in individual_feature_names]
    
    # 5. Build matrix
    num_pairs = len(pairs_clean)
    num_feat = len(pair_feature_names)
    
    print(f"Allocating {num_pairs} x {num_feat}")
    X_np = np.zeros((num_pairs, num_feat), dtype=np.float32)
    y_np = pairs_clean["label"].values.astype(np.float32)
    
    for i, (idx, row) in enumerate(pairs_clean.iterrows()):
        f1 = protein_features[row["protein1"]]
        f2 = protein_features[row["protein2"]]
        X_np[i, :] = np.concatenate([f1 * f2, np.abs(f1 - f2)])
    
    # 6. Save with all required fields
    cache_path = f"cache/{dataset}_{dataset}_pairs_same_go_facebook_esm2_t33_650m_ur50d_hadamard_abs_v3_features.h5"
    print(f"Saving to {cache_path}")
    
    with h5py.File(cache_path, "w") as hf:
        hf.create_dataset("X_data", data=X_np)
        hf.create_dataset("y_data", data=y_np)
        hf.create_dataset("X_index", data=np.arange(num_pairs))
        hf.create_dataset("y_index", data=np.arange(num_pairs))
        hf.attrs["y_name"] = "label"
        
        # Save columns in both formats
        hf.create_dataset("X_cols", data=np.array([c.encode("utf-8") for c in pair_feature_names], dtype="S"))
        
        dt = h5py.special_dtype(vlen=str)
        ds = hf.create_dataset("feature_names_arr", (len(pair_feature_names),), dtype=dt)
        ds[:] = pair_feature_names

    print("Success.")

# Mock class if not available in imports
class MockEmbeddingComputer:
    def __init__(self, embedding_dim=1280): self.embedding_dim = embedding_dim

if __name__ == "__main__":
    create_clean_cache("yeast")
