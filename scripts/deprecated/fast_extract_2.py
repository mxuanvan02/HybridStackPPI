import os
import sys
import numpy as np
import pandas as pd
import h5py
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import json

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from hybridstack.data_utils import load_data, create_feature_matrix
from hybridstack.feature_engine import FeatureEngine, EmbeddingComputer, InterpretableFeatureExtractor

GLOBAL_MOTIFS = None
GLOBAL_H5 = "cache/esm2/esm2_embeddings_v4.h5"

def init_worker(motifs_dict):
    global GLOBAL_MOTIFS
    GLOBAL_MOTIFS = motifs_dict

def extract_single(args):
    seq_id, sequence = args
    handcraft_ext = InterpretableFeatureExtractor()
    handcraft_feats = handcraft_ext.extract(sequence)
    
    with h5py.File(GLOBAL_H5, "r", swmr=True) as h5f:
        seq_upper = sequence.upper()
        matrix_key = f"{seq_upper}_matrix_v2"
        global_key = f"{seq_upper}_global_v2"
        
        if matrix_key in h5f and global_key in h5f:
            embedding_matrix = h5f[matrix_key][:]
            global_embedding = h5f[global_key][:]
        else:
            embedding_matrix = np.zeros((len(sequence), 1280), dtype=np.float32)
            global_embedding = np.zeros(1280, dtype=np.float32)
            
    motif_binary_vector = []
    local_embedding_vectors = []
    
    if embedding_matrix.shape[0] == 0:
        motif_binary_vector = [0] * len(GLOBAL_MOTIFS)
        final_local_embedding = np.zeros(2 * 1280, dtype=np.float32)
    else:
        for elm_id, pattern in GLOBAL_MOTIFS.items():
            matches = list(pattern.finditer(sequence))
            if matches:
                motif_binary_vector.append(1)
                for match in matches:
                    start, end = match.span()
                    start = min(start, embedding_matrix.shape[0] - 1)
                    end = min(end, embedding_matrix.shape[0])
                    if start < end:
                        motif_embs = embedding_matrix[start:end]
                        if motif_embs.shape[0] > 0:
                            local_embedding_vectors.append(motif_embs)
            else:
                motif_binary_vector.append(0)
                
        if local_embedding_vectors:
            all_motif_embs = np.vstack(local_embedding_vectors)
            max_pool = np.max(all_motif_embs, axis=0)
            mean_pool = np.mean(all_motif_embs, axis=0)
            final_local_embedding = np.concatenate([max_pool, mean_pool])
        else:
            length = embedding_matrix.shape[0]
            start_idx = length // 4
            end_idx = length - start_idx
            if start_idx < end_idx:
                region = embedding_matrix[start_idx:end_idx]
            else:
                region = embedding_matrix
            max_pool = region.max(axis=0)
            mean_pool = region.mean(axis=0)
            final_local_embedding = np.concatenate([max_pool, mean_pool])
            
    combined_vector = np.concatenate([
        handcraft_feats, 
        np.array(motif_binary_vector, dtype=np.float32), 
        global_embedding, 
        final_local_embedding
    ])
    return seq_id, combined_vector

def main():
    global GLOBAL_MOTIFS
    h5_cache_path = "cache/esm2/esm2_embeddings_v4.h5"
    
    embedding_computer = EmbeddingComputer(model_name="facebook/esm2_t33_650M_UR50D")
    feature_engine = FeatureEngine(h5_cache_path=h5_cache_path, embedding_computer=embedding_computer)
    feature_names = feature_engine.get_feature_names()
    GLOBAL_MOTIFS = feature_engine.motifs

    for dataset in ["human", "yeast"]:
        print(f"Processing {dataset}...", flush=True)
        fasta_path = f"data/BioGrid/{dataset.capitalize()}/{dataset}_dict.fasta"
        pairs_path = f"data/BioGrid/{dataset.capitalize()}/{dataset}_pairs_same_go.tsv"
        
        seqs, pairs_df = load_data(fasta_path, pairs_path)
        needed_seqs = {seq_id: seq for seq_id, seq in seqs.items() if seq_id in set(pairs_df["protein1"]).union(set(pairs_df["protein2"]))}
        
        print(f"Extracting features for {len(needed_seqs)} sequences...", flush=True)
        tasks = [(seq_id, seq) for seq_id, seq in needed_seqs.items()]
        
        protein_features = {}
        with mp.Pool(processes=6) as pool:
            for seq_id, combined_vector in tqdm(pool.imap_unordered(extract_single, tasks), total=len(tasks)):
                protein_features[seq_id] = combined_vector
                
        print("Creating feature matrix...", flush=True)
        X_df, y_s = create_feature_matrix(pairs_df, protein_features, feature_names, pairing_strategy="hadamard_abs")
        
        cache_name = f"cache/{dataset}_{dataset}_pairs_same_go_facebook_esm2_t33_650m_ur50d_hadamard_abs_v3_features.h5"
        print(f"Saving to {cache_name}...", flush=True)
        with h5py.File(cache_name, "w") as hf:
            hf.create_dataset("X_data", data=np.ascontiguousarray(X_df.values, dtype=np.float32))
            hf.create_dataset("y_data", data=np.ascontiguousarray(y_s.values, dtype=np.float32))
            dt = h5py.special_dtype(vlen=str)
            ds = hf.create_dataset("feature_names_arr", (len(X_df.columns),), dtype=dt)
            ds[:] = list(X_df.columns)
            
        print(f"Done {dataset}!", flush=True)

if __name__ == "__main__":
    main()
