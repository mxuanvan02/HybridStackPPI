import os
import sys
import numpy as np
import pandas as pd
import h5py
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import re
import io
from collections import Counter

# --- Constants copied from feature_engine.py ---
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
CTD_GROUPS = {
    "hydrophobicity": {"polar": "RKEDQN", "neutral": "GASTPHY", "hydrophobic": "CVLIMFW"},
    "normalized_vdw": {"small": "GASTPDC", "medium": "NVEQIL", "large": "MHKFRYW"},
    "polarity": {"neutral": "LIFWCMVY", "polar": "PATGS", "charged": "HQRKNED"},
    "polarizability": {"low": "GASDT", "medium": "CPNVEQIL", "high": "KMHFRYW"},
    "charge": {"positive": "KR", "neutral": "ANCQGHILMFPSTWYV", "negative": "DE"},
    "secondary_structure": {"helix": "EALMQKRH", "strand": "VIYW", "coil": "GNPSD"},
    "solvent_accessibility": {"buried": "ALFCGIVW", "exposed": "MSPTHY", "intermediate": "NQRKDE"},
}
PHYSICO_INDICES = {
    "hydrophobicity": {"A": 0.62, "C": 0.29, "D": -0.90, "E": -0.74, "F": 1.19, "G": 0.48, "H": -0.40, "I": 1.38, "K": -1.50, "L": 1.06, "M": 0.64, "N": -0.78, "P": 0.12, "Q": -0.85, "R": -2.53, "S": -0.18, "T": -0.05, "V": 1.08, "W": 0.81, "Y": 0.26},
    "hydrophilicity": {"A": -0.5, "C": -1.0, "D": 3.0, "E": 3.0, "F": -2.5, "G": 0.0, "H": -0.5, "I": -1.8, "K": 3.0, "L": -1.8, "M": -1.3, "N": 0.2, "P": 0.0, "Q": 0.2, "R": 3.0, "S": 0.3, "T": -0.4, "V": -1.5, "W": -3.4, "Y": -2.3},
}

# --- Handcrafted Extractors copied from feature_engine.py ---
class AACExtractor:
    def __init__(self):
        self.feature_names = [f"AAC_{aa}" for aa in AMINO_ACIDS]
    def compute(self, sequence: str) -> np.ndarray:
        seq = sequence.upper()
        length = len(seq) or 1
        counts = Counter(seq)
        return np.array([counts.get(aa, 0) / length for aa in AMINO_ACIDS], dtype=np.float32)

class DPCExtractor:
    def __init__(self):
        self.dipeptides = [aa1 + aa2 for aa1 in AMINO_ACIDS for aa2 in AMINO_ACIDS]
        self.feature_names = [f"DPC_{dp}" for dp in self.dipeptides]
    def compute(self, sequence: str) -> np.ndarray:
        seq = sequence.upper()
        length = len(seq) - 1 or 1
        counts = Counter(seq[j : j + 2] for j in range(len(seq) - 1))
        return np.array([counts.get(dp, 0) / length for dp in self.dipeptides], dtype=np.float32)

class CTDExtractor:
    def __init__(self):
        self.prop_mappers = {}
        self.feature_names = []
        for prop, groups in CTD_GROUPS.items():
            mapper = {}
            for i, aas in enumerate(groups.values()):
                for aa in aas:
                    mapper[aa] = i
            self.prop_mappers[prop] = mapper
        for prop in CTD_GROUPS.keys():
            self.feature_names.extend([f"CTD_{prop}_C_{i+1}" for i in range(3)])
            self.feature_names.extend([f"CTD_{prop}_T_{t}" for t in ["12", "13", "23"]])
            for i in range(3):
                self.feature_names.extend([f"CTD_{prop}_D_{i+1}_{q}" for q in [0, 25, 50, 75, 100]])
    def compute(self, sequence: str) -> np.ndarray:
        seq = sequence.upper()
        features = []
        seq_len = len(seq) or 1
        for prop, mapper in self.prop_mappers.items():
            seq_groups = [mapper.get(aa, -1) for aa in seq]
            counts = Counter(seq_groups)
            features.extend([counts.get(i, 0) / seq_len for i in range(3)])
            trans = {"12": 0, "13": 0, "23": 0}
            for i in range(seq_len - 1):
                g1, g2 = seq_groups[i], seq_groups[i + 1]
                if g1 != -1 and g2 != -1 and g1 != g2:
                    pair = tuple(sorted((g1, g2)))
                    if pair == (0, 1): trans["12"] += 1
                    elif pair == (0, 2): trans["13"] += 1
                    elif pair == (1, 2): trans["23"] += 1
            features.extend([v / (seq_len - 1 or 1) for v in trans.values()])
            for gid in range(3):
                positions = [i for i, g in enumerate(seq_groups) if g == gid]
                if positions:
                    quartiles = np.percentile(positions, [0, 25, 50, 75, 100]) / (seq_len - 1 or 1)
                    features.extend(quartiles)
                else:
                    features.extend([0.0] * 5)
        return np.array(features, dtype=np.float32)

class PAACExtractor:
    def __init__(self, lambda_val: int = 10, weight: float = 0.05):
        self.lambda_val = lambda_val
        self.weight = weight
        self.aac_extractor = AACExtractor()
        self.feature_names = [f"PAAC_AAC_{aa}" for aa in AMINO_ACIDS]
        self.feature_names.extend([f"PAAC_lambda_{i+1}" for i in range(lambda_val)])
    def compute(self, sequence: str) -> np.ndarray:
        seq = sequence.upper()
        seq_len = len(seq) or 1
        aac = self.aac_extractor.compute(seq)
        hydro = [PHYSICO_INDICES["hydrophobicity"].get(aa, 0) for aa in seq]
        hydrophil = [PHYSICO_INDICES["hydrophilicity"].get(aa, 0) for aa in seq]
        theta = []
        for lag in range(1, self.lambda_val + 1):
            if seq_len > lag:
                corr = sum((hydro[i] - hydro[i + lag]) ** 2 + (hydrophil[i] - hydrophil[i + lag]) ** 2 for i in range(seq_len - lag))
                theta.append(corr / (seq_len - lag))
            else: theta.append(0.0)
        denominator = 1 + self.weight * sum(theta)
        return np.concatenate([aac / denominator, (self.weight * np.array(theta)) / denominator]).astype(np.float32)

class MoranAutocorrelation:
    def __init__(self, max_lag: int = 30):
        self.max_lag = max_lag
        self.properties = list(PHYSICO_INDICES.keys())
        self.feature_names = [f"Moran_{p}_lag{l}" for p in self.properties for l in range(1, max_lag + 1)]
    def compute(self, sequence: str) -> np.ndarray:
        seq = sequence.upper()
        seq_len = len(seq)
        all_features = []
        for prop_name in self.properties:
            prop_seq = np.array([PHYSICO_INDICES[prop_name].get(aa, 0) for aa in seq], dtype=np.float32)
            if seq_len < 2:
                all_features.extend([0.0] * self.max_lag)
                continue
            mean_prop = np.mean(prop_seq)
            std_dev = np.std(prop_seq) + 1e-9
            norm_prop_seq = (prop_seq - mean_prop) / std_dev
            for lag in range(1, self.max_lag + 1):
                if seq_len > lag:
                    all_features.append(np.sum(norm_prop_seq[:-lag] * norm_prop_seq[lag:]) / (seq_len - lag))
                else: all_features.append(0.0)
        return np.array(all_features, dtype=np.float32)

class InterpretableFeatureExtractor:
    def __init__(self):
        self.extractors = {"AAC": AACExtractor(), "DPC": DPCExtractor(), "CTD": CTDExtractor(), "PAAC": PAACExtractor(), "Moran": MoranAutocorrelation()}
    def extract(self, sequence: str) -> np.ndarray:
        return np.concatenate([ext.compute(sequence) for ext in self.extractors.values()])

# --- Load Data copied from data_utils.py ---
def load_data_local(fasta_path: str, pairs_path: str):
    sequences = {}
    with open(fasta_path, "r") as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                header = line.split()[0][1:]
                sequences[header] = ""
            elif header: sequences[header] += line
    pairs_df = pd.read_csv(pairs_path, sep="\t", header=None, names=["protein1", "protein2", "label"])
    return sequences, pairs_df

# --- Workers and Globals ---
GLOBAL_MOTIFS = None
GLOBAL_H5 = "cache/esm2/esm2_embeddings_v4.h5"

def init_worker(motifs_dict):
    global GLOBAL_MOTIFS
    GLOBAL_MOTIFS = motifs_dict

def extract_single(args):
    seq_id, sequence = args
    handcraft_feats = InterpretableFeatureExtractor().extract(sequence)
    with h5py.File(GLOBAL_H5, "r", swmr=True) as h5f:
        seq_upper = sequence.upper()
        m_key, g_key = f"{seq_upper}_matrix_v2", f"{seq_upper}_global_v2"
        if m_key in h5f and g_key in h5f:
            emb_matrix, global_emb = h5f[m_key][:], h5f[g_key][:]
        else:
            emb_matrix = np.zeros((len(sequence), 1280), dtype=np.float32)
            global_emb = np.zeros(1280, dtype=np.float32)
    motif_vec, local_embs = [], []
    if emb_matrix.shape[0] == 0:
        motif_vec = [0.0] * len(GLOBAL_MOTIFS)
        local_motif_emb = np.zeros(2560, dtype=np.float32)
    else:
        for pattern in GLOBAL_MOTIFS.values():
            matches = list(pattern.finditer(sequence))
            if matches:
                motif_vec.append(1.0)
                for m in matches:
                    s, e = m.span()
                    if s < emb_matrix.shape[0]:
                        local_embs.append(emb_matrix[s:min(e, emb_matrix.shape[0])])
            else: motif_vec.append(0.0)
        if local_embs:
            stacked = np.vstack(local_embs)
            local_motif_emb = np.concatenate([np.max(stacked, axis=0), np.mean(stacked, axis=0)])
        else:
            l = emb_matrix.shape[0]
            reg = emb_matrix[l//4 : l-l//4] if l//4 < l-l//4 else emb_matrix
            local_motif_emb = np.concatenate([np.max(reg, axis=0), np.mean(reg, axis=0)])
    return seq_id, np.concatenate([handcraft_feats, np.array(motif_vec, dtype=np.float32), global_emb, local_motif_emb])

def main():
    # Load motifs once without importing full hybridstack if possible, or just import here
    from hybridstack.feature_engine import FeatureEngine
    engine = FeatureEngine(h5_cache_path=GLOBAL_H5, embedding_computer=None)
    motifs_dict = engine.motifs
    feature_names = engine.get_feature_names()

    for dataset in ["human", "yeast"]:
        print(f"Processing {dataset}...", flush=True)
        f_path = f"data/BioGrid/{dataset.capitalize()}/{dataset}_dict.fasta"
        p_path = f"data/BioGrid/{dataset.capitalize()}/{dataset}_pairs_same_go.tsv"
        seqs, pairs_df = load_data_local(f_path, p_path)
        needed = {sid: s for sid, s in seqs.items() if sid in set(pairs_df["protein1"]).union(set(pairs_df["protein2"]))}
        print(f"Extracting for {len(needed)} sequences...", flush=True)
        tasks = [(sid, s) for sid, s in needed.items()]
        protein_features = {}
        with mp.Pool(processes=8, initializer=init_worker, initargs=(motifs_dict,)) as pool:
            for sid, vec in tqdm(pool.imap_unordered(extract_single, tasks), total=len(tasks)):
                protein_features[sid] = vec
        print("Creating matrix...", flush=True)
        from hybridstack.data_utils import create_feature_matrix
        X_df, y_s = create_feature_matrix(pairs_df, protein_features, feature_names, pairing_strategy="hadamard_abs")
        c_name = f"cache/{dataset}_{dataset}_pairs_same_go_facebook_esm2_t33_650m_ur50d_hadamard_abs_v3_features.h5"
        with h5py.File(c_name, "w") as hf:
            hf.create_dataset("X_data", data=X_df.values.astype(np.float32))
            hf.create_dataset("y_data", data=y_s.values.astype(np.float32))
            dt = h5py.special_dtype(vlen=str)
            ds = hf.create_dataset("feature_names_arr", (len(X_df.columns),), dtype=dt)
            ds[:] = list(X_df.columns)
        print(f"Done {dataset}!", flush=True)

if __name__ == "__main__":
    main()
