
import os
import sys
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engine import FeatureEngine, EmbeddingComputer

def load_fasta(fasta_path):
    sequences = {}
    if not os.path.exists(fasta_path):
        print(f"⚠️ Warning: File not found {fasta_path}")
        return {}
    
    with open(fasta_path, "r") as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                header = line[1:].split()[0]
                sequences[header] = ""
            elif header:
                sequences[header] += line
    return sequences

def main():
    # Configuration
    CACHE_PATH = "/media/SAS/Van/HybridStackPPI/cache/esm2_embeddings.h5"
    MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
    
    FASTA_FILES = [
        PROJECT_ROOT / "data/data/yeast/sequences.fasta"
    ]
    
    # 1. Collect all unique sequences
    print("📂 Collecting all sequences...")
    all_sequences = {}
    for f_path in FASTA_FILES:
        seqs = load_fasta(f_path)
        all_sequences.update(seqs)
    
    print(f"📊 Total unique protein IDs: {len(all_sequences)}")
    
    # 2. Initialize Engine
    print(f"🚀 Initializing ESM2 Model: {MODEL_NAME}...")
    embedding_computer = EmbeddingComputer(model_name=MODEL_NAME)
    # Note: FeatureEngine handles the per-sequence caching logic
    feature_engine = FeatureEngine(h5_cache_path=CACHE_PATH, embedding_computer=embedding_computer)
    
    # 3. Trigger extraction
    # This will use FeatureEngine logic to check cache and compute if missing
    print("⚙️ Starting extraction/caching process...")
    # We call extract_all_features which internally calls _get_or_compute_embeddings
    # and saves results to the H5 file.
    feature_engine.extract_all_features(all_sequences)
    
    print(f"\n✨ All done! Cache saved at: {CACHE_PATH}")

if __name__ == "__main__":
    main()
