#!/usr/bin/env python3
"""
HybridStack-PPI Global Pre-computation Script
==============================================
Ensures ALL protein sequences from ALL datasets (Yeast, Human) 
are pre-computed and stored in the centralized ESM2 cache.

This script should be run after updating CDHIT_Reduced datasets 
or when starting with a fresh environment.
"""

import os
import sys
import warnings
from pathlib import Path
from tqdm import tqdm

# Suppress noisy logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engine import FeatureEngine, EmbeddingComputer
from scripts.config import DATASETS, ESM_MODEL_NAME, ESM_CACHE_PATH

def load_fasta(fasta_path):
    """Load sequences from FASTA file."""
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
    print("\n" + "=" * 80)
    print("  HybridStack-PPI Global Pre-computation (Centralized Cache)")
    print("=" * 80)
    print(f"  Target Cache: {ESM_CACHE_PATH}")
    print(f"  ESM Model:    {ESM_MODEL_NAME}")
    print("-" * 80)

    # 1. Collect all unique sequences from all configured datasets
    print("📂 Step 1: Collecting sequences from all datasets...")
    all_sequences = {}
    
    for dataset_name, paths in DATASETS.items():
        print(f"   - Processing {dataset_name.upper()}...")
        
        # Check raw fasta
        raw_seqs = load_fasta(paths['fasta'])
        all_sequences.update(raw_seqs)
        print(f"     Loaded {len(raw_seqs):,} from raw FASTA")
        
        # Check clean fasta (CDHIT Reduced) - prioritizing these IDs
        if 'clean_fasta' in paths:
            clean_seqs = load_fasta(paths['clean_fasta'])
            all_sequences.update(clean_seqs)
            print(f"     Loaded {len(clean_seqs):,} from clean FASTA")
            
    print(f"\n📊 Total unique protein IDs to cache: {len(all_sequences):,}")
    
    if len(all_sequences) == 0:
        print("❌ Error: No sequences found. Check your paths in config.py.")
        return

    # 2. Initialize Engine
    print(f"\n🚀 Step 2: Initializing Deep Learning Engine...")
    try:
        embedding_computer = EmbeddingComputer(model_name=ESM_MODEL_NAME)
        # FeatureEngine handles the per-sequence caching/H5 logic
        feature_engine = FeatureEngine(h5_cache_path=str(ESM_CACHE_PATH), 
                                      embedding_computer=embedding_computer)
    except Exception as e:
        print(f"❌ Error initializing model/cache: {e}")
        return
    
    # 3. Trigger extraction
    print("\n⚙️  Step 3: Running extraction & caching (Only missing sequences)...")
    # extract_all_features internally checks if ID is in cache before computing
    feature_engine.extract_all_features(all_sequences)
    
    print("\n" + "=" * 80)
    print(f"✨ SUCCESS: Global pre-computation complete!")
    print(f"   Total Proteins Cached: {len(all_sequences):,}")
    print(f"   Cache Location: {ESM_CACHE_PATH}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
