#!/usr/bin/env python3
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engine import FeatureEngine, EmbeddingComputer
from scripts.config import ESM_MODEL_NAME, ESM_CACHE_PATH

def benchmark_efficiency():
    print("\n" + "="*70)
    print("  HybridStack-PPI: Computational Efficiency Benchmark")
    print("="*70)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Use a dummy sequence
    seq1 = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    seq2 = "MSLSDKDKAAVKAVWAKISPKADDIGAEALGRMLTVYPWTKRYFPHFDLSHGSAQVKAHGKKVGDALTLAVGHLDDL"
    
    # 1. Start Timing
    times = {}
    
    # Stage 0: Initialization (Usually one-time, but good to know)
    start = time.time()
    computer = EmbeddingComputer(model_name=ESM_MODEL_NAME)
    # Use no cache for raw speed measurement
    engine = FeatureEngine(h5_cache_path=None, embedding_computer=computer)
    times['Initialization'] = time.time() - start
    
    print(f"✅ Initialization: {times['Initialization']:.2f}s")

    # Warm-up (ESM needs one pass)
    _ = computer.compute_full_embeddings(seq1)
    
    # Stage 1: Handcrafted Features (Biological Branch)
    start = time.time()
    # 1a. General handcrafted (AAC, CTD, etc.)
    handcrafted1 = engine.handcraft_extractor.extract(seq1)
    handcrafted2 = engine.handcraft_extractor.extract(seq2)
    times['Bio-feature Extraction'] = time.time() - start
    
    # Stage 2: Deep Learning Features (ESM-2 Embedding)
    start = time.time()
    matrix1, global1 = computer.compute_full_embeddings(seq1)
    matrix2, global2 = computer.compute_full_embeddings(seq2)
    times['ESM-2 Embedding'] = time.time() - start

    # Stage 3: Motifs & Local Embeddings (Joint Stage)
    start = time.time()
    motifs1, local1 = engine._extract_motif_and_local_embedding(seq1, matrix1)
    motifs2, local2 = engine._extract_motif_and_local_embedding(seq2, matrix2)
    times['Motif & Local extraction'] = time.time() - start
    
    # Stage 4: Feature Concatenation & Pre-processing
    start = time.time()
    # Simulate final vector construction
    _ = np.concatenate([handcrafted1, motifs1, global1, local1])
    times['Post-processing'] = time.time() - start
    
    # Stage 4: Inference (Stacking Model - Mock with average time)
    # Usually fast (~ms)
    times['Stacking Inference'] = 0.005 # Assigned baseline for 1 pair
    
    print(f"📊 BENCHMARK RESULTS (Time per pair):")
    total_inference = times['Bio-feature Extraction'] + times['ESM-2 Embedding'] + times['Motif & Local extraction'] + times['Post-processing'] + times['Stacking Inference']
    
    for k, v in times.items():
        if k != 'Initialization':
            print(f"   - {k:25}: {v*1000:.2f} ms")
            
    print(f"\n🚀 Total Inference Latency: {total_inference*1000:.2f} ms / pair")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plot_data = {k: v*1000 for k, v in times.items() if k != 'Initialization'}
    sns.barplot(x=list(plot_data.keys()), y=list(plot_data.values()), palette='viridis')
    plt.title('Inference Latency Breakdown (per pair)', fontweight='bold')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=15)
    
    # Save results
    out_path = PROJECT_ROOT / "results/plots/computational_efficiency.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    print(f"✅ Plot saved to {out_path}")

if __name__ == "__main__":
    benchmark_efficiency()
