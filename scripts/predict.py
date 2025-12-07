#!/usr/bin/env python3
"""
HybridStack-PPI Prediction Script
===================================
Predict protein-protein interaction from two protein sequences.

Usage:
    python scripts/predict.py --seq1 "MKTVRQERL..." --seq2 "MSRSLLLRFL..."
    python scripts/predict.py --fasta1 protein1.fasta --fasta2 protein2.fasta
    python scripts/predict.py --model-path models/saved/hybridstack_model.pkl
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_sequence_from_fasta(fasta_path: str) -> str:
    """Load a single protein sequence from a FASTA file."""
    sequence = ""
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                continue
            sequence += line
    return sequence


def identify_elm_motifs(sequence: str, motifs: dict) -> list:
    """Identify ELM motifs in a protein sequence."""
    identified = []
    for motif_name, pattern in motifs.items():
        match = pattern.search(sequence)
        if match:
            identified.append({
                "motif": motif_name,
                "start": match.start(),
                "end": match.end(),
                "sequence": match.group()
            })
    return identified


def main():
    parser = argparse.ArgumentParser(
        description="Predict protein-protein interaction using HybridStack-PPI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--seq1", type=str, help="First protein sequence (amino acid string)")
    input_group.add_argument("--fasta1", type=str, help="Path to FASTA file for first protein")
    
    parser.add_argument("--seq2", type=str, help="Second protein sequence")
    parser.add_argument("--fasta2", type=str, help="Path to FASTA file for second protein")
    
    # Model options
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/saved/hybridstack_model.pkl",
        help="Path to trained model file (default: models/saved/hybridstack_model.pkl)"
    )
    parser.add_argument(
        "--esm-model",
        type=str,
        default="facebook/esm2_t33_650M_UR50D",
        help="ESM-2 model name (default: facebook/esm2_t33_650M_UR50D)"
    )
    parser.add_argument(
        "--h5-cache",
        type=str,
        default="cache/esm2_embeddings.h5",
        help="Path to ESM embedding cache (default: cache/esm2_embeddings.h5)"
    )
    
    # Output options
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load sequences
    if args.seq1:
        seq1 = args.seq1.upper()
    else:
        seq1 = load_sequence_from_fasta(args.fasta1).upper()
    
    if args.seq2:
        seq2 = args.seq2.upper()
    elif args.fasta2:
        seq2 = load_sequence_from_fasta(args.fasta2).upper()
    else:
        parser.error("Second protein sequence is required (--seq2 or --fasta2)")
    
    # Validate sequences
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    for i, seq in enumerate([seq1, seq2], 1):
        invalid_chars = set(seq) - valid_aa
        if invalid_chars:
            print(f"Warning: Protein {i} contains non-standard amino acids: {invalid_chars}")
    
    print("=" * 70)
    print("HybridStack-PPI Prediction")
    print("=" * 70)
    print(f"\nProtein 1: {len(seq1)} amino acids")
    print(f"  {seq1[:50]}..." if len(seq1) > 50 else f"  {seq1}")
    print(f"\nProtein 2: {len(seq2)} amino acids")
    print(f"  {seq2[:50]}..." if len(seq2) > 50 else f"  {seq2}")
    
    # Import components
    print("\n[1/4] Loading feature extraction components...")
    from hybridstack.feature_engine import (
        EmbeddingComputer,
        InterpretableFeatureExtractor,
        FeatureEngine,
    )
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"\nWarning: Trained model not found at {model_path}")
        print("Please train a model first using reproduce_results.py or provide a valid model path.")
        print("\nProceeding with feature extraction only...")
        model = None
    else:
        import pickle
        print(f"\n[2/4] Loading trained model from {model_path}...")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    
    # Initialize feature extraction
    print("\n[3/4] Extracting features...")
    
    # Bio features (handcrafted)
    bio_extractor = InterpretableFeatureExtractor()
    bio_feats1 = bio_extractor.extract(seq1)
    bio_feats2 = bio_extractor.extract(seq2)
    bio_combined = np.concatenate([bio_feats1, bio_feats2])
    
    if args.verbose:
        print(f"  Bio features (per protein): {len(bio_feats1)}")
        print(f"  Bio features (combined): {len(bio_combined)}")
    
    # ESM-2 embeddings
    print("  Computing ESM-2 embeddings (this may take a moment)...")
    embedding_computer = EmbeddingComputer(model_name=args.esm_model)
    _, esm_vec1 = embedding_computer.compute_full_embeddings(seq1)
    _, esm_vec2 = embedding_computer.compute_full_embeddings(seq2)
    esm_combined = np.concatenate([esm_vec1, esm_vec2])
    
    if args.verbose:
        print(f"  ESM-2 embeddings (per protein): {len(esm_vec1)}")
        print(f"  ESM-2 embeddings (combined): {len(esm_combined)}")
    
    # Identify motifs
    print("\n[4/4] Identifying ELM motifs...")
    
    # Load ELM motifs from FeatureEngine
    try:
        feature_engine = FeatureEngine(args.h5_cache, embedding_computer)
        motifs = feature_engine.motifs
    except Exception:
        motifs = {}
        print("  Warning: Could not load ELM motifs")
    
    motifs_p1 = identify_elm_motifs(seq1, motifs)
    motifs_p2 = identify_elm_motifs(seq2, motifs)
    
    # Make prediction
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    if model is not None:
        combined_features = np.concatenate([bio_combined, esm_combined])
        
        # Ensure feature size matches model expectation
        try:
            proba = model.predict_proba(combined_features.reshape(1, -1))[0]
            interaction_prob = proba[1]  # Probability of positive class
            prediction = "INTERACTING" if interaction_prob > 0.5 else "NON-INTERACTING"
            
            print(f"\n🔬 Prediction: {prediction}")
            print(f"   Interaction Probability: {interaction_prob:.4f}")
            print(f"   Non-interaction Probability: {proba[0]:.4f}")
        except Exception as e:
            print(f"\n⚠️  Prediction failed: {e}")
            print("   This may be due to feature dimension mismatch.")
    else:
        print("\n⚠️  No model loaded - cannot make prediction.")
        print("   Train a model first or provide a valid model path.")
    
    # ELM Motifs
    print("\n📋 Identified ELM Motifs:")
    print("-" * 40)
    print(f"\nProtein 1 ({len(motifs_p1)} motifs):")
    if motifs_p1:
        for m in motifs_p1[:10]:  # Show first 10
            print(f"  - {m['motif']}: {m['sequence']} (pos {m['start']}-{m['end']})")
        if len(motifs_p1) > 10:
            print(f"  ... and {len(motifs_p1) - 10} more")
    else:
        print("  No motifs identified")
    
    print(f"\nProtein 2 ({len(motifs_p2)} motifs):")
    if motifs_p2:
        for m in motifs_p2[:10]:
            print(f"  - {m['motif']}: {m['sequence']} (pos {m['start']}-{m['end']})")
        if len(motifs_p2) > 10:
            print(f"  ... and {len(motifs_p2) - 10} more")
    else:
        print("  No motifs identified")
    
    # Branch contribution (estimated from feature dimensions)
    print("\n📊 Feature Contribution:")
    print("-" * 40)
    bio_dim = len(bio_combined)
    esm_dim = len(esm_combined)
    total_dim = bio_dim + esm_dim
    
    print(f"  Bio (Interpretable) Branch: {bio_dim} features ({100*bio_dim/total_dim:.1f}%)")
    print(f"  Deep (ESM-2) Branch: {esm_dim} features ({100*esm_dim/total_dim:.1f}%)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
