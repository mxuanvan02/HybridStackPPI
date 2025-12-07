#!/usr/bin/env python3
"""
RAPPPID Inference Script for BioGrid Datasets
==============================================

Run inference on Yeast/Human BioGrid data using pretrained RAPPPID model.

Usage:
    conda run -n rapppid_env python scripts/run_rapppid_inference.py --dataset yeast
    conda run -n rapppid_env python scripts/run_rapppid_inference.py --dataset human
    conda run -n rapppid_env python scripts/run_rapppid_inference.py --all
"""

import sys
import argparse
from pathlib import Path

# Add RAPPPID to path
PROJECT_ROOT = Path('/media/SAS/Van/HybridStackPPI')
RAPPPID_DIR = PROJECT_ROOT / 'external_tools/RAPPPID'
sys.path.insert(0, str(RAPPPID_DIR / 'rapppid'))

import numpy as np
import torch
import sentencepiece as sp
from pytorch_lightning.utilities import seed as pl_seed
from Bio import SeqIO
from tqdm import tqdm

# Import RAPPPID modules
from train import LSTMAWD


def load_model_and_tokenizer():
    """Load pretrained RAPPPID model and SentencePiece tokenizer."""
    weights_dir = RAPPPID_DIR / 'data/pretrained_weights/1690837077.519848_red-dreamy'
    
    chkpt_path = weights_dir / '1690837077.519848_red-dreamy.ckpt'
    spm_path = weights_dir / 'spm.model'
    
    print(f"Loading model from: {chkpt_path}")
    model = LSTMAWD.load_from_checkpoint(str(chkpt_path))
    model.eval()
    
    print(f"Loading SentencePiece from: {spm_path}")
    spp = sp.SentencePieceProcessor(model_file=str(spm_path))
    
    return model, spp


def encode_seq(spp, seq, trunc_len=1500):
    """Encode a protein sequence using SentencePiece."""
    toks = spp.encode(seq, enable_sampling=False, alpha=0.1, nbest_size=-1)
    
    if trunc_len:
        if len(toks) > trunc_len:
            toks = toks[:trunc_len]
        else:
            pad_len = trunc_len - len(toks)
            toks = np.pad(toks, (0, pad_len), 'constant')
    
    return torch.tensor(toks).long()


def load_biogrid_data(dataset='yeast'):
    """Load BioGrid pairs and sequences."""
    data_dir = PROJECT_ROOT / f'data/BioGrid/{dataset.capitalize()}'
    
    pairs_file = data_dir / f'{dataset}_pairs.tsv'
    fasta_file = data_dir / f'{dataset}_dict.fasta'
    
    print(f"Loading sequences from: {fasta_file}")
    seqs = {}
    for record in SeqIO.parse(fasta_file, 'fasta'):
        seqs[record.id] = str(record.seq)
    print(f"Loaded {len(seqs)} sequences")
    
    print(f"Loading pairs from: {pairs_file}")
    pairs = []
    with open(pairs_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                p1, p2, label = parts[0], parts[1], int(parts[2])
                if p1 in seqs and p2 in seqs:
                    pairs.append((p1, p2, label))
    print(f"Loaded {len(pairs)} valid pairs")
    
    return pairs, seqs


def run_inference(dataset='yeast', batch_size=32, max_pairs=None):
    """Run RAPPPID inference on dataset."""
    print(f"\n{'='*60}")
    print(f"RAPPPID Inference on {dataset.upper()} Dataset")
    print(f"{'='*60}")
    
    # Set seed for reproducibility
    seed = 8675309
    pl_seed.seed_everything(seed, workers=True)
    
    # Load model
    model, spp = load_model_and_tokenizer()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Load data
    pairs, seqs = load_biogrid_data(dataset)
    
    if max_pairs:
        pairs = pairs[:max_pairs]
        print(f"Limited to {max_pairs} pairs for testing")
    
    # Run inference
    predictions = []
    labels = []
    
    print(f"\nRunning inference on {len(pairs)} pairs...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(pairs), batch_size)):
            batch_pairs = pairs[i:i+batch_size]
            
            # Encode sequences
            seqs_a = []
            seqs_b = []
            batch_labels = []
            
            for p1, p2, label in batch_pairs:
                seqs_a.append(encode_seq(spp, seqs[p1]))
                seqs_b.append(encode_seq(spp, seqs[p2]))
                batch_labels.append(label)
            
            # Stack and move to device
            toks_a = torch.stack(seqs_a).to(device)
            toks_b = torch.stack(seqs_b).to(device)
            
            # Get embeddings
            emb_a = model(toks_a)
            emb_b = model(toks_b)
            
            # Predict
            logits = model.class_head(emb_a, emb_b).float()
            probs = torch.sigmoid(logits)
            
            predictions.extend(probs.cpu().numpy().flatten().tolist())
            labels.extend(batch_labels)
    
    # Calculate metrics
    predictions = np.array(predictions)
    labels = np.array(labels)
    pred_binary = (predictions > 0.5).astype(int)
    
    # Metrics
    tp = np.sum((pred_binary == 1) & (labels == 1))
    tn = np.sum((pred_binary == 0) & (labels == 0))
    fp = np.sum((pred_binary == 1) & (labels == 0))
    fn = np.sum((pred_binary == 0) & (labels == 1))
    
    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # MCC
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0
    
    # Results
    results = {
        'dataset': dataset,
        'total_pairs': len(pairs),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'mcc': mcc,
        'binding_predicted': int(np.sum(pred_binary == 1)),
        'non_binding_predicted': int(np.sum(pred_binary == 0))
    }
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RAPPPID {dataset.upper()} Results")
    print(f"{'='*60}")
    print(f"Total pairs evaluated: {len(pairs)}")
    print(f"Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"MCC:         {mcc:.4f}")
    print(f"\nPredictions: {int(np.sum(pred_binary == 1))} binding, {int(np.sum(pred_binary == 0))} non-binding")
    
    # Save results
    output_file = PROJECT_ROOT / f'results/rapppid_{dataset}_inference.txt'
    with open(output_file, 'w') as f:
        f.write(f"RAPPPID {dataset.upper()} Inference Results\n")
        f.write(f"{'='*40}\n")
        f.write(f"Model: 1690837077.519848_red-dreamy.ckpt\n")
        f.write(f"Total pairs: {len(pairs)}\n\n")
        f.write(f"Metrics:\n")
        f.write(f"  Accuracy:    {accuracy:.4f}\n")
        f.write(f"  Precision:   {precision:.4f}\n")
        f.write(f"  Recall:      {recall:.4f}\n")
        f.write(f"  Specificity: {specificity:.4f}\n")
        f.write(f"  F1-Score:    {f1:.4f}\n")
        f.write(f"  MCC:         {mcc:.4f}\n")
    
    print(f"\nResults saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run RAPPPID inference on BioGrid data')
    parser.add_argument('--dataset', choices=['yeast', 'human'], help='Dataset to process')
    parser.add_argument('--all', action='store_true', help='Run on both datasets')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--max-pairs', type=int, help='Limit number of pairs (for testing)')
    
    args = parser.parse_args()
    
    if args.all:
        results_yeast = run_inference('yeast', args.batch_size, args.max_pairs)
        results_human = run_inference('human', args.batch_size, args.max_pairs)
        
        print(f"\n{'='*60}")
        print("RAPPPID BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"Yeast - Accuracy: {results_yeast['accuracy']:.4f}, F1: {results_yeast['f1_score']:.4f}")
        print(f"Human - Accuracy: {results_human['accuracy']:.4f}, F1: {results_human['f1_score']:.4f}")
    elif args.dataset:
        run_inference(args.dataset, args.batch_size, args.max_pairs)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
