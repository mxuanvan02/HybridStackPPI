#!/usr/bin/env python3
"""Check how many Human proteins are already in cache v4."""
import h5py
from pathlib import Path

# Load Human sequences
human_fasta = Path("/media/SAS/Van/HybridStackPPI/data/BioGrid/Human/human_dict.fasta")
sequences = {}
current_id = None
current_seq = []

with open(human_fasta) as f:
    for line in f:
        line = line.strip()
        if line.startswith(">"):
            if current_id:
                sequences[current_id] = "".join(current_seq)
            current_id = line[1:]
            current_seq = []
        else:
            current_seq.append(line)
    if current_id:
        sequences[current_id] = "".join(current_seq)

print(f"Total Human proteins: {len(sequences)}")

# Check cache
cache_path = "/media/SAS/Van/HybridStackPPI/cache/esm2/esm2_embeddings_v4.h5"
with h5py.File(cache_path, "r") as f:
    cache_keys = set(f.keys())
    
found = 0
missing = 0
missing_examples = []

for pid, seq in sequences.items():
    global_key = f"{seq.upper()}_global_v2"
    if global_key in cache_keys:
        found += 1
    else:
        missing += 1
        if len(missing_examples) < 3:
            missing_examples.append(pid)

print(f"Found in cache: {found}")
print(f"Missing: {missing}")
print(f"Coverage: {found/len(sequences)*100:.1f}%")
if missing_examples:
    print(f"Missing examples: {missing_examples}")
