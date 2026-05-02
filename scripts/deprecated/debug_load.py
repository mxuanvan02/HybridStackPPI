import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

print("Importing load_data...")
from hybridstack.data_utils import load_data
print("Imported load_data.")

fasta_path = "data/BioGrid/Human/human_dict.fasta"
pairs_path = "data/BioGrid/Human/human_pairs_same_go.tsv"

print(f"Loading data from {fasta_path}...")
seqs, pairs_df = load_data(fasta_path, pairs_path)
print(f"Loaded {len(seqs)} sequences and {len(pairs_df)} pairs.")
