#!/usr/bin/env python3
"""
HybridStack-PPI Configuration
==============================
Centralized configuration for all paths and parameters.
All scripts should import from this module to ensure consistency.
"""

from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"

# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================
DATASETS = {
    "yeast": {
        "name": "Yeast",
        "full_name": "BioGRID Saccharomyces cerevisiae",
        # Raw data
        "fasta": DATA_DIR / "BioGrid/Yeast/yeast_dict.fasta",
        "pairs": DATA_DIR / "BioGrid/Yeast/yeast_pairs.tsv",
        # Clean data (CD-HIT Reduced)
        "clean_fasta": DATA_DIR / "BioGrid/Yeast/CDHIT_Reduced/yeast_clean.fasta",
        "clean_pairs": DATA_DIR / "BioGrid/Yeast/CDHIT_Reduced/yeast_clean_pairs.tsv",
        # Cluster file (CD-HIT 40%) - Standard Path
        "clstr": DATA_DIR / "BioGrid/Yeast/CDHIT_Reduced/yeast_clean.clstr",
        "cluster_map": CACHE_DIR / "yeast_cluster_map.csv",
        # Pre-computed feature cache (FULL feature matrix)
        "feature_cache": CACHE_DIR / "yeast_yeast_pairs_facebook_esm2_t33_650m_ur50d_concat_v2_features.h5",
    },
    "human": {
        "name": "Human",
        "full_name": "BioGRID Homo sapiens",
        # Raw data
        "fasta": DATA_DIR / "BioGrid/Human/human_dict.fasta",
        "pairs": DATA_DIR / "BioGrid/Human/human_pairs.tsv",
        # Clean data (CD-HIT Reduced)
        "clean_fasta": DATA_DIR / "BioGrid/Human/CDHIT_Reduced/human_clean.fasta",
        "clean_pairs": DATA_DIR / "BioGrid/Human/CDHIT_Reduced/human_clean_pairs.tsv",
        # Cluster file (CD-HIT 40%) - Standard Path
        "clstr": DATA_DIR / "BioGrid/Human/CDHIT_Reduced/human_clean.clstr",
        "cluster_map": CACHE_DIR / "human_cluster_map.csv",
        # Pre-computed feature cache (FULL feature matrix)
        "feature_cache": CACHE_DIR / "human_human_pairs_facebook_esm2_t33_650m_ur50d_concat_v2_features.h5",
    },
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
ESM_CACHE_PATH = CACHE_DIR / "esm2/esm2_embeddings_v4.h5"

# =============================================================================
# CROSS-VALIDATION SETTINGS
# =============================================================================
CV_N_SPLITS = 5
CV_RANDOM_STATE = 42

# =============================================================================
# FEATURE SELECTION PARAMETERS
# =============================================================================
# Interpretable branch (permissive)
INTERP_IMPORTANCE_QUANTILE = 0.97
INTERP_CORR_THRESHOLD = 0.95
INTERP_VARIANCE_THRESHOLD = 0.0

# Embedding branch (strict)
EMBED_IMPORTANCE_QUANTILE = 0.92
EMBED_CORR_THRESHOLD = 0.85
EMBED_VARIANCE_THRESHOLD = 0.01

# =============================================================================
# CD-HIT SETTINGS
# =============================================================================
CDHIT_IDENTITY_THRESHOLD = 0.4  # 40% sequence identity
CDHIT_WORD_SIZE = 2

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_dataset_config(dataset_name: str) -> dict:
    """
    Get configuration for a dataset.
    
    Args:
        dataset_name: 'yeast' or 'human'
        
    Returns:
        Dict with all paths for the dataset
        
    Raises:
        ValueError: If dataset_name is not valid
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be one of: {list(DATASETS.keys())}")
    
    config = DATASETS[dataset_name].copy()
    # Convert Path objects to strings for compatibility
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
    
    return config


def validate_cache_files(dataset_name: str) -> bool:
    """
    Check if all required cache files exist for a dataset.
    
    Args:
        dataset_name: 'yeast' or 'human'
        
    Returns:
        True if all files exist, False otherwise
    """
    config = DATASETS[dataset_name.lower()]
    
    required_files = [
        config["clstr"],
        config["feature_cache"],
    ]
    
    missing = []
    for f in required_files:
        if not Path(f).exists():
            missing.append(str(f))
    
    if missing:
        print(f"⚠️ Missing cache files for {dataset_name}:")
        for f in missing:
            print(f"   - {f}")
        return False
    
    print(f"✅ All cache files exist for {dataset_name}")
    return True


# =============================================================================
# PRINT CONFIG ON IMPORT (Optional)
# =============================================================================
if __name__ == "__main__":
    print("HybridStack-PPI Configuration")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"ESM Model: {ESM_MODEL_NAME}")
    print(f"CV Splits: {CV_N_SPLITS}")
    print()
    
    for name in DATASETS:
        print(f"\n{name.upper()} Dataset:")
        validate_cache_files(name)
