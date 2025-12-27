import os
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.feature_engine import EmbeddingComputer, FeatureEngine
from src.logger import PipelineLogger


def load_data(fasta_path: str, pairs_path: str):
    """Load protein sequences from FASTA and interaction pairs from TSV."""
    sequences: dict[str, str] = {}
    with open(fasta_path, "r") as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                header = line.split()[0][1:]
                sequences[header] = ""
            elif header:
                sequences[header] += line

    pairs_df = pd.read_csv(pairs_path, sep="\t", header=None, names=["protein1", "protein2", "label"])
    return sequences, pairs_df


def deduplicate_sequences_and_pairs(sequences: Dict[str, str], pairs_df: pd.DataFrame, logger=None):
    """
    Map duplicate protein sequences to a single canonical ID and update pairs.
    This prevents leakage where identical sequences have different IDs in Train/Val.
    """
    if logger:
        logger.phase("Deduplicating sequences and updating pairs")

    # 1. Group IDs by sequence
    seq_to_ids = {}
    for pid, seq in sequences.items():
        if seq not in seq_to_ids:
            seq_to_ids[seq] = []
        seq_to_ids[seq].append(pid)

    # 2. Map each original ID to a canonical ID (the first ID found for that sequence)
    id_map = {}
    clean_sequences = {}
    for seq, ids in seq_to_ids.items():
        canonical_id = ids[0]
        clean_sequences[canonical_id] = seq
        for pid in ids:
            id_map[pid] = canonical_id

    if logger:
        logger.info(f"Unique sequences: {len(clean_sequences)} (from {len(sequences)} total IDs)")

    # 3. Update pairs_df with canonical IDs
    clean_pairs = pairs_df.copy()
    
    # Filter out pairs where IDs are not in our map (if any)
    valid_mask = clean_pairs["protein1"].isin(id_map) & clean_pairs["protein2"].isin(id_map)
    n_invalid = len(clean_pairs) - valid_mask.sum()
    if n_invalid > 0 and logger:
        logger.warning(f"Dropping {n_invalid} pairs with IDs not found in FASTA")
    
    clean_pairs = clean_pairs[valid_mask].copy()
    clean_pairs["protein1"] = clean_pairs["protein1"].map(id_map)
    clean_pairs["protein2"] = clean_pairs["protein2"].map(id_map)

    # 4. Use canonicalize_pairs to drop actual duplicates (now that IDs are unified)
    clean_pairs = canonicalize_pairs(clean_pairs, dataset_name="Unified-Sequence Dataset", logger=logger)

    return clean_sequences, clean_pairs


def canonicalize_pairs(pairs_df: pd.DataFrame, dataset_name: str = "Dataset", logger=None):
    """
    Sort protein1/protein2 alphabetically and drop duplicates to prevent data leakage.
    
    Args:
        pairs_df: DataFrame with columns [protein1, protein2, label]
        dataset_name: Name of dataset for logging
        logger: Optional PipelineLogger instance
        
    Returns:
        Canonicalized DataFrame with duplicates removed
    """
    if logger:
        logger.phase(f"Canonicalizing pairs for {dataset_name}")
    
    # Sort protein IDs so that protein1 <= protein2 alphabetically
    pairs_canonical = pairs_df.copy()
    swap_mask = pairs_canonical["protein1"] > pairs_canonical["protein2"]
    pairs_canonical.loc[swap_mask, ["protein1", "protein2"]] = pairs_canonical.loc[swap_mask, ["protein2", "protein1"]].values
    
    # Find duplicates before dropping
    duplicates = pairs_canonical.duplicated(subset=["protein1", "protein2"], keep="first")
    n_duplicates = duplicates.sum()
    
    if n_duplicates > 0 and logger:
        logger.warning(f"Found {n_duplicates} duplicate pairs in {dataset_name} (will keep first occurrence)")
        
        # Check for conflicting labels
        dup_indices = pairs_canonical[duplicates].index
        for idx in dup_indices:
            dup_row = pairs_canonical.loc[idx]
            first_occurrence = pairs_canonical[
                (pairs_canonical["protein1"] == dup_row["protein1"]) & 
                (pairs_canonical["protein2"] == dup_row["protein2"])
            ].iloc[0]
            
            if first_occurrence["label"] != dup_row["label"]:
                if logger:
                    logger.warning(
                        f"Conflicting labels for pair ({dup_row['protein1']}, {dup_row['protein2']}): "
                        f"keeping label={first_occurrence['label']}, dropping label={dup_row['label']}"
                    )
    
    # Drop duplicates
    pairs_clean = pairs_canonical.drop_duplicates(subset=["protein1", "protein2"], keep="first").copy()
    pairs_clean.reset_index(drop=True, inplace=True)
    
    if logger:
        logger.info(f"{dataset_name}: {len(pairs_df)} → {len(pairs_clean)} pairs after deduplication")
    
    return pairs_clean


def create_feature_matrix(
    pairs_df: pd.DataFrame, protein_features: dict, feature_names: List[str], pairing_strategy: str = "concat"
):
    """Build interaction-level feature matrix using concat or avg/diff strategy."""
    X_data, y_data, pair_indices = [], [], []

    if pairing_strategy == "concat":
        p1_feature_names = [f"P1_{name}" for name in feature_names]
        p2_feature_names = [f"P2_{name}" for name in feature_names]
        pair_feature_names = p1_feature_names + p2_feature_names
    elif pairing_strategy == "avgdiff":
        avg_feature_names = [f"Avg_{name}" for name in feature_names]
        diff_feature_names = [f"Diff_{name}" for name in feature_names]
        pair_feature_names = avg_feature_names + diff_feature_names
    else:
        raise ValueError(f"Unknown pairing strategy: {pairing_strategy}")

    for idx, row in pairs_df.iterrows():
        p1_id, p2_id, label = row["protein1"], row["protein2"], row["label"]

        # Canonicalize order to respect symmetry of PPI
        if p1_id > p2_id:
            p1_id, p2_id = p2_id, p1_id

        try:
            p1_feats = protein_features[p1_id].astype(np.float64)
            p2_feats = protein_features[p2_id].astype(np.float64)
        except KeyError:
            continue

        if pairing_strategy == "concat":
            pair_features = np.concatenate([p1_feats, p2_feats])
        elif pairing_strategy == "avgdiff":
            avg_feats = (p1_feats + p2_feats) / 2.0
            diff_feats = np.abs(p1_feats - p2_feats)
            pair_features = np.concatenate([avg_feats, diff_feats])

        X_data.append(pair_features)
        y_data.append(label)
        pair_indices.append(idx)

    X_df = pd.DataFrame(X_data, columns=pair_feature_names, index=pair_indices)
    y_s = pd.Series(y_data, index=pair_indices, name="label")
    return X_df, y_s


def get_cache_filename(
    pairs_path: str,
    pairing_strategy: str = "concat",
    esm_model_name: str = "esm2_t33_650M_UR50D",
    root_cache_dir: str = "cache",
    cache_version: str = "v2",
    suffix: str = "",
) -> str:
    """Build a dataset-specific cache filename from pairs path and ESM model."""
    parent_dir_name = os.path.basename(os.path.dirname(pairs_path)).lower()
    file_name_without_ext = os.path.splitext(os.path.basename(pairs_path))[0].lower()

    if parent_dir_name and parent_dir_name != "data":
        dataset_id = f"{parent_dir_name}_{file_name_without_ext}"
    else:
        dataset_id = file_name_without_ext

    safe_esm_model = esm_model_name.replace("/", "_").lower()
    suffix_str = f"_{suffix}" if suffix else ""
    filename = f"{dataset_id}_{safe_esm_model}_{pairing_strategy}_{cache_version}{suffix_str}_features.h5"

    os.makedirs(root_cache_dir, exist_ok=True)
    return os.path.join(root_cache_dir, filename)


def save_feature_matrix_h5(X: pd.DataFrame, y: pd.Series, file_path: str):
    """Save X (DataFrame) and y (Series) to H5."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        print(f"  [Cache] Saving feature matrix to {file_path}...")
        with h5py.File(file_path, "w") as hf:
            hf.create_dataset("X_data", data=X.values)
            hf.create_dataset("X_cols", data=np.array(X.columns, dtype="S"))
            hf.create_dataset("X_index", data=X.index)

            hf.create_dataset("y_data", data=y.values)
            hf.create_dataset("y_index", data=y.index)
            hf.attrs["y_name"] = y.name if y.name else "label"
        print("  [Cache] Save complete.")
    except Exception as exc:  # noqa: BLE001
        print(f"  [Cache] ❌ ERROR saving cache to {file_path}: {exc}")


def load_feature_matrix_h5(file_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load X (DataFrame) and y (Series) from H5."""
    print(f"  [Cache] Loading feature matrix from {file_path}...")
    with h5py.File(file_path, "r") as hf:
        X_data = hf["X_data"][:]
        X_cols = [col.decode("utf-8") for col in hf["X_cols"][:]]
        X_index = hf["X_index"][:]
        X = pd.DataFrame(X_data, columns=X_cols, index=X_index)

        y_data = hf["y_data"][:]
        y_index = hf["y_index"][:]
        y_name = hf.attrs["y_name"]
        y = pd.Series(y_data, index=y_index, name=y_name)
    print(f"  [Cache] Load complete. X={X.shape}, y={y.shape}")
    return X, y


def split_pairs_no_overlap(pairs_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split train/test so no protein appears in both sets.
    Pairs containing proteins from different sets are dropped.
    """
    proteins = list(set(pairs_df["protein1"]).union(set(pairs_df["protein2"])))
    rng = np.random.RandomState(random_state)
    rng.shuffle(proteins)
    cutoff = max(1, int(len(proteins) * (1 - test_size)))
    train_proteins = set(proteins[:cutoff])
    test_proteins = set(proteins[cutoff:])

    train_idx, test_idx = [], []
    dropped = 0
    for idx, row in pairs_df.iterrows():
        p1, p2 = row["protein1"], row["protein2"]
        if p1 in train_proteins and p2 in train_proteins:
            train_idx.append(idx)
        elif p1 in test_proteins and p2 in test_proteins:
            test_idx.append(idx)
        else:
            dropped += 1
    return train_idx, test_idx, dropped


def make_protein_folds(pairs_df: pd.DataFrame, n_splits: int = 5, random_state: int = 42):
    """
    Create folds based on proteins (no overlap across folds).
    Pairs with proteins from different folds are dropped.
    """
    proteins = list(set(pairs_df["protein1"]).union(set(pairs_df["protein2"])))
    rng = np.random.RandomState(random_state)
    rng.shuffle(proteins)
    fold_assign = {prot: i % n_splits for i, prot in enumerate(proteins)}

    pair_fold: dict[int, int | None] = {}
    dropped = 0
    for idx, row in pairs_df.iterrows():
        f1 = fold_assign.get(row["protein1"])
        f2 = fold_assign.get(row["protein2"])
        if f1 is None or f2 is None or f1 != f2:
            pair_fold[idx] = None
            dropped += 1
        else:
            pair_fold[idx] = f1

    folds = []
    for fold_id in range(n_splits):
        val_idx = [idx for idx, f in pair_fold.items() if f == fold_id]
        train_idx = [idx for idx, f in pair_fold.items() if f is not None and f != fold_id]
        folds.append((train_idx, val_idx))

    return folds, dropped


def _get_or_compute_esm_global_only(seq_id: str, sequence: str, h5_path: str, embedding_computer: EmbeddingComputer):
    """Get or compute global ESM2 vector for a sequence."""
    seq_upper = sequence.upper()
    global_key = f"{seq_upper}_global_v2"

    with h5py.File(h5_path, "a") as h5f:
        if global_key in h5f:
            return h5f[global_key][:]

        _, global_vec = embedding_computer.compute_full_embeddings(sequence)
        h5f.create_dataset(global_key, data=global_vec)
        return global_vec


def build_esm_only_pair_matrix(fasta_path: str, pairs_path: str, h5_cache_path: str):
    """
    Build ESM2-only feature matrix using cached global vectors.
    """
    logger = PipelineLogger()
    logger.phase("Building ESM2-only feature matrix (READ CACHE ONLY)")

    sequences, pairs_df = load_data(fasta_path, pairs_path)
    logger.info(f"Số chuỗi protein: {len(sequences)}")
    logger.info(f"Số cặp tương tác: {len(pairs_df)}")

    if not os.path.exists(h5_cache_path):
        raise FileNotFoundError(
            f"H5 cache không tồn tại: {h5_cache_path}\n"
            "Bạn cần chạy feature extraction (ESM2) trước."
        )

    h5f = h5py.File(h5_cache_path, "r")
    global_vectors = {}

    for pid, seq in sequences.items():
        seq_upper = seq.upper()
        key = f"{seq_upper}_global_v2"

        if key not in h5f:
            raise KeyError(
                f"Global embedding của protein {pid} chưa có trong cache.\n"
                f"File cache: {h5_cache_path}"
            )

        global_vectors[pid] = h5f[key][:]

    h5f.close()

    X_rows, y_rows, idx_rows = [], [], []

    for idx, row in pairs_df.iterrows():
        p1, p2, label = row["protein1"], row["protein2"], row["label"]

        if p1 not in global_vectors or p2 not in global_vectors:
            continue

        v1 = global_vectors[p1]
        v2 = global_vectors[p2]
        X_rows.append(np.concatenate([v1, v2]))
        y_rows.append(label)
        idx_rows.append(idx)

    X_arr = np.vstack(X_rows)
    X_df = pd.DataFrame(X_arr, index=idx_rows, columns=[f"ESM_f{i}" for i in range(X_arr.shape[1])])
    y_s = pd.Series(y_rows, index=idx_rows, name="label")

    logger.result("ESM-only matrix shape", X_df.shape)
    return X_df, y_s


def get_protein_based_splits(pairs_df, n_splits=5, random_state=42):
    """
    Protein-level splits to avoid leakage: a protein appears in only one fold.
    """
    unique_proteins = list(set(pairs_df["protein1"]) | set(pairs_df["protein2"]))
    unique_proteins.sort()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    print(f"Generating {n_splits}-fold splits based on {len(unique_proteins)} unique proteins...")

    for fold_idx, (train_prot_idx, val_prot_idx) in enumerate(kf.split(unique_proteins)):
        train_prots = set(unique_proteins[i] for i in train_prot_idx)
        val_prots = set(unique_proteins[i] for i in val_prot_idx)

        train_mask = pairs_df.apply(
            lambda x: (x["protein1"] in train_prots) and (x["protein2"] in train_prots), axis=1
        )
        val_mask = pairs_df.apply(
            lambda x: (x["protein1"] in val_prots) and (x["protein2"] in val_prots), axis=1
        )

        train_indices = pairs_df[train_mask].index.to_numpy()
        val_indices = pairs_df[val_mask].index.to_numpy()
        splits.append((train_indices, val_indices))

        print(
            f"  Fold {fold_idx+1}: Train Pairs={len(train_indices)}, Val Pairs={len(val_indices)} "
            f"(Leakage Check: {len(set(train_indices) & set(val_indices))} overlap)"
        )

    return splits


def load_cluster_map(cluster_path: str) -> Dict[str, str]:
    """
    Load a protein->cluster mapping file.

    Accepts TSV/CSV with two columns: protein_id, cluster_id.
    Extra columns are ignored; duplicate proteins keep the first mapping.
    """
    if not os.path.exists(cluster_path):
        raise FileNotFoundError(f"Cluster map file not found: {cluster_path}")

    df = pd.read_csv(cluster_path, sep=None, engine="python", header=None)
    if df.shape[1] < 2:
        raise ValueError(f"Cluster map file must have at least 2 columns, found {df.shape[1]}")

    protein_col, cluster_col = 0, 1
    if {"protein", "cluster"}.issubset({c.lower() for c in df.columns.astype(str)}):
        protein_col = [c.lower() for c in df.columns].index("protein")
        cluster_col = [c.lower() for c in df.columns].index("cluster")

    mapping = {}
    for _, row in df.iterrows():
        protein_id = str(row[protein_col]).strip()
        cluster_id = str(row[cluster_col]).strip()
        if protein_id and protein_id not in mapping:
            mapping[protein_id] = cluster_id

    if not mapping:
        raise ValueError(f"No cluster mappings were parsed from {cluster_path}")

    return mapping


def reduce_by_clusters(pairs_df: pd.DataFrame, cluster_map: Dict[str, str], logger=None):
    """
    Reduce redundancy by mapping each protein to its cluster centroid/ID.
    Interactions between members of the same clusters will be consolidated.
    
    Example: (A1, B1) and (A2, B1) where A1, A2 are in Cluster X will become (Cluster X, B1).
    Duplicates are then dropped.
    """
    if logger:
        logger.phase("Reducing redundancy via CD-HIT clusters")
        logger.info(f"Original pairs: {len(pairs_df)}")

    reduced_df = pairs_df.copy()
    
    # Map each protein to its cluster ID. If not in map, it's a singleton.
    reduced_df["protein1"] = reduced_df["protein1"].map(lambda x: cluster_map.get(x, x))
    reduced_df["protein2"] = reduced_df["protein2"].map(lambda x: cluster_map.get(x, x))

    # Re-canonicalize (since mapping might change alphabetical order or create duplicates)
    # also we need to handle protein1 == protein2 if they map to the same cluster (self-interactions)
    # but the user didn't specify, we'll keep them if they were there or if they became so.
    
    # Sort protein IDs so that protein1 <= protein2 alphabetically
    swap_mask = reduced_df["protein1"] > reduced_df["protein2"]
    reduced_df.loc[swap_mask, ["protein1", "protein2"]] = reduced_df.loc[swap_mask, ["protein2", "protein1"]].values

    # Drop duplicates at the cluster interaction level
    final_df = reduced_df.drop_duplicates(subset=["protein1", "protein2"]).copy()
    final_df.reset_index(drop=True, inplace=True)

    if logger:
        logger.info(f"Redundancy reduction: {len(pairs_df)} -> {len(final_df)} cluster-level pairs")
        n_dropped = len(pairs_df) - len(final_df)
        logger.info(f"Dropped {n_dropped} redundant interactions.")

    return final_df


def get_cluster_based_splits(pairs_df: pd.DataFrame, cluster_map: Dict[str, str], n_splits: int = 5, random_state: int = 42):
    """
    Build splits ensuring proteins from the same sequence cluster never leak across folds.
    Unmapped proteins are treated as their own singleton clusters.
    """
    proteins = set(pairs_df["protein1"]).union(set(pairs_df["protein2"]))
    protein_to_cluster = {prot: cluster_map.get(prot, f"unmapped::{prot}") for prot in proteins}

    unique_clusters = sorted(set(protein_to_cluster.values()))
    if len(unique_clusters) < n_splits:
        raise ValueError(f"Not enough unique clusters ({len(unique_clusters)}) for {n_splits}-fold CV.")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []

    for fold_idx, (train_cluster_idx, val_cluster_idx) in enumerate(kf.split(unique_clusters), start=1):
        train_clusters = {unique_clusters[i] for i in train_cluster_idx}
        val_clusters = {unique_clusters[i] for i in val_cluster_idx}

        train_mask = pairs_df.apply(
            lambda x: protein_to_cluster[x["protein1"]] in train_clusters
            and protein_to_cluster[x["protein2"]] in train_clusters,
            axis=1,
        )
        val_mask = pairs_df.apply(
            lambda x: protein_to_cluster[x["protein1"]] in val_clusters
            and protein_to_cluster[x["protein2"]] in val_clusters,
            axis=1,
        )

        train_indices = pairs_df[train_mask].index.to_numpy()
        val_indices = pairs_df[val_mask].index.to_numpy()
        splits.append((train_indices, val_indices))

        print(
            f"  Cluster Fold {fold_idx}: Train Pairs={len(train_indices)}, Val Pairs={len(val_indices)} "
            f"(clusters train={len(train_clusters)}, val={len(val_clusters)})"
        )

    return splits


def run_cdhit_clustering(
    fasta_path: str,
    output_prefix: str,
    identity_cutoff: float = 0.4,
    word_size: int = 2,
    threads: int = 0,
    memory_mb: int = 0,
) -> str:
    """
    Run CD-HIT clustering on protein sequences.
    
    CD-HIT clusters sequences based on sequence identity to prevent data leakage.
    Proteins with >identity_cutoff similarity will be grouped in same cluster.
    
    Args:
        fasta_path: Path to input FASTA file
        output_prefix: Prefix for output files (will create .clstr file)
        identity_cutoff: Sequence identity threshold (0.0-1.0), default 0.4 (40%)
        word_size: Word size for CD-HIT (2 for ~40%, 3 for ~50%, 4 for ~60%, 5 for ~90%)
        threads: Number of threads (0 = all available)
        memory_mb: Memory limit in MB (0 = unlimited)
    
    Returns:
        Path to the .clstr output file
        
    Raises:
        FileNotFoundError: If cd-hit binary not found
        subprocess.CalledProcessError: If CD-HIT execution fails
    """
    import subprocess
    import shutil
    
    # Check if cd-hit is available
    if not shutil.which("cd-hit"):
        raise FileNotFoundError(
            "cd-hit binary not found in PATH. "
            "Install with: conda install -c bioconda cd-hit"
        )
    
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"Input FASTA file not found: {fasta_path}")
    
    print(f"\n🔧 Running CD-HIT clustering...")
    print(f"   Input: {fasta_path}")
    print(f"   Identity cutoff: {identity_cutoff}")
    print(f"   Word size: {word_size}")
    
    # Build CD-HIT command
    cmd = [
        "cd-hit",
        "-i", fasta_path,
        "-o", output_prefix,
        "-c", str(identity_cutoff),
        "-n", str(word_size),
        "-d", "0",  # Full description length in .clstr
        "-M", str(memory_mb),
        "-T", str(threads),
    ]
    
    # Run CD-HIT
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"   ✅ CD-HIT completed successfully")
        
        cluster_file = f"{output_prefix}.clstr"
        if os.path.exists(cluster_file):
            print(f"   Output: {cluster_file}")
            return cluster_file
        else:
            raise FileNotFoundError(f"Expected cluster file not found: {cluster_file}")
            
    except subprocess.CalledProcessError as e:
        print(f"   ❌ CD-HIT failed with error:")
        print(f"   {e.stderr}")
        raise


def parse_cdhit_clusters(cluster_file: str) -> Dict[str, List[str]]:
    """
    Parse CD-HIT .clstr file to extract cluster assignments.
    
    Args:
        cluster_file: Path to .clstr file from CD-HIT
    
    Returns:
        Dictionary mapping cluster_id -> list of protein IDs
        
    Example:
        clusters = parse_cdhit_clusters("output.clstr")
        # {'cluster_0': ['protein1', 'protein5'], 'cluster_1': ['protein2', ...]}
    """
    clusters = {}
    current_cluster = None
    current_members = []
    
    with open(cluster_file, "r") as f:
        for line in f:
            line = line.strip()
            
            if line.startswith(">Cluster"):
                # Save previous cluster
                if current_cluster is not None:
                    clusters[current_cluster] = current_members
                
                # Start new cluster
                current_cluster = f"cluster_{line.split()[1]}"
                current_members = []
                
            elif line:
                # Extract protein ID from cluster member line
                # Format: "0	2799aa, >protein_id... *" or "0	2799aa, >protein_id... at 95%"
                parts = line.split(">")
                if len(parts) > 1:
                    protein_id = parts[1].split("...")[0].strip()
                    current_members.append(protein_id)
    
    # Save last cluster
    if current_cluster is not None:
        clusters[current_cluster] = current_members
    
    print(f"\n📊 Parsed {len(clusters)} clusters")
    print(f"   Total proteins: {sum(len(members) for members in clusters.values())}")
    
    return clusters


def get_sota_consistent_splits(
    fasta_path: str,
    test_ratio: float = 0.2,
    identity_cutoff: float = 0.4,
    random_state: int = 42,
    cache_dir: str = "cache",
) -> tuple[List[str], List[str]]:
    """
    Create train/test splits using CD-HIT clustering for SOTA consistency.
    
    This ensures fair comparison with LPBERT and other methods that use
    sequence identity-based splitting to avoid data leakage.
    
    Args:
        fasta_path: Path to FASTA file with all proteins
        test_ratio: Fraction of clusters to use for test (0.0-1.0)
        identity_cutoff: CD-HIT sequence identity threshold (0.4 = 40%)
        random_state: Random seed for reproducibility
        cache_dir: Directory to store CD-HIT outputs
    
    Returns:
        (train_protein_ids, test_protein_ids): Two lists of protein IDs
        
    Example:
        train_ids, test_ids = get_sota_consistent_splits(
            "data/BioGrid/Human/human_dict.fasta",
            test_ratio=0.2,
            identity_cutoff=0.4
        )
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate output path
    base_name = os.path.splitext(os.path.basename(fasta_path))[0]
    output_prefix = os.path.join(cache_dir, f"{base_name}_cdhit_{int(identity_cutoff*100)}")
    cluster_file = f"{output_prefix}.clstr"
    
    # Run CD-HIT if not cached
    if not os.path.exists(cluster_file):
        print(f"🔄 CD-HIT cache not found, running clustering...")
        run_cdhit_clustering(fasta_path, output_prefix, identity_cutoff)
    else:
        print(f"✅ Using cached CD-HIT clusters: {cluster_file}")
    
    # Parse clusters
    clusters = parse_cdhit_clusters(cluster_file)
    
    # Split clusters into train/test
    cluster_ids = list(clusters.keys())
    rng = np.random.RandomState(random_state)
    rng.shuffle(cluster_ids)
    
    n_test_clusters = max(1, int(len(cluster_ids) * test_ratio))
    test_clusters = cluster_ids[:n_test_clusters]
    train_clusters = cluster_ids[n_test_clusters:]
    
    # Collect protein IDs
    train_protein_ids = []
    test_protein_ids = []
    
    for cluster_id in train_clusters:
        train_protein_ids.extend(clusters[cluster_id])
    
    for cluster_id in test_clusters:
        test_protein_ids.extend(clusters[cluster_id])
    
    print(f"\n📊 Split Summary:")
    print(f"   Train clusters: {len(train_clusters)} ({len(train_protein_ids)} proteins)")
    print(f"   Test clusters:  {len(test_clusters)} ({len(test_protein_ids)} proteins)")
    print(f"   Test ratio: {len(test_protein_ids) / (len(train_protein_ids) + len(test_protein_ids)):.2%}")
    
    return train_protein_ids, test_protein_ids


def create_cluster_based_cv_splits(
    fasta_path: str,
    n_splits: int = 5,
    identity_cutoff: float = 0.4,
    random_state: int = 42,
    cache_dir: str = "cache",
) -> List[tuple[List[str], List[str]]]:
    """
    Create cross-validation splits using CD-HIT clustering.
    
    Similar to get_sota_consistent_splits but returns multiple folds for CV.
    
    Args:
        fasta_path: Path to FASTA file
        n_splits: Number of CV folds
        identity_cutoff: CD-HIT sequence identity threshold
        random_state: Random seed
        cache_dir: Directory for CD-HIT outputs
    
    Returns:
        List of (train_protein_ids, val_protein_ids) tuples
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate output path
    base_name = os.path.splitext(os.path.basename(fasta_path))[0]
    output_prefix = os.path.join(cache_dir, f"{base_name}_cdhit_{int(identity_cutoff*100)}")
    cluster_file = f"{output_prefix}.clstr"
    
    # Run CD-HIT if not cached
    if not os.path.exists(cluster_file):
        print(f"🔄 CD-HIT cache not found, running clustering...")
        run_cdhit_clustering(fasta_path, output_prefix, identity_cutoff)
    else:
        print(f"✅ Using cached CD-HIT clusters: {cluster_file}")
    
    # Parse clusters
    clusters = parse_cdhit_clusters(cluster_file)
    
    # Create KFold splits on clusters
    cluster_ids = list(clusters.keys())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    splits = []
    for fold_idx, (train_cluster_idx, val_cluster_idx) in enumerate(kf.split(cluster_ids), 1):
        train_clusters = [cluster_ids[i] for i in train_cluster_idx]
        val_clusters = [cluster_ids[i] for i in val_cluster_idx]
        
        # Collect protein IDs
        train_proteins = []
        val_proteins = []
        
        for c_id in train_clusters:
            train_proteins.extend(clusters[c_id])
        for c_id in val_clusters:
            val_proteins.extend(clusters[c_id])
        
        splits.append((train_proteins, val_proteins))
        
        print(f"   Fold {fold_idx}: {len(train_proteins)} train, {len(val_proteins)} val")
    
    return splits


__all__ = [
    "load_data",
    "canonicalize_pairs",
    "create_feature_matrix",
    "get_cache_filename",
    "save_feature_matrix_h5",
    "load_feature_matrix_h5",
    "load_cluster_map",
    "run_cdhit_clustering",
    "parse_cdhit_clusters",
    "get_sota_consistent_splits",
    "create_cluster_based_cv_splits",
    "deduplicate_sequences_and_pairs",
]
