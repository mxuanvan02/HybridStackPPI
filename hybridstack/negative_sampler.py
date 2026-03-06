"""
negative_sampler.py
===================
Generate biologically-informed negative protein pairs for PPI prediction.

The current dataset uses mutant-derived negatives (protein_A ↔ B_mutant_N),
which trivially simplifies the classification task (ESM-2 detects mutations).
This module replaces those with four strategies, ordered by difficulty:

    Strategy           Difficulty  Description
    ────────────────────────────────────────────────────────────────
    random             Easy        Random pairs from same universe
    same_compartment   Hard        Same subcellular location, NOT in BioGRID
    same_go            Hard        Same GO Biological Process, NOT in BioGRID
    negatome           Hardest     Experimentally confirmed non-interactions

Difficulty rationale
────────────────────
"Hard" means the discriminating feature (co-localisation, shared function)
is the SAME as in true positives.  The model cannot exploit physical
separation or functional distance — it must learn actual molecular
interaction signals.

    Easy (wrong):   nucleus ↔ extracellular  → trivially non-interacting
    HARD (correct): nucleus A ↔ nucleus B    → co-localised, yet do NOT interact

Design
──────
- NegativeSampler.fit(positives_df, protein_ids) → no side effects
- NegativeSampler.sample() → pd.DataFrame
- No file I/O inside the class.  I/O is handled by generate_negatives.py.
- Mutant proteins are excluded from the sampling universe in all strategies.
- O(1) positive-pair lookup via sorted-tuple-keyed set.

Author: HybridStack-PPI Team
"""

import re
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ─── Tuning constants ─────────────────────────────────────────────────────────

# Candidate pairs generated per vectorized batch (controls RAM vs speed trade-off).
# 200k integers × 2 arrays × 8 bytes ≈ 3.2 MB — negligible overhead.
_BATCH_SIZE = 200_000

# UniProt REST endpoints.
_UNIPROT_IDMAPPING_RUN    = "https://rest.uniprot.org/idmapping/run"
_UNIPROT_IDMAPPING_POLL   = "https://rest.uniprot.org/idmapping/status/{job_id}"
_UNIPROT_IDMAPPING_RESULT = "https://rest.uniprot.org/idmapping/stream/{job_id}"
_MAX_RETRIES = 30
_POLL_SLEEP  = 3.0    # seconds between polls
_RATE_SLEEP  = 0.5    # seconds between batch requests

# ─── Helpers ──────────────────────────────────────────────────────────────────

# Regex to identify mutant IDs added during dataset construction.
# e.g. "5371_reviewed_mutant_1", "375_reviewed_mutant_3"
_MUTANT_PATTERN = re.compile(r"_mutant_\d+", re.IGNORECASE)


def _canonical_pair(p1: str, p2: str) -> Tuple[str, str]:
    """
    Return a canonical (sorted) tuple key for an unordered protein pair.

    sorted-tuple is used instead of frozenset: both have O(1) lookup,
    but string comparison (<) is faster than frozenset hashing in Python 3.12+.
    """
    return (p1, p2) if p1 < p2 else (p2, p1)


def _is_mutant(protein_id: str) -> bool:
    """Return True if the protein ID represents a mutant / variant."""
    return bool(_MUTANT_PATTERN.search(protein_id))


def _strip_suffix(protein_id: str) -> str:
    """
    Strip dataset-specific suffix to obtain the raw numeric Gene ID
    used for external API lookups.

    Examples
    --------
    '2624_reviewed' → '2624'
    '5371_reviewed' → '5371'
    """
    return protein_id.split("_")[0]


# ─── Main class ───────────────────────────────────────────────────────────────

class NegativeSampler:
    """
    Generate hard negative PPI pairs using one of four strategies.

    Parameters
    ----------
    strategy : {'random', 'same_compartment', 'same_go', 'negatome'}
        Sampling strategy.  See module docstring for difficulty explanation.
    n_negatives : int or None
        Number of negative pairs to generate.  Defaults to the size of the
        positive set supplied in ``fit()``.
    random_state : int
        Random seed for reproducibility.
    max_attempts_multiplier : int
        Safety cap: max sampling attempts = n_negatives × multiplier.
        Prevents infinite loops when the universe is sparse.
    uniprot_batch_size : int
        Proteins submitted per UniProt API request.
    annotation_cache_path : str or None
        Path to a pre-downloaded annotation TSV (protein_id <TAB> value).
        When supplied, the API is bypassed entirely.
    negatome_path : str or None
        Path to a Negatome 2.0 TSV file (protein1 <TAB> protein2, no header).
        Required when strategy == 'negatome'.
    """

    STRATEGIES = ("random", "diff_compartment", "same_compartment", "same_go", "negatome")

    def __init__(
        self,
        strategy: str = "same_compartment",
        n_negatives: Optional[int] = None,
        random_state: int = 42,
        max_attempts_multiplier: int = 20,
        uniprot_batch_size: int = 500,
        annotation_cache_path: Optional[str] = None,
        negatome_path: Optional[str] = None,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from {self.STRATEGIES}."
            )

        self.strategy = strategy
        self.n_negatives = n_negatives
        self.random_state = random_state
        self.max_attempts_multiplier = max_attempts_multiplier
        self.uniprot_batch_size = uniprot_batch_size
        self.annotation_cache_path = annotation_cache_path
        self.negatome_path = negatome_path

        # Set after fit():
        self._positive_set: Set[Tuple[str, str]] = set()
        self._universe: np.ndarray = np.array([], dtype=object)
        self._degrees: Dict[str, int] = {}
        self._n_targets: int = 0
        # protein_id → annotation group label (compartment or GO term)
        self._annotation: Dict[str, str] = {}
        self._fitted = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, positives_df: pd.DataFrame, protein_ids: List[str]) -> "NegativeSampler":
        """
        Learn the positive interaction set and the protein universe.

        Parameters
        ----------
        positives_df : pd.DataFrame
            Columns: ['protein1', 'protein2']. Only positive pairs (label==1).
        protein_ids : list[str]
            All protein IDs from the FASTA file (mutants filtered automatically).

        Returns
        -------
        self
        """
        # ── 1. Build protein universe (exclude mutants) ────────────────────────
        raw = [p for p in protein_ids if not _is_mutant(p)]
        unique = list(dict.fromkeys(raw))   # deduplicate, preserve order
        # numpy object-array → supports O(1) integer-array indexing in batch sampling
        self._universe = np.array(unique, dtype=object)

        # ── 2. Build O(1) positive-pair lookup & Degree Map (Harmonious) ──────
        degrees = {}
        self._positive_set = set()
        for row in positives_df.itertuples(index=False):
            self._positive_set.add(_canonical_pair(row.protein1, row.protein2))
            degrees[row.protein1] = degrees.get(row.protein1, 0) + 1
            degrees[row.protein2] = degrees.get(row.protein2, 0) + 1
        self._degrees = degrees

        # ── 3. Target count ────────────────────────────────────────────────────
        self._n_targets = self.n_negatives if self.n_negatives is not None else len(positives_df)

        # ── 4. Load annotations for annotation-based strategies ───────────────
        if self.strategy in ("same_compartment", "diff_compartment", "same_go"):
            self._annotation = self._load_annotations()

        print(
            f"[NegativeSampler] strategy={self.strategy} | "
            f"universe={len(self._universe):,} proteins | "
            f"positives={len(self._positive_set):,} | "
            f"target={self._n_targets:,}"
        )
        if self._annotation:
            annotated = sum(1 for p in unique if p in self._annotation)
            print(f"[NegativeSampler] annotation coverage: "
                  f"{annotated:,}/{len(unique):,} proteins "
                  f"({annotated/max(len(unique),1)*100:.1f}%)")

        self._fitted = True
        return self

    def sample(self) -> pd.DataFrame:
        """
        Generate negative pairs.

        Returns
        -------
        pd.DataFrame  columns=['protein1', 'protein2', 'label']
            All rows have label == 0.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before sample().")

        dispatch = {
            "random":           self._sample_random,
            "diff_compartment": self._sample_diff_group,
            "same_compartment": self._sample_same_group,
            "same_go":          self._sample_same_group,
            "negatome":         self._sample_negatome,
        }
        result = dispatch[self.strategy]()
        print(f"[NegativeSampler] Generated {len(result):,} negatives "
              f"(strategy={self.strategy}).")
        return result

    # ── Private: sampling strategies ──────────────────────────────────────────

    def _sample_random(self) -> pd.DataFrame:
        """
        Strategy: RANDOM (baseline – easy)
        ────────────────────────────────────
        Randomly sample pairs from the protein universe that are NOT in the
        known positive set.

        Performance
        -----------
        Vectorized batch generation: one ``rng.randint`` call produces
        _BATCH_SIZE pairs.  Self-pairs are discarded via numpy boolean mask
        before the Python-level acceptance loop.  With ~7,800 non-mutant
        proteins and ~31k positives, the rejection rate is < 0.1 % →
        each batch accepts almost every candidate.
        """
        rng  = np.random.RandomState(self.random_state)
        
        # Build expanded universe based on harmonious degrees (degree + 1)
        expanded_universe = []
        for p in self._universe:
            expanded_universe.extend([p] * (self._degrees.get(p, 0) + 1))
        expanded_universe = np.array(expanded_universe, dtype=object)
        n_expanded = len(expanded_universe)

        cap  = self._n_targets * self.max_attempts_multiplier
        accepted: List[Tuple[str, str]] = []
        total_attempts = 0

        with tqdm(total=self._n_targets, desc="RANDOM negatives", unit="pair") as pbar:
            while len(accepted) < self._n_targets and total_attempts < cap:
                batch = min(_BATCH_SIZE, cap - total_attempts)
                i_arr = rng.randint(0, n_expanded, size=batch)
                j_arr = rng.randint(0, n_expanded, size=batch)

                p1_arr = expanded_universe[i_arr]
                p2_arr = expanded_universe[j_arr]

                # Use boolean mask to discard identical protein selections
                valid = p1_arr != p2_arr
                p1_arr, p2_arr = p1_arr[valid], p2_arr[valid]
                total_attempts += batch

                for p1, p2 in zip(p1_arr, p2_arr):
                    if len(accepted) >= self._n_targets:
                        break
                    key = _canonical_pair(p1, p2)
                    if key not in self._positive_set:
                        accepted.append(key)
                        pbar.update(1)

        self._warn_if_short(len(accepted), total_attempts)
        return self._to_dataframe(accepted)

    def _sample_same_group(self) -> pd.DataFrame:
        """
        Strategy: SAME_COMPARTMENT / SAME_GO  (hard)
        ───────────────────────────────────────────────
        Sample pairs where BOTH proteins share the same annotation group
        (subcellular compartment or GO Biological Process term) but are NOT
        in the known positive set.

        Why this is harder
        ------------------
        Co-localised proteins have a physical OPPORTUNITY to interact.
        Proteins in the same biological process share functional context.
        The model cannot trivially exploit compartment/process separation;
        it must learn actual molecular binding features to distinguish
        true interactors from co-localised non-interactors.

        Performance
        -----------
        Pre-builds numpy arrays per group to enable O(1) random element
        access.  Generates group-index batches vectorially; only the
        set-membership check remains sequential.
        """
        if not self._annotation:
            print("[NegativeSampler] ⚠ No annotations available. "
                  "Falling back to random sampling.")
            return self._sample_random()

        # Group proteins that are ANNOTATED.
        groups: Dict[str, List[str]] = {}
        for pid in self._universe.tolist():
            ann = self._annotation.get(pid)
            if ann:
                groups.setdefault(ann, []).append(pid)

        # Retain only groups with ≥ 2 proteins (need pairs).
        groups = {g: members for g, members in groups.items() if len(members) >= 2}

        if not groups:
            print("[NegativeSampler] ⚠ No annotation groups with ≥2 proteins. "
                  "Falling back to random sampling.")
            return self._sample_random()

        group_names  = list(groups.keys())
        n_groups     = len(group_names)
        
        # Expanded versions for harmonious (degree-preserving) sampling
        groups_expanded_np = {}
        group_indices_expanded = []
        
        for i, g in enumerate(group_names):
            expanded_members = []
            for m in groups[g]:
                expanded_members.extend([m] * (self._degrees.get(m, 0) + 1))
            groups_expanded_np[g] = np.array(expanded_members, dtype=object)
            group_indices_expanded.extend([i] * len(expanded_members))
            
        group_expanded_sizes = np.array([len(groups_expanded_np[g]) for g in group_names])
        max_expanded_size = group_expanded_sizes.max()
        group_indices_expanded = np.array(group_indices_expanded, dtype=int)
        n_total_expanded = len(group_indices_expanded)

        annotated_coverage = sum(len(m) for m in groups.values())
        print(f"[NegativeSampler] {n_groups} annotation groups, "
              f"{annotated_coverage:,} annotated proteins available for pairing.")

        rng  = np.random.RandomState(self.random_state)
        cap  = self._n_targets * self.max_attempts_multiplier
        accepted: List[Tuple[str, str]] = []
        total_attempts = 0

        with tqdm(total=self._n_targets,
                  desc=f"{self.strategy.upper()} negatives", unit="pair") as pbar:
            while len(accepted) < self._n_targets and total_attempts < cap:
                batch = min(_BATCH_SIZE, cap - total_attempts)

                # Pick groups proportionally to their total degree mass
                gi_arr = group_indices_expanded[rng.randint(0, n_total_expanded, size=batch)]
                
                # Random member indices within each chosen group using expanded sizes
                p1_idx = rng.randint(0, max_expanded_size, size=batch)
                p2_idx = rng.randint(0, max_expanded_size, size=batch)
                total_attempts += batch

                for k in range(len(gi_arr)):
                    if len(accepted) >= self._n_targets:
                        break
                    gi   = gi_arr[k]
                    gname = group_names[gi]
                    gsz  = group_expanded_sizes[gi]
                    
                    a    = p1_idx[k] % gsz
                    b    = p2_idx[k] % gsz
                    
                    p1 = groups_expanded_np[gname][a]
                    p2 = groups_expanded_np[gname][b]
                    
                    if p1 == p2:
                        continue   # self-pair or same protein picked twice
                        
                    key = _canonical_pair(p1, p2)
                    if key not in self._positive_set:
                        accepted.append(key)
                        pbar.update(1)

        self._warn_if_short(len(accepted), total_attempts)
        return self._to_dataframe(accepted)

    def _sample_diff_group(self) -> pd.DataFrame:
        """
        Samples negatives by pairing proteins from completely different groups 
        (e.g. different subcellular compartments).
        Implements Pan et al. (Psub) baseline dataset.
        Maintains degree-preserving (harmonious) mapping.
        """
        if not self._annotation:
            self._annotation = self._load_annotations()

        annotated_expanded = []
        group_assignments = []
        
        for p in self._universe:
            g = self._annotation.get(p)
            if g and g != "unknown":
                weight = self._degrees.get(p, 0) + 1
                annotated_expanded.extend([p] * weight)
                group_assignments.extend([g] * weight)
                
        annotated_expanded = np.array(annotated_expanded, dtype=object)
        group_assignments = np.array(group_assignments, dtype=object)
        
        n_expanded = len(annotated_expanded)
        if n_expanded == 0:
            print("[NegativeSampler] Failed to get annotations required for diff_compartment")
            return pd.DataFrame(columns=["protein1", "protein2", "label"])

        unique_groups = len(set(group_assignments))
        print(f"[NegativeSampler] diff_compartment: {unique_groups} groups available.")

        rng  = np.random.RandomState(self.random_state)
        cap  = self._n_targets * self.max_attempts_multiplier
        accepted: List[Tuple[str, str]] = []
        total_attempts = 0

        with tqdm(total=self._n_targets,
                  desc=f"{self.strategy.upper()} negatives", unit="pair") as pbar:
            while len(accepted) < self._n_targets and total_attempts < cap:
                batch = min(_BATCH_SIZE, cap - total_attempts)

                i_arr = rng.randint(0, n_expanded, size=batch)
                j_arr = rng.randint(0, n_expanded, size=batch)

                p1_arr = annotated_expanded[i_arr]
                p2_arr = annotated_expanded[j_arr]
                g1_arr = group_assignments[i_arr]
                g2_arr = group_assignments[j_arr]

                valid = g1_arr != g2_arr
                p1_arr, p2_arr = p1_arr[valid], p2_arr[valid]
                total_attempts += batch

                for p1, p2 in zip(p1_arr, p2_arr):
                    if len(accepted) >= self._n_targets:
                        break
                    key = _canonical_pair(p1, p2)
                    if key not in self._positive_set:
                        accepted.append(key)
                        pbar.update(1)

        self._warn_if_short(len(accepted), total_attempts)
        return self._to_dataframe(accepted)

    def _sample_negatome(self) -> pd.DataFrame:
        """
        Strategy: NEGATOME (hardest)
        ──────────────────────────────
        Load experimentally confirmed non-interacting pairs from
        Negatome 2.0 (Blohm et al., 2014; DOI:10.1093/nar/gkt1199).

        Only pairs where BOTH proteins are present in the current
        protein universe are kept.  Pairs overlapping the positive set
        are discarded.

        Negatome TSV format (tab-separated, no header):
            protein1_id  <TAB>  protein2_id
        IDs must match the format in this dataset (e.g. '2624_reviewed').

        Parameters required
        -------------------
        negatome_path : str  (set at construction time)
            Path to a Negatome file mapped to dataset IDs.
        """
        if not self.negatome_path:
            raise ValueError(
                "strategy='negatome' requires negatome_path to be set."
            )

        print(f"[NegativeSampler] Loading Negatome from: {self.negatome_path}")
        neo_df = pd.read_csv(
            self.negatome_path,
            sep="\t",
            header=None,
            names=["protein1", "protein2"],
        )

        universe_set = set(self._universe.tolist())
        accepted: List[Tuple[str, str]] = []

        for row in neo_df.itertuples(index=False):
            p1, p2 = row.protein1, row.protein2
            # Both proteins must be in the wild-type universe.
            if p1 not in universe_set or p2 not in universe_set:
                continue
            key = _canonical_pair(p1, p2)
            if key in self._positive_set:
                continue   # confirmed positive — discard as negative
            accepted.append(key)
            if len(accepted) >= self._n_targets:
                break

        if len(accepted) < self._n_targets:
            short = self._n_targets - len(accepted)
            print(
                f"[NegativeSampler] ⚠ Negatome yielded only {len(accepted):,} valid pairs "
                f"(needed {self._n_targets:,}, short by {short:,}). "
                "Consider supplementing with same_compartment negatives."
            )
        return self._to_dataframe(accepted)

    # ── Private: annotation loading ───────────────────────────────────────────

    def _load_annotations(self) -> Dict[str, str]:
        """
        Return a dict  protein_id → annotation_value.

        Resolution order:
          1. Pre-downloaded TSV cache (fastest, reproducible).
          2. Live UniProt REST API (requires network; results not cached here).
        """
        if self.annotation_cache_path:
            return self._load_annotation_from_cache(self.annotation_cache_path)
        print(f"[NegativeSampler] No cache supplied; querying UniProt REST API…")
        return self._fetch_annotations_from_uniprot()

    def _load_annotation_from_cache(self, path: str) -> Dict[str, str]:
        """Load annotation from a pre-downloaded two-column TSV (no header).

        For subcellular location annotations, the raw UniProt string often
        contains multiple locations separated by periods, commas, or
        semicolons (e.g. "Nucleus. Nucleus, nucleoplasm. Cytoplasm").
        Using the full string as a group key creates hundreds of tiny groups
        with high false-negative contamination.

        This method normalizes the annotation to the PRIMARY compartment
        (the first term before any delimiter), producing a small number of
        large groups (Nucleus ~2100, Cytoplasm ~2200, Cell membrane ~700)
        where the false-negative rate stays below ~2%.
        """
        _PRIMARY_COMPARTMENT_DELIMITERS = re.compile(r"[.,;]")

        try:
            df = pd.read_csv(path, sep="\t", header=None,
                             names=["protein_id", "annotation"])
            raw_mapping = dict(zip(df["protein_id"], df["annotation"]))

            # Normalize to primary compartment
            mapping = {}
            for pid, ann in raw_mapping.items():
                primary = _PRIMARY_COMPARTMENT_DELIMITERS.split(str(ann), maxsplit=1)[0].strip()
                if primary:
                    mapping[pid] = primary

            # Log group distribution for diagnostics
            from collections import Counter
            group_counts = Counter(mapping.values())
            top_groups = group_counts.most_common(5)
            top_str = ", ".join(f"{g}={c}" for g, c in top_groups)
            print(f"[NegativeSampler] Cache loaded: {len(mapping):,} entries ← {path}")
            print(f"[NegativeSampler] Primary compartments (top-5): {top_str}")
            return mapping
        except Exception as exc:
            print(f"[NegativeSampler] ⚠ Failed to read cache '{path}': {exc}")
            return {}

    def _fetch_annotations_from_uniprot(self) -> Dict[str, str]:
        """
        Fetch annotations via UniProt ID Mapping (GeneID → UniProtKB).

        Annotation field:
          same_compartment → cc_scl_term  (subcellular location)
          same_go          → go_p         (GO Biological Process)
        """
        field_map = {
            "same_compartment": "cc_scl_term",
            "diff_compartment": "cc_scl_term",
            "same_go":          "go_p",
        }
        return_field = field_map[self.strategy]

        gene_ids = list(dict.fromkeys(
            _strip_suffix(p) for p in self._universe.tolist()
        ))
        annotation: Dict[str, str] = {}

        for start in range(0, len(gene_ids), self.uniprot_batch_size):
            batch = gene_ids[start : start + self.uniprot_batch_size]
            try:
                job_id = self._submit_idmapping(batch)
                results = self._wait_and_fetch(job_id, return_field)
                for gene_id, value in results.items():
                    annotation[f"{gene_id}_reviewed"] = value
                time.sleep(_RATE_SLEEP)
            except Exception as exc:
                print(f"[NegativeSampler] ⚠ Batch {start // self.uniprot_batch_size + 1} "
                      f"failed: {exc}")

        return annotation

    def _submit_idmapping(self, gene_ids: List[str]) -> str:
        resp = requests.post(
            _UNIPROT_IDMAPPING_RUN,
            data={"from": "GeneID", "to": "UniProtKB",
                  "ids": ",".join(gene_ids)},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["jobId"]

    def _wait_and_fetch(self, job_id: str, return_field: str) -> Dict[str, str]:
        poll_url = _UNIPROT_IDMAPPING_POLL.format(job_id=job_id)
        for _ in range(_MAX_RETRIES):
            resp = requests.get(poll_url, timeout=15)
            resp.raise_for_status()
            status = resp.json().get("jobStatus", "")
            if status == "FINISHED":
                break
            if status == "FAILED":
                raise RuntimeError(f"UniProt job {job_id} failed.")
            time.sleep(_POLL_SLEEP)
        else:
            raise TimeoutError(f"UniProt job {job_id} timed out.")

        result_url = _UNIPROT_IDMAPPING_RESULT.format(job_id=job_id)
        resp = requests.get(
            result_url,
            params={"format": "tsv", "fields": f"accession,{return_field}"},
            timeout=120,
        )
        resp.raise_for_status()

        mapping: Dict[str, str] = {}
        for line in resp.text.strip().split("\n")[1:]:
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1].strip():
                mapping[parts[0]] = parts[1].split(";")[0].strip()
        return mapping

    # ── Private: utilities ────────────────────────────────────────────────────

    @staticmethod
    def _to_dataframe(accepted: List[Tuple[str, str]]) -> pd.DataFrame:
        """Convert a list of canonical pairs to a labelled DataFrame."""
        return pd.DataFrame(
            [(p1, p2, 0) for p1, p2 in accepted],
            columns=["protein1", "protein2", "label"],
        )

    def _warn_if_short(self, n_accepted: int, n_attempts: int) -> None:
        if n_accepted < self._n_targets:
            print(
                f"[NegativeSampler] ⚠ Only {n_accepted:,} negatives generated "
                f"after {n_attempts:,} attempts (target={self._n_targets:,}). "
                "Consider increasing max_attempts_multiplier."
            )

    # ── Static loaders ────────────────────────────────────────────────────────

    @staticmethod
    def load_protein_ids_from_fasta(fasta_path: str) -> List[str]:
        """
        Return all protein IDs from a FASTA file (in order, no dedup).

        Only the first whitespace-delimited token after '>' is returned.
        Mutant IDs are NOT filtered here; filtering occurs in ``fit()``.
        """
        ids: List[str] = []
        with open(fasta_path, "r") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(">"):
                    ids.append(line[1:].split()[0])
        return ids

    @staticmethod
    def load_protein_ids_from_tsv(tsv_path: str) -> List[str]:
        """
        Return protein IDs from a two-column TSV (protein_id <TAB> sequence).

        Equivalent to ``load_protein_ids_from_fasta`` but for dict.tsv files.
        """
        ids: List[str] = []
        with open(tsv_path, "r") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    ids.append(stripped.split("\t")[0])
        return ids

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def universe_size(self) -> int:
        """Non-mutant proteins in the sampling universe."""
        return len(self._universe)

    @property
    def positive_count(self) -> int:
        """Known positive pairs (used for leakage check)."""
        return len(self._positive_set)


__all__ = ["NegativeSampler"]
