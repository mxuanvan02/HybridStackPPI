#!/usr/bin/env python3
"""
Case Study Annotator
====================
Map UniProt IDs to gene names and extract motif information for Case Study pairs.

Author: HybridStackPPI Team
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hybridstack.data_utils import load_feature_matrix_h5


# Yeast UniProt ID to Gene Name mapping (SGD IDs)
# These are the Case Study proteins
YEAST_GENE_MAP = {
    # Top Case Study #1
    "851784": {"gene": "RAV2", "name": "RAVE complex subunit RAV2", "function": "Regulator of V-ATPase assembly"},
    "854509": {"gene": "VMA4", "name": "V-type proton ATPase subunit E", "function": "Vacuolar H+-ATPase subunit"},
    
    # Case Study #2
    "853985": {"gene": "UBX5", "name": "UBX domain-containing protein 5", "function": "Ubiquitin regulatory pathway"},
    "852053": {"gene": "CDC48", "name": "Cell division control protein 48", "function": "AAA-ATPase, protein extraction"},
    
    # Case Study #3
    "854985": {"gene": "VPS33", "name": "Vacuolar protein sorting 33", "function": "HOPS/CORVET complex"},
    "856760": {"gene": "VAM6", "name": "Vacuole morphology protein 6", "function": "HOPS complex, vacuolar fusion"},
    
    # Case Study #4 (self-interaction)
    "853078": {"gene": "NUP60", "name": "Nucleoporin NUP60", "function": "Nuclear pore complex"},
    
    # Case Study #5
    "852726": {"gene": "UFD1", "name": "Ubiquitin fusion degradation protein 1", "function": "ERAD pathway"},
    
    # Case Study #6
    "850607": {"gene": "UBC13", "name": "Ubiquitin-conjugating enzyme E2 13", "function": "DNA damage tolerance"},
    "851991": {"gene": "MMS2", "name": "Methyl methanesulfonate sensitivity 2", "function": "Ubiquitin-conjugating enzyme variant"},
}


def lookup_uniprot_ids(ids: list) -> dict:
    """
    Lookup gene names from UniProt API for yeast proteins.
    Falls back to local mapping if API fails.
    """
    results = {}
    
    for uid in ids:
        # Clean the ID
        clean_id = uid.replace("_reviewed", "").replace("_mutant", "").split("_")[0]
        
        # Try local mapping first
        if clean_id in YEAST_GENE_MAP:
            results[uid] = YEAST_GENE_MAP[clean_id]
            continue
        
        # Try UniProt API
        try:
            url = f"https://rest.uniprot.org/uniprotkb/{clean_id}.json"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                gene_names = data.get("genes", [{}])
                gene = gene_names[0].get("geneName", {}).get("value", clean_id) if gene_names else clean_id
                protein_name = data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "Unknown")
                results[uid] = {"gene": gene, "name": protein_name, "function": ""}
            else:
                results[uid] = {"gene": clean_id, "name": "Unknown", "function": ""}
        except Exception as e:
            results[uid] = {"gene": clean_id, "name": "Unknown", "function": ""}
        
        time.sleep(0.1)  # Respect API rate limits
    
    return results


def get_motif_info_for_pair(X_df, feature_names, pair_idx, protein1, protein2):
    """
    Extract detected motifs for a specific protein pair.
    """
    motif_keywords = ["LIG_", "MOD_", "DOC_", "DEG_", "CLV_", "TRG_"]
    
    p1_motifs = []
    p2_motifs = []
    
    for col in feature_names:
        if not any(kw in col.upper() for kw in motif_keywords):
            continue
        
        value = X_df.iloc[pair_idx][col]
        if value != 0:
            # Extract motif name
            motif_name = col.replace("P1_Motif_", "").replace("P2_Motif_", "")
            motif_name = col.replace("P1_", "").replace("P2_", "")
            
            if col.startswith("P1_"):
                p1_motifs.append(motif_name)
            elif col.startswith("P2_"):
                p2_motifs.append(motif_name)
    
    return p1_motifs, p2_motifs


def annotate_case_studies():
    """Main function to annotate case study pairs."""
    
    from scripts.config import get_dataset_config
    
    config = get_dataset_config("yeast")
    
    print(f"\n{'='*80}")
    print("CASE STUDY ANNOTATION")
    print(f"{'='*80}")
    
    # Load predictions
    preds_path = PROJECT_ROOT / "results" / "Yeast_branch_predictions.csv"
    preds_df = pd.read_csv(preds_path)
    print(f"\n[1/4] Loaded predictions: {len(preds_df)} rows")
    
    # Load feature matrix for motif extraction
    print("\n[2/4] Loading feature matrix...")
    X_df, y_s = load_feature_matrix_h5(config['feature_cache'])
    feature_names = list(X_df.columns)
    
    # Find Case Study pairs
    case_studies = preds_df[
        (preds_df['y_true'] == 1) &
        (preds_df['embed_proba'] < 0.5) &
        (preds_df['hybrid_proba'] > 0.5)
    ].copy()
    case_studies['boost'] = case_studies['hybrid_proba'] - case_studies['embed_proba']
    case_studies = case_studies.sort_values('boost', ascending=False)
    
    print(f"\n[3/4] Found {len(case_studies)} Case Study candidates")
    
    # Get unique proteins
    all_proteins = set(case_studies['protein1']).union(set(case_studies['protein2']))
    print(f"   Unique proteins: {len(all_proteins)}")
    
    # Lookup gene names
    print("\n[4/4] Looking up gene names...")
    gene_info = lookup_uniprot_ids(list(all_proteins))
    
    # === ANNOTATED OUTPUT ===
    print(f"\n{'='*80}")
    print("ANNOTATED CASE STUDIES FOR PAPER")
    print(f"{'='*80}")
    
    latex_cases = []
    
    for rank, (idx, row) in enumerate(case_studies.head(6).iterrows(), 1):
        p1, p2 = row['protein1'], row['protein2']
        p1_clean = p1.replace("_reviewed", "").split("_")[0]
        p2_clean = p2.replace("_reviewed", "").split("_")[0]
        
        # Get gene info
        g1 = gene_info.get(p1, YEAST_GENE_MAP.get(p1_clean, {"gene": p1_clean, "name": "Unknown", "function": ""}))
        g2 = gene_info.get(p2, YEAST_GENE_MAP.get(p2_clean, {"gene": p2_clean, "name": "Unknown", "function": ""}))
        
        # Get motifs (need to find matching row in original feature matrix)
        # Find pairs in original data
        pairs_df = pd.read_csv(config['pairs'], sep='\t', header=None, names=['protein1', 'protein2', 'label'])
        pair_matches = pairs_df[(pairs_df['protein1'] == p1) & (pairs_df['protein2'] == p2)]
        
        if len(pair_matches) > 0:
            pair_idx = pair_matches.index[0]
            if pair_idx < len(X_df):
                p1_motifs, p2_motifs = get_motif_info_for_pair(X_df, feature_names, pair_idx, p1, p2)
            else:
                p1_motifs, p2_motifs = [], []
        else:
            p1_motifs, p2_motifs = [], []
        
        print(f"\n{'─'*80}")
        print(f"CASE STUDY #{rank}")
        print(f"{'─'*80}")
        print(f"  Protein A: {g1['gene']} ({p1_clean})")
        print(f"            {g1['name']}")
        print(f"            Function: {g1['function']}")
        print(f"  Protein B: {g2['gene']} ({p2_clean})")
        print(f"            {g2['name']}")
        print(f"            Function: {g2['function']}")
        print(f"\n  Prediction Probabilities:")
        print(f"    - Interp Branch: {row['interp_proba']:.4f}")
        print(f"    - Embed Branch:  {row['embed_proba']:.4f} (< 0.5, WRONG)")
        print(f"    - Hybrid Final:  {row['hybrid_proba']:.4f} (> 0.5, CORRECT)")
        print(f"    - Confidence Boost: +{row['boost']:.4f}")
        
        if p1_motifs or p2_motifs:
            print(f"\n  Detected Motifs:")
            if p1_motifs:
                print(f"    {g1['gene']}: {', '.join(p1_motifs[:5])}")
            if p2_motifs:
                print(f"    {g2['gene']}: {', '.join(p2_motifs[:5])}")
        
        # Build LaTeX snippet
        latex_cases.append({
            'rank': rank,
            'gene1': g1['gene'],
            'gene2': g2['gene'],
            'name1': g1['name'],
            'name2': g2['name'],
            'func1': g1['function'],
            'func2': g2['function'],
            'interp_prob': row['interp_proba'],
            'embed_prob': row['embed_proba'],
            'hybrid_prob': row['hybrid_proba'],
            'boost': row['boost'],
            'p1_motifs': p1_motifs[:3] if p1_motifs else [],
            'p2_motifs': p2_motifs[:3] if p2_motifs else [],
        })
    
    # === SUMMARY ===
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Found {len(case_studies)} case studies where Hybrid outperformed Embed-only branch")
    if len(case_studies) > 0:
        print(f"Best case: {case_studies.iloc[0]['protein1']} <-> {case_studies.iloc[0]['protein2']}")
        print(f"  Embed prob: {case_studies.iloc[0]['embed_proba']:.4f}")
        print(f"  Hybrid prob: {case_studies.iloc[0]['hybrid_proba']:.4f}")
        print(f"  Boost: +{case_studies.iloc[0]['boost']:.4f}")
    
    return case_studies


if __name__ == "__main__":
    annotate_case_studies()
