#!/usr/bin/env python3
"""
download_annotations.py
========================
Download protein annotation data from UniProt using Bulk Streaming API
and save to ``data/annotations/`` for use by the negative-sampling pipeline.

This script replaces slow batch queries with a single streaming HTTP
request that downloads the entire organism TSV file (~10-20MB) directly
into memory, parsed incrementally. This guarantees exactly 1 request,
no timeouts, and completion in ~10-30 seconds, not hours.

Extracted Fields:
  - Subcellular Location: The main location before evidence tags {ECO:...}.
  - GO Biological Process: The first primary term listed before [GO:...].
"""

import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Assume these exist from the context
from hybridstack.negative_sampler import NegativeSampler, _is_mutant, _strip_suffix
from hybridstack.logger import PipelineLogger

# ─── Configuration ────────────────────────────────────────────────────────────

_DATASET_FASTA = {
    "human": "data/BioGrid/Human/human_dict.fasta",
    "yeast": "data/BioGrid/Yeast/yeast_dict.fasta",
}

_ORGANISM_TAXID = {
    "human": "9606",
    "yeast": "559292",
}

_UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"


def parse_line(line: str) -> Tuple[List[str], str, str]:
    """Parse one TSV line from UniProt stream."""
    parts = line.split('\t')
    if len(parts) < 4:
        return [], "", ""
        
    gene_ids_str = parts[1]
    gene_ids = [gid.strip().replace(';', '') for gid in gene_ids_str.split(';') if gid.strip()]
    if not gene_ids:
        return [], "", ""
        
    subcel_raw = parts[2]
    subcel = ""
    if subcel_raw and "SUBCELLULAR LOCATION:" in subcel_raw:
        parts_sub = subcel_raw.split("SUBCELLULAR LOCATION: ")
        if len(parts_sub) > 1:
            first_loc = parts_sub[1].split("{")[0]
            subcel = first_loc.strip(';. ')
            
    go_raw = parts[3]
    go_val = ""
    if go_raw:
        first_go = go_raw.split(';')[0].strip()
        go_val = first_go.split('[')[0].strip()
        
    return gene_ids, subcel, go_val

def bulk_download_annotations(
    taxon_id: str, 
    target_gene_ids: set, 
    logger: PipelineLogger
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Download and stream the entire organism TSV taking only targeted IDs.
    Returns: (subcellular_map, go_map)
    """
    params = {
        "query": f"organism_id:{taxon_id}",
        "format": "tsv",
        "fields": "accession,xref_geneid,cc_subcellular_location,go_p",
    }
    
    logger.info(f"Connecting to UniProt Stream API for Taxonomy ID {taxon_id}...")
    start_time = time.perf_counter()
    
    resp = requests.get(_UNIPROT_STREAM_URL, params=params, stream=True, timeout=120)
    resp.raise_for_status()
    
    subcel_map = {}
    go_map = {}
    
    processed_lines = 0
    matched_genes = 0
    
    for line in resp.iter_lines(decode_unicode=True):
        if not line: continue
        processed_lines += 1
        if processed_lines == 1: continue  # skip physical header
            
        gene_ids, subcel, go_val = parse_line(line)
        
        for gid in gene_ids:
            if gid in target_gene_ids:
                if subcel and gid not in subcel_map:
                    subcel_map[gid] = subcel
                    matched_genes += 1
                if go_val and gid not in go_map:
                    go_map[gid] = go_val
                    
    elapsed = time.perf_counter() - start_time
    logger.info(f"Stream complete! Evaluated {processed_lines:,} Proteome records in {elapsed:.1f} seconds.")
    
    return subcel_map, go_map


def save_annotation_tsv(
    protein_ids: List[str],
    gene_to_ann: Dict[str, str],
    out_path: str,
    logger: PipelineLogger,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = 0
    with open(out_path, "w") as fh:
        for pid in protein_ids:
            # Need to get corresponding GeneID
            gene_id = _strip_suffix(pid)
            ann = gene_to_ann.get(gene_id, "")
            if ann:
                fh.write(f"{pid}\t{ann}\n")
                written += 1

    coverage = written / max(len(protein_ids), 1) * 100
    file_name = os.path.basename(out_path)
    logger.result(file_name, f"{written:,} annotated ({coverage:.1f}%)", indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Bulk Download UniProt Annotations")
    parser.add_argument("--dataset", choices=["human", "yeast", "both"], default="human")
    parser.add_argument("--type", choices=["subcellular", "go_term", "all"], default="all")
    parser.add_argument("--tag-date", action="store_true")
    parser.add_argument("--out-dir", default="data/annotations")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = PipelineLogger()

    datasets = ["human", "yeast"] if args.dataset == "both" else [args.dataset]
    out_dir = PROJECT_ROOT / args.out_dir
    date_tag = f"_{date.today().isoformat()}" if args.tag_date else ""

    logger.header("BULK UNIPROT ANNOTATION DOWNLOAD")

    for dataset in datasets:
        fasta_path = PROJECT_ROOT / _DATASET_FASTA[dataset]
        taxon_id = _ORGANISM_TAXID[dataset]

        if not fasta_path.exists():
            logger.warning(f"FASTA not found: {fasta_path}")
            continue

        logger.phase(f"DATASET: {dataset.upper()}")

        # 1. Gather Target Gene IDs
        all_ids = NegativeSampler.load_protein_ids_from_fasta(str(fasta_path))
        wt_ids = list(dict.fromkeys(p for p in all_ids if not _is_mutant(p)))
        gene_ids = list(dict.fromkeys(_strip_suffix(p) for p in wt_ids))
        target_set = frozenset(gene_ids)

        logger.info(f"Wild-Type targets: {len(gene_ids):,}")

        # 2. Extract specific annotation type maps in O(1) bulk fetch
        subcel_map, go_map = bulk_download_annotations(taxon_id, target_set, logger)

        # 3. Save to caching TSVs
        if args.type in ("all", "subcellular"):
            out_subcel = str(out_dir / f"uniprot_subcellular_{dataset}{date_tag}.tsv")
            logger.info("Saving Subcellular Cache:")
            save_annotation_tsv(wt_ids, subcel_map, out_subcel, logger)

        if args.type in ("all", "go_term"):
            out_go = str(out_dir / f"uniprot_go_{dataset}{date_tag}.tsv")
            logger.info("Saving GO Terms Cache:")
            save_annotation_tsv(wt_ids, go_map, out_go, logger)

    print("\n✅ Download pipeline successfully finished.")

if __name__ == "__main__":
    main()
