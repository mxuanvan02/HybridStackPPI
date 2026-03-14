#!/usr/bin/env python3
import requests
from pathlib import Path
import sys
import os

def get_uniprot_mapping(accessions):
    """Batch query UniProt for Accession -> GeneID mapping."""
    url = "https://rest.uniprot.org/uniprotkb/stream"
    mappings = {}
    chunk_size = 50 # smaller chunks
    for i in range(0, len(accessions), chunk_size):
        chunk = accessions[i : i + chunk_size]
        # Try a different query format
        acc_query = " OR ".join([f"accession:{acc}" for acc in chunk])
        params = {
            "query": f"({acc_query})",
            "format": "tsv",
            "fields": "accession,xref_geneid"
        }
        print(f"Querying UniProt for chunk {i//chunk_size + 1} ({len(chunk)} accs)...")
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code != 200:
                print(f"Error {resp.status_code}: {resp.text[:200]}")
                continue
        except Exception as e:
            print(f"Error querying UniProt: {e}")
            continue
        
        lines = resp.text.strip().split("\n")
        if len(lines) <= 1: continue
        
        for line in lines[1:]: # skip header
            parts = line.split("\t")
            if len(parts) < 2: continue
            acc = parts[0]
            gene_ids_str = parts[1]
            gene_ids = [gid.strip().replace(';', '') for gid in gene_ids_str.split(';') if gid.strip()]
            if gene_ids:
                mappings[acc] = gene_ids[0]
    return mappings

def main():
    negatome_raw = "data/annotations/negatome_manual_stringent.txt"
    human_fasta = "data/BioGrid/Human/human_dict.fasta"
    output_mapped = "data/annotations/negatome_human.tsv"
    
    if not os.path.exists(negatome_raw):
        print(f"Error: {negatome_raw} not found.")
        return

    # 1. Load raw pairs
    print(f"Loading raw Negatome pairs...")
    unique_accs = set()
    pairs = []
    with open(negatome_raw, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                p1, p2 = parts[0], parts[1]
                # Strip isoform suffix if present (e.g., P12345-2 -> P12345)
                p1_base = p1.split("-")[0]
                p2_base = p2.split("-")[0]
                pairs.append((p1_base, p2_base))
                unique_accs.add(p1_base)
                unique_accs.add(p2_base)
    
    unique_accs = sorted(list(unique_accs))
    print(f"Found {len(unique_accs)} unique base accessions.")

    # 2. Get UniProt -> GeneID mapping
    mapping = get_uniprot_mapping(unique_accs)
    print(f"Mapped {len(mapping)} accessions to GeneIDs.")

    # 3. Load project universe IDs
    print(f"Loading universe IDs from {human_fasta}...")
    universe_ids = []
    if os.path.exists(human_fasta):
        with open(human_fasta, "r") as f:
            for line in f:
                if line.startswith(">"):
                    universe_ids.append(line[1:].strip())
    
    gene_to_internal = {}
    for uid in universe_ids:
        if "_mutant_" not in uid:
            gene_id = uid.split("_")[0]
            gene_to_internal[gene_id] = uid
    print(f"Universe has {len(gene_to_internal)} wild-type proteins.")

    # 4. Map Negatome pairs
    mapped_count = 0
    with open(output_mapped, "w") as f:
        for p1, p2 in pairs:
            gid1 = mapping.get(p1)
            gid2 = mapping.get(p2)
            
            if gid1 and gid2:
                int1 = gene_to_internal.get(gid1)
                int2 = gene_to_internal.get(gid2)
                
                if int1 and int2:
                    f.write(f"{int1}\t{int2}\n")
                    mapped_count += 1
    
    print(f"Successfully mapped {mapped_count} pairs to {output_mapped}.")

if __name__ == "__main__":
    main()
