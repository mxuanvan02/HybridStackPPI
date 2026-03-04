# Protein Annotation Cache Files

This directory stores pre-downloaded annotation data from external databases.
These files make the `subcellular` and `go_term` negative-sampling strategies
**fully reproducible** without live API calls.

## File naming convention

```
uniprot_{annotation_type}_{organism}.tsv
```

| File | Strategy | Content |
|------|----------|---------|
| `uniprot_subcellular_human.tsv` | `subcellular` | Subcellular location per human protein |
| `uniprot_subcellular_yeast.tsv` | `subcellular` | Subcellular location per yeast protein |
| `uniprot_go_human.tsv` | `go_term` | GO Biological Process per human protein |
| `uniprot_go_yeast.tsv` | `go_term` | GO Biological Process per yeast protein |

## File format

Tab-separated, **no header**, two columns:

```
{protein_id}    {annotation_value}
```

where `{protein_id}` matches the IDs in `human_dict.fasta` / `yeast_dict.fasta`
(e.g. `2624_reviewed`).

Example rows for `uniprot_subcellular_human.tsv`:
```
2624_reviewed   Nucleus
5371_reviewed   Cytoplasm
6118_reviewed   Cell membrane
```

## How to generate these files

Run the download helper script:

```bash
# Download subcellular localization for Human proteins (≈ 5 minutes):
python scripts/download_annotations.py --dataset human --type subcellular

# Download GO terms for Human proteins:
python scripts/download_annotations.py --dataset human --type go_term

# Download both for both organisms:
python scripts/download_annotations.py --dataset both --type all
```

The script queries UniProt REST API in batches of 500 IDs and saves the
result to this directory.  Subsequent runs use the cached file automatically.

## Version pinning

Each downloaded file records the retrieval date in its filename when using
`--tag-date`:

```bash
python scripts/download_annotations.py --dataset human --type subcellular --tag-date
# → uniprot_subcellular_human_2025-04-01.tsv
```

Pin to a specific file by passing `--annotation-cache` to `generate_negatives.py`.
