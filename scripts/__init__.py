"""Scripts for running HybridStack-PPI experiments."""

from src.logger import PipelineLogger
from src.metrics import (
    display_full_metrics,
    plot_evaluation_results,
    plot_feature_importance_for_paper,
    print_paper_style_results,
    plot_hybrid_feature_importance,
    save_feature_importance_table,
    plot_roc_pr_curves,
)
from src.data_utils import (
    load_data,
    create_feature_matrix,
    get_cache_filename,
    save_feature_matrix_h5,
    load_feature_matrix_h5,
    build_esm_only_pair_matrix,
    get_protein_based_splits,
    get_cluster_based_splits,
    load_cluster_map,
)
from .run import run_experiment, run_ablation_study, run_estackppi_esm_only_ablation, set_seed

__all__ = [
    "PipelineLogger",
    "display_full_metrics",
    "plot_evaluation_results",
    "plot_feature_importance_for_paper",
    "print_paper_style_results",
    "plot_hybrid_feature_importance",
    "save_feature_importance_table",
    "plot_roc_pr_curves",
    "load_data",
    "create_feature_matrix",
    "get_cache_filename",
    "save_feature_matrix_h5",
    "load_feature_matrix_h5",
    "build_esm_only_pair_matrix",
    "get_protein_based_splits",
    "get_cluster_based_splits",
    "load_cluster_map",
    "run_experiment",
    "run_ablation_study",
    "run_estackppi_esm_only_ablation",
    "set_seed",
]
