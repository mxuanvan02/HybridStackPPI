import gc
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.feature_engine import EmbeddingComputer, FeatureEngine
from src.logger import PipelineLogger
from src.metrics import (
    display_full_metrics,
    plot_feature_importance_for_paper,
    print_paper_style_results,
    plot_hybrid_feature_importance,
    save_feature_importance_table,
    plot_cv_roc_pr_curves,
    plot_cv_metric_distribution,
    plot_decision_power,
    generate_latex_table,
)
from src.data_utils import (
    load_data,
    canonicalize_pairs,
    create_feature_matrix,
    get_cache_filename,
    save_feature_matrix_h5,
    load_feature_matrix_h5,
    build_esm_only_pair_matrix,
    get_protein_based_splits,
    get_cluster_based_splits,
    load_cluster_map,
    reduce_by_clusters,
)
from src.builders import (
    create_embed_only_pipeline,
    create_embed_only_stacking_pipeline,
    create_esm_lgbm_raw_pipeline,
    create_esm_lgbm_selector_pipeline,
    create_esm_only_stacking_pipeline,
    create_esm_lr_pipeline,
    create_esm_global_lr_pipeline,
    create_interp_lr_pipeline,
    create_embed_lr_pipeline,
    create_interp_only_pipeline,
    create_interp_only_stacking_pipeline,
    create_stacking_pipeline,
    define_stacking_columns,
)

warnings.filterwarnings("ignore", category=UserWarning)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _log_selector_state(logger: PipelineLogger, model_pipeline, prefix: str = ""):
    """
    Log how many features remain after selector and list a short preview.
    Supports simple pipelines and stacking branches.
    """
    def _print_selected(name, selector_obj):
        if not selector_obj or not getattr(selector_obj, "selected_features_", None):
            return
        feats = selector_obj.selected_features_
        preview = ", ".join(feats[:15])
        suffix = " ..." if len(feats) > 15 else ""
        logger.info(f"{prefix}{name}: kept {len(feats)} features")
        logger.info(f"{prefix}{name} preview: {preview}{suffix}")

    if hasattr(model_pipeline, "named_steps"):
        _print_selected("Selector", model_pipeline.named_steps.get("selector"))

    if hasattr(model_pipeline, "estimators_"):
        for branch_name, est in model_pipeline.estimators_:
            if hasattr(est, "named_steps"):
                pre = est.named_steps.get("preprocessor")
                if pre and getattr(pre, "transformers_", None):
                    sel = getattr(pre.transformers_[0][1], "named_steps", {}).get("selector")
                    _print_selected(f"{branch_name} selector", sel)


def _extract_importances(model_pipeline, default_feature_names: list[str]):
    """
    Retrieve feature importances and the corresponding feature names from either a flat
    pipeline or the interpretable branch of a stacking classifier.
    """
    feat_names = list(default_feature_names)
    importances = None

    if hasattr(model_pipeline, "named_steps"):
        sel = model_pipeline.named_steps.get("selector")
        if sel and getattr(sel, "selected_features_", None):
            feat_names = sel.selected_features_
        model = model_pipeline.named_steps.get("model")
        if model is not None and hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

    if importances is None and hasattr(model_pipeline, "estimators_"):
        for _, est in model_pipeline.estimators_:
            if not hasattr(est, "named_steps"):
                continue
            model = est.named_steps.get("model")
            if model is not None and hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                pre = est.named_steps.get("preprocessor")
                if pre and getattr(pre, "transformers_", None):
                    sel = getattr(pre.transformers_[0][1], "named_steps", {}).get("selector")
                    if sel and getattr(sel, "selected_features_", None):
                        feat_names = sel.selected_features_
                break

    return importances, feat_names


def _save_feature_artifacts(model_pipeline, feature_names: list[str], logger: PipelineLogger, title: str):
    """
    Persist feature-importance related artifacts if the underlying estimator exposes them.
    """
    try:
        importances, feat_names = _extract_importances(model_pipeline, feature_names)
        if importances is None:
            logger.warning("Model does not expose feature_importances_; skipping importance artifacts.")
            return

        save_feature_importance_table(importances, feat_names, top_k=50, path="feature_importance_top.csv")
        plot_feature_importance_for_paper(
            model_pipeline,
            feat_names,
            title=title,
            save_path="feature_importance_paper.png",
        )
        plot_hybrid_feature_importance(
            importances,
            feat_names,
            top_k=20,
            save_path="feature_importance_hybrid.png",
        )
        logger.info("Saved feature_importance_top.csv, feature_importance_paper.png, feature_importance_hybrid.png")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not generate importance artifacts: {exc}")


def run_experiment(
    fasta_path: str,
    pairs_path: str,
    h5_cache_path: str,
    model_factory: callable,
    pairing_strategy: str = "concat",
    model_params: dict = None,
    param_grid: dict = None,
    test_fasta_path: str = None,
    test_pairs_path: str = None,
    n_splits: int = 1,
    test_size: float = 0.2,
    esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
    random_state: int = 42,
    n_jobs: int = -1,
    cluster_map: dict | None = None,
    cluster_path: str | None = None,
    cache_version: str = "v2",
) -> dict:
    """
    Unified experiment runner with independent-test or protein-level CV.
    """
    logger = PipelineLogger()

    def _load_and_clean(fasta_file: str, pairs_file: str, dataset_name: str):
        seqs, pairs = load_data(fasta_file, pairs_file)
        pairs = canonicalize_pairs(pairs, dataset_name=dataset_name, logger=logger)
        return seqs, pairs

    if test_fasta_path and test_pairs_path:
        logger.header(f"EXPERIMENT: INDEPENDENT TEST (Strategy: {pairing_strategy})")

        train_sequences, train_pairs_df = _load_and_clean(fasta_path, pairs_path, dataset_name="Train")
        test_sequences, test_pairs_df = _load_and_clean(test_fasta_path, test_pairs_path, dataset_name="Test")

        # --- Redundancy Reduction for Independent Test ---
        cluster_mapping = None
        if cluster_path:
            try:
                cluster_mapping = load_cluster_map(cluster_path)
                logger.info(f"Loaded cluster map from {cluster_path}")
                train_pairs_df = reduce_by_clusters(train_pairs_df, cluster_mapping, logger=logger)
                test_pairs_df = reduce_by_clusters(test_pairs_df, cluster_mapping, logger=logger)
                
                # Reverse map: Cluster ID -> Rep Protein ID
                cluster_to_rep = {}
                for pid, cid in cluster_mapping.items():
                    if cid not in cluster_to_rep and (pid in train_sequences or pid in test_sequences):
                        cluster_to_rep[cid] = pid

                used_train = set(train_pairs_df["protein1"]).union(set(train_pairs_df["protein2"]))
                train_sequences = {cid: (train_sequences[cluster_to_rep[cid]] if cid in cluster_to_rep else train_sequences[cid]) 
                                   for cid in used_train if cid in cluster_to_rep or cid in train_sequences}
                
                used_test = set(test_pairs_df["protein1"]).union(set(test_pairs_df["protein2"]))
                test_sequences = {cid: (test_sequences[cluster_to_rep[cid]] if cid in cluster_to_rep else test_sequences[cid]) 
                                  for cid in used_test if cid in cluster_to_rep or cid in test_sequences}
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Clustering reduction failed: {exc}")

        suffix = "cluster_v3" if cluster_path else ""
        train_cache_path = get_cache_filename(
            pairs_path, pairing_strategy, esm_model_name, cache_version=cache_version, suffix=suffix
        )
        test_cache_path = get_cache_filename(
            test_pairs_path, pairing_strategy, esm_model_name, cache_version=cache_version, suffix=suffix
        )

        feature_engine = None
        single_feature_names = None

        def _ensure_feature_engine():
            nonlocal feature_engine, single_feature_names
            if feature_engine is None:
                embedding_computer = EmbeddingComputer(model_name=esm_model_name)
                feature_engine = FeatureEngine(h5_cache_path, embedding_computer)
                single_feature_names = feature_engine.get_feature_names()

        def _load_or_build(cache_path, sequences, pairs_df, split_name: str):
            if os.path.exists(cache_path):
                logger.phase(f"Loading {split_name} Features from Cache")
                X_df_cached, y_s_cached = load_feature_matrix_h5(cache_path)
                if len(X_df_cached) == len(pairs_df):
                    return X_df_cached, y_s_cached
                logger.warning(
                    f"{split_name} cache rows ({len(X_df_cached)}) do not match cleaned pairs ({len(pairs_df)}). "
                    "Recomputing to avoid duplicated pairs."
                )

            logger.phase(f"{split_name} Cache NOT FOUND or stale. Running Extraction...")
            _ensure_feature_engine()
            protein_features = feature_engine.extract_all_features(sequences)
            X_df, y_s = create_feature_matrix(pairs_df, protein_features, single_feature_names, pairing_strategy)
            save_feature_matrix_h5(X_df, y_s, cache_path)
            return X_df, y_s

        X_train, y_train = _load_or_build(train_cache_path, train_sequences, train_pairs_df, "TRAIN")
        X_test, y_test = _load_or_build(test_cache_path, test_sequences, test_pairs_df, "TEST")

        logger.phase("Training Model")
        model_pipeline = model_factory(n_jobs=n_jobs)
        model_pipeline.fit(X_train, y_train)
        _log_selector_state(logger, model_pipeline, prefix="[Train] ")
        _save_feature_artifacts(
            model_pipeline,
            feature_names=list(X_train.columns),
            logger=logger,
            title="HybridStack-PPI Feature Importance (Top 20)",
        )

        logger.phase("Evaluating on Independent Test Set")
        y_pred = model_pipeline.predict(X_test)
        y_proba = model_pipeline.predict_proba(X_test)[:, 1]

        metrics = display_full_metrics(y_test, y_pred, y_proba, title="Independent Test Results")
        try:
            plot_roc_pr_curves(y_test, y_proba, title="Independent Test", prefix="independent")
            logger.info("Saved independent_roc.png and independent_pr.png")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not plot ROC/PR for independent test: {exc}")
        return metrics

    sequences, pairs_df = _load_and_clean(fasta_path, pairs_path, dataset_name="Train")
    
    # --- STEP 0: Sequence-level Redundancy Reduction via CD-HIT ---
    cluster_mapping = None
    if cluster_path:
        try:
            cluster_mapping = load_cluster_map(cluster_path)
            logger.info(f"Loaded cluster map from {cluster_path} with {len(cluster_mapping)} entries.")
            
            # Reduce sequences and pairs to cluster representatives
            pairs_df = reduce_by_clusters(pairs_df, cluster_mapping, logger=logger)
            
            # Create a reverse map: Cluster ID -> Representative Protein ID
            # Since cluster_mapping is Protein -> Cluster, we pick the first protein encountered for each cluster.
            cluster_to_rep = {}
            for prot_id, clstr_id in cluster_mapping.items():
                if clstr_id not in cluster_to_rep and prot_id in sequences:
                    cluster_to_rep[clstr_id] = prot_id
            
            # New sequences dict where keys are Cluster IDs and values are representatives' sequences
            new_sequences = {}
            used_clusters = set(pairs_df["protein1"]).union(set(pairs_df["protein2"]))
            for clstr_id in used_clusters:
                rep_id = cluster_to_rep.get(clstr_id)
                if rep_id:
                    new_sequences[clstr_id] = sequences[rep_id]
                else:
                    # If clstr_id is actually a singleton protein ID (from reduce_by_clusters mapping fallback)
                    if clstr_id in sequences:
                        new_sequences[clstr_id] = sequences[clstr_id]

            sequences = new_sequences
            logger.info(f"Reduced sequences for extraction: {len(sequences)} cluster representatives.")
            
            # --- STEP 0.1: Save Reduced Datasets for Reproducibility ---
            reduced_data_dir = os.path.join(os.path.dirname(fasta_path), "CDHIT_Reduced")
            os.makedirs(reduced_data_dir, exist_ok=True)
            
            dataset_tag = "yeast" if "yeast" in fasta_path.lower() else "human"
            clean_fasta_path = os.path.join(reduced_data_dir, f"{dataset_tag}_clean.fasta")
            clean_pairs_path = os.path.join(reduced_data_dir, f"{dataset_tag}_clean_pairs.tsv")
            
            if not os.path.exists(clean_fasta_path) or not os.path.exists(clean_pairs_path):
                logger.info(f"💾 Saving reduced datasets to {reduced_data_dir}...")
                # Save FASTA
                with open(clean_fasta_path, "w") as f:
                    for pid, seq in sequences.items():
                        f.write(f">{pid}\n{seq}\n")
                # Save TSV
                pairs_df.to_csv(clean_pairs_path, sep="\t", header=False, index=False)
                logger.info(f"✅ Saved {len(sequences)} seqs to FASTA and {len(pairs_df)} pairs to TSV.")
            else:
                logger.info(f"ℹ️ Clean datasets already exist in {reduced_data_dir}. Skipping save.")

        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not apply clustering reduction: {exc}")
            cluster_mapping = None

    suffix = "cluster_v3" if cluster_path else ""
    cache_path = get_cache_filename(pairs_path, pairing_strategy, esm_model_name, cache_version=cache_version, suffix=suffix)

    feature_engine = None
    single_feature_names = None

    def _ensure_feature_engine():
        nonlocal feature_engine, single_feature_names
        if feature_engine is None:
            embedding_computer = EmbeddingComputer(model_name=esm_model_name)
            feature_engine = FeatureEngine(h5_cache_path, embedding_computer)
            single_feature_names = feature_engine.get_feature_names()

    need_recompute = True
    if os.path.exists(cache_path):
        logger.phase("Loading Features from Cache")
        X_df, y_s = load_feature_matrix_h5(cache_path)
        if len(X_df) == len(pairs_df):
            need_recompute = False
        else:
            logger.warning(
                f"Cache rows ({len(X_df)}) do not match cleaned pairs ({len(pairs_df)}). "
                "Recomputing to avoid duplicated pairs."
            )

    if need_recompute:
        logger.phase("Cache NOT FOUND or stale. Running Full Feature Extraction")
        _ensure_feature_engine()
        protein_features = feature_engine.extract_all_features(sequences)
        X_df, y_s = create_feature_matrix(pairs_df, protein_features, single_feature_names, pairing_strategy)
        save_feature_matrix_h5(X_df, y_s, cache_path)

    pairs_df_for_split = pairs_df
    if n_splits > 1:
        if cluster_mapping:
            logger.header(f"EXPERIMENT: {n_splits}-FOLD CV (CLUSTER-LEVEL SPLIT)")
            # Note: since pairs_df is already cluster-level, we could use standard split
            # but using get_cluster_based_splits is safer/more explicit.
            splits = get_cluster_based_splits(
                pairs_df_for_split, cluster_mapping, n_splits=n_splits, random_state=random_state
            )
        else:
            logger.header(f"EXPERIMENT: {n_splits}-FOLD CV (PROTEIN-LEVEL SPLIT - NO LEAKAGE)")
            splits = get_protein_based_splits(pairs_df_for_split, n_splits=n_splits, random_state=random_state)

        fold_metrics_list = []
        cv_results = []  # NEW: Collect y_true and y_proba for CV visualization

        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            import time
            fold_start = time.time()
            logger.info(f"--- Fold {fold_idx + 1}/{n_splits} ---")
            logger.info(f"  📊 Train: {len(train_indices)} pairs, Val: {len(val_indices)} pairs")
            
            X_train_fold, X_val_fold = X_df.iloc[train_indices], X_df.iloc[val_indices]
            y_train_fold, y_val_fold = y_s.iloc[train_indices], y_s.iloc[val_indices]

            logger.info(f"  🔧 Building model pipeline...")
            model_pipeline = model_factory(n_jobs=n_jobs)
            
            logger.info(f"  🏋️ Training model (this may take a while)...")
            train_start = time.time()
            model_pipeline.fit(X_train_fold, y_train_fold)
            train_time = time.time() - train_start
            logger.info(f"  ✅ Training complete in {train_time:.1f}s")
            
            _log_selector_state(logger, model_pipeline, prefix=f"[Fold {fold_idx + 1}] ")

            # --- Latency Benchmark (Fold 1 only) ---
            if fold_idx == 0:
                logger.header("⏱️ INFERENCE LATENCY BREAKDOWN (CPU)")
                bench_samples = X_val_fold.sample(min(50, len(X_val_fold)), random_state=42)
                
                # 1. Feature Extraction (Interpretable) + Embedding Lookup + Classification
                total_times = []
                # Prime CPU
                model_pipeline.predict_proba(bench_samples.iloc[[0]])
                
                # We need a representative sequence for extraction benchmark
                # We'll take a pair from the original pairs_df mapping to these indices
                test_pair = pairs_df.iloc[val_indices[0]]
                p1, p2 = test_pair['protein1'], test_pair['protein2']
                seq1, seq2 = sequences.get(p1, "M"), sequences.get(p2, "M")
                
                logger.info(f"  ℹ️  Benchmarking protein pair: {p1} ({len(seq1)} aa) / {p2} ({len(seq2)} aa)")
                
                # A. Bio-feature Extraction (simulated on one pair)
                t0 = time.time()
                _ensure_feature_engine()
                # Mock extraction for one pair
                feature_engine.extract_all_features({p1: seq1, p2: seq2})
                bio_time = (time.time() - t0) * 1000 # ms
                
                # B. ESM-2 Embedding (simulated lookup/compute)
                t0 = time.time()
                feature_engine.embedding_computer.compute_full_embeddings(seq1)
                embed_time = (time.time() - t0) * 1000 # ms
                
                # C. Stacking Classification (Inference)
                t0 = time.time()
                for i in range(len(bench_samples)):
                    model_pipeline.predict_proba(bench_samples.iloc[[i]])
                stack_time = ((time.time() - t0) / len(bench_samples)) * 1000 # ms
                
                logger.info(f"  👉 Bio-feature Extraction:  {bio_time:.2f} ms")
                logger.info(f"  👉 ESM-2 Embedding:        {embed_time:.2f} ms")
                logger.info(f"  👉 Stacking Classification: {stack_time:.2f} ms")
                logger.info(f"  🚀 Total Latency/Pair:      {bio_time + embed_time + stack_time:.2f} ms")

            logger.info(f"  🔮 Predicting on validation set...")
            y_pred_val = model_pipeline.predict(X_val_fold)
            y_proba_val = model_pipeline.predict_proba(X_val_fold)[:, 1]
            
            # NEW: Collect results for CV visualization
            cv_results.append({
                'y_true': y_val_fold.values if hasattr(y_val_fold, 'values') else y_val_fold,
                'y_proba': y_proba_val,
            })

            if fold_idx == n_splits - 1:
                logger.info("Generating Publication-Quality Feature Importance Plot...")
                _save_feature_artifacts(
                    model_pipeline,
                    feature_names=list(X_train_fold.columns),
                    logger=logger,
                    title="HybridStack-PPI Feature Importance (Top 20)",
                )
                try:
                    plot_roc_pr_curves(
                        y_val_fold,
                        y_proba_val,
                        title=f"Fold {fold_idx + 1}",
                        prefix=f"fold{fold_idx + 1}",
                    )
                    logger.info(f"Saved fold{fold_idx + 1}_roc.png and fold{fold_idx + 1}_pr.png")
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Could not plot ROC/PR for fold {fold_idx + 1}: {exc}")

            metrics = display_full_metrics(y_val_fold, y_pred_val, y_proba_val, title=f"Fold {fold_idx + 1}")
            fold_metrics_list.append(metrics)

            # --- Decision Power (Branch Contribution) ---
            if hasattr(model_pipeline.named_steps.get('model'), 'final_estimator_'):
                meta = model_pipeline.named_steps['model'].final_estimator_
                if hasattr(meta, 'coef_'):
                    coefs = np.abs(meta.coef_[0])
                    total_coef = np.sum(coefs)
                    if total_coef > 0:
                        contributions = coefs / total_coef * 100
                        logger.info(f"  💡 Decision Power: Biological={contributions[0]:.1f}%, Deep Learning={contributions[1]:.1f}%")
                        if fold_idx == 0:
                            dname = fasta_path.split('/')[-2].upper()
                            plot_decision_power({"Biological": contributions[0], "Deep Learning": contributions[1]}, dataset_name=dname)
            
            fold_time = time.time() - fold_start
            logger.info(f"  ⏱️ Fold {fold_idx + 1} complete in {fold_time:.1f}s | Acc: {metrics['Accuracy']:.2%} | AUC: {metrics['ROC-AUC']:.4f}")

            # Cleanup immediately after each fold to prevent RAM bloat
            del X_train_fold, X_val_fold, y_train_fold, y_val_fold, model_pipeline
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Final CV Visualization Summary
        logger.header("📊 GENERATING CROSS-VALIDATION SUMMARY")
        dname = fasta_path.split('/')[-2]
        plot_cv_roc_pr_curves(cv_results, title=f"{n_splits}-Fold CV ({dname})", prefix=f"{dname.lower()}_cv")
        plot_cv_metric_distribution(fold_metrics_list, prefix=f"{dname.lower()}_cv")
        generate_latex_table(fold_metrics_list, method_name=f"HybridStack-PPI ({dname})")

        return metrics
        
        try:
            plot_cv_metric_distribution(
                fold_metrics_list,
                save_dir='results/plots',
                title=f'{n_splits}-Fold CV'
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not generate CV metric distribution: {exc}")
        
        return pd.DataFrame(fold_metrics_list).mean().to_dict()

    logger.warning("Running simple Train/Test split (Random). Be careful of Data Leakage!")
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_s, test_size=test_size, random_state=random_state, stratify=y_s
    )
    model_pipeline = model_factory(n_jobs=n_jobs)
    model_pipeline.fit(X_train, y_train)
    _log_selector_state(logger, model_pipeline, prefix="[Train/Test] ")
    y_pred_test = model_pipeline.predict(X_test)
    y_proba_test = model_pipeline.predict_proba(X_test)[:, 1]
    metrics = display_full_metrics(y_test, y_pred_test, y_proba_test)
    try:
        plot_roc_pr_curves(y_test, y_proba_test, title="Train/Test Split", prefix="train_test")
        logger.info("Saved train_test_roc.png and train_test_pr.png")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not plot ROC/PR for train/test: {exc}")
    return metrics


def run_ablation_study(
    fasta_path: str,
    pairs_path: str,
    h5_cache_path: str,
    esm_model_name: str,
    n_splits: int = 5,
    n_jobs: int = -1,
):
    """
    Wrapper to run all ablation experiments.
    
    Structure:
    - 3 Feature types: Interpretable, Embedding (Full), ESM2-Global
    - For each: LR baseline (No Selector, No Stacking) vs Stacking+Selector
    - Plus: Hybrid model (2 branches → Stacking)
    """
    logger = PipelineLogger()
    logger.header("🚀 STARTING FULL ABLATION STUDY 🚀")

    all_results = []

    logger.phase("Initializing FeatureEngine (for column names)")
    try:
        embedding_computer = EmbeddingComputer(model_name=esm_model_name)
        feature_engine = FeatureEngine(h5_cache_path=h5_cache_path, embedding_computer=embedding_computer)
        interp_cols_concat, embed_cols_concat = define_stacking_columns(feature_engine, "concat")
        esm2_global_cols = [c for c in embed_cols_concat if "Global_ESM" in c]
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Cannot initialize engine, possibly API error. Error: {exc}")
        return

    # =========================================================================
    # INTERPRETABLE FEATURES
    # =========================================================================
    logger.phase("Running Ablation 1: Interpretable → LR (Baseline)")
    model_factory_1 = lambda n_jobs=-1: create_interp_lr_pipeline(interp_cols_concat, n_jobs)
    res1 = run_experiment(
        fasta_path, pairs_path, h5_cache_path, model_factory_1,
        n_splits=n_splits, n_jobs=n_jobs, esm_model_name=esm_model_name, pairing_strategy="concat",
    )
    res1["Model"] = "1. Interp → LR (Baseline)"
    all_results.append(res1)
    del model_factory_1, res1; gc.collect()

    logger.phase("Running Ablation 2: Interpretable → Selector + Stacking → LR")
    model_factory_2 = lambda n_jobs=-1: create_interp_only_stacking_pipeline(interp_cols_concat, n_jobs, use_selector=True)
    res2 = run_experiment(
        fasta_path, pairs_path, h5_cache_path, model_factory_2,
        n_splits=n_splits, n_jobs=n_jobs, esm_model_name=esm_model_name, pairing_strategy="concat",
    )
    res2["Model"] = "2. Interp → Selector+Stacking → LR"
    all_results.append(res2)
    del model_factory_2, res2; gc.collect()

    # =========================================================================
    # EMBEDDING FEATURES (Global + Local)
    # =========================================================================
    logger.phase("Running Ablation 3: Embedding → LR (Baseline)")
    model_factory_3 = lambda n_jobs=-1: create_embed_lr_pipeline(embed_cols_concat, n_jobs)
    res3 = run_experiment(
        fasta_path, pairs_path, h5_cache_path, model_factory_3,
        n_splits=n_splits, n_jobs=n_jobs, esm_model_name=esm_model_name, pairing_strategy="concat",
    )
    res3["Model"] = "3. Embed → LR (Baseline)"
    all_results.append(res3)
    del model_factory_3, res3; gc.collect()

    logger.phase("Running Ablation 4: Embedding → Selector + Stacking → LR")
    model_factory_4 = lambda n_jobs=-1: create_embed_only_stacking_pipeline(embed_cols_concat, n_jobs, use_selector=True)
    res4 = run_experiment(
        fasta_path, pairs_path, h5_cache_path, model_factory_4,
        n_splits=n_splits, n_jobs=n_jobs, esm_model_name=esm_model_name, pairing_strategy="concat",
    )
    res4["Model"] = "4. Embed → Selector+Stacking → LR"
    all_results.append(res4)
    del model_factory_4, res4; gc.collect()

    # =========================================================================
    # ESM2-GLOBAL ONLY FEATURES
    # =========================================================================
    logger.phase("Running Ablation 5: ESM2-Global → LR (Baseline)")
    model_factory_5 = lambda n_jobs=-1: create_esm_global_lr_pipeline(esm2_global_cols, n_jobs)
    res5 = run_experiment(
        fasta_path, pairs_path, h5_cache_path, model_factory_5,
        n_splits=n_splits, n_jobs=n_jobs, esm_model_name=esm_model_name, pairing_strategy="concat",
    )
    res5["Model"] = "5. ESM2-Global → LR (Baseline)"
    all_results.append(res5)
    del model_factory_5, res5; gc.collect()

    logger.phase("Running Ablation 6: ESM2-Global → Selector + Stacking → LR")
    model_factory_6 = lambda n_jobs=-1: create_esm_only_stacking_pipeline(esm2_global_cols, n_jobs, use_selector=True)
    res6 = run_experiment(
        fasta_path, pairs_path, h5_cache_path, model_factory_6,
        n_splits=n_splits, n_jobs=n_jobs, esm_model_name=esm_model_name, pairing_strategy="concat",
    )
    res6["Model"] = "6. ESM2-Global → Selector+Stacking → LR"
    all_results.append(res6)
    del model_factory_6, res6; gc.collect()

    # =========================================================================
    # HYBRID MODEL (2 BRANCHES)
    # =========================================================================
    logger.phase("Running Ablation 7: Hybrid (Interp+Embed) → Stacking → LR")
    model_factory_7 = lambda n_jobs=-1: create_stacking_pipeline(interp_cols_concat, embed_cols_concat, n_jobs, use_selector=False)
    res7 = run_experiment(
        fasta_path, pairs_path, h5_cache_path, model_factory_7,
        n_splits=n_splits, n_jobs=n_jobs, esm_model_name=esm_model_name, pairing_strategy="concat",
    )
    res7["Model"] = "7. Hybrid (Interp+Embed) → Stacking → LR"
    all_results.append(res7)
    del model_factory_7, res7; gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.header("📊 ABLATION STUDY FINAL RESULTS 📊")
    results_df = pd.DataFrame(all_results)
    results_df = results_df.set_index("Model")

    cols_order = [
        "Accuracy",
        "ROC-AUC",
        "PR-AUC",
        "F1 Score",
        "MCC",
        "Precision",
        "Recall (Sensitivity)",
        "Specificity",
    ]
    cols_to_show = [col for col in cols_order if col in results_df.columns]

    print(results_df[cols_to_show].to_string(float_format="%.4f"))
    return results_df


def run_estackppi_esm_only_ablation(
    fasta_path: str,
    pairs_path: str,
    h5_cache_path: str,
    esm_model_name: str,
    n_splits: int = 5,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Mini ablation for E-StackPPI (ESM2-only), 3 models.
    """
    logger = PipelineLogger()
    logger.header("🚀 E-STACKPPI MINI ABLATION (ESM2-ONLY) 🚀")

    X_df, y_s = build_esm_only_pair_matrix(fasta_path=fasta_path, pairs_path=pairs_path, h5_cache_path=h5_cache_path)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    model_builders = [
        ("1. ESM-only + LR (no selector)", create_esm_lr_pipeline),
        ("2. ESM-only + LGBM (no selector)", create_esm_lgbm_raw_pipeline),
        ("3. ESM-only + LGBM (3-stage selector)", create_esm_lgbm_selector_pipeline),
    ]

    all_results = []

    for model_name, builder in model_builders:
        logger.phase(f"MODEL: {model_name}")
        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_df, y_s), start=1):
            logger.info(f"--- Fold {fold_idx}/{n_splits} ---")

            X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
            y_train, y_val = y_s.iloc[train_idx], y_s.iloc[val_idx]

            model = builder(n_jobs=n_jobs)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]

            metrics = display_full_metrics(y_val, y_pred, y_proba, title=f"{model_name} - Fold {fold_idx}")
            fold_metrics.append(metrics)

        fold_df = pd.DataFrame(fold_metrics)
        mean_scores = fold_df.mean().to_dict()
        mean_scores["Model"] = model_name
        all_results.append(mean_scores)

        logger.info(f"Dọn bộ nhớ sau {model_name}...")
        del model, fold_metrics, fold_df
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results).set_index("Model")
    cols_order = [
        "Accuracy",
        "ROC-AUC",
        "PR-AUC",
        "F1 Score",
        "MCC",
        "Precision",
        "Recall (Sensitivity)",
        "Specificity",
    ]
    cols_to_show = [c for c in cols_order if c in results_df.columns]

    logger.header("📊 E-STACKPPI MINI ABLATION – FINAL RESULTS 📊")
    print(results_df[cols_to_show].to_string(float_format="%.4f"))

    return results_df


def create_stacking_pipeline_for_notebook(
    pairing_strategy: str,
    n_jobs: int = -1,
    h5_cache_path: str = "cache/esm2_embeddings.h5",
    esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
):
    """
    Convenience helper: build stacking pipeline with columns derived from FeatureEngine.
    Mirrors the notebook setup to avoid duplicated boilerplate.
    """
    embedding_computer = EmbeddingComputer(model_name=esm_model_name)
    feature_engine = FeatureEngine(h5_cache_path=h5_cache_path, embedding_computer=embedding_computer)
    interp_cols, embed_cols = define_stacking_columns(feature_engine, pairing_strategy=pairing_strategy)
    return create_stacking_pipeline(interp_cols=interp_cols, embed_cols=embed_cols, n_jobs=n_jobs, use_selector=True)
