import gc
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from hybridstack.feature_engine import EmbeddingComputer, FeatureEngine
from hybridstack.logger import PipelineLogger
from hybridstack.metrics import (
    display_full_metrics,
    plot_feature_importance_for_paper,
    print_paper_style_results,
    plot_hybrid_feature_importance,
    save_feature_importance_table,
    plot_roc_pr_curves,
    plot_cv_roc_pr_curves,
    plot_cv_metric_distribution,
    plot_f1_threshold_curve,
    plot_oof_confusion_matrix,
)
from hybridstack.data_utils import (
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
)
from hybridstack.builders import (
    create_embed_only_pipeline,
    create_esm_lgbm_raw_pipeline,
    create_esm_lgbm_selector_pipeline,
    create_esm_lr_pipeline,
    create_interp_only_pipeline,
    create_stacking_pipeline,
    define_stacking_columns,
    create_early_fusion_pipeline,
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
    split_strategy: str = "protein",
    cluster_path: str = None,
    cache_version: str = "v1",
    output_dir: str = "results",
) -> dict:
    """
    Execute a full classification experiment using a given model factory.

    Args:
        output_dir: Root directory for all output artifacts (models, plots, CSV).
                     Each dataset should have its own output_dir (e.g. results/human/).
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

        train_cache_path = get_cache_filename(
            pairs_path, pairing_strategy, esm_model_name, cache_version=cache_version
        )
        test_cache_path = get_cache_filename(
            test_pairs_path, pairing_strategy, esm_model_name, cache_version=cache_version
        )

        feature_engine = None
        single_feature_names = None

        def _ensure_feature_engine():
            nonlocal feature_engine, single_feature_names
            if feature_engine is None:
                embedding_computer = EmbeddingComputer(model_name=esm_model_name)
                feature_engine = FeatureEngine(h5_cache_path, embedding_computer)
                single_feature_names = feature_engine.get_feature_names()

        def _load_or_build(cache_path, seqs, pairs_df_sub, split_name: str, full_pairs_path: str):
            if os.path.exists(cache_path):
                logger.phase(f"Loading {split_name} Features from Cache")
                X_df_cached, y_s_cached = load_feature_matrix_h5(cache_path)
                if len(X_df_cached) == len(pairs_df_sub):
                    return X_df_cached, y_s_cached
                logger.warning(
                    f"{split_name} cache rows ({len(X_df_cached)}) do not match cleaned pairs ({len(pairs_df_sub)}). "
                    "Recomputing to avoid duplicated pairs."
                )

            logger.phase(f"{split_name} Cache NOT FOUND or stale. Running Extraction...")
            _ensure_feature_engine()
            
            # --- CACHE RECOVERY LOGIC ---
            protein_features = {}
            if pairing_strategy == "concat" and ("_random" in full_pairs_path or "_same_compartment" in full_pairs_path or "_same_go" in full_pairs_path or "_negatome" in full_pairs_path):
                from hybridstack.data_utils import extract_protein_features_from_pair_cache
                old_pairs_path = full_pairs_path.replace("_random", "").replace("_same_compartment", "").replace("_same_go", "").replace("_negatome", "")
                old_cache_path = get_cache_filename(old_pairs_path, pairing_strategy, esm_model_name, cache_version=cache_version)
                
                if os.path.exists(old_cache_path):
                    logger.info(f"Attempting to recover protein features from {old_cache_path}...")
                    protein_features = extract_protein_features_from_pair_cache(old_cache_path, old_pairs_path)
                    
                    if protein_features:
                        first_dim = len(next(iter(protein_features.values())))
                        expected_dim = len(single_feature_names)
                        if first_dim != expected_dim:
                            logger.warning(f"Dimension mismatch in recovered cache ({first_dim} vs expected {expected_dim}). Resetting recovery.")
                            protein_features = {}
            
            if protein_features:
                required_seqs = set(pairs_df_sub["protein1"]).union(set(pairs_df_sub["protein2"]))
                missing_sequences = {seq_id: seq for seq_id, seq in seqs.items() if seq_id in required_seqs and seq_id not in protein_features}
                if missing_sequences:
                    logger.info(f"Computing features for {len(missing_sequences)} missing sequences...")
                    computed_features = feature_engine.extract_all_features(missing_sequences)
                    protein_features.update(computed_features)
                else:
                    logger.info("All required sequence features recovered from old cache.")
            else:
                required_seqs = set(pairs_df_sub["protein1"]).union(set(pairs_df_sub["protein2"]))
                needed_seqs = {seq_id: seq for seq_id, seq in seqs.items() if seq_id in required_seqs}
                logger.info(f"Computing features for all {len(needed_seqs)} needed sequences...")
                protein_features = feature_engine.extract_all_features(needed_seqs)
            
            # --- END CACHE RECOVERY ---
            
            X_df, y_s = create_feature_matrix(pairs_df_sub, protein_features, single_feature_names, pairing_strategy)
            save_feature_matrix_h5(X_df, y_s, cache_path)
            return X_df, y_s

        X_train, y_train = _load_or_build(train_cache_path, train_sequences, train_pairs_df, "TRAIN", pairs_path)
        X_test, y_test = _load_or_build(test_cache_path, test_sequences, test_pairs_df, "TEST", test_pairs_path)

        logger.phase("Training Model")
        X_train_np = np.ascontiguousarray(X_train.to_numpy(dtype=np.float32))
        y_train_np = np.ascontiguousarray(y_train.to_numpy(dtype=np.float32))
        X_test_np = np.ascontiguousarray(X_test.to_numpy(dtype=np.float32))

        model_pipeline = model_factory(n_jobs=n_jobs, feature_names=list(X_train.columns))
        model_pipeline.fit(X_train_np, y_train_np)
        _log_selector_state(logger, model_pipeline, prefix="[Train] ")
        _save_feature_artifacts(
            model_pipeline,
            feature_names=list(X_train.columns),
            logger=logger,
            title="HybridStack-PPI Feature Importance (Top 20)",
        )

        logger.phase("Evaluating on Independent Test Set")
        # --- NEW: Dynamic Thresholding (F1-Optimized on Train Set) ---
        from sklearn.metrics import precision_recall_curve
        y_train_proba = model_pipeline.predict_proba(X_train_np)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_train_np, y_train_proba)
        
        # Calculate F1 score for each threshold safely
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        f1_scores = np.nan_to_num(f1_scores)
        
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        logger.info(f"Optimal Threshold on Train Set (F1-Maximized): {optimal_threshold:.4f}")
        
        y_proba = model_pipeline.predict_proba(X_test_np)[:, 1]
        y_pred = (y_proba >= optimal_threshold).astype(int)

        metrics = display_full_metrics(y_test, y_pred, y_proba, title="Independent Test Results")
        try:
            plot_roc_pr_curves(y_test, y_proba, title="Independent Test", prefix="independent")
            logger.info("Saved independent_roc.png and independent_pr.png")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not plot ROC/PR for independent test: {exc}")
        return metrics

    sequences, pairs_df = _load_and_clean(fasta_path, pairs_path, dataset_name="Train")
    cache_path = get_cache_filename(pairs_path, pairing_strategy, esm_model_name, cache_version=cache_version)

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
        
        # --- CACHE RECOVERY LOGIC ---
        protein_features = {}
        if pairing_strategy == "concat" and ("_random" in pairs_path or "_same_compartment" in pairs_path or "_same_go" in pairs_path or "_negatome" in pairs_path):
            from hybridstack.data_utils import extract_protein_features_from_pair_cache
            old_pairs_path = pairs_path.replace("_random", "").replace("_same_compartment", "").replace("_same_go", "").replace("_negatome", "")
            old_cache_path = get_cache_filename(old_pairs_path, pairing_strategy, esm_model_name, cache_version=cache_version)
            
            if os.path.exists(old_cache_path):
                logger.info(f"Attempting to recover protein features from {old_cache_path}...")
                protein_features = extract_protein_features_from_pair_cache(old_cache_path, old_pairs_path)
                
                if protein_features:
                    first_dim = len(next(iter(protein_features.values())))
                    expected_dim = len(single_feature_names)
                    if first_dim != expected_dim:
                        logger.warning(f"Dimension mismatch in recovered cache ({first_dim} vs expected {expected_dim}). Resetting recovery.")
                        protein_features = {}
        
        if protein_features:
            required_seqs = set(pairs_df["protein1"]).union(set(pairs_df["protein2"]))
            missing_sequences = {seq_id: seq for seq_id, seq in sequences.items() if seq_id in required_seqs and seq_id not in protein_features}
            if missing_sequences:
                logger.info(f"Computing features for {len(missing_sequences)} missing sequences...")
                computed_features = feature_engine.extract_all_features(missing_sequences)
                protein_features.update(computed_features)
            else:
                logger.info("All required sequence features recovered from old cache.")
        else:
            required_seqs = set(pairs_df["protein1"]).union(set(pairs_df["protein2"]))
            needed_seqs = {seq_id: seq for seq_id, seq in sequences.items() if seq_id in required_seqs}
            logger.info(f"Computing features for all {len(needed_seqs)} needed sequences...")
            protein_features = feature_engine.extract_all_features(needed_seqs)
        
        # --- END CACHE RECOVERY ---
        
        X_df, y_s = create_feature_matrix(pairs_df, protein_features, single_feature_names, pairing_strategy)
        save_feature_matrix_h5(X_df, y_s, cache_path)

    pairs_df_for_split = pairs_df
    cluster_mapping = None
    if cluster_mapping is None and cluster_path:
        try:
            cluster_mapping = load_cluster_map(cluster_path)
            logger.info(f"Loaded cluster map from {cluster_path} with {len(cluster_mapping)} entries.")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not load cluster map from {cluster_path}: {exc}")
            cluster_mapping = None

    if n_splits > 1:
        if cluster_mapping:
            logger.header(f"EXPERIMENT: {n_splits}-FOLD CV (CLUSTER-LEVEL SPLIT)")
            splits = get_cluster_based_splits(
                pairs_df_for_split, cluster_mapping, n_splits=n_splits, random_state=random_state
            )
        else:
            logger.header(f"EXPERIMENT: {n_splits}-FOLD CV (PROTEIN-LEVEL SPLIT - NO LEAKAGE)")
            splits = get_protein_based_splits(pairs_df_for_split, n_splits=n_splits, random_state=random_state)

        fold_metrics_list = []
        cv_results = []  # NEW: Collect y_true and y_proba for CV visualization

        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            logger.info(f"--- Fold {fold_idx + 1}/{n_splits} ---")
            X_train_fold, X_val_fold = X_df.iloc[train_indices], X_df.iloc[val_indices]
            y_train_fold, y_val_fold = y_s.iloc[train_indices], y_s.iloc[val_indices]

            # Strip Pandas abstraction entirely before fitting to unleash Scikit/Joblib 
            # automatic memory mapping (mmap) efficiency. This completely solves OOM.
            X_train_np = np.ascontiguousarray(X_train_fold.to_numpy(dtype=np.float32))
            X_val_np = np.ascontiguousarray(X_val_fold.to_numpy(dtype=np.float32))
            y_train_np = np.ascontiguousarray(y_train_fold.to_numpy(dtype=np.float32))
            
            model_pipeline = model_factory(n_jobs=n_jobs, feature_names=list(X_train_fold.columns))
            model_pipeline.fit(X_train_np, y_train_np)
            
            # Khởi tạo thư mục và lưu Model của Fold hiện tại
            import joblib
            models_dir = os.path.join(output_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f"model_fold{fold_idx + 1}.joblib")
            joblib.dump(model_pipeline, model_path)
            logger.info(f"Saved {model_path}")

            _log_selector_state(logger, model_pipeline, prefix=f"[Fold {fold_idx + 1}] ")

            y_proba_val = model_pipeline.predict_proba(X_val_np)[:, 1]
            y_pred_val = (y_proba_val >= 0.5).astype(int)
            
            # NEW: Collect results for CV visualization & OOF Thresholding
            cv_results.append({
                'fold_id': fold_idx + 1,
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

        # Print standard CV summary
        print_paper_style_results(fold_metrics_list)

        # Ensure output directories exist
        plots_dir = os.path.join(output_dir, "plots")
        models_dir = os.path.join(output_dir, "models")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        # =========================================================================
        # OOF DYNAMIC THRESHOLDING (F1-Optimized — Gold Master)
        # =========================================================================
        from sklearn.metrics import (
            precision_recall_curve, accuracy_score, precision_score,
            recall_score, f1_score, matthews_corrcoef,
        )
        all_y_true = np.concatenate([res['y_true'] for res in cv_results])
        all_y_proba = np.concatenate([res['y_proba'] for res in cv_results])
        all_fold_ids = np.concatenate(
            [np.full(len(res['y_true']), res['fold_id']) for res in cv_results]
        )

        precisions, recalls, thresholds = precision_recall_curve(all_y_true, all_y_proba)

        with np.errstate(divide='ignore', invalid='ignore'):
            oof_f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)

        optimal_idx = np.argmax(oof_f1_scores)
        optimal_threshold = thresholds[optimal_idx]

        y_pred_optimal = (all_y_proba >= optimal_threshold).astype(int)
        opt_acc  = accuracy_score(all_y_true, y_pred_optimal)
        opt_prec = precision_score(all_y_true, y_pred_optimal, zero_division=0)
        opt_rec  = recall_score(all_y_true, y_pred_optimal, zero_division=0)
        opt_f1   = f1_score(all_y_true, y_pred_optimal, zero_division=0)
        opt_mcc  = matthews_corrcoef(all_y_true, y_pred_optimal)

        logger.header("OOF DYNAMIC THRESHOLDING (F1-Optimized)")
        print(f"  Optimal F1 Threshold: {optimal_threshold:.4f}")
        print(f"  Accuracy:  {opt_acc*100:.2f}%")
        print(f"  Precision: {opt_prec*100:.2f}%")
        print(f"  Recall:    {opt_rec*100:.2f}%")
        print(f"  F1-Score:  {opt_f1*100:.2f}%")
        print(f"  MCC:       {opt_mcc*100:.2f}%\n")

        # --- Save OOF Predictions CSV ---
        try:
            oof_pred_df = pd.DataFrame({
                'fold_id': all_fold_ids.astype(int),
                'y_true': all_y_true.astype(int),
                'y_proba': all_y_proba,
                'y_pred_default': (all_y_proba >= 0.5).astype(int),
                'y_pred_optimal': y_pred_optimal,
            })
            oof_pred_path = os.path.join(output_dir, "oof_predictions.csv")
            oof_pred_df.to_csv(oof_pred_path, index=False)
            logger.info(f"Saved OOF predictions to {oof_pred_path}")
        except Exception as e:
            logger.warning(f"Could not save OOF predictions: {e}")

        # --- Save OOF Optimal Metrics CSV ---
        try:
            oof_metrics = {
                'Optimal Threshold': [optimal_threshold],
                'Accuracy': [opt_acc],
                'Precision': [opt_prec],
                'Recall': [opt_rec],
                'F1 Score': [opt_f1],
                'MCC': [opt_mcc],
            }
            oof_metrics_path = os.path.join(output_dir, "oof_optimal_metrics.csv")
            pd.DataFrame(oof_metrics).to_csv(oof_metrics_path, index=False)
            logger.info(f"Saved OOF optimal metrics to {oof_metrics_path}")
        except Exception as e:
            logger.warning(f"Could not save OOF optimal metrics: {e}")

        # --- Save raw fold metrics ---
        try:
            fold_df = pd.DataFrame(fold_metrics_list)
            fold_df.index = [f"Fold {i+1}" for i in range(len(fold_metrics_list))]
            fold_csv = os.path.join(output_dir, "fold_metrics.csv")
            fold_df.to_csv(fold_csv, index_label="Fold")
            logger.info(f"Saved raw fold metrics to {fold_csv}")
        except Exception as e:
            logger.warning(f"Could not save raw fold metrics: {e}")

        # =========================================================================
        # VISUALIZATION PHASE
        # =========================================================================
        logger.phase("Generating Cross-Validation Visualizations")
        try:
            cv_stats = plot_cv_roc_pr_curves(
                cv_results,
                save_dir=plots_dir,
                title=f'{n_splits}-Fold Cross-Validation',
            )
            logger.info(f"CV ROC-AUC: {cv_stats['mean_roc_auc']:.4f} ± {cv_stats['std_roc_auc']:.4f}")
            logger.info(f"CV PR-AUC:  {cv_stats['mean_pr_auc']:.4f} ± {cv_stats['std_pr_auc']:.4f}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not generate CV ROC/PR curves: {exc}")

        try:
            plot_cv_metric_distribution(
                fold_metrics_list,
                save_dir=plots_dir,
                title=f'{n_splits}-Fold CV',
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not generate CV metric distribution: {exc}")

        try:
            plot_f1_threshold_curve(
                thresholds, oof_f1_scores, optimal_idx,
                save_dir=plots_dir,
                title="OOF F1 Score vs Decision Threshold",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not generate F1 threshold curve: {exc}")

        try:
            plot_oof_confusion_matrix(
                all_y_true, y_pred_optimal,
                save_dir=plots_dir,
                threshold_label=optimal_threshold,
                title="OOF Confusion Matrix (Global)",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not generate OOF confusion matrix: {exc}")

        # --- Train and save FINAL model on 100% data ---
        logger.phase("Training Final Model on 100% Dataset")
        final_model = model_factory(n_jobs=n_jobs, feature_names=list(X_df.columns))
        X_df_np = np.ascontiguousarray(X_df.to_numpy(dtype=np.float32))
        y_s_np = np.ascontiguousarray(y_s.to_numpy(dtype=np.float32))
        final_model.fit(X_df_np, y_s_np)
        import joblib
        final_model_path = os.path.join(models_dir, "final_model.joblib")
        joblib.dump(final_model, final_model_path)
        logger.info(f"Saved {final_model_path}")

        return pd.DataFrame(fold_metrics_list).mean().to_dict()

    logger.warning("Running simple Train/Test split (Random). Be careful of Data Leakage!")
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_s, test_size=test_size, random_state=random_state, stratify=y_s
    )
    model_pipeline = model_factory(n_jobs=n_jobs)
    X_train_np = np.ascontiguousarray(X_train.to_numpy(dtype=np.float32))
    y_train_np = np.ascontiguousarray(y_train.to_numpy(dtype=np.float32))
    model_pipeline.fit(X_train_np, y_train_np)
    
    # Save train/test split model
    import joblib
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "train_test_model.joblib")
    joblib.dump(model_pipeline, model_path)
    logger.info(f"Saved {model_path}")

    _log_selector_state(logger, model_pipeline, prefix="[Train/Test] ")
    X_test_np = np.ascontiguousarray(X_test.to_numpy(dtype=np.float32))
    y_pred_test = model_pipeline.predict(X_test_np)
    y_proba_test = model_pipeline.predict_proba(X_test_np)[:, 1]
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
    pairing_strategy: str = "hadamard_abs",
    n_splits: int = 5,
    n_jobs: int = -1,
    output_dir: str = "results",
    cache_version: str = "v3",
    reuse_cached_ablations: bool = True,
    rerun_c_series: bool = False,
):
    """
    Run ablation study and summarize all cached/rerun variants.

    The reference A4 (Full Stacking with hadamard_abs) is assumed to already exist
    from the main reproduce_results.py run and will be loaded automatically.

    Args:
        pairing_strategy: Primary pairing used for ablation runs (default: hadamard_abs).
        output_dir: Root output directory (e.g. results/human). Ablation artifacts
                    will be saved under {output_dir}/ablation/{ablation_id}/.
        reuse_cached_ablations: Load existing fold_metrics.csv for A/B variants.
        rerun_c_series: Force rerun for C1/C2/C3 even when cached results exist.
    """
    logger = PipelineLogger()
    logger.header("🚀 STARTING ABLATION STUDY 🚀")
    logger.info(f"Primary pairing: {pairing_strategy}")

    all_results = []
    ablation_root = os.path.join(output_dir, "ablation")

    # --- Initialize FeatureEngine for column names ---
    logger.phase("Initializing FeatureEngine (for column names)")
    try:
        embedding_computer = EmbeddingComputer(model_name=esm_model_name)
        feature_engine = FeatureEngine(h5_cache_path=h5_cache_path, embedding_computer=embedding_computer)
        interp_cols_primary, embed_cols_primary = define_stacking_columns(feature_engine, pairing_strategy)
        interp_cols_concat, embed_cols_concat = define_stacking_columns(feature_engine, "concat")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Cannot initialize FeatureEngine: {exc}")
        return

    def _load_ablation_from_cache(ablation_id: str, label: str) -> bool:
        existing_csv = os.path.join(ablation_root, ablation_id, "fold_metrics.csv")
        if not os.path.exists(existing_csv):
            return False
        try:
            ref_df = pd.read_csv(existing_csv)
            if "Fold" in ref_df.columns:
                ref_df = ref_df.drop("Fold", axis=1)
            ref_mean = ref_df.mean().to_dict()
            ref_mean["Model"] = label
            ref_mean["Time (s)"] = "—"
            all_results.append(ref_mean)
            logger.info(f"✅ Loaded cached {label}")
            return True
        except Exception as exc:
            logger.warning(f"Could not load cached {label}: {exc}")
            return False

    # --- Helper: run one ablation and collect results ---
    def _run_one(ablation_id: str, label: str, model_factory, pairing: str, allow_cache: bool = False):
        if allow_cache and _load_ablation_from_cache(ablation_id, label):
            return
        abl_dir = os.path.join(ablation_root, ablation_id)
        os.makedirs(abl_dir, exist_ok=True)

        logger.phase(f"Running {label}")
        t0 = time.time()
        try:
            res = run_experiment(
                fasta_path,
                pairs_path,
                h5_cache_path,
                model_factory,
                n_splits=n_splits,
                n_jobs=n_jobs,
                esm_model_name=esm_model_name,
                pairing_strategy=pairing,
                cache_version=cache_version,
                output_dir=abl_dir,
            )
            elapsed = time.time() - t0
            res["Model"] = label
            res["Time (s)"] = f"{elapsed:.0f}"
            all_results.append(res)
            logger.info(f"✅ {label} completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
        except Exception as exc:
            elapsed = time.time() - t0
            logger.warning(f"❌ {label} FAILED after {elapsed:.0f}s: {exc}")
            import traceback
            traceback.print_exc()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if reuse_cached_ablations:
        logger.phase("Loading cached A/B ablations (if available)")
        for abl_id, abl_label in [
            ("A1_InterpOnly", "A1: Interp-Only (Handcraft+Motif)"),
            ("A2_EmbedOnly", "A2: Embed-Only (ESM-2+Local)"),
            ("B1_InterLR", "B1: Inter-LR (Raw feat + Linear)"),
            ("B2_EmbedLR", "B2: Embed-LR (Raw feat + Linear)"),
            ("B3_GlobalLR", "B3: Global-LR (Global ESM2 + Linear)"),
        ]:
            _load_ablation_from_cache(abl_id, abl_label)

    # =====================================================================
    # NEW ABLATIONS: C1 (Selector), C2 (Fusion), C3 (Meta-Learner)
    # =====================================================================
    _run_one("C1_NoSelector", "C1: No Selector (Embed-LGBM)", 
             lambda n_jobs=-1, feature_names=None, _ec=embed_cols_primary: create_embed_only_pipeline(
                 _ec, n_jobs, use_selector=False, feature_names=feature_names, estimator_type="lgbm"
             ), pairing=pairing_strategy, allow_cache=(not rerun_c_series))
    
    _run_one("C2_EarlyFusion", "C2: Early Fusion (Flat-Stacking)", 
             lambda n_jobs=-1, feature_names=None, _ic=interp_cols_primary, _ec=embed_cols_primary: create_early_fusion_pipeline(
                 _ic, _ec, n_jobs, feature_names=feature_names
             ), pairing=pairing_strategy, allow_cache=(not rerun_c_series))

    _run_one("C3_TreeMeta", "C3: Tree Meta-Learner (Stacking LGBM)", 
             lambda n_jobs=-1, feature_names=None, _ic=interp_cols_primary, _ec=embed_cols_primary: create_stacking_pipeline(
                 _ic, _ec, n_jobs, use_selector=True, cv_n_jobs=1, feature_names=feature_names, meta_learner_type="lgbm"
             ), pairing=pairing_strategy, allow_cache=(not rerun_c_series))

    # =====================================================================
    # A4 Reference: Load existing Full Stacking hadamard_abs results (if available)
    # =====================================================================
    ref_fold_csv = os.path.join(output_dir, "fold_metrics.csv")
    if os.path.exists(ref_fold_csv):
        logger.phase("Loading A4 Reference: Full Stacking (Hadamard+Abs) from main run")
        try:
            ref_df = pd.read_csv(ref_fold_csv)
            if "Fold" in ref_df.columns:
                ref_df = ref_df.drop("Fold", axis=1)
            ref_mean = ref_df.mean().to_dict()
            ref_mean["Model"] = "A4: Full Stacking (Symmetric) [ref]"
            ref_mean["Time (s)"] = "—"
            all_results.append(ref_mean)
            logger.info("✅ Loaded reference A4 from existing fold_metrics.csv")
        except Exception as exc:
            logger.warning(f"Could not load reference metrics: {exc}")
    else:
        logger.warning(f"Reference fold_metrics.csv not found at {ref_fold_csv}. A4 reference skipped.")

    # =====================================================================
    # Final Summary
    # =====================================================================
    logger.header("📊 ABLATION STUDY FINAL RESULTS 📊")
    results_df = pd.DataFrame(all_results)
    results_df = results_df.set_index("Model")

    cols_order = [
        "Accuracy",
        "Precision",
        "Recall (Sensitivity)",
        "F1 Score",
        "Specificity",
        "MCC",
        "ROC-AUC",
        "PR-AUC",
        "Time (s)",
    ]
    cols_to_show = [col for col in cols_order if col in results_df.columns]

    # Format as percentage for display
    display_df = results_df[cols_to_show].copy()
    metric_cols = [c for c in cols_to_show if c != "Time (s)"]
    for c in metric_cols:
        display_df[c] = display_df[c].apply(
            lambda x: f"{x*100:.2f}" if isinstance(x, (int, float)) else x
        )

    print(display_df.to_string())

    # Save to CSV
    summary_csv = os.path.join(ablation_root, "ablation_summary.csv")
    os.makedirs(ablation_root, exist_ok=True)
    display_df.to_csv(summary_csv)
    logger.info(f"💾 Saved ablation summary to {summary_csv}")

    # LaTeX output
    print("\n📄 LaTeX Table Rows:")
    print("-" * 100)
    for model_name, row in display_df.iterrows():
        parts = [str(model_name)]
        for c in ["Accuracy", "F1 Score", "MCC", "ROC-AUC", "PR-AUC"]:
            parts.append(str(row.get(c, "—")))
        print("  " + " & ".join(parts) + " \\\\")
    print("-" * 100)

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
    cv_n_jobs: int = 1,
    feature_names: list[str] | None = None
):
    """
    Convenience helper: build stacking pipeline with columns derived from FeatureEngine.
    Mirrors the notebook setup to avoid duplicated boilerplate.
    """
    embedding_computer = EmbeddingComputer(model_name=esm_model_name)
    feature_engine = FeatureEngine(h5_cache_path=h5_cache_path, embedding_computer=embedding_computer)
    interp_cols, embed_cols = define_stacking_columns(feature_engine, pairing_strategy=pairing_strategy)
    return create_stacking_pipeline(
        interp_cols=interp_cols, 
        embed_cols=embed_cols, 
        n_jobs=n_jobs, 
        use_selector=True,
        cv_n_jobs=cv_n_jobs,
        feature_names=feature_names
    )
