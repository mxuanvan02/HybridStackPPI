from typing import List
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.feature_engine import FeatureEngine
from src.selectors import CumulativeFeatureSelector

__all__ = [
    "create_stacking_pipeline",
    "create_embed_only_stacking_pipeline",
    "create_interp_only_stacking_pipeline",
    "create_embed_only_pipeline",
    "create_interp_only_pipeline",
    "create_lgbm_pipeline",
    "create_svm_pipeline",
    "create_esm_lr_pipeline",
    "create_esm_global_lr_pipeline",
    "create_interp_lr_pipeline",
    "create_embed_lr_pipeline",
    "create_esm_lgbm_raw_pipeline",
    "create_esm_lgbm_selector_pipeline",
    "define_stacking_columns",
]

INTERP_Q = 0.97
INTERP_CORR = 0.95
INTERP_VAR = 0.0

EMBED_Q = 0.92
EMBED_CORR = 0.85
EMBED_VAR = 0.01


def _create_lgbm_branch(
    columns: List[str],
    use_selector: bool,
    selector_params: dict,
    lgbm_params: dict,
    use_scaler: bool = False,
) -> Pipeline:
    """Creates a standardized LightGBM pipeline branch."""
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    if use_selector:
        steps.append(("selector", CumulativeFeatureSelector(**selector_params)))
    else:
        steps.append(("passthrough", "passthrough"))

    return Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [("trans", Pipeline(steps), columns)], remainder="drop"
                ),
            ),
            ("model", LGBMClassifier(**lgbm_params)),
        ]
    )


def create_stacking_pipeline(
    interp_cols: List[str], embed_cols: List[str], n_jobs: int = 1, use_selector: bool = True
) -> Pipeline:
    """Creates the main HybridStack-PPI pipeline."""
    lgbm_defaults = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 20,
        "n_jobs": n_jobs,
        "class_weight": "balanced",
        "verbose": -1,
    }

    interp_base = _create_lgbm_branch(
        columns=interp_cols,
        use_selector=use_selector,
        selector_params={
            "importance_quantile": INTERP_Q,
            "corr_threshold": INTERP_CORR,
            "variance_threshold": INTERP_VAR,
            "verbose": True,
        },
        lgbm_params=lgbm_defaults | {"random_state": 42},
        use_scaler=False,
    )

    embed_base = _create_lgbm_branch(
        columns=embed_cols,
        use_selector=use_selector,
        selector_params={
            "importance_quantile": EMBED_Q,
            "corr_threshold": EMBED_CORR,
            "variance_threshold": EMBED_VAR,
            "verbose": True,
        },
        lgbm_params=lgbm_defaults | {"random_state": 123},
        use_scaler=True,
    )

    stacking = StackingClassifier(
        estimators=[("interp_branch", interp_base), ("embed_branch", embed_base)],
        final_estimator=LogisticRegression(random_state=42, class_weight="balanced"),
        cv=5,
        n_jobs=n_jobs,
    )

    return Pipeline([("ensemble", stacking)])


def create_embed_only_stacking_pipeline(
    embed_cols: List[str], n_jobs: int = 1, use_selector: bool = True
) -> Pipeline:
    """Creates a stacking pipeline with only embedding features."""
    lgbm_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 20,
        "n_jobs": n_jobs,
        "class_weight": "balanced",
        "verbose": -1,
    }
    selector_params = {
        "variance_threshold": EMBED_VAR,
        "importance_quantile": EMBED_Q,
        "corr_threshold": EMBED_CORR,
        "verbose": False,
    }

    estimators = []
    for seed in [42, 123]:
        branch = _create_lgbm_branch(
            columns=embed_cols,
            use_selector=use_selector,
            selector_params=selector_params,
            lgbm_params=lgbm_params | {"random_state": seed},
            use_scaler=True,
        )
        estimators.append((f"lgbm_s{seed}", branch))

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=42, class_weight="balanced"),
        cv=5,
        n_jobs=n_jobs,
    )
    return Pipeline([("ensemble", stacking)])


def create_interp_only_stacking_pipeline(
    interp_cols: List[str], n_jobs: int = 1, use_selector: bool = True
) -> Pipeline:
    """Creates a stacking pipeline with only interpretable features."""
    lgbm_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 20,
        "n_jobs": n_jobs,
        "class_weight": "balanced",
        "verbose": -1,
    }
    selector_params = {
        "variance_threshold": INTERP_VAR,
        "importance_quantile": INTERP_Q,
        "corr_threshold": INTERP_CORR,
        "verbose": False,
    }

    estimators = []
    for seed in [42, 123]:
        branch = _create_lgbm_branch(
            columns=interp_cols,
            use_selector=use_selector,
            selector_params=selector_params,
            lgbm_params=lgbm_params | {"random_state": seed},
            use_scaler=False,
        )
        estimators.append((f"lgbm_s{seed}", branch))

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=42, class_weight="balanced"),
        cv=5,
        n_jobs=n_jobs,
    )
    return Pipeline([("ensemble", stacking)])


def create_embed_only_pipeline(
    full_cols: List[str], n_jobs: int = 1, use_selector: bool = True
) -> Pipeline:
    """Creates a single-learner pipeline with embedding features."""
    steps = [("scaler", StandardScaler())]
    if use_selector:
        steps.append(
            (
                "selector",
                CumulativeFeatureSelector(
                    variance_threshold=EMBED_VAR,
                    importance_quantile=EMBED_Q,
                    corr_threshold=EMBED_CORR,
                    verbose=True,
                ),
            )
        )

    pre = ColumnTransformer([("trans", Pipeline(steps), full_cols)], remainder="drop")
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=20,
        random_state=42,
        n_jobs=n_jobs,
        class_weight="balanced",
        verbose=-1,
    )
    return Pipeline([("pre", pre), ("model", model)])


def create_interp_only_pipeline(
    interp_cols: List[str], n_jobs: int = 1, use_selector: bool = True
) -> Pipeline:
    """Creates a single-learner pipeline with interpretable features."""
    steps = []
    if use_selector:
        steps.append(
            (
                "selector",
                CumulativeFeatureSelector(
                    variance_threshold=INTERP_VAR,
                    importance_quantile=INTERP_Q,
                    corr_threshold=INTERP_CORR,
                    verbose=True,
                ),
            )
        )
    else:
        steps.append(("passthrough", "passthrough"))

    pre = ColumnTransformer([("trans", Pipeline(steps), interp_cols)], remainder="drop")
    return Pipeline(
        [
            ("pre", pre),
            (
                "model",
                LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    num_leaves=20,
                    random_state=42,
                    n_jobs=n_jobs,
                    class_weight="balanced",
                    verbose=-1,
                ),
            ),
        ]
    )


def create_lgbm_pipeline(n_jobs: int = 1, use_selector: bool = True) -> Pipeline:
    """Creates a generic LightGBM pipeline."""
    steps = [("scaler", StandardScaler())]
    if use_selector:
        steps.append(
            (
                "selector",
                CumulativeFeatureSelector(
                    variance_threshold=EMBED_VAR,
                    importance_quantile=EMBED_Q,
                    corr_threshold=EMBED_CORR,
                ),
            )
        )
    return Pipeline(
        steps
        + [
            (
                "model",
                LGBMClassifier(
                    n_estimators=500,
                    n_jobs=n_jobs,
                    random_state=42,
                    verbose=-1,
                    class_weight="balanced",
                ),
            )
        ]
    )


def create_svm_pipeline(n_jobs: int = 1) -> Pipeline:
    """Creates a generic SVM pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="rbf",
                    probability=True,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _create_baseline_pipeline(
    columns: List[str], n_jobs: int = 1, use_scaler: bool = True
) -> Pipeline:
    """Creates a baseline logistic regression pipeline."""
    steps = [("scaler", StandardScaler())] if use_scaler else []
    pre = ColumnTransformer([("trans", Pipeline(steps), columns)], remainder="drop")
    return Pipeline(
        [
            ("pre", pre),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


def create_esm_lr_pipeline(n_jobs: int = 1) -> Pipeline:
    """Creates an ESM logistic regression pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000, random_state=42, class_weight="balanced"
                ),
            ),
        ]
    )


def create_esm_global_lr_pipeline(embed_cols: List[str], n_jobs: int = 1) -> Pipeline:
    """Creates an ESM global logistic regression pipeline."""
    return _create_baseline_pipeline(embed_cols, n_jobs=n_jobs)


def create_interp_lr_pipeline(interp_cols: List[str], n_jobs: int = 1) -> Pipeline:
    """Creates an interpretable logistic regression pipeline."""
    return _create_baseline_pipeline(interp_cols, n_jobs=n_jobs)


def create_embed_lr_pipeline(embed_cols: List[str], n_jobs: int = 1) -> Pipeline:
    """Creates an embedding logistic regression pipeline."""
    return _create_baseline_pipeline(embed_cols, n_jobs=n_jobs)


def create_esm_lgbm_raw_pipeline(n_jobs: int = 1) -> Pipeline:
    """Creates an ESM LightGBM pipeline without feature selection."""
    return create_lgbm_pipeline(n_jobs, False)


def create_esm_lgbm_selector_pipeline(n_jobs: int = 1) -> Pipeline:
    """Creates an ESM LightGBM pipeline with feature selection."""
    return create_lgbm_pipeline(n_jobs, True)


def define_stacking_columns(
    feature_engine: FeatureEngine, pairing_strategy: str = "concat"
) -> tuple[List[str], List[str]]:
    """Defines the columns for the stacking pipeline."""
    h, m = (
        feature_engine.handcraft_extractor.get_feature_names(),
        feature_engine.motif_names,
    )
    g, l = feature_engine.global_emb_names, feature_engine.local_emb_names
    i_names, e_names = h + m, g + l
    p1, p2 = ("P1_", "P2_") if pairing_strategy == "concat" else ("Avg_", "Diff_")
    ic = [f"{p1}{n}" for n in i_names] + [f"{p2}{n}" for n in i_names]
    ec = [f"{p1}{n}" for n in e_names] + [f"{p2}{n}" for n in e_names]
    return ic, ec
