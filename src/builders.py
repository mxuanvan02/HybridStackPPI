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

# =============================================================================
# HIGH-FIDELITY ELITE PARAMETERS - AGREED VERSION
# =============================================================================
# Interp Branch: Bio-features need more permissive thresholds and NO scaling
INTERP_Q = 0.97
INTERP_CORR = 0.95
INTERP_VAR = 0.0

# Embed Branch: Embedding vectors (high-dim) need strict Elite selection + Scaling
EMBED_Q = 0.92
EMBED_CORR = 0.85
EMBED_VAR = 0.01

def create_stacking_pipeline(
    interp_cols: List[str], 
    embed_cols: List[str], 
    n_jobs: int = 1, 
    use_selector: bool = True
) -> Pipeline:
    """
    Proposed HybridStack-PPI (V0).
    Branch 1: Interpretable (Raw Signals, permissive selection)
    Branch 2: Embedding (Normalized, strict Elite selection)
    """
    
    # 1. Interp Branch
    interp_steps = []
    if use_selector:
        interp_steps.append(("selector", CumulativeFeatureSelector(
            importance_quantile=INTERP_Q, 
            corr_threshold=INTERP_CORR, 
            variance_threshold=INTERP_VAR, 
            verbose=True
        )))
    else:
        interp_steps.append(("passthrough", "passthrough"))
    
    interp_base = Pipeline([
        ("pre", ColumnTransformer([("trans", Pipeline(interp_steps), interp_cols)], remainder="drop")),
        ("model", LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=20, random_state=42, n_jobs=n_jobs, class_weight="balanced", verbose=-1))
    ])

    # 2. Embed Branch
    embed_steps = [("scaler", StandardScaler())]
    if use_selector:
        embed_steps.append(("selector", CumulativeFeatureSelector(
            importance_quantile=EMBED_Q, 
            corr_threshold=EMBED_CORR, 
            variance_threshold=EMBED_VAR, 
            verbose=True
        )))
    
    embed_branch = Pipeline(embed_steps)
    embed_base = Pipeline([
        ("pre", ColumnTransformer([("trans", Pipeline(embed_steps), embed_cols)], remainder="drop")),
        ("model", LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=20, random_state=123, n_jobs=n_jobs, class_weight="balanced", verbose=-1))
    ])

    stacking = StackingClassifier(
        estimators=[("interp_branch", interp_base), ("embed_branch", embed_base)],
        final_estimator=LogisticRegression(random_state=42, class_weight="balanced"),
        cv=3,
        n_jobs=n_jobs
    )
    
    return Pipeline([("ensemble", stacking)])

def create_embed_only_stacking_pipeline(embed_cols: List[str], n_jobs: int = 1, use_selector: bool = True) -> Pipeline:
    """Variants V1, V2, V5: Uses the Embedding branch Elite strategy across different seeds."""
    def _make_branch(seed):
        steps = [("scaler", StandardScaler())]
        if use_selector:
            steps.append(("selector", CumulativeFeatureSelector(variance_threshold=EMBED_VAR, importance_quantile=EMBED_Q, corr_threshold=EMBED_CORR, verbose=False)))
        
        return Pipeline([
            ("pre", ColumnTransformer([("trans", Pipeline(steps), embed_cols)], remainder="drop")),
            ("model", LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=20, random_state=seed, n_jobs=n_jobs, class_weight="balanced", verbose=-1))
        ])

    stacking = StackingClassifier(
        estimators=[("lgbm_s1", _make_branch(42)), ("lgbm_s2", _make_branch(123))],
        final_estimator=LogisticRegression(random_state=42, class_weight="balanced"),
        cv=3,
        n_jobs=n_jobs
    )
    return Pipeline([("ensemble", stacking)])

def create_embed_only_pipeline(full_cols: List[str], n_jobs: int = 1, use_selector: bool = True) -> Pipeline:
    """V4 Variant: Single Learner fallback (uses global scaling for safety when hybrid)."""
    steps = [("scaler", StandardScaler())]
    if use_selector:
        # Use a balanced mix for single learner: strict on embeddings/general
        steps.append(("selector", CumulativeFeatureSelector(variance_threshold=EMBED_VAR, importance_quantile=EMBED_Q, corr_threshold=EMBED_CORR, verbose=True)))
    
    pre = ColumnTransformer([("trans", Pipeline(steps), full_cols)], remainder="drop")
    model = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=20, random_state=42, n_jobs=n_jobs, class_weight="balanced", verbose=-1)
    return Pipeline([("pre", pre), ("model", model)])

# --- Baselines & Compatibility ---

def create_lgbm_pipeline(n_jobs: int = 1, use_selector: bool = True) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if use_selector:
        steps.append(("selector", CumulativeFeatureSelector(variance_threshold=EMBED_VAR, importance_quantile=EMBED_Q, corr_threshold=EMBED_CORR)))
    return Pipeline(steps + [("model", LGBMClassifier(n_estimators=500, n_jobs=n_jobs, random_state=42, verbose=-1, class_weight="balanced"))])

def create_svm_pipeline(n_jobs: int = 1) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"))
    ])

def create_esm_lr_pipeline(n_jobs: int = 1) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"))])

def create_esm_global_lr_pipeline(embed_cols: List[str], n_jobs: int = 1) -> Pipeline:
    """ESM-Global embeddings → Logistic Regression (simplest baseline)."""
    steps = [("scaler", StandardScaler())]
    pre = ColumnTransformer([("trans", Pipeline(steps), embed_cols)], remainder="drop")
    return Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced", n_jobs=n_jobs))])

def create_interp_lr_pipeline(interp_cols: List[str], n_jobs: int = 1) -> Pipeline:
    """Interpretable features → Logistic Regression (baseline)."""
    steps = [("scaler", StandardScaler())]
    pre = ColumnTransformer([("trans", Pipeline(steps), interp_cols)], remainder="drop")
    return Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced", n_jobs=n_jobs))])

def create_embed_lr_pipeline(embed_cols: List[str], n_jobs: int = 1) -> Pipeline:
    """Full Embedding (Global+Local) → Logistic Regression (baseline)."""
    steps = [("scaler", StandardScaler())]
    pre = ColumnTransformer([("trans", Pipeline(steps), embed_cols)], remainder="drop")
    return Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced", n_jobs=n_jobs))])

def create_esm_lgbm_raw_pipeline(n_jobs: int = 1) -> Pipeline: return create_lgbm_pipeline(n_jobs, False)
def create_esm_lgbm_selector_pipeline(n_jobs: int = 1) -> Pipeline: return create_lgbm_pipeline(n_jobs, True)

def create_interp_only_pipeline(interp_cols: List[str], n_jobs: int = 1, use_selector: bool = True) -> Pipeline:
    steps = []
    if use_selector:
        steps.append(("selector", CumulativeFeatureSelector(variance_threshold=INTERP_VAR, importance_quantile=INTERP_Q, corr_threshold=INTERP_CORR, verbose=True)))
    else:
        steps.append(("passthrough", "passthrough"))
    pre = ColumnTransformer([("trans", Pipeline(steps), interp_cols)], remainder="drop")
    return Pipeline([("pre", pre), ("model", LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=20, random_state=42, n_jobs=n_jobs, class_weight="balanced", verbose=-1))])

def define_stacking_columns(feature_engine: FeatureEngine, pairing_strategy: str = "concat") -> tuple[List[str], List[str]]:
    h, m = feature_engine.handcraft_extractor.get_feature_names(), feature_engine.motif_names
    g, l = feature_engine.global_emb_names, feature_engine.local_emb_names
    i_names, e_names = h + m, g + l
    p1, p2 = ("P1_", "P2_") if pairing_strategy == "concat" else ("Avg_", "Diff_")
    ic = [f"{p1}{n}" for n in i_names] + [f"{p2}{n}" for n in i_names]
    ec = [f"{p1}{n}" for n in e_names] + [f"{p2}{n}" for n in e_names]
    return ic, ec

def create_esm_only_stacking_pipeline(embed_cols: List[str], n_jobs: int = 1, use_selector: bool = True) -> Pipeline:
    return create_embed_only_stacking_pipeline(embed_cols, n_jobs, use_selector)
def create_interp_only_stacking_pipeline(interp_cols: List[str], n_jobs: int = 1, use_selector: bool = True) -> Pipeline:
    def _make_branch(seed):
        steps = []
        if use_selector: steps.append(("selector", CumulativeFeatureSelector(variance_threshold=INTERP_VAR, importance_quantile=INTERP_Q, corr_threshold=INTERP_CORR, verbose=False)))
        else: steps.append(("passthrough", "passthrough"))
        return Pipeline([
            ("pre", ColumnTransformer([("trans", Pipeline(steps), interp_cols)], remainder="drop")),
            ("model", LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=20, random_state=seed, n_jobs=n_jobs, class_weight="balanced", verbose=-1))
        ])
    return Pipeline([("ensemble", StackingClassifier(estimators=[("s1", _make_branch(42)), ("s2", _make_branch(123))], final_estimator=LogisticRegression(), cv=3, n_jobs=n_jobs))])
