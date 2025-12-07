import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold


class CumulativeFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Three-stage selector: variance -> importance -> correlation.
    Stages can be toggled for ablation.
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        importance_quantile: float = 0.95,
        corr_threshold: float = 0.97,
        use_variance: bool = True,
        use_importance: bool = True,
        use_corr: bool = True,
        importance_estimator=None,  # NEW: Allow custom estimator
        verbose: bool = True,
    ):
        """
        Initialize the selector.
        
        Args:
            variance_threshold: Minimum variance to keep feature (Stage 1)
            importance_quantile: Cumulative importance quantile (Stage 2)
            corr_threshold: Max correlation threshold (Stage 3)
            use_variance: Whether to apply variance filter
            use_importance: Whether to apply importance filter
            use_corr: Whether to apply correlation filter
            importance_estimator: Custom estimator for importance (default: LGBMClassifier)
            verbose: Whether to print progress
        """
        self.variance_threshold = variance_threshold
        self.importance_quantile = importance_quantile
        self.corr_threshold = corr_threshold
        self.use_variance = use_variance
        self.use_importance = use_importance
        self.use_corr = use_corr
        self.importance_estimator = importance_estimator  # Store the custom estimator
        self.verbose = verbose
        self.estimator_ = None
        self.selected_features_ = None
        self.selected_features_stage1_ = None
        self.selected_features_stage2_ = None
        self._feature_names_in: list[str] | None = None

    def _check_X(self, X) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            generic_cols = [f"f_{i}" for i in range(X.shape[1])]
            self._feature_names_in = generic_cols
            return pd.DataFrame(X, columns=generic_cols)
        if isinstance(X, pd.DataFrame):
            self._feature_names_in = X.columns.tolist()
            return X
        raise TypeError(f"Unsupported input type: {type(X)}")

    def _compute_max_correlation_with_selected(
        self, X_df: pd.DataFrame, feature: str, selected_features: list[str]
    ) -> float:
        """
        Compute the maximum absolute correlation between a single feature
        and a list of already-selected features.
        
        This is the core optimization: instead of computing the full N×N correlation matrix,
        we only compute correlations for the current feature against selected features.
        
        Args:
            X_df: DataFrame containing all features
            feature: Name of the feature to check
            selected_features: List of already-selected feature names
            
        Returns:
            Maximum absolute correlation value
        """
        if not selected_features:
            return 0.0
        
        # Extract feature column and selected columns
        feature_col = X_df[feature].values
        selected_cols = X_df[selected_features].values
        
        # Compute correlation with each selected feature
        # Using numpy for performance: corr(x,y) = cov(x,y) / (std(x) * std(y))
        correlations = []
        for i in range(selected_cols.shape[1]):
            corr = np.corrcoef(feature_col, selected_cols[:, i])[0, 1]
            correlations.append(abs(corr))
        
        return max(correlations) if correlations else 0.0

    def fit(self, X, y: pd.Series, **fit_params):
        X_df = self._check_X(X)
        initial_count = X_df.shape[1]

        if self.verbose:
            print(f"\n🔥 Starting Cumulative Feature Selection (Initial: {initial_count})")
            print(
                f"   [Config] use_variance={self.use_variance}, "
                f"use_importance={self.use_importance}, use_corr={self.use_corr}"
            )

        # Stage 1: Variance
        if self.use_variance:
            selector_var = VarianceThreshold(threshold=self.variance_threshold)
            selector_var.fit(X_df)
            X_stage1 = X_df.loc[:, selector_var.get_support()]
            if self.verbose:
                print(f"  - Stage 1 (Variance, thresh={self.variance_threshold}): "
                      f"{initial_count} -> {X_stage1.shape[1]} features")
        else:
            X_stage1 = X_df
            if self.verbose:
                print("  - Stage 1 (Variance): SKIPPED (use_variance=False)")

        if X_stage1.shape[1] == 0:
            print("  [Selector] Warning: no features left after Stage 1.")
            self.selected_features_ = []
            return self

        # Stage 2: Cumulative Importance
        if self.use_importance:
            if self.verbose:
                print(f"  - Stage 2 (Cumulative Importance): Pre-selecting from {X_stage1.shape[1]} features...")
            
            # Use custom estimator if provided, otherwise default to LGBM
            if self.importance_estimator is not None:
                self.estimator_ = clone(self.importance_estimator)
            else:
                self.estimator_ = LGBMClassifier(n_jobs=-1, random_state=42, verbose=-1)
            
            self.estimator_.fit(X_stage1, y)
            importances = pd.Series(self.estimator_.feature_importances_, index=X_stage1.columns)
            importances = importances.sort_values(ascending=False)
            
            if importances.empty:
                print("  [Selector] Warning: No features with importance > 0. Keeping the best one.")
                self.selected_features_stage2_ = X_stage1.columns[:1].tolist()
                X_stage2 = X_stage1[self.selected_features_stage2_]
            else:
                cumulative_importances = importances.cumsum()
                total_importance = cumulative_importances.iloc[-1]
                cutoff_value = total_importance * self.importance_quantile
                selected_count = (cumulative_importances <= cutoff_value).sum() + 1
                final_k = max(1, min(selected_count, len(importances)))
                
                self.selected_features_stage2_ = importances.head(final_k).index.tolist()
                X_stage2 = X_stage1[self.selected_features_stage2_]
            
            if self.verbose:
                print(f"  - Stage 2 (Cumulative Importance, q={self.importance_quantile}): "
                      f"{X_stage1.shape[1]} -> {X_stage2.shape[1]} features")
        else:
            X_stage2 = X_stage1
            self.selected_features_stage2_ = X_stage1.columns.tolist() # Store all features from Stage 1
            if self.verbose:
                print("  - Stage 2 (Cumulative Importance): SKIPPED (use_importance=False)")

        if X_stage2.shape[1] == 0:
            print("  [Selector] Warning: no features left after Stage 2.")
            self.selected_features_ = []
            return self

        # Stage 3: Correlation (Greedy Filter - Optimized for Large Feature Sets)
        if self.use_corr:
            if self.verbose:
                print(f"  - Stage 3 (Greedy Correlation Filter, thresh={self.corr_threshold}): "
                      f"Cleaning up {X_stage2.shape[1]} features...")

            # If we already have importance ranking from Stage 2, use it
            # Otherwise, use feature order as-is
            if self.use_importance and hasattr(self, 'estimator_') and hasattr(self.estimator_, 'feature_importances_'):
                # FIX: estimator_ was fitted on X_stage1, so importances match X_stage1 columns.
                # We must filter these importances to only include features that survived to X_stage2.
                all_importances = pd.Series(self.estimator_.feature_importances_, index=X_stage1.columns)
                stage2_importances = all_importances.loc[X_stage2.columns]
                feature_order = stage2_importances.sort_values(ascending=False).index.tolist()
            else:
                feature_order = X_stage2.columns.tolist()

            # Greedy selection: keep feature if max correlation with selected < threshold
            selected_features = []
            for feature in feature_order:
                if len(selected_features) == 0:
                    # Always keep the first (most important) feature
                    selected_features.append(feature)
                else:
                    # Compute correlation only with already-selected features
                    max_corr = self._compute_max_correlation_with_selected(
                        X_stage2, feature, selected_features
                    )
                    if max_corr < self.corr_threshold:
                        selected_features.append(feature)
                    elif self.verbose and len(selected_features) < 10:  # Log first few drops
                        pass  # Suppress individual drop messages for cleaner output

            X_final = X_stage2[selected_features]
            n_dropped = X_stage2.shape[1] - len(selected_features)

            if self.verbose:
                print(
                    f"  ✅ [Selector] Completed. Final count: {initial_count} -> {X_final.shape[1]} "
                    f"(dropped {n_dropped} correlated)"
                )
        else:
            X_final = X_stage2
            if self.verbose:
                print("  - Stage 3 (Correlation Filter): SKIPPED (use_corr=False)")
                print(
                    f"  ✅ [Selector] Completed. Final count: {initial_count} -> {X_final.shape[1]} "
                    "(no correlation pruning)"
                )

        self.selected_features_ = X_final.columns.tolist()
        return self

    def transform(self, X) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise RuntimeError("Selector has not been fitted yet.")

        X_df = self._check_X(X)

        if not self.selected_features_:
            return pd.DataFrame(index=X_df.index)

        missing_cols = set(self.selected_features_) - set(X_df.columns)
        if missing_cols:
            print(f"Warning: {len(missing_cols)} selected columns missing in new data. They will be ignored.")
            cols_to_keep = [col for col in self.selected_features_ if col in X_df.columns]
            return X_df[cols_to_keep]

        return X_df[self.selected_features_]

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)
