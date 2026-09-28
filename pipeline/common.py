"""
common.py — shared configuration and utilities for the delirium pipeline.

Everything the numbered stages need in common lives here:
  • repo-relative paths (portable across machines and OSes — no hardcoded paths)
  • the reproducibility seed and the final model hyperparameters (one source of truth)
  • the MICE imputation wrapper used inside the preprocessing pipelines
"""

from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (enables the import below)
from sklearn.impute import IterativeImputer as SklearnIterativeImputer


# ─────────────────────────────────────────────────────────────────────────────
# Paths — resolved relative to the repository, so the pipeline runs anywhere.
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"      # all generated figures/tables/models (git-ignored)
REPORT_DIR = REPO_ROOT / "reports"      # the human-readable result docs (tracked)

OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

CLEANED_CSV = DATA_DIR / "cleaned.csv"                 # de-identified model input (tracked)
BRONZE_CSV = DATA_DIR / "source_file_bronze.csv"       # pre-de-id join (local only, git-ignored)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility + final hyperparameters.
# RF_BEST_PARAMS / LR_BEST_PARAMS are the values selected by the nested CV in
# stage 01 and reused by stages 02–10. Defined once here so they never drift.
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42

RF_BEST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
}

LR_BEST_PARAMS = {
    "C": 1.0,               # most frequent across folds (6/10) -- current CV run
    "penalty": "l1",
    "solver": "liblinear",
    "max_iter": 500,
    "random_state": RANDOM_STATE,
}


# ─────────────────────────────────────────────────────────────────────────────
# MICE imputation wrapper (numeric features).
# ─────────────────────────────────────────────────────────────────────────────
class FancyIterativeImputerWrapper(BaseEstimator, TransformerMixin):
    """scikit-learn transformer wrapping IterativeImputer (MICE-like) for numeric data.

    Optionally uses the OUTCOME (y) as an auxiliary variable in the imputation
    model. When ``use_outcome=True`` and ``y`` is supplied at ``fit`` time, the
    outcome is appended as an extra column so the chained-equation regressors
    can learn predictor<->outcome relationships. To avoid outcome leakage into
    the imputed predictors, the outcome is treated as UNKNOWN (NaN) at
    ``transform`` time -- i.e. the true outcome of the rows being imputed is
    never used; only the outcome-informed imputation model (learned at fit) is
    applied. The outcome column is dropped from the output, so the transformed
    shape always equals the number of predictor columns.

    ``n_burn_in`` (default 0): extra chained-equation cycles run BEFORE the
    ``max_iterations`` that produce the retained imputation, i.e. the imputer
    is fit with ``max_iter = n_burn_in + max_iterations`` total cycles.
    scikit-learn's IterativeImputer only ever returns the state after its
    final cycle (there is no native concept of discarding early draws), so
    this is the correct/only way to map "burn-in" onto this algorithm: it
    gives the round-robin regression chain additional cycles to stabilise
    before the recorded values are taken, which matters most when
    ``sample_posterior=True`` (each cycle then involves a stochastic draw).
    """

    def __init__(self, max_iterations=10, random_state=42,
                 use_outcome=True, sample_posterior=False, n_burn_in=0):
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.use_outcome = use_outcome
        self.sample_posterior = sample_posterior
        self.n_burn_in = n_burn_in

    def fit(self, X, y=None):
        X_array = np.array(X, dtype=np.float64)
        self.n_predictors_ = X_array.shape[1]
        self.uses_outcome_ = bool(self.use_outcome) and (y is not None)
        if self.uses_outcome_:
            y_col = np.asarray(y, dtype=np.float64).reshape(-1, 1)
            X_array = np.hstack([X_array, y_col])   # outcome is the last column

        self.imputer_ = SklearnIterativeImputer(
            max_iter=self.n_burn_in + self.max_iterations,
            random_state=self.random_state,
            sample_posterior=self.sample_posterior,
        )
        self.imputer_.fit(X_array)
        return self

    def transform(self, X, y=None):
        X_array = np.array(X, dtype=np.float64)
        if self.uses_outcome_:
            # Outcome is UNKNOWN at transform time (avoids leakage): append an
            # all-NaN outcome column so the imputer applies the outcome-informed
            # model without using the true outcome of these rows.
            nan_col = np.full((X_array.shape[0], 1), np.nan)
            X_array = np.hstack([X_array, nan_col])
            X_trans = self.imputer_.transform(X_array)
            return X_trans[:, : self.n_predictors_]     # drop the outcome column
        return self.imputer_.transform(X_array)

    def get_feature_names_out(self, input_features=None):
        # MICE does not change the columns, so pass the names straight through.
        if input_features is not None:
            return np.asarray(input_features)
        return np.asarray([f"x{i}" for i in range(self.n_predictors_)])
