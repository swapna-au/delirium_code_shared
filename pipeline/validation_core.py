"""Shared methods for clinical prediction model development and validation.

The functions in this module deliberately separate:
  * fitting the complete development algorithm;
  * evaluating a fitted algorithm; and
  * Harrell bootstrap optimism correction.

All data-dependent preprocessing is fitted only in the model-training portion
of a development sample. The fitted transform is then applied unchanged to the
calibration portion and to evaluation data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.special import expit

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import FancyIterativeImputerWrapper


EPS = 1e-7


@dataclass
class FittedDevelopmentAlgorithm:
    """One fitted version of the prespecified model-development algorithm."""

    preprocessor: Pipeline
    uncalibrated: Mapping[str, object]
    calibrated: Mapping[str, object]


def _safe_model_params(params: Mapping, seed: int) -> dict:
    """Avoid nested CPU oversubscription when bootstrap iterations are parallel."""
    out = dict(params)
    out.setdefault("random_state", seed)
    if "n_jobs" in out:
        out["n_jobs"] = 1
    return out


def build_preprocessor(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    seed: int,
    *,
    repeat_feature_selection: bool = True,
) -> Pipeline:
    """Build the preprocessing/selection algorithm used in every resample.

    The existing two-stage selector is retained to preserve the prespecified
    development algorithm. It must be repeated during bootstrap validation
    because it is outcome-dependent. Set ``repeat_feature_selection=False``
    only for a clearly labelled conditional sensitivity analysis.
    """

    numeric_transformer = Pipeline(
        [
            (
                "fancy_imputer",
                FancyIterativeImputerWrapper(
                    max_iterations=5,
                    n_burn_in=3,
                    random_state=seed,
                    sample_posterior=True,
                    use_outcome=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )
    steps = [
        (
            "preprocessor",
            ColumnTransformer(
                [
                    ("num", numeric_transformer, list(numeric_cols)),
                    ("cat", categorical_transformer, list(categorical_cols)),
                ]
            ),
        )
    ]
    if repeat_feature_selection:
        steps.extend(
            [
                (
                    "lasso_fs",
                    SelectFromModel(
                        LogisticRegressionCV(
                            cv=5,
                            penalty="l1",
                            solver="liblinear",
                            random_state=seed,
                            max_iter=500,
                            n_jobs=1,
                        ),
                        threshold="mean",
                    ),
                ),
                (
                    "rf_fs",
                    SelectFromModel(
                        RandomForestClassifier(
                            n_estimators=50,
                            random_state=seed,
                            n_jobs=1,
                        ),
                        threshold="mean",
                    ),
                ),
            ]
        )
    return Pipeline(steps)


def _make_calibrator(estimator):
    """Construct a prefit Platt calibrator across supported sklearn versions."""
    try:
        # scikit-learn >=1.6: ``cv='prefit'`` was deprecated and subsequently
        # removed. FrozenEstimator explicitly marks the fitted base estimator.
        from sklearn.frozen import FrozenEstimator

        return CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid")
    except ImportError:
        # Compatibility with older project environments.
        return CalibratedClassifierCV(estimator, method="sigmoid", cv="prefit")


def fit_development_algorithm(
    X_raw: pd.DataFrame,
    y: np.ndarray,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    rf_params: Mapping,
    lr_params: Mapping,
    *,
    seed: int,
    calibration_fraction: float = 0.20,
    repeat_feature_selection: bool = True,
) -> FittedDevelopmentAlgorithm:
    """Fit preprocessing, both models and Platt scaling without leakage."""

    y = np.asarray(y, dtype=int)
    all_idx = np.arange(len(y))
    model_idx, calibration_idx = train_test_split(
        all_idx,
        test_size=calibration_fraction,
        random_state=seed,
        stratify=y,
    )

    preprocessor = build_preprocessor(
        numeric_cols,
        categorical_cols,
        seed,
        repeat_feature_selection=repeat_feature_selection,
    )
    X_model = preprocessor.fit_transform(X_raw.iloc[model_idx], y[model_idx])
    X_calibration = preprocessor.transform(X_raw.iloc[calibration_idx])

    rf = RandomForestClassifier(**_safe_model_params(rf_params, seed))
    lr = LogisticRegression(**_safe_model_params(lr_params, seed))
    rf.fit(X_model, y[model_idx])
    lr.fit(X_model, y[model_idx])

    rf_calibrated = _make_calibrator(rf)
    lr_calibrated = _make_calibrator(lr)
    rf_calibrated.fit(X_calibration, y[calibration_idx])
    lr_calibrated.fit(X_calibration, y[calibration_idx])

    return FittedDevelopmentAlgorithm(
        preprocessor=preprocessor,
        uncalibrated={"rf": rf, "lr": lr},
        calibrated={"rf": rf_calibrated, "lr": lr_calibrated},
    )


def predict_models(
    fitted: FittedDevelopmentAlgorithm,
    X_raw: pd.DataFrame,
) -> Dict[str, Dict[str, np.ndarray]]:
    X_transformed = fitted.preprocessor.transform(X_raw)
    return {
        model: {
            "uncalibrated": fitted.uncalibrated[model].predict_proba(X_transformed)[:, 1],
            "calibrated": fitted.calibrated[model].predict_proba(X_transformed)[:, 1],
        }
        for model in ("rf", "lr")
    }


def calibration_statistics(y_true, y_prob) -> Tuple[float, float, float]:
    """Return calibration slope, logistic calibration intercept and CITL.

    ``intercept`` and ``slope`` come from a logistic calibration model with
    both parameters estimated. ``citl`` is obtained from a logistic model with
    the prediction logit included as an offset (slope fixed at one).
    """

    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), EPS, 1.0 - EPS)
    if len(np.unique(y)) < 2 or np.std(p) == 0:
        return np.nan, np.nan, np.nan
    lp = np.log(p / (1.0 - p))

    def neg_log_likelihood(beta):
        eta = beta[0] + beta[1] * lp
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    result = minimize(
        neg_log_likelihood,
        x0=np.array([0.0, 1.0]),
        method="BFGS",
    )
    if not result.success and not np.all(np.isfinite(result.x)):
        intercept = slope = np.nan
    else:
        intercept, slope = map(float, result.x)

    def offset_score(alpha):
        return float(np.mean(expit(alpha + lp)) - np.mean(y))

    try:
        citl = float(brentq(offset_score, -30.0, 30.0))
    except ValueError:
        citl = np.nan
    return slope, intercept, citl


def performance_metrics(y_true, y_prob, threshold: float = 0.50) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    pred = p >= threshold
    slope, intercept, citl = calibration_statistics(y, p)
    return {
        "auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        "brier": brier_score_loss(y, p),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "citl": citl,
        "accuracy": accuracy_score(y, pred),
        "sensitivity": recall_score(y, pred, zero_division=0),
        "ppv": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }


def threshold_metrics(y_true, y_prob, thresholds: Iterable[float]) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    t = np.asarray(list(thresholds), dtype=float)
    pred = p[:, None] >= t[None, :]
    case = y[:, None] == 1
    tp = (pred & case).sum(axis=0).astype(float)
    fp = (pred & ~case).sum(axis=0).astype(float)
    tn = (~pred & ~case).sum(axis=0).astype(float)
    fn = (~pred & case).sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "flag_rate": (tp + fp) / len(y),
    }


def net_benefit(y_true, y_prob, thresholds: Iterable[float]) -> np.ndarray:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    t = np.asarray(list(thresholds), dtype=float)
    pred = p[:, None] >= t[None, :]
    case = y[:, None] == 1
    tp = (pred & case).sum(axis=0)
    fp = (pred & ~case).sum(axis=0)
    return (tp / len(y)) - (fp / len(y)) * (t / (1.0 - t))


def ordinary_bootstrap_indices(
    n: int,
    rng: np.random.RandomState,
    *,
    groups: Optional[Sequence] = None,
) -> np.ndarray:
    """Ordinary individual bootstrap, or cluster bootstrap when groups exist."""

    if groups is None:
        return rng.randint(0, n, size=n)

    groups = np.asarray(groups)
    unique_groups = pd.unique(groups)
    sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
    group_to_rows = {
        group: np.flatnonzero(groups == group)
        for group in unique_groups
    }
    return np.concatenate([group_to_rows[group] for group in sampled_groups])


def fixed_prediction_ci(
    y_true,
    y_prob,
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Evaluation-sample uncertainty for apparent predictions; no model refit."""

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    rng = np.random.RandomState(seed)
    keys = ("auc", "brier", "calibration_slope", "calibration_intercept", "citl")
    values = {key: [] for key in keys}
    for _ in range(n_resamples):
        idx = rng.randint(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        metrics = performance_metrics(y[idx], p[idx])
        for key in keys:
            values[key].append(metrics[key])
    return {
        key: (
            float(np.nanpercentile(values[key], 2.5)),
            float(np.nanpercentile(values[key], 97.5)),
        )
        for key in keys
    }
