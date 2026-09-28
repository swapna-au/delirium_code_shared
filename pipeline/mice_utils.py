"""
mice_utils.py
=============
Helpers for multiple imputation (MICE) with Rubin's-rules pooling.

The pipeline generates m imputed datasets by fitting the MICE imputer with
sample_posterior=True and a different random_state per imputation, so each
imputation is a proper draw from the posterior predictive distribution. Model
performance is estimated on each imputed dataset and combined with Rubin's rules.
"""

import numpy as np
from scipy import stats


def rubin_pool(estimates, variances=None, conf=0.95):
    """
    Combine per-imputation estimates with Rubin's (1987) rules.

    Parameters
    ----------
    estimates : array-like, length m
        Point estimate of the quantity (e.g. AUC) on each imputed dataset.
    variances : array-like, length m, optional
        Within-imputation sampling variance of each estimate (e.g. the squared
        standard error from cross-validation). If None, only between-imputation
        variance is used (appropriate when a within-imputation SE is unavailable).
    conf : float
        Confidence level for the returned interval.

    Returns
    -------
    dict with keys:
        estimate : pooled point estimate (mean of the m estimates)
        se       : pooled standard error (sqrt of total variance)
        ci_low, ci_high : confidence-interval bounds
        df       : Barnard-Rubin-style degrees of freedom
        within, between, total : the variance components
    """
    Q = np.asarray(estimates, dtype=float)
    Q = Q[~np.isnan(Q)]
    m = len(Q)
    Qbar = Q.mean()
    # Between-imputation variance
    B = Q.var(ddof=1) if m > 1 else 0.0
    # Within-imputation variance
    if variances is not None:
        U = np.asarray(variances, dtype=float)
        U = U[~np.isnan(U)]
        Ubar = U.mean() if len(U) else 0.0
    else:
        Ubar = 0.0
    # Total variance (Rubin)
    T = Ubar + (1.0 + 1.0 / m) * B if m > 1 else Ubar
    se = np.sqrt(T) if T > 0 else 0.0
    # Degrees of freedom (Rubin 1987); fall back to large df if no within var
    if T > 0 and B > 0:
        r = (1.0 + 1.0 / m) * B / max(Ubar, 1e-12)
        df = (m - 1) * (1.0 + 1.0 / r) ** 2
    else:
        df = 1e6
    tcrit = stats.t.ppf(1 - (1 - conf) / 2, df)
    return {
        "estimate": Qbar,
        "se": se,
        "ci_low": Qbar - tcrit * se,
        "ci_high": Qbar + tcrit * se,
        "df": df,
        "within": Ubar,
        "between": B,
        "total": T,
        "m": m,
    }


def pooled_str(pool, nd=4):
    """Format a rubin_pool() result as 'estimate [low-high]'."""
    return f"{pool['estimate']:.{nd}f} [{pool['ci_low']:.{nd}f}-{pool['ci_high']:.{nd}f}]"
