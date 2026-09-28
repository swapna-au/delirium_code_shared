"""
00b_missingness_analysis.py — missingness characterisation + Little's MCAR test.

Characterises missing data BEFORE imputation:
  • per-variable missing counts / percentages (predictors AND outcome)
  • missingness heatmap (pattern visualisation)
  • Little's (1988) MCAR test on the numeric predictors

Little's test assesses whether data are Missing Completely At Random (MCAR). A
small p-value (< 0.05) rejects MCAR (data are not MCAR -> MAR/MNAR more likely,
supporting the use of multiple imputation rather than complete-case analysis).
The test assumes multivariate normality of the continuous variables and is
applied to the numeric predictors only.

Run on cleaned.csv, i.e. the joined data BEFORE any imputation stage runs.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(__file__))
from common import CLEANED_CSV, OUTPUT_DIR, RANDOM_STATE

sns.set_style("whitegrid")

numeric_cols = ["age_patient", "x4at_score", "total_acb_score", "unique_drug_count"]
categorical_cols = [
    "sex_patient", "race", "nurse_verified_continence_risk",
    "nurse_verified_nutritional_risk", "nurse_verified_risk_of_cogn_impairment",
    "nurse_verified_risk_of_falls", "nurse_verified_risk_of_infection",
    "nurse_verified_risk_of_pressure_injury", "boc_risk_nominal",
    "critical_hb_status", "hb_status", "cr_positive", "wcc_status",
    "hip_fracture", "dementia", "copd", "atrial_fibrillation",
    "psychiatric_disorders", "peripheral_pulse_rate_status",
    "respiratory_rate_status", "temperature_tympanic_status",
    "new_change_in_behaviour_thinking", "admission_type_pharmacy",
]
OUTCOME_COL = "delirium"


def little_mcar_test(data, max_iter=300, tol=1e-6):
    """
    Little's (1988) MCAR test via EM. `data` is an (n, p) float array with
    np.nan for missing. Returns (chi2 statistic, degrees of freedom, p-value).
    """
    X = np.asarray(data, dtype=float)
    n, p = X.shape
    R = ~np.isnan(X)
    patterns, inv = np.unique(R, axis=0, return_inverse=True)

    mu = np.nanmean(X, axis=0)
    cc = X[np.all(R, axis=1)]
    Sigma = np.cov(cc, rowvar=False) if cc.shape[0] > p else np.diag(np.nanvar(X, axis=0))
    Sigma = np.atleast_2d(Sigma) + 1e-6 * np.eye(p)

    for _ in range(max_iter):
        T1 = np.zeros(p)
        T2 = np.zeros((p, p))
        for k, pat in enumerate(patterns):
            idx = np.where(inv == k)[0]
            obs = np.where(pat)[0]
            mis = np.where(~pat)[0]
            m = len(idx)
            if len(obs) == 0:
                continue
            Xo = X[np.ix_(idx, obs)]
            if len(mis) == 0:
                Ex = X[idx]
                T1 += Ex.sum(0)
                T2 += Ex.T @ Ex
                continue
            So_o_inv = np.linalg.pinv(Sigma[np.ix_(obs, obs)])
            Sm_o = Sigma[np.ix_(mis, obs)]
            B = Sm_o @ So_o_inv
            cond_mean = mu[mis] + (Xo - mu[obs]) @ B.T
            cond_cov = Sigma[np.ix_(mis, mis)] - B @ Sm_o.T
            Ex = np.empty((m, p))
            Ex[:, obs] = Xo
            Ex[:, mis] = cond_mean
            T1 += Ex.sum(0)
            T2 += Ex.T @ Ex
            T2[np.ix_(mis, mis)] += m * cond_cov
        mu_new = T1 / n
        Sigma_new = T2 / n - np.outer(mu_new, mu_new)
        if (np.max(np.abs(mu_new - mu)) < tol
                and np.max(np.abs(Sigma_new - Sigma)) < tol):
            mu, Sigma = mu_new, Sigma_new
            break
        mu, Sigma = mu_new, Sigma_new + 1e-8 * np.eye(p)

    d2, df_stat = 0.0, 0
    for k, pat in enumerate(patterns):
        idx = np.where(inv == k)[0]
        obs = np.where(pat)[0]
        if len(obs) == 0:
            continue
        m = len(idx)
        xbar = X[np.ix_(idx, obs)].mean(0)
        diff = xbar - mu[obs]
        d2 += m * diff @ np.linalg.pinv(Sigma[np.ix_(obs, obs)]) @ diff
        df_stat += len(obs)
    df_stat -= p
    pval = float(1 - chi2.cdf(d2, df_stat)) if df_stat > 0 else np.nan
    return float(d2), int(df_stat), pval


def main() -> None:
    print("Loading data...")
    df = pd.read_csv(CLEANED_CSV)
    all_cols = [c for c in numeric_cols + categorical_cols + [OUTCOME_COL] if c in df.columns]

    # ── Per-variable missingness summary (predictors AND outcome) ──────────
    miss = pd.DataFrame({
        "Variable": all_cols,
        "Type": ["outcome" if c == OUTCOME_COL else
                  "numeric" if c in numeric_cols else "categorical" for c in all_cols],
        "N_missing": [int(df[c].isna().sum()) for c in all_cols],
        "Pct_missing": [round(100 * df[c].isna().mean(), 2) for c in all_cols],
    }).sort_values("Pct_missing", ascending=False)

    miss_path = OUTPUT_DIR / "missingness_summary.csv"
    miss.to_csv(miss_path, index=False)
    print(f"[OK] Saved: {miss_path.name}")
    print("\n" + miss.to_string(index=False))

    # ── Missingness heatmap (predictors only, sampled rows) ─────────────────
    predictor_cols = [c for c in numeric_cols + categorical_cols if c in df.columns]
    sample = df[predictor_cols].sample(n=min(500, len(df)), random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(sample.isna(), cbar=False, cmap="viridis", yticklabels=False, ax=ax)
    ax.set_title("Missingness pattern (random sample of rows)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Variable")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "missingness_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] Saved: missingness_heatmap.png")

    # ── Missingness bar chart ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    mplot = miss[(miss["Pct_missing"] > 0) & (miss["Variable"] != OUTCOME_COL)]
    if len(mplot):
        ax.barh(mplot["Variable"][::-1], mplot["Pct_missing"][::-1], color="#1f77b4", alpha=0.85)
    ax.set_xlabel("% missing")
    ax.set_title("Percentage missing by variable", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "missingness_barchart.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] Saved: missingness_barchart.png")

    # ── Little's MCAR test (numeric predictors) ─────────────────────────────
    num = df[numeric_cols].apply(pd.to_numeric, errors="coerce").values
    stat, dfree, pval = little_mcar_test(num)
    n_patterns = len(np.unique(~np.isnan(num), axis=0))
    verdict = ("Reject MCAR (data are NOT MCAR; MAR/MNAR more likely — "
               "multiple imputation appropriate)") if (not np.isnan(pval) and pval < 0.05) \
        else "Do not reject MCAR (consistent with data being MCAR)"

    mcar = pd.DataFrame({
        "Metric": ["Variables tested", "Missingness patterns",
                   "Chi-square statistic", "Degrees of freedom", "p-value", "Conclusion"],
        "Value": [", ".join(numeric_cols), n_patterns,
                  round(stat, 3), dfree,
                  ("<0.001" if (not np.isnan(pval) and pval < 0.001) else round(pval, 4)),
                  verdict],
    })
    mcar_path = OUTPUT_DIR / "missingness_mcar_test.csv"
    mcar.to_csv(mcar_path, index=False)

    print("\n" + "=" * 70)
    print("LITTLE'S MCAR TEST (numeric predictors)")
    print("=" * 70)
    print(mcar.to_string(index=False))
    print(f"\n[OK] Saved: {mcar_path.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
