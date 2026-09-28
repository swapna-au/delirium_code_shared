"""Harrell bootstrap internal validation for RF and logistic regression.

This is the single computationally intensive script in the reporting pipeline.
It repeats the complete prespecified development algorithm in each bootstrap
sample, evaluates the fitted algorithm both in that bootstrap sample and in the
original development data, and subtracts mean optimism from apparent
performance.

Outputs include:
  * apparent predictions;
  * apparent, bootstrap-test, optimism and corrected performance;
  * corrected threshold performance;
  * corrected subgroup performance;
  * corrected decision-curve net benefit;
  * convergence and Monte Carlo error diagnostics.

No confidence interval is attached to an optimism-corrected estimate. A valid
interval requires a separate two-stage bootstrap. Apparent-performance CIs are
provided from an evaluation-only bootstrap of fixed predictions.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from common import CLEANED_CSV, LR_BEST_PARAMS, OUTPUT_DIR, RF_BEST_PARAMS
from validation_core import (
    fit_development_algorithm,
    fixed_prediction_ci,
    net_benefit,
    ordinary_bootstrap_indices,
    performance_metrics,
    predict_models,
    threshold_metrics,
)


RANDOM_STATE = 42
QUICK_TEST = os.getenv("QUICK_TEST", "0") == "1"
N_BOOTSTRAPS = 20 if QUICK_TEST else int(os.getenv("N_BOOTSTRAPS", "200"))
N_JOBS = int(os.getenv("BOOTSTRAP_N_JOBS", "1"))
CALIBRATION_FRACTION = 0.20
APPARENT_CI_RESAMPLES = 200 if QUICK_TEST else 1000
CLUSTER_COLUMN = os.getenv("BOOTSTRAP_CLUSTER_COLUMN") or None

REPORTING_THRESHOLDS = np.array(
    [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
)
DCA_THRESHOLDS = np.linspace(0.01, 0.50, 50)

NUMERIC_COLS = [
    "age_patient",
    "x4at_score",
    "total_acb_score",
    "unique_drug_count",
]
CATEGORICAL_COLS = [
    "sex_patient",
    "race",
    "nurse_verified_continence_risk",
    "nurse_verified_nutritional_risk",
    "nurse_verified_risk_of_cogn_impairment",
    "nurse_verified_risk_of_falls",
    "nurse_verified_risk_of_infection",
    "nurse_verified_risk_of_pressure_injury",
    "boc_risk_nominal",
    "critical_hb_status",
    "hb_status",
    "cr_positive",
    "wcc_status",
    "hip_fracture",
    "dementia",
    "copd",
    "atrial_fibrillation",
    "psychiatric_disorders",
    "peripheral_pulse_rate_status",
    "respiratory_rate_status",
    "temperature_tympanic_status",
    "new_change_in_behaviour_thinking",
    "admission_type_pharmacy",
]
PRIMARY_METRICS = (
    "auc",
    "brier",
    "calibration_slope",
    "calibration_intercept",
    "citl",
)


def load_data():
    df = pd.read_csv(CLEANED_CSV)
    df = df.drop(columns=["disch_disposition"], errors="ignore")
    df = df.dropna(subset=["delirium"]).reset_index(drop=True)
    missing = [
        column
        for column in NUMERIC_COLS + CATEGORICAL_COLS
        if column not in df.columns
    ]
    if missing:
        raise KeyError(f"Required predictors are missing: {missing}")
    if CLUSTER_COLUMN and CLUSTER_COLUMN not in df.columns:
        raise KeyError(f"Cluster column {CLUSTER_COLUMN!r} is not present")
    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y = df["delirium"].astype(int).to_numpy()
    groups = df[CLUSTER_COLUMN].to_numpy() if CLUSTER_COLUMN else None
    return df, X, y, groups


def subgroup_masks(df):
    masks = {("Overall", "All"): np.ones(len(df), dtype=bool)}
    if "sex_patient" in df:
        for value in sorted(df["sex_patient"].dropna().astype(str).unique()):
            if value != "O":
                masks[("Sex", value)] = df["sex_patient"].astype(str).eq(value).to_numpy()
    if "age_patient" in df:
        age_groups = pd.cut(
            df["age_patient"],
            bins=[-np.inf, 65, 75, 85, np.inf],
            right=False,
            labels=["<65", "65–74", "75–84", "85+"],
        )
        for value in age_groups.cat.categories:
            masks[("Age", str(value))] = age_groups.eq(value).to_numpy()
    return masks


def evaluate_predictions(y, predictions, threshold=0.50):
    return {
        model: {
            basis: performance_metrics(y, values, threshold=threshold)
            for basis, values in model_predictions.items()
        }
        for model, model_predictions in predictions.items()
    }


def bootstrap_iteration(
    iteration,
    X,
    y,
    groups,
    masks,
):
    seed = RANDOM_STATE + 10_000 + iteration
    rng = np.random.RandomState(seed)
    boot_idx = ordinary_bootstrap_indices(len(y), rng, groups=groups)
    X_boot = X.iloc[boot_idx].reset_index(drop=True)
    y_boot = y[boot_idx]

    fitted = fit_development_algorithm(
        X_boot,
        y_boot,
        NUMERIC_COLS,
        CATEGORICAL_COLS,
        RF_BEST_PARAMS,
        LR_BEST_PARAMS,
        seed=seed,
        calibration_fraction=CALIBRATION_FRACTION,
        repeat_feature_selection=True,
    )
    pred_boot = predict_models(fitted, X_boot)
    pred_original = predict_models(fitted, X)
    metrics_boot = evaluate_predictions(y_boot, pred_boot)
    metrics_original = evaluate_predictions(y, pred_original)

    result = {
        "iteration": iteration + 1,
        "metrics_boot": metrics_boot,
        "metrics_original": metrics_original,
        "threshold": {},
        "dca": {},
        "subgroups": {},
    }

    for model in ("rf", "lr"):
        p_boot = pred_boot[model]["calibrated"]
        p_original = pred_original[model]["calibrated"]
        result["threshold"][model] = {
            "boot": threshold_metrics(y_boot, p_boot, REPORTING_THRESHOLDS),
            "original": threshold_metrics(y, p_original, REPORTING_THRESHOLDS),
        }
        result["dca"][model] = {
            "boot": net_benefit(y_boot, p_boot, DCA_THRESHOLDS),
            "original": net_benefit(y, p_original, DCA_THRESHOLDS),
        }

        subgroup_result = {}
        for key, original_mask in masks.items():
            bootstrap_mask = original_mask[boot_idx]
            if (
                bootstrap_mask.sum() < 20
                or original_mask.sum() < 20
                or len(np.unique(y_boot[bootstrap_mask])) < 2
                or len(np.unique(y[original_mask])) < 2
            ):
                continue
            subgroup_result[key] = {
                "boot": performance_metrics(
                    y_boot[bootstrap_mask],
                    p_boot[bootstrap_mask],
                    threshold=0.15,
                ),
                "original": performance_metrics(
                    y[original_mask],
                    p_original[original_mask],
                    threshold=0.15,
                ),
            }
        result["subgroups"][model] = subgroup_result
    return result


def corrected_scalar(apparent, boot_values, original_values):
    optimism = np.asarray(boot_values, dtype=float) - np.asarray(
        original_values, dtype=float
    )
    valid = optimism[np.isfinite(optimism)]
    return {
        "bootstrap_test_mean": float(np.nanmean(original_values)),
        "optimism": float(np.mean(valid)),
        "optimism_sd": float(np.std(valid, ddof=1)),
        "optimism_mcse": float(np.std(valid, ddof=1) / np.sqrt(len(valid))),
        "corrected": float(apparent - np.mean(valid)),
    }


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    df, X, y, groups = load_data()
    masks = subgroup_masks(df)

    print("=" * 78)
    print("HARRELL BOOTSTRAP INTERNAL VALIDATION")
    print("=" * 78)
    print(f"Observations: {len(y):,}; events: {y.sum():,} ({y.mean():.2%})")
    print(f"Bootstrap repetitions: {N_BOOTSTRAPS}; outer workers: {N_JOBS}")
    print(
        "Resampling unit: "
        + (f"cluster ({CLUSTER_COLUMN})" if CLUSTER_COLUMN else "individual episode")
    )
    if QUICK_TEST:
        print("*** QUICK_TEST output is not suitable for reporting ***")

    # Apparent performance uses exactly the same development algorithm as every
    # bootstrap repetition.
    apparent_fitted = fit_development_algorithm(
        X,
        y,
        NUMERIC_COLS,
        CATEGORICAL_COLS,
        RF_BEST_PARAMS,
        LR_BEST_PARAMS,
        seed=RANDOM_STATE,
        calibration_fraction=CALIBRATION_FRACTION,
        repeat_feature_selection=True,
    )
    apparent_predictions = predict_models(apparent_fitted, X)
    apparent_metrics = evaluate_predictions(y, apparent_predictions)

    prediction_table = pd.DataFrame(
        {
            "row_id": np.arange(len(y)),
            "outcome": y,
            "rf_uncalibrated": apparent_predictions["rf"]["uncalibrated"],
            "rf_calibrated": apparent_predictions["rf"]["calibrated"],
            "lr_uncalibrated": apparent_predictions["lr"]["uncalibrated"],
            "lr_calibrated": apparent_predictions["lr"]["calibrated"],
        }
    )
    prediction_table.to_csv(output_dir / "apparent_predictions.csv", index=False)

    apparent_cis = {}
    for model in ("rf", "lr"):
        for basis in ("uncalibrated", "calibrated"):
            apparent_cis[(model, basis)] = fixed_prediction_ci(
                y,
                apparent_predictions[model][basis],
                n_resamples=APPARENT_CI_RESAMPLES,
                seed=RANDOM_STATE + (0 if model == "rf" else 100) + (0 if basis == "uncalibrated" else 10),
            )

    started = time.time()
    results = Parallel(n_jobs=N_JOBS, prefer="processes", verbose=5)(
        delayed(bootstrap_iteration)(b, X, y, groups, masks)
        for b in range(N_BOOTSTRAPS)
    )
    print(f"Bootstrap completed in {(time.time() - started) / 60:.1f} minutes")

    # Primary tidy performance table.
    summary_rows = []
    iteration_rows = []
    for model in ("rf", "lr"):
        for basis in ("uncalibrated", "calibrated"):
            for metric in PRIMARY_METRICS:
                boot_values = [
                    r["metrics_boot"][model][basis][metric] for r in results
                ]
                original_values = [
                    r["metrics_original"][model][basis][metric] for r in results
                ]
                corrected = corrected_scalar(
                    apparent_metrics[model][basis][metric],
                    boot_values,
                    original_values,
                )
                ci_low, ci_high = apparent_cis[(model, basis)][metric]
                summary_rows.append(
                    {
                        "model": model.upper(),
                        "basis": basis,
                        "metric": metric,
                        "apparent": apparent_metrics[model][basis][metric],
                        "apparent_ci_low": ci_low,
                        "apparent_ci_high": ci_high,
                        # Two distinct resampling counts sit behind this row:
                        # the apparent CI comes from evaluation-only resampling
                        # of fixed predictions, the corrected value from the
                        # Harrell optimism loop. Recorded so downstream figures
                        # and tables cite them without restating constants.
                        "apparent_ci_resamples": APPARENT_CI_RESAMPLES,
                        "optimism_repetitions": N_BOOTSTRAPS,
                        **corrected,
                    }
                )
                for b, (boot_value, original_value) in enumerate(
                    zip(boot_values, original_values), start=1
                ):
                    iteration_rows.append(
                        {
                            "iteration": b,
                            "model": model.upper(),
                            "basis": basis,
                            "metric": metric,
                            "bootstrap_sample": boot_value,
                            "test_on_original": original_value,
                            "optimism": boot_value - original_value,
                        }
                    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "bootstrap_performance_summary.csv", index=False)
    pd.DataFrame(iteration_rows).to_csv(
        output_dir / "bootstrap_iteration_metrics.csv", index=False
    )

    # Convergence: cumulative corrected AUC and Brier, with MCSE of optimism.
    convergence_rows = []
    checkpoints = sorted(
        set([x for x in (25, 50, 75, 100, 150, 200) if x <= N_BOOTSTRAPS] + [N_BOOTSTRAPS])
    )
    for n in checkpoints:
        for model in ("rf", "lr"):
            for metric in ("auc", "brier"):
                optimism = np.array(
                    [
                        r["metrics_boot"][model]["calibrated"][metric]
                        - r["metrics_original"][model]["calibrated"][metric]
                        for r in results[:n]
                    ]
                )
                convergence_rows.append(
                    {
                        "n_iterations": n,
                        "model": model.upper(),
                        "metric": metric,
                        "corrected": apparent_metrics[model]["calibrated"][metric]
                        - optimism.mean(),
                        "optimism_mcse": optimism.std(ddof=1) / np.sqrt(n)
                        if n > 1
                        else np.nan,
                    }
                )
    pd.DataFrame(convergence_rows).to_csv(
        output_dir / "bootstrap_convergence_diagnostic.csv", index=False
    )

    # Threshold performance: one correction from the same primary bootstrap.
    threshold_rows = []
    for model in ("rf", "lr"):
        apparent_threshold = threshold_metrics(
            y, apparent_predictions[model]["calibrated"], REPORTING_THRESHOLDS
        )
        for i, threshold in enumerate(REPORTING_THRESHOLDS):
            row = {"model": model.upper(), "threshold": threshold}
            for metric in ("sensitivity", "specificity", "ppv", "npv", "flag_rate"):
                boot_values = [
                    r["threshold"][model]["boot"][metric][i] for r in results
                ]
                original_values = [
                    r["threshold"][model]["original"][metric][i] for r in results
                ]
                row[metric] = corrected_scalar(
                    apparent_threshold[metric][i], boot_values, original_values
                )["corrected"]
            threshold_rows.append(row)
    threshold_df = pd.DataFrame(threshold_rows)
    for model in ("RF", "LR"):
        threshold_df.query("model == @model").drop(columns="model").to_csv(
            output_dir / f"threshold_performance_{model.lower()}.csv",
            index=False,
        )

    # Subgroup point estimates. Uncertainty is not presented because a valid
    # optimism-corrected CI requires a two-stage bootstrap.
    subgroup_rows = []
    for model in ("rf", "lr"):
        p = apparent_predictions[model]["calibrated"]
        for key, mask in masks.items():
            if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
                continue
            apparent = performance_metrics(y[mask], p[mask], threshold=0.15)
            available = [
                r["subgroups"][model][key]
                for r in results
                if key in r["subgroups"][model]
            ]
            if not available:
                continue
            row = {
                "grouping": key[0],
                "group": key[1],
                "model": model.upper(),
                "n": int(mask.sum()),
                "events": int(y[mask].sum()),
            }
            for metric in (
                "auc",
                "brier",
                "calibration_slope",
                "sensitivity",
                "ppv",
            ):
                row[metric] = corrected_scalar(
                    apparent[metric],
                    [item["boot"][metric] for item in available],
                    [item["original"][metric] for item in available],
                )["corrected"]
            subgroup_rows.append(row)
    pd.DataFrame(subgroup_rows).to_csv(
        output_dir / "subgroup_performance_extended.csv", index=False
    )

    # Decision curves. Corrected point estimates only.
    prevalence = y.mean()
    treat_all = prevalence - (1.0 - prevalence) * DCA_THRESHOLDS / (
        1.0 - DCA_THRESHOLDS
    )
    dca = {
        "threshold": DCA_THRESHOLDS,
        "nb_treat_all": treat_all,
        "nb_treat_none": np.zeros_like(DCA_THRESHOLDS),
    }
    for model in ("rf", "lr"):
        apparent_nb = net_benefit(
            y, apparent_predictions[model]["calibrated"], DCA_THRESHOLDS
        )
        optimism = np.vstack(
            [
                r["dca"][model]["boot"] - r["dca"][model]["original"]
                for r in results
            ]
        )
        dca[f"nb_{model}_corrected"] = apparent_nb - optimism.mean(axis=0)
        dca[f"nb_{model}_optimism_mcse"] = optimism.std(axis=0, ddof=1) / np.sqrt(
            N_BOOTSTRAPS
        )
    pd.DataFrame(dca).to_csv(
        output_dir / "bootstrap_decision_curve_analysis.csv", index=False
    )

    print("\nPrimary calibrated optimism-corrected performance")
    print(
        summary.query("basis == 'calibrated'")[
            [
                "model",
                "metric",
                "apparent",
                "optimism",
                "optimism_mcse",
                "corrected",
            ]
        ].round(4).to_string(index=False)
    )
    print(f"\nOutputs saved to {output_dir}")


if __name__ == "__main__":
    main()
