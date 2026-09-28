"""Methodologically consistent, runtime-conscious sensitivity analyses.

Only variants that change the predictor set are refitted here. The primary
"with 4AT" result is read from script 02 rather than recomputed. Each variant
uses Harrell optimism correction and repeats preprocessing, feature selection,
model fitting and Platt calibration in every bootstrap sample.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import CLEANED_CSV, LR_BEST_PARAMS, OUTPUT_DIR, RF_BEST_PARAMS
from validation_core import (
    fit_development_algorithm,
    ordinary_bootstrap_indices,
    performance_metrics,
    predict_models,
)


RANDOM_STATE = 942
QUICK_TEST = os.getenv("QUICK_TEST", "0") == "1"
N_BOOTSTRAPS = 10 if QUICK_TEST else int(os.getenv("SENSITIVITY_BOOTSTRAPS", "100"))
N_JOBS = int(os.getenv("BOOTSTRAP_N_JOBS", "1"))
CALIBRATION_FRACTION = 0.20

NUMERIC_ALL = ["age_patient", "x4at_score", "total_acb_score", "unique_drug_count"]
CATEGORICAL_ALL = [
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

VARIANTS = {
    "Without 4AT": (
        [column for column in NUMERIC_ALL if column != "x4at_score"],
        CATEGORICAL_ALL,
    ),
    "Without symptom variables": (
        NUMERIC_ALL,
        [
            column
            for column in CATEGORICAL_ALL
            if column not in {"boc_risk_nominal", "new_change_in_behaviour_thinking"}
        ],
    ),
}


def fit_and_evaluate(X, y, numeric_cols, categorical_cols, seed):
    fitted = fit_development_algorithm(
        X,
        y,
        numeric_cols,
        categorical_cols,
        RF_BEST_PARAMS,
        LR_BEST_PARAMS,
        seed=seed,
        calibration_fraction=CALIBRATION_FRACTION,
        repeat_feature_selection=True,
    )
    predictions = predict_models(fitted, X)
    return {
        model: performance_metrics(y, predictions[model]["calibrated"])
        for model in ("rf", "lr")
    }


def one_iteration(iteration, X, y, numeric_cols, categorical_cols):
    seed = RANDOM_STATE + 10_000 + iteration
    rng = np.random.RandomState(seed)
    idx = ordinary_bootstrap_indices(len(y), rng)
    X_boot = X.iloc[idx].reset_index(drop=True)
    y_boot = y[idx]
    fitted = fit_development_algorithm(
        X_boot,
        y_boot,
        numeric_cols,
        categorical_cols,
        RF_BEST_PARAMS,
        LR_BEST_PARAMS,
        seed=seed,
        calibration_fraction=CALIBRATION_FRACTION,
        repeat_feature_selection=True,
    )
    pred_boot = predict_models(fitted, X_boot)
    pred_original = predict_models(fitted, X)
    return {
        model: {
            "bootstrap": performance_metrics(
                y_boot, pred_boot[model]["calibrated"]
            ),
            "original": performance_metrics(
                y, pred_original[model]["calibrated"]
            ),
        }
        for model in ("rf", "lr")
    }


def run_variant(label, df, numeric_cols, categorical_cols):
    X = df[numeric_cols + categorical_cols].copy()
    y = df["delirium"].astype(int).to_numpy()
    apparent = fit_and_evaluate(
        X, y, numeric_cols, categorical_cols, RANDOM_STATE
    )
    results = Parallel(n_jobs=N_JOBS, prefer="processes", verbose=5)(
        delayed(one_iteration)(b, X, y, numeric_cols, categorical_cols)
        for b in range(N_BOOTSTRAPS)
    )
    rows = []
    for model in ("rf", "lr"):
        for metric in ("auc", "brier"):
            optimism = np.array(
                [
                    result[model]["bootstrap"][metric]
                    - result[model]["original"][metric]
                    for result in results
                ]
            )
            rows.append(
                {
                    "variant": label,
                    "model": model.upper(),
                    "metric": metric,
                    "apparent": apparent[model][metric],
                    "optimism": optimism.mean(),
                    "optimism_mcse": optimism.std(ddof=1) / np.sqrt(len(optimism)),
                    "corrected": apparent[model][metric] - optimism.mean(),
                    "bootstrap_repetitions": N_BOOTSTRAPS,
                }
            )
    return rows


def main():
    output_dir = Path(OUTPUT_DIR)
    df = pd.read_csv(CLEANED_CSV)
    df = df.drop(columns=["disch_disposition"], errors="ignore")
    df = df.dropna(subset=["delirium"]).reset_index(drop=True)

    rows = []
    primary_path = output_dir / "bootstrap_performance_summary.csv"
    if primary_path.exists():
        primary = pd.read_csv(primary_path)
        primary_iterations_path = output_dir / "bootstrap_iteration_metrics.csv"
        primary_repetitions = (
            int(pd.read_csv(primary_iterations_path, usecols=["iteration"])["iteration"].max())
            if primary_iterations_path.exists()
            else np.nan
        )
        for _, row in primary.query(
            "basis == 'calibrated' and metric in ['auc', 'brier']"
        ).iterrows():
            rows.append(
                {
                    "variant": "Primary model",
                    "model": row.model,
                    "metric": row.metric,
                    "apparent": row.apparent,
                    "optimism": row.optimism,
                    "optimism_mcse": row.optimism_mcse,
                    "corrected": row.corrected,
                    "bootstrap_repetitions": primary_repetitions,
                }
            )

    for label, (numeric_cols, categorical_cols) in VARIANTS.items():
        rows.extend(run_variant(label, df, numeric_cols, categorical_cols))

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "sensitivity_analysis_summary.csv", index=False)
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
