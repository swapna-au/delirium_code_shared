"""Unit-of-analysis sensitivity check: episode-level (i.i.d.) bootstrap vs.
patient-level (cluster) bootstrap, using the same full predictor set and
Harrell optimism-correction methodology as the primary model. The
episode-level comparator is the primary model's already-computed corrected
AUC (bootstrap_performance_summary.csv, 200 repetitions) -- not recomputed
here. Only the patient-level cluster-bootstrap arm is run (using
validation_core.ordinary_bootstrap_indices(..., groups=patient_cluster_id),
which resamples whole patients with replacement so all episodes for a
resampled patient are carried together, preventing within-patient leakage
across the bootstrap/original split).
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
    ordinary_bootstrap_indices,
    performance_metrics,
    predict_models,
)

RANDOM_STATE = 942
QUICK_TEST = os.getenv("QUICK_TEST", "0") == "1"
N_BOOTSTRAPS = 10 if QUICK_TEST else int(os.getenv("SENSITIVITY_BOOTSTRAPS", "100"))
N_JOBS = int(os.getenv("BOOTSTRAP_N_JOBS", "1"))
CALIBRATION_FRACTION = 0.20
CLUSTER_COLUMN = "patient_cluster_id"

NUMERIC_COLS = ["age_patient", "x4at_score", "total_acb_score", "unique_drug_count"]
CATEGORICAL_COLS = [
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


def load_data():
    df = pd.read_csv(CLEANED_CSV)
    df = df.drop(columns=["disch_disposition"], errors="ignore")
    df = df.dropna(subset=["delirium"]).reset_index(drop=True)
    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y = df["delirium"].astype(int).to_numpy()
    groups = df[CLUSTER_COLUMN].to_numpy()
    return df, X, y, groups


def one_iteration(iteration, X, y, groups):
    seed = RANDOM_STATE + 10_000 + iteration
    rng = np.random.RandomState(seed)
    idx = ordinary_bootstrap_indices(len(y), rng, groups=groups)
    X_boot = X.iloc[idx].reset_index(drop=True)
    y_boot = y[idx]
    fitted = fit_development_algorithm(
        X_boot, y_boot, NUMERIC_COLS, CATEGORICAL_COLS,
        RF_BEST_PARAMS, LR_BEST_PARAMS,
        seed=seed, calibration_fraction=CALIBRATION_FRACTION,
        repeat_feature_selection=True,
    )
    pred_boot = predict_models(fitted, X_boot)
    pred_original = predict_models(fitted, X)
    return {
        model: {
            "bootstrap": performance_metrics(y_boot, pred_boot[model]["calibrated"]),
            "original": performance_metrics(y, pred_original[model]["calibrated"]),
        }
        for model in ("rf", "lr")
    }


def main():
    output_dir = Path(OUTPUT_DIR)
    df, X, y, groups = load_data()
    n_patients = len(np.unique(groups))
    print("=" * 78)
    print("UNIT-OF-ANALYSIS SENSITIVITY CHECK: patient-level cluster bootstrap")
    print("=" * 78)
    print(f"Episodes: {len(y):,}; unique patients: {n_patients:,}")
    print(f"Bootstrap repetitions: {N_BOOTSTRAPS}; workers: {N_JOBS}")
    if QUICK_TEST:
        print("*** QUICK_TEST output is not suitable for reporting ***")

    X_full = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    apparent_fitted = fit_development_algorithm(
        X_full, y, NUMERIC_COLS, CATEGORICAL_COLS,
        RF_BEST_PARAMS, LR_BEST_PARAMS,
        seed=42, calibration_fraction=CALIBRATION_FRACTION,
        repeat_feature_selection=True,
    )
    apparent_predictions = predict_models(apparent_fitted, X_full)
    apparent = {
        model: performance_metrics(y, apparent_predictions[model]["calibrated"])
        for model in ("rf", "lr")
    }

    started = time.time()
    results = Parallel(n_jobs=N_JOBS, prefer="processes", verbose=5)(
        delayed(one_iteration)(b, X, y, groups) for b in range(N_BOOTSTRAPS)
    )
    print(f"Bootstrap completed in {(time.time() - started) / 60:.1f} minutes")

    rows = []
    # Episode-level comparator: primary model's already-computed corrected value.
    primary_path = output_dir / "bootstrap_performance_summary.csv"
    primary = pd.read_csv(primary_path)
    for model in ("RF", "LR"):
        for metric in ("auc",):
            row = primary.query(
                "model == @model and basis == 'calibrated' and metric == @metric"
            ).iloc[0]
            rows.append({
                "variant": "Episode-level (primary)", "model": model, "metric": metric,
                "apparent": row.apparent, "optimism": row.optimism,
                "optimism_mcse": row.optimism_mcse, "corrected": row.corrected,
                "bootstrap_repetitions": 200,
            })

    for model in ("rf", "lr"):
        for metric in ("auc",):
            optimism = np.array([
                r[model]["bootstrap"][metric] - r[model]["original"][metric] for r in results
            ])
            rows.append({
                "variant": "Patient-level (cluster bootstrap)", "model": model.upper(), "metric": metric,
                "apparent": apparent[model][metric], "optimism": optimism.mean(),
                "optimism_mcse": optimism.std(ddof=1) / np.sqrt(len(optimism)),
                "corrected": apparent[model][metric] - optimism.mean(),
                "bootstrap_repetitions": N_BOOTSTRAPS,
            })

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "unit_of_analysis_summary.csv", index=False)
    print(result.round(4).to_string(index=False))
    print(f"\n✓ Saved: {output_dir / 'unit_of_analysis_summary.csv'}")


if __name__ == "__main__":
    main()
