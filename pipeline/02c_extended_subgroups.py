"""Extended subgroup analysis (Dementia, Cognitive impairment risk, Hip
fracture, Admission type) on the current Harrell bootstrap optimism-correction
pipeline, matching the methodology used for the primary Sex/Age subgroups in
02_bootstrap_validation.py. Sex/Age are NOT recomputed here (already produced
by 02_bootstrap_validation.py); this script's rows are APPENDED to
subgroup_performance_extended.csv.

Apparent predictions are loaded from apparent_predictions.csv (produced by
02_bootstrap_validation.py with the same RANDOM_STATE=42, so refitting would
reproduce identical apparent predictions -- loading avoids the redundant fit).
The bootstrap loop uses the same seed scheme (RANDOM_STATE + 10_000 +
iteration) as 02_bootstrap_validation.py for consistency, and evaluates all
four new subgroup masks from a single shared 200-iteration loop (one model
refit per iteration, sliced by all masks -- cheap relative to the refit cost).
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

RANDOM_STATE = 42
QUICK_TEST = os.getenv("QUICK_TEST", "0") == "1"
N_BOOTSTRAPS = 20 if QUICK_TEST else int(os.getenv("N_BOOTSTRAPS", "200"))
N_JOBS = int(os.getenv("BOOTSTRAP_N_JOBS", "1"))
CALIBRATION_FRACTION = 0.20
SUBGROUP_THRESHOLD = 0.15
MIN_GROUP_N = 20

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
    return df, X, y


def extended_masks(df):
    masks = {
        ("Dementia", "No"): df["dementia"].eq(0).to_numpy(),
        ("Dementia", "Yes"): df["dementia"].eq(1).to_numpy(),
        ("Cognitive impairment risk", "Yes"): df["nurse_verified_risk_of_cogn_impairment"].eq("Yes").to_numpy(),
        ("Cognitive impairment risk", "No"): df["nurse_verified_risk_of_cogn_impairment"].eq("No").to_numpy(),
        ("Hip fracture", "No"): df["hip_fracture"].eq(0).to_numpy(),
        ("Hip fracture", "Yes"): df["hip_fracture"].eq(1).to_numpy(),
        ("Admission type", "Medical"): df["admission_type_pharmacy"].eq("Medical").to_numpy(),
        ("Admission type", "Surgical"): df["admission_type_pharmacy"].eq("Surgical").to_numpy(),
    }
    return {k: v for k, v in masks.items() if v.sum() >= MIN_GROUP_N}


def bootstrap_iteration(iteration, X, y, masks):
    seed = RANDOM_STATE + 10_000 + iteration
    rng = np.random.RandomState(seed)
    boot_idx = ordinary_bootstrap_indices(len(y), rng, groups=None)
    X_boot = X.iloc[boot_idx].reset_index(drop=True)
    y_boot = y[boot_idx]

    fitted = fit_development_algorithm(
        X_boot, y_boot, NUMERIC_COLS, CATEGORICAL_COLS,
        RF_BEST_PARAMS, LR_BEST_PARAMS,
        seed=seed, calibration_fraction=CALIBRATION_FRACTION,
        repeat_feature_selection=True,
    )
    pred_boot = predict_models(fitted, X_boot)
    pred_original = predict_models(fitted, X)

    result = {}
    for model in ("rf", "lr"):
        p_boot = pred_boot[model]["calibrated"]
        p_original = pred_original[model]["calibrated"]
        subgroup_result = {}
        for key, original_mask in masks.items():
            bootstrap_mask = original_mask[boot_idx]
            if (
                bootstrap_mask.sum() < MIN_GROUP_N
                or original_mask.sum() < MIN_GROUP_N
                or len(np.unique(y_boot[bootstrap_mask])) < 2
                or len(np.unique(y[original_mask])) < 2
            ):
                continue
            subgroup_result[key] = {
                "boot": performance_metrics(y_boot[bootstrap_mask], p_boot[bootstrap_mask], threshold=SUBGROUP_THRESHOLD),
                "original": performance_metrics(y[original_mask], p_original[original_mask], threshold=SUBGROUP_THRESHOLD),
            }
        result[model] = subgroup_result
    return result


def corrected_scalar(apparent, boot_values, original_values):
    optimism = np.asarray(boot_values, dtype=float) - np.asarray(original_values, dtype=float)
    valid = optimism[np.isfinite(optimism)]
    return float(apparent - np.mean(valid))


def main():
    output_dir = Path(OUTPUT_DIR)
    df, X, y = load_data()
    masks = extended_masks(df)
    print(f"Groupings: {list(masks.keys())}")

    predictions = pd.read_csv(output_dir / "apparent_predictions.csv")
    assert len(predictions) == len(y), "apparent_predictions.csv row count mismatch -- rerun 02 first"

    print("=" * 78)
    print("EXTENDED SUBGROUP ANALYSIS (Dementia / Cognitive impairment / Hip fracture / Admission type)")
    print("=" * 78)
    print(f"Bootstrap repetitions: {N_BOOTSTRAPS}; workers: {N_JOBS}")
    if QUICK_TEST:
        print("*** QUICK_TEST output is not suitable for reporting ***")

    started = time.time()
    results = Parallel(n_jobs=N_JOBS, prefer="processes", verbose=5)(
        delayed(bootstrap_iteration)(b, X, y, masks) for b in range(N_BOOTSTRAPS)
    )
    print(f"Bootstrap completed in {(time.time() - started) / 60:.1f} minutes")

    rows = []
    for model, col in [("rf", "rf_calibrated"), ("lr", "lr_calibrated")]:
        p = predictions[col].to_numpy()
        for key, mask in masks.items():
            apparent = performance_metrics(y[mask], p[mask], threshold=SUBGROUP_THRESHOLD)
            available = [r[model][key] for r in results if key in r[model]]
            if not available:
                continue
            row = {
                "grouping": key[0], "group": key[1], "model": model.upper(),
                "n": int(mask.sum()), "events": int(y[mask].sum()),
            }
            for metric in ("auc", "brier", "calibration_slope", "sensitivity", "ppv"):
                row[metric] = corrected_scalar(
                    apparent[metric],
                    [item["boot"][metric] for item in available],
                    [item["original"][metric] for item in available],
                )
            rows.append(row)

    new_df = pd.DataFrame(rows)
    existing_path = output_dir / "subgroup_performance_extended.csv"
    existing = pd.read_csv(existing_path)
    existing = existing[~existing["grouping"].isin(new_df["grouping"].unique())]
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(existing_path, index=False)
    print(f"\n✓ Appended to {existing_path}")
    print(new_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
