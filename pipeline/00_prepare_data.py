"""
00_prepare_data.py — build the de-identified model input (cleaned.csv) from the
raw joined extract (source_file_bronze.csv).

This is the Python port of the cleaning + de-identification step that previously
lived in logistic_regression.rmd (janitor::clean_names + column drops + recodes).

  INPUT   data/source_file_bronze.csv   (NOT in the repo — contains identifiers)
  OUTPUT  data/cleaned.csv              (de-identified, tracked)

De-identification: every direct/indirect identifier is dropped here (encounter
and URN ids, visit number, admission/discharge datetimes, ward display name,
medical service, reason for visit, length of stay, encounter type/status).
Only the columns in COLUMN_MAP below survive into cleaned.csv, PLUS one derived
column: `patient_cluster_id`, a one-way salted hash of Urn (see below). This is
NOT a reversible identifier -- it exists only so cleaned.csv can support
patient-level grouped cross-validation (repeated admissions by the same patient
land in the same CV fold), which requires knowing which episodes share a
patient WITHOUT retaining the identifier itself.

Run automatically by run_all.py when the bronze file is present; otherwise the
pipeline starts from the committed cleaned.csv.
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import BRONZE_CSV, CLEANED_CSV

# Fixed, non-secret salt: this is NOT a security control (Urn is a small,
# structured ID space and a motivated party with access to the source system
# could still dictionary-attack this hash). Its purpose is narrower: prevent
# the *raw* identifier or a *directly recognisable* transform of it from
# appearing in cleaned.csv, while still letting episodes be grouped by patient.
_PATIENT_HASH_SALT = "delirium-risk-prediction-patient-cluster-v1"


def _hash_patient_id(urn) -> str:
    """One-way pseudonymous cluster ID for grouped CV. Not reversible without
    brute-forcing the Urn space; not intended as a security/privacy control."""
    if pd.isna(urn):
        return np.nan
    raw = f"{_PATIENT_HASH_SALT}:{int(urn)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

# ── Columns kept from bronze, in output order:  raw bronze name -> clean name ──
# Everything NOT listed here is dropped; the identifier columns dropped for
# de-identification are named in the module docstring above.
COLUMN_MAP = {
    "Disch Disposition": "disch_disposition",
    "Age_patient": "age_patient",
    "Sex_patient": "sex_patient",
    "Race": "race",
    "Admission Type_pharmacy": "admission_type_pharmacy",
    "Mean Cr Result Value": "mean_cr_result_value",
    "Cr Result Val mg/dl": "cr_result_val_mg_dl",
    "Cr Positive": "cr_positive",
    "Mean_Hb_Result_Val": "mean_hb_result_val",
    "lowest_Hb_Result_Val": "lowest_hb_result_val",
    "Critical_Hb_Status": "critical_hb_status",
    "Hb_Status": "hb_status",
    "Mean_WCC_Result_Val": "mean_wcc_result_val",
    "WCC_Status": "wcc_status",
    "4at Score": "x4at_score",
    "Nurse Verified Continence Risk": "nurse_verified_continence_risk",
    "Nurse Verified Nutritional Risk": "nurse_verified_nutritional_risk",
    "Nurse Verified Risk of Cogn Impairment": "nurse_verified_risk_of_cogn_impairment",
    "Nurse Verified Risk of Falls": "nurse_verified_risk_of_falls",
    "Nurse Verified Risk of Infection": "nurse_verified_risk_of_infection",
    "Nurse Verified Risk of Pressure Injury": "nurse_verified_risk_of_pressure_injury",
    "BOC Risk (ordinal)": "boc_risk_ordinal",
    "BOC Risk (nominal)": "boc_risk_nominal",
    "4at Score Categorical": "x4at_score_categorical",
    "Peripheral Pulse Rate Max": "peripheral_pulse_rate_max",
    "Peripheral Pulse Rate Mean": "peripheral_pulse_rate_mean",
    "Respiratory Rate Max": "respiratory_rate_max",
    "Respiratory Rate Mean": "respiratory_rate_mean",
    "Temperature Tympanic Max": "temperature_tympanic_max",
    "Temperature Tympanic Mean": "temperature_tympanic_mean",
    "Family Worry": "family_worry",
    "New Change in Behaviour / Thinking": "new_change_in_behaviour_thinking",
    "Peripheral Pulse Rate Status": "peripheral_pulse_rate_status",
    "Respiratory Rate Status": "respiratory_rate_status",
    "Temperature Tympanic Status": "temperature_tympanic_status",
    "Atrial Fibrillation": "atrial_fibrillation",
    "Delirium": "delirium",
    "COPD": "copd",
    "Psychiatric disorders": "psychiatric_disorders",
    "Hip Fracture": "hip_fracture",
    "Dementia": "dementia",
    "SHX Lives with_numerical": "shx_lives_with_numerical",
    "SHx Carer Status_numerical": "s_hx_carer_status_numerical",
    "SHx Home Accommodation Ownership_numerical": "s_hx_home_accommodation_ownership_numerical",
    "AHA_visual impairment": "aha_visual_impairment",
    "is_hearing_impaired": "is_hearing_impaired",
    "Total_ACB_Score": "total_acb_score",
    "Unique_Drug_Count": "unique_drug_count",
}


def main() -> None:
    if not BRONZE_CSV.exists():
        raise SystemExit(f"bronze file not found: {BRONZE_CSV}\n"
                         f"(it is not in the repo — place it there to rebuild cleaned.csv)")

    bronze = pd.read_csv(BRONZE_CSV)

    missing = [c for c in COLUMN_MAP if c not in bronze.columns]
    if missing:
        raise SystemExit(f"bronze is missing expected columns: {missing}")

    df = bronze[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)

    # Derive the pseudonymous patient-cluster ID from Urn BEFORE Urn itself is
    # dropped (Urn is not in COLUMN_MAP, so it never otherwise reaches df).
    if "Urn" in bronze.columns:
        df["patient_cluster_id"] = bronze["Urn"].apply(_hash_patient_id)
    else:
        print("  WARNING: 'Urn' not found in bronze -- patient_cluster_id will be all-NaN "
              "(patient-level grouped CV will not be possible)")
        df["patient_cluster_id"] = np.nan

    # Recode: lives alone (<=1) vs with someone (>1); NaN stays NaN.
    df["shx_lives_with_numerical"] = df["shx_lives_with_numerical"].map(
        lambda v: np.nan if pd.isna(v) else ("alone" if v <= 1 else "someone")
    )

    # Drop rows with no delirium label.
    df = df.dropna(subset=["delirium"])

    df.to_csv(CLEANED_CSV, index=False)
    n_patients = df["patient_cluster_id"].nunique()
    n_repeated = int((df["patient_cluster_id"].value_counts() > 1).sum())
    print(f"OK: Wrote {CLEANED_CSV}  ({len(df):,} episodes, {df.shape[1]} columns)")
    print(f"  delirium cases: {int(df['delirium'].sum()):,} ({100 * df['delirium'].mean():.1f}%)")
    print(f"  unique patients: {n_patients:,}  |  patients with >1 admission: {n_repeated:,}")


if __name__ == "__main__":
    main()
