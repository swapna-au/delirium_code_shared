# %%
"""
SELECTED FEATURES PER FOLD
============================
Reproduces the exact per-fold preprocessing + two-stage feature selection used in
01_cross_validation.py and exports the NAMES of the features selected in each of
the 10 outer folds (the CV script only saved the count, not the names).

Outputs:
  • selected_features_per_fold.csv       (fold x feature indicator matrix)
  • selected_features_frequency.csv      (how many folds each feature was selected in)

Author: [Your name]
Date: July 2026
"""

import numpy as np
import pandas as pd
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from common import FancyIterativeImputerWrapper, OUTPUT_DIR, CLEANED_CSV, RF_BEST_PARAMS, LR_BEST_PARAMS

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier

# ==============================================================================
# CONFIGURATION — must match 01_cross_validation.py
# ==============================================================================

RANDOM_STATE = 42
N_SPLITS = 10
CALIB_TEST_SPLIT = 0.2
setwd = str(OUTPUT_DIR)
output_dir = setwd

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

# ==============================================================================
# LOAD DATA
# ==============================================================================

print("Loading data...")
df = pd.read_csv(CLEANED_CSV)
df.drop(columns=["disch_disposition"], inplace=True, errors="ignore")
df = df.dropna(subset=["delirium"])
X = df[numeric_cols + categorical_cols].copy()
y = df["delirium"].copy()

# ==============================================================================
# BUILD PREPROCESS + SELECTION PIPELINE (identical to 01)
# ==============================================================================

def build_pipeline():
    numeric_transformer = Pipeline([
        ("fancy_imputer", FancyIterativeImputerWrapper(max_iterations=5, n_burn_in=3, random_state=RANDOM_STATE)),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])
    lasso_selector = SelectFromModel(
        LogisticRegressionCV(cv=5, penalty="l1", solver="liblinear",
                             random_state=RANDOM_STATE, max_iter=500))
    rf_selector = SelectFromModel(
        RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE))
    return Pipeline([
        ("preprocessor", preprocessor),
        ("lasso_fs", lasso_selector),
        ("rf_fs", rf_selector),
    ])

# ==============================================================================
# LOOP OVER FOLDS AND RECORD SELECTED FEATURE NAMES
# ==============================================================================

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_features = {}   # fold -> list of selected feature names
fold_num = 0

for train_idx, val_idx in skf.split(X, y):
    fold_num += 1
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]

    # 80/20 split (feature selection is fitted on the 80% model-training split)
    X_model_train, _, y_model_train, _ = train_test_split(
        X_train_fold, y_train_fold, test_size=CALIB_TEST_SPLIT,
        random_state=RANDOM_STATE, stratify=y_train_fold)

    pipe = build_pipeline()
    pipe.fit(X_model_train, y_model_train)

    # Final selected feature names after both selection stages
    names = list(pipe.get_feature_names_out())
    # Tidy the ColumnTransformer prefixes (num__ / cat__)
    names = [n.replace("num__", "").replace("cat__", "") for n in names]
    fold_features[fold_num] = names
    print(f"Fold {fold_num:2d}: {len(names)} features -> {names}")

# ==============================================================================
# BUILD OUTPUT TABLES
# ==============================================================================

# Union of all features ever selected
all_feats = sorted({f for feats in fold_features.values() for f in feats})

# Indicator matrix: rows=features, cols=folds
matrix = pd.DataFrame(index=all_feats, columns=[f"Fold_{i}" for i in range(1, N_SPLITS + 1)])
for i in range(1, N_SPLITS + 1):
    sel = set(fold_features.get(i, []))
    matrix[f"Fold_{i}"] = [1 if f in sel else 0 for f in all_feats]

matrix["Times_selected"] = matrix.sum(axis=1)
matrix = matrix.sort_values("Times_selected", ascending=False)
matrix.index.name = "Feature"

matrix.to_csv(os.path.join(output_dir, "selected_features_per_fold.csv"))
print("\n[OK] Saved: selected_features_per_fold.csv")

freq = matrix[["Times_selected"]].copy()
freq["Selected_in_all_folds"] = (freq["Times_selected"] == N_SPLITS)
freq.to_csv(os.path.join(output_dir, "selected_features_frequency.csv"))
print("[OK] Saved: selected_features_frequency.csv")

print("\n" + "=" * 70)
print("FEATURE SELECTION FREQUENCY ACROSS 10 FOLDS")
print("=" * 70)
print(freq.to_string())

print("\nFeatures selected in ALL 10 folds (stable core):")
core = freq[freq["Times_selected"] == N_SPLITS].index.tolist()
for f in core:
    print(f"  - {f}")

print("\nDone.")
