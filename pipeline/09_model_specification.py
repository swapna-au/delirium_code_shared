# %%
"""
MODEL SPECIFICATION EXTRACTOR
===============================
Documents the FULL specification of the final models so they can be reported
(TRIPOD item 15) and re-applied / externally validated by others.

Extracts and saves:
  • Preprocessing, feature-selection, and hyperparameter configuration
  • The selected feature set
  • Logistic Regression: intercept + coefficients + odds ratios (per SD)
  • Random Forest: feature importances + hyperparameters
  • Platt scaling parameters (a, b) for each calibrated model
  • Environment (sklearn / numpy versions, random seed)
  • Serialised fitted pipelines (joblib .pkl) for external validation/deployment

Outputs:
  • model_specification.md
  • model_spec_lr_coefficients.csv
  • model_spec_rf_importances.csv
  • model_spec_config.csv
  • final_model_rf.pkl / final_model_lr.pkl   (full preprocess+select+model+calibrator)

Run AFTER 02_bootstrap_validation.py (uses the same best hyperparameters).

Author: [Your name]
Date: July 2026
"""

import numpy as np
import pandas as pd
import os, sys, sklearn, joblib

sys.path.insert(0, os.path.dirname(__file__))
from common import FancyIterativeImputerWrapper, OUTPUT_DIR, CLEANED_CSV, RF_BEST_PARAMS, LR_BEST_PARAMS

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# ==============================================================================
# CONFIGURATION — must match the bootstrap / pipeline files
# ==============================================================================

RANDOM_STATE = 42
setwd = str(OUTPUT_DIR)
output_dir = setwd

MICE_MAX_ITER  = 5
LASSO_CV       = 5
RF_SELECTOR_N  = 50

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
# LOAD + PREPROCESS + FIT (identical recipe to the pipeline)
# ==============================================================================

print("Loading data...")
df = pd.read_csv(CLEANED_CSV)
df.drop(columns=["disch_disposition"], inplace=True, errors="ignore")
df = df.dropna(subset=["delirium"])
X = df[numeric_cols + categorical_cols].copy()
y = df["delirium"].values

numeric_transformer = Pipeline([
    ("fancy_imputer", FancyIterativeImputerWrapper(max_iterations=MICE_MAX_ITER, n_burn_in=3, random_state=RANDOM_STATE)),
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
    LogisticRegressionCV(cv=LASSO_CV, penalty="l1", solver="liblinear",
                         random_state=RANDOM_STATE, max_iter=500))
rf_selector = SelectFromModel(
    RandomForestClassifier(n_estimators=RF_SELECTOR_N, random_state=RANDOM_STATE))
preproc_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("lasso_fs", lasso_selector),
    ("rf_fs", rf_selector),
])

print("Preprocessing (MICE + feature selection)...")
X_pp = preproc_pipeline.fit_transform(X, y)
feature_names = [n.replace("num__", "").replace("cat__", "")
                 for n in preproc_pipeline.get_feature_names_out()]
print(f"Selected features ({len(feature_names)}): {feature_names}")

idx_all = np.arange(len(y))
idx_train, idx_calib = train_test_split(
    idx_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

rf = RandomForestClassifier(**RF_BEST_PARAMS)
rf.fit(X_pp[idx_train], y[idx_train])
rf_cal = CalibratedClassifierCV(rf, method='sigmoid', cv='prefit')
rf_cal.fit(X_pp[idx_calib], y[idx_calib])

lr = LogisticRegression(**LR_BEST_PARAMS)
lr.fit(X_pp[idx_train], y[idx_train])
lr_cal = CalibratedClassifierCV(lr, method='sigmoid', cv='prefit')
lr_cal.fit(X_pp[idx_calib], y[idx_calib])

# ==============================================================================
# LOGISTIC REGRESSION SPECIFICATION (interpretable equation)
# ==============================================================================

lr_coef = lr.coef_.ravel()
lr_intercept = float(lr.intercept_[0])
lr_spec = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient (log-odds, per SD)': np.round(lr_coef, 4),
    'Odds Ratio (per SD)': np.round(np.exp(lr_coef), 4),
})
lr_spec = lr_spec[lr_spec['Coefficient (log-odds, per SD)'] != 0]  # L1 may zero some out
lr_spec = lr_spec.sort_values('Coefficient (log-odds, per SD)', key=abs, ascending=False)
lr_spec.loc[len(lr_spec)] = ['(Intercept)', round(lr_intercept, 4), np.nan]
lr_spec.to_csv(os.path.join(output_dir, 'model_spec_lr_coefficients.csv'), index=False)
print("\n[OK] Saved: model_spec_lr_coefficients.csv")

# ==============================================================================
# RANDOM FOREST SPECIFICATION (importances)
# ==============================================================================

rf_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': np.round(rf.feature_importances_, 4),
}).sort_values('Importance', ascending=False)
rf_imp.to_csv(os.path.join(output_dir, 'model_spec_rf_importances.csv'), index=False)
print("[OK] Saved: model_spec_rf_importances.csv")

# ==============================================================================
# PLATT SCALING PARAMETERS
# ==============================================================================

def platt_params(cal_model):
    """Extract sigmoid (Platt) a, b across sklearn versions."""
    try:
        cc = cal_model.calibrated_classifiers_[0]
        # attribute name varies by version: 'calibrators' or 'calibrators_'
        cals = getattr(cc, 'calibrators', None) or getattr(cc, 'calibrators_', None)
        sig = cals[0]
        return float(getattr(sig, 'a_', np.nan)), float(getattr(sig, 'b_', np.nan))
    except Exception as e:
        print(f"  (could not extract Platt params: {e})")
        return np.nan, np.nan

rf_a, rf_b = platt_params(rf_cal)
lr_a, lr_b = platt_params(lr_cal)

# ==============================================================================
# CONFIGURATION TABLE
# ==============================================================================

config = pd.DataFrame({
    'Component': [
        'Numeric imputation', 'Numeric scaling', 'Categorical imputation',
        'Categorical encoding', 'Feature selection stage 1', 'Feature selection stage 2',
        'RF hyperparameters', 'LR hyperparameters', 'Probability calibration',
        'Train/calibration split', 'Random seed', 'scikit-learn version', 'numpy version',
        'N selected features',
    ],
    'Specification': [
        f'MICE (IterativeImputer), max_iter={MICE_MAX_ITER}',
        'StandardScaler (zero mean, unit variance)',
        'Most frequent value',
        'One-hot (drop first, ignore unknown)',
        f'L1 logistic regression (LogisticRegressionCV, cv={LASSO_CV})',
        f'Random forest importance (SelectFromModel, n_estimators={RF_SELECTOR_N})',
        str(RF_BEST_PARAMS), str(LR_BEST_PARAMS),
        "Platt scaling (sigmoid), fitted on held-out 20%",
        '80% model training / 20% calibration (stratified)',
        str(RANDOM_STATE), sklearn.__version__, np.__version__,
        str(len(feature_names)),
    ],
})
config.to_csv(os.path.join(output_dir, 'model_spec_config.csv'), index=False)
print("[OK] Saved: model_spec_config.csv")

# ==============================================================================
# SERIALISE FULL FITTED PIPELINES (preprocess + select + model + calibrator)
# ==============================================================================

rf_full = Pipeline([("preprocess", preproc_pipeline), ("calibrated_rf", rf_cal)])
lr_full = Pipeline([("preprocess", preproc_pipeline), ("calibrated_lr", lr_cal)])
joblib.dump(rf_full, os.path.join(output_dir, 'final_model_rf.pkl'))
joblib.dump(lr_full, os.path.join(output_dir, 'final_model_lr.pkl'))
print("[OK] Saved: final_model_rf.pkl, final_model_lr.pkl")

# ==============================================================================
# MARKDOWN SUMMARY
# ==============================================================================

lines = ["# Final Model Specification\n",
         "## Development configuration\n"]
for _, r in config.iterrows():
    lines.append(f"- **{r['Component']}**: {r['Specification']}")
lines.append("\n## Selected features\n")
for f in feature_names:
    lines.append(f"- {f}")
lines.append("\n## Logistic Regression — coefficients (standardised, per SD)\n")
lines.append("| Feature | Coefficient | Odds Ratio |")
lines.append("|---|---|---|")
for _, r in lr_spec.iterrows():
    orv = "" if pd.isna(r['Odds Ratio (per SD)']) else f"{r['Odds Ratio (per SD)']:.3f}"
    lines.append(f"| {r['Feature']} | {r['Coefficient (log-odds, per SD)']:.4f} | {orv} |")
lines.append(f"\nPlatt scaling (LR): a={lr_a:.4f}, b={lr_b:.4f}")
lines.append("\n## Random Forest — feature importances\n")
lines.append("| Feature | Importance |")
lines.append("|---|---|")
for _, r in rf_imp.iterrows():
    lines.append(f"| {r['Feature']} | {r['Importance']:.4f} |")
lines.append(f"\nPlatt scaling (RF): a={rf_a:.4f}, b={rf_b:.4f}")
lines.append("\n## Applying the model\n")
lines.append("- `final_model_rf.pkl` / `final_model_lr.pkl` contain the complete pipeline "
             "(preprocessing + feature selection + model + Platt calibration). Load with "
             "`joblib.load(...)` and call `.predict_proba(X_raw)[:, 1]` on raw feature data "
             "with the same columns.")
lines.append("- Coefficients are on STANDARDISED predictors (per 1 SD); to obtain a raw-scale "
             "equation, back-transform using the StandardScaler means/SDs stored in the pipeline.")

with open(os.path.join(output_dir, 'model_specification.md'), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("[OK] Saved: model_specification.md")

print("\n" + "="*70)
print("MODEL SPECIFICATION")
print("="*70)
print(config.to_string(index=False))
print("\nLogistic Regression coefficients:")
print(lr_spec.to_string(index=False))
print("\nRandom Forest importances:")
print(rf_imp.to_string(index=False))
print(f"\nPlatt scaling — RF: a={rf_a:.4f}, b={rf_b:.4f} | LR: a={lr_a:.4f}, b={lr_b:.4f}")
print("\nDone.")
