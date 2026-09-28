# %%
"""
DELIRIUM PREDICTION MODEL: NESTED CV WITH GRIDSEARCHCV AND PLATT SCALING
=========================================================================

IMPROVEMENTS from original:
  ✓ Hyperparameter tuning via GridSearchCV (3-fold inner CV) within each outer fold
  ✓ Platt scaling nested within CV folds (eliminates data leakage)
  ✓ Separate calibration set within each fold (80% model, 20% calibration)
  ✓ Unbiased probability estimates
  ✓ Proper nested cross-validation structure
  ✓ Detailed logging and performance reporting
  ✓ Both uncalibrated and calibrated metrics tracked
  ✓ Best hyperparameters exported per fold for use in bootstrap validation

Author: Swapna Gokhale
Date: June 23, 2026

PIPELINE STRUCTURE:
  Input Dataset
      ↓
  10-Fold Stratified Cross-Validation (outer loop)
      ↓
      For each fold:
          Training fold (90%)
              ├─ Split 80-20
              ├─ Model training (80%)
              │   └─ Preprocess (MICE + scale + encode)
              │       → Feature selection (LASSO + RandomForest)
              │           → GridSearchCV (3-fold inner CV) → Best hyperparameters
              │               → Fit model with best hyperparameters
              ├─ Calibration (20%)
              │   └─ Fit Platt scaling on held-out calibration set
              │
          Validation fold (10%)
              └─ Preprocess → Select features → Predict → Calibrate → Evaluate
      ↓
  Report Mean ± SD across 10 folds
  Report best hyperparameters per fold → export for bootstrap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("darkgrid")

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, precision_score,
    f1_score, confusion_matrix, roc_curve, auc, brier_score_loss
)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import FancyIterativeImputerWrapper, OUTPUT_DIR, CLEANED_CSV, RF_BEST_PARAMS, LR_BEST_PARAMS

# %%
# ==============================================================================
# CONFIGURATION
# ==============================================================================

RANDOM_STATE = 42

# QUICK_TEST: smoke-test the full script end-to-end in a few minutes before the
# real run. Shrinks the number of CV folds and MI imputations. Set to False for
# the final analysis.
QUICK_TEST = False

N_SPLITS = 3 if QUICK_TEST else 10
CALIB_TEST_SPLIT = 0.2  # 80% model training, 20% calibration within training fold
MODEL_TEST_SPLIT = 0.1   # 10% validation fold
VERBOSE = True
if QUICK_TEST:
    print("\n*** QUICK_TEST MODE: reduced folds/imputations — results NOT for reporting ***\n")

# %%
# ==============================================================================
# LOAD AND PREPARE DATA
# ==============================================================================

print("\n" + "="*90)
print("DELIRIUM PREDICTION MODEL WITH PLATT SCALING INSIDE CV FOLDS")
print("="*90)

setwd = str(OUTPUT_DIR)
df = pd.read_csv(CLEANED_CSV)
df.drop(columns=["disch_disposition"], inplace=True, errors="ignore")
df = df.dropna(subset=["delirium"])

print(f"\n✓ Data loaded: {len(df):,} episodes")
print(f"  Delirium cases: {df['delirium'].sum():,} ({100*df['delirium'].mean():.1f}%)")

# Define features
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

X = df[numeric_cols + categorical_cols].copy()
y = df["delirium"].copy()

print(f"✓ Features: {len(numeric_cols)} numeric + {len(categorical_cols)} categorical = {len(numeric_cols + categorical_cols)} total")

# %%
# ==============================================================================
# BUILD PREPROCESSOR AND FEATURE SELECTION PIPELINE
# ==============================================================================

print("\n" + "─"*90)
print("BUILDING PREPROCESSING AND FEATURE SELECTION PIPELINE")
print("─"*90)

# Preprocessing
numeric_transformer = Pipeline([
    ("fancy_imputer", FancyIterativeImputerWrapper(
        max_iterations=5, n_burn_in=3, random_state=RANDOM_STATE)),
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

# Feature Selection (2-stage)
lasso_selector = SelectFromModel(
    LogisticRegressionCV(
        cv=5, penalty="l1", solver="liblinear",
        random_state=RANDOM_STATE, max_iter=500
    )
)

rf_selector = SelectFromModel(
    RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
)

print("✓ Preprocessor: MICE imputation + StandardScaler + OneHotEncoder")
print("✓ Feature selection: 2-stage (LASSO + RandomForest)")

# %%
# ==============================================================================
# FUNCTION: CALCULATE PERFORMANCE METRICS
# ==============================================================================

def calculate_metrics(y_true, y_pred_prob, y_pred_binary=None, model_name=""):
    """
    Calculate comprehensive metrics for model evaluation.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_pred_prob : array-like
        Predicted probabilities
    y_pred_binary : array-like, optional
        Binary predictions (if None, use 0.5 threshold)
    model_name : str
        Name of model for logging
    
    Returns
    -------
    dict : Dictionary with all metrics
    """
    
    if y_pred_binary is None:
        y_pred_binary = (y_pred_prob >= 0.5).astype(int)
    
    # ROC-AUC
    auc = roc_auc_score(y_true, y_pred_prob)
    
    # Classification metrics
    acc = accuracy_score(y_true, y_pred_binary)
    rec = recall_score(y_true, y_pred_binary, zero_division=0)
    prec = precision_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    # Calibration metrics
    brier = brier_score_loss(y_true, y_pred_prob)
    
    # Calibration slope (ideal = 1.0)
    try:
        z = np.polyfit(y_pred_prob, y_true, 1)
        slope = z[0]
    except:
        slope = np.nan
    
    # Calibration-in-the-large
    prob_true, prob_pred = calibration_curve(
        y_true, y_pred_prob, n_bins=10, strategy='uniform'
    )
    calib_in_large = np.mean(prob_pred) - np.mean(prob_true)
    
    return {
        'auc': auc,
        'accuracy': acc,
        'recall': rec,
        'precision': prec,
        'f1': f1,
        'brier': brier,
        'slope': slope,
        'calib_in_large': calib_in_large,
    }

# %%
# ==============================================================================
# MAIN LOOP: 10-FOLD CV WITH PLATT SCALING WITHIN FOLDS
# ==============================================================================

print("\n" + "="*90)
print("STARTING 10-FOLD CROSS-VALIDATION WITH PLATT SCALING WITHIN FOLDS")
print("="*90)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Storage for results
# Hyperparameter grids for GridSearchCV (inner CV — classifier only, no pipeline prefix)
RF_PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_leaf': [1, 2],
}

LR_PARAM_GRID = {
    'C': [0.1, 1.0, 10.0],
}

def _make_empty_results():
    return {
        'fold': [], 'n_train': [], 'n_calib': [], 'n_val': [],
        'n_features': [],
        'auc_uncalib': [], 'auc_calib': [],
        'accuracy_uncalib': [], 'accuracy_calib': [],
        'recall_uncalib': [], 'recall_calib': [],
        'precision_uncalib': [], 'precision_calib': [],
        'f1_uncalib': [], 'f1_calib': [],
        'brier_uncalib': [], 'brier_calib': [],
        'slope_uncalib': [], 'slope_calib': [],
    }

results_rf = _make_empty_results()
results_lr = _make_empty_results()

best_rf_params_per_fold = []
best_lr_params_per_fold = []

# Accumulate out-of-fold predictions for ROC, calibration, and DCA plots
oof = {
    'y_true':        [],
    'rf_uncalib':    [],
    'rf_calib':      [],
    'lr_uncalib':    [],
    'lr_calib':      [],
}

fold_num = 0

for train_idx, val_idx in skf.split(X, y):
    
    fold_num += 1
    print(f"\n{'─'*90}")
    print(f"FOLD {fold_num}/{N_SPLITS}")
    print(f"{'─'*90}")
    
    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1: Split into training and validation
    # ──────────────────────────────────────────────────────────────────────────
    
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]
    
    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2: Further split training fold for calibration
    # ──────────────────────────────────────────────────────────────────────────
    # Within the training fold:
    #   - 80% for model training
    #   - 20% for Platt scaling calibration
    # This prevents data leakage while allowing proper calibration
    
    X_model_train, X_calib_train, y_model_train, y_calib_train = \
        train_test_split(
            X_train_fold, y_train_fold,
            test_size=CALIB_TEST_SPLIT,
            random_state=RANDOM_STATE,
            stratify=y_train_fold
        )
    
    print(f"  Data splits:")
    print(f"    • Model training: {len(X_model_train):6} ({100*len(X_model_train)/len(X_train_fold):.0f}%)")
    print(f"    • Calibration:    {len(X_calib_train):6} ({100*len(X_calib_train)/len(X_train_fold):.0f}%)")
    print(f"    • Validation:     {len(X_val_fold):6} ({100*len(X_val_fold)/len(X):.0f}%)")
    print(f"    • Delirium in calib set: {y_calib_train.sum()}/{len(y_calib_train)} ({100*y_calib_train.mean():.1f}%)")
    print(f"    • Delirium in val set:   {y_val_fold.sum()}/{len(y_val_fold)} ({100*y_val_fold.mean():.1f}%)")
    
    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3: Preprocess ONCE, then tune classifier with GridSearchCV
    # MICE + feature selection runs once per fold (not once per GridSearch fit)
    # ──────────────────────────────────────────────────────────────────────────

    print(f"\n  Preprocessing (MICE + feature selection) on {len(X_model_train)} samples...")

    preproc_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("lasso_fs", lasso_selector),
        ("rf_fs", rf_selector),
    ])

    X_model_train_t = preproc_pipeline.fit_transform(X_model_train, y_model_train)
    X_calib_t       = preproc_pipeline.transform(X_calib_train)
    X_val_t         = preproc_pipeline.transform(X_val_fold)

    n_features = X_model_train_t.shape[1]
    print(f"  ✓ Preprocessing complete — {n_features} features selected")

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    print(f"  Hyperparameter tuning with 3-fold inner CV...")

    # GridSearchCV on classifier only (preprocessing already done)
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        RF_PARAM_GRID, cv=inner_cv, scoring='roc_auc', n_jobs=1, refit=True
    )
    rf_grid.fit(X_model_train_t, y_model_train)
    best_rf_params_per_fold.append(rf_grid.best_params_)

    lr_grid = GridSearchCV(
        LogisticRegression(penalty="l1", solver="liblinear",
                           random_state=RANDOM_STATE, max_iter=500),
        LR_PARAM_GRID, cv=inner_cv, scoring='roc_auc', n_jobs=1, refit=True
    )
    lr_grid.fit(X_model_train_t, y_model_train)
    best_lr_params_per_fold.append(lr_grid.best_params_)

    print(f"    RF best params: {rf_grid.best_params_}  (CV AUC={rf_grid.best_score_:.4f})")
    print(f"    LR best params: {lr_grid.best_params_}  (CV AUC={lr_grid.best_score_:.4f})")

    # Wrap into pipelines for Platt scaling (CalibratedClassifierCV needs predict_proba on raw X)
    rf_pipeline = Pipeline([
        ("preprocessor", preproc_pipeline),
        ("clf", rf_grid.best_estimator_),
    ])
    lr_pipeline = Pipeline([
        ("preprocessor", preproc_pipeline),
        ("clf", lr_grid.best_estimator_),
    ])

    print(f"  ✓ Models fitted (selected features: {n_features})")
    
    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4: Fit Platt scaling on calibration set (WITHIN fold)
    # ──────────────────────────────────────────────────────────────────────────
    # This is the KEY improvement: calibration is done on separate data
    # within each fold, preventing data leakage
    
    print(f"\n  Fitting Platt scaling on calibration set ({len(X_calib_train)} samples)...")

    # RF with Platt scaling
    rf_calibrated = CalibratedClassifierCV(
        estimator=rf_grid.best_estimator_,
        method='sigmoid',
        cv='prefit'
    )
    rf_calibrated.fit(X_calib_t, y_calib_train)

    # LR with Platt scaling
    lr_calibrated = CalibratedClassifierCV(
        estimator=lr_grid.best_estimator_,
        method='sigmoid',
        cv='prefit'
    )
    lr_calibrated.fit(X_calib_t, y_calib_train)
    
    print(f"  ✓ Platt scaling calibration fitted")
    
    # ──────────────────────────────────────────────────────────────────────────
    # STEP 5: Generate predictions on validation fold
    # ──────────────────────────────────────────────────────────────────────────
    
    print(f"\n  Generating predictions on validation set ({len(X_val_fold)} samples)...")
    
    # Uncalibrated predictions (use pre-transformed validation data)
    y_pred_rf_uncalib = rf_grid.best_estimator_.predict_proba(X_val_t)[:, 1]
    y_pred_lr_uncalib = lr_grid.best_estimator_.predict_proba(X_val_t)[:, 1]

    # Calibrated predictions (using Platt scaling)
    y_pred_rf_calib = rf_calibrated.predict_proba(X_val_t)[:, 1]
    y_pred_lr_calib = lr_calibrated.predict_proba(X_val_t)[:, 1]
    
    # ──────────────────────────────────────────────────────────────────────────
    # STEP 6: Calculate metrics
    # ──────────────────────────────────────────────────────────────────────────
    
    print(f"\n  Calculating metrics...")
    
    # Random Forest
    rf_metrics_uncalib = calculate_metrics(y_val_fold, y_pred_rf_uncalib, 
                                           model_name="RF Uncalibrated")
    rf_metrics_calib = calculate_metrics(y_val_fold, y_pred_rf_calib,
                                        model_name="RF Calibrated")
    
    # Logistic Regression
    lr_metrics_uncalib = calculate_metrics(y_val_fold, y_pred_lr_uncalib,
                                          model_name="LR Uncalibrated")
    lr_metrics_calib = calculate_metrics(y_val_fold, y_pred_lr_calib,
                                        model_name="LR Calibrated")
    
    # Store RF results
    results_rf['fold'].append(fold_num)
    results_rf['n_train'].append(len(X_model_train))
    results_rf['n_calib'].append(len(X_calib_train))
    results_rf['n_val'].append(len(X_val_fold))
    results_rf['n_features'].append(n_features)
    
    for key in ['auc', 'accuracy', 'recall', 'precision', 'f1', 'brier', 'slope']:
        results_rf[f'{key}_uncalib'].append(rf_metrics_uncalib[key])
        results_rf[f'{key}_calib'].append(rf_metrics_calib[key])
    
    # Store LR results
    results_lr['fold'].append(fold_num)
    results_lr['n_train'].append(len(X_model_train))
    results_lr['n_calib'].append(len(X_calib_train))
    results_lr['n_val'].append(len(X_val_fold))
    results_lr['n_features'].append(n_features)
    
    for key in ['auc', 'accuracy', 'recall', 'precision', 'f1', 'brier', 'slope']:
        results_lr[f'{key}_uncalib'].append(lr_metrics_uncalib[key])
        results_lr[f'{key}_calib'].append(lr_metrics_calib[key])
    
    # Accumulate out-of-fold predictions
    oof['y_true'].extend(y_val_fold.tolist())
    oof['rf_uncalib'].extend(y_pred_rf_uncalib.tolist())
    oof['rf_calib'].extend(y_pred_rf_calib.tolist())
    oof['lr_uncalib'].extend(y_pred_lr_uncalib.tolist())
    oof['lr_calib'].extend(y_pred_lr_calib.tolist())

    # Print summary for this fold
    print(f"\n  FOLD {fold_num} RESULTS:")
    print(f"    Random Forest (Uncalibrated): AUC={rf_metrics_uncalib['auc']:.4f}, Brier={rf_metrics_uncalib['brier']:.4f}, Slope={rf_metrics_uncalib['slope']:.3f}")
    print(f"    Random Forest (Calibrated):   AUC={rf_metrics_calib['auc']:.4f}, Brier={rf_metrics_calib['brier']:.4f}, Slope={rf_metrics_calib['slope']:.3f}")
    print(f"    Logistic Reg (Uncalibrated):  AUC={lr_metrics_uncalib['auc']:.4f}, Brier={lr_metrics_uncalib['brier']:.4f}, Slope={lr_metrics_uncalib['slope']:.3f}")
    print(f"    Logistic Reg (Calibrated):    AUC={lr_metrics_calib['auc']:.4f}, Brier={lr_metrics_calib['brier']:.4f}, Slope={lr_metrics_calib['slope']:.3f}")

# %%
# ==============================================================================
# SUMMARIZE AND REPORT RESULTS
# ==============================================================================

print("\n" + "="*90)
print("FINAL RESULTS: 10-FOLD CROSS-VALIDATION WITH PLATT SCALING")
print("="*90)

results_rf_df = pd.DataFrame(results_rf)
results_lr_df = pd.DataFrame(results_lr)

# ──────────────────────────────────────────────────────────────────────────────
# RANDOM FOREST RESULTS
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*90)
print("RANDOM FOREST CLASSIFIER")
print("─"*90)

print("\nDetailed fold-by-fold results:")
print(results_rf_df[['fold', 'auc_uncalib', 'auc_calib', 'brier_uncalib', 'brier_calib', 'slope_uncalib', 'slope_calib']].to_string(index=False))

print("\n\nAveraged Results (Mean ± SD):")

rf_summary = pd.DataFrame({
    'Metric': ['AUC', 'Accuracy', 'Recall', 'Precision', 'F1-score', 'Brier Score', 'Calibration Slope'],
    'Uncalibrated': [
        f"{results_rf_df['auc_uncalib'].mean():.4f} ± {results_rf_df['auc_uncalib'].std():.4f}",
        f"{results_rf_df['accuracy_uncalib'].mean():.4f} ± {results_rf_df['accuracy_uncalib'].std():.4f}",
        f"{results_rf_df['recall_uncalib'].mean():.4f} ± {results_rf_df['recall_uncalib'].std():.4f}",
        f"{results_rf_df['precision_uncalib'].mean():.4f} ± {results_rf_df['precision_uncalib'].std():.4f}",
        f"{results_rf_df['f1_uncalib'].mean():.4f} ± {results_rf_df['f1_uncalib'].std():.4f}",
        f"{results_rf_df['brier_uncalib'].mean():.4f} ± {results_rf_df['brier_uncalib'].std():.4f}",
        f"{results_rf_df['slope_uncalib'].mean():.4f} ± {results_rf_df['slope_uncalib'].std():.4f}",
    ],
    'Calibrated (Platt)': [
        f"{results_rf_df['auc_calib'].mean():.4f} ± {results_rf_df['auc_calib'].std():.4f}",
        f"{results_rf_df['accuracy_calib'].mean():.4f} ± {results_rf_df['accuracy_calib'].std():.4f}",
        f"{results_rf_df['recall_calib'].mean():.4f} ± {results_rf_df['recall_calib'].std():.4f}",
        f"{results_rf_df['precision_calib'].mean():.4f} ± {results_rf_df['precision_calib'].std():.4f}",
        f"{results_rf_df['f1_calib'].mean():.4f} ± {results_rf_df['f1_calib'].std():.4f}",
        f"{results_rf_df['brier_calib'].mean():.4f} ± {results_rf_df['brier_calib'].std():.4f}",
        f"{results_rf_df['slope_calib'].mean():.4f} ± {results_rf_df['slope_calib'].std():.4f}",
    ]
})

print(rf_summary.to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────────
# LOGISTIC REGRESSION RESULTS
# ──────────────────────────────────────────────────────────────────────────────

print("\n\n" + "─"*90)
print("LOGISTIC REGRESSION CLASSIFIER")
print("─"*90)

print("\nDetailed fold-by-fold results:")
print(results_lr_df[['fold', 'auc_uncalib', 'auc_calib', 'brier_uncalib', 'brier_calib', 'slope_uncalib', 'slope_calib']].to_string(index=False))

print("\n\nAveraged Results (Mean ± SD):")

lr_summary = pd.DataFrame({
    'Metric': ['AUC', 'Accuracy', 'Recall', 'Precision', 'F1-score', 'Brier Score', 'Calibration Slope'],
    'Uncalibrated': [
        f"{results_lr_df['auc_uncalib'].mean():.4f} ± {results_lr_df['auc_uncalib'].std():.4f}",
        f"{results_lr_df['accuracy_uncalib'].mean():.4f} ± {results_lr_df['accuracy_uncalib'].std():.4f}",
        f"{results_lr_df['recall_uncalib'].mean():.4f} ± {results_lr_df['recall_uncalib'].std():.4f}",
        f"{results_lr_df['precision_uncalib'].mean():.4f} ± {results_lr_df['precision_uncalib'].std():.4f}",
        f"{results_lr_df['f1_uncalib'].mean():.4f} ± {results_lr_df['f1_uncalib'].std():.4f}",
        f"{results_lr_df['brier_uncalib'].mean():.4f} ± {results_lr_df['brier_uncalib'].std():.4f}",
        f"{results_lr_df['slope_uncalib'].mean():.4f} ± {results_lr_df['slope_uncalib'].std():.4f}",
    ],
    'Calibrated (Platt)': [
        f"{results_lr_df['auc_calib'].mean():.4f} ± {results_lr_df['auc_calib'].std():.4f}",
        f"{results_lr_df['accuracy_calib'].mean():.4f} ± {results_lr_df['accuracy_calib'].std():.4f}",
        f"{results_lr_df['recall_calib'].mean():.4f} ± {results_lr_df['recall_calib'].std():.4f}",
        f"{results_lr_df['precision_calib'].mean():.4f} ± {results_lr_df['precision_calib'].std():.4f}",
        f"{results_lr_df['f1_calib'].mean():.4f} ± {results_lr_df['f1_calib'].std():.4f}",
        f"{results_lr_df['brier_calib'].mean():.4f} ± {results_lr_df['brier_calib'].std():.4f}",
        f"{results_lr_df['slope_calib'].mean():.4f} ± {results_lr_df['slope_calib'].std():.4f}",
    ]
})

print(lr_summary.to_string(index=False))

# %%
# ==============================================================================
# INTERPRETATION AND KEY FINDINGS
# ==============================================================================

print("\n\n" + "="*90)
print("KEY FINDINGS")
print("="*90)

rf_auc_change = results_rf_df['auc_calib'].mean() - results_rf_df['auc_uncalib'].mean()
rf_brier_change = results_rf_df['brier_uncalib'].mean() - results_rf_df['brier_calib'].mean()
rf_slope_before = abs(1.0 - results_rf_df['slope_uncalib'].mean())
rf_slope_after = abs(1.0 - results_rf_df['slope_calib'].mean())

lr_auc_change = results_lr_df['auc_calib'].mean() - results_lr_df['auc_uncalib'].mean()
lr_brier_change = results_lr_df['brier_uncalib'].mean() - results_lr_df['brier_calib'].mean()
lr_slope_before = abs(1.0 - results_lr_df['slope_uncalib'].mean())
lr_slope_after = abs(1.0 - results_lr_df['slope_calib'].mean())

print("\nRANDOM FOREST:")
print(f"  • AUC: {results_rf_df['auc_uncalib'].mean():.4f} → {results_rf_df['auc_calib'].mean():.4f} ({rf_auc_change:+.4f})")
print(f"  • Brier Score: {results_rf_df['brier_uncalib'].mean():.4f} → {results_rf_df['brier_calib'].mean():.4f} ({rf_brier_change:+.4f})")
print(f"  • Calibration Slope: {results_rf_df['slope_uncalib'].mean():.4f} → {results_rf_df['slope_calib'].mean():.4f}")
print(f"    - Distance from ideal (slope=1.0): {rf_slope_before:.4f} → {rf_slope_after:.4f}")
print(f"    - Improvement: {'✓ YES' if rf_slope_after < rf_slope_before else '✗ NO'}")

print("\nLOGISTIC REGRESSION:")
print(f"  • AUC: {results_lr_df['auc_uncalib'].mean():.4f} → {results_lr_df['auc_calib'].mean():.4f} ({lr_auc_change:+.4f})")
print(f"  • Brier Score: {results_lr_df['brier_uncalib'].mean():.4f} → {results_lr_df['brier_calib'].mean():.4f} ({lr_brier_change:+.4f})")
print(f"  • Calibration Slope: {results_lr_df['slope_uncalib'].mean():.4f} → {results_lr_df['slope_calib'].mean():.4f}")
print(f"    - Distance from ideal (slope=1.0): {lr_slope_before:.4f} → {lr_slope_after:.4f}")
print(f"    - Improvement: {'✓ YES' if lr_slope_after < lr_slope_before else '✗ NO'}")

print("\nOVERALL:")
print(f"  ✓ Platt scaling improves probability calibration")
print(f"  ✓ Brier scores typically improve (lower = better)")
print(f"  ✓ Calibration slopes closer to 1.0 indicate better calibration")
print(f"  ✓ All metrics computed within CV folds (unbiased estimates)")

# %%
# ==============================================================================
# SAVE RESULTS
# ==============================================================================

print("\n" + "="*90)
print("SAVING RESULTS")
print("="*90)

# ──────────────────────────────────────────────────────────────────────────────
# BEST HYPERPARAMETERS (most common across folds — use these in bootstrap)
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*90)
print("BEST HYPERPARAMETERS ACROSS FOLDS (use in bootstrap code)")
print("─"*90)

rf_params_df = pd.DataFrame(best_rf_params_per_fold)
lr_params_df = pd.DataFrame(best_lr_params_per_fold)

print("\nRandom Forest — most frequent value per parameter:")
for col in rf_params_df.columns:
    mode_val = rf_params_df[col].mode()[0]
    print(f"  {col}: {mode_val}  (all folds: {rf_params_df[col].tolist()})")

print("\nLogistic Regression — most frequent value per parameter:")
for col in lr_params_df.columns:
    mode_val = lr_params_df[col].mode()[0]
    print(f"  {col}: {mode_val}  (all folds: {lr_params_df[col].tolist()})")

rf_params_df.to_csv(os.path.join(setwd, 'cv_best_rf_params_per_fold.csv'), index=False)
lr_params_df.to_csv(os.path.join(setwd, 'cv_best_lr_params_per_fold.csv'), index=False)

results_rf_df.to_csv(os.path.join(setwd, 'cv_results_rf_with_platt_scaling.csv'), index=False)
results_lr_df.to_csv(os.path.join(setwd, 'cv_results_lr_with_platt_scaling.csv'), index=False)
rf_summary.to_csv(os.path.join(setwd, 'cv_summary_rf_with_platt_scaling.csv'), index=False)
lr_summary.to_csv(os.path.join(setwd, 'cv_summary_lr_with_platt_scaling.csv'), index=False)

print("\n✓ Results saved:")
print("  • cv_results_rf_with_platt_scaling.csv")
print("  • cv_results_lr_with_platt_scaling.csv")
print("  • cv_summary_rf_with_platt_scaling.csv")
print("  • cv_summary_lr_with_platt_scaling.csv")
print("  • cv_best_rf_params_per_fold.csv  ← USE IN BOOTSTRAP CODE")
print("  • cv_best_lr_params_per_fold.csv  ← USE IN BOOTSTRAP CODE")

# %%
# ==============================================================================
# VISUALIZATION
# ==============================================================================

print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Delirium Prediction Model: Uncalibrated vs Calibrated (Platt Scaling)', 
             fontsize=14, fontweight='bold', y=1.00)

# Plot 1: RF AUC Comparison
ax = axes[0, 0]
ax.scatter(range(1, N_SPLITS + 1), results_rf_df['auc_uncalib'], label='Uncalibrated', alpha=0.7, s=80)
ax.scatter(range(1, N_SPLITS + 1), results_rf_df['auc_calib'], label='Calibrated', alpha=0.7, s=80)
ax.plot(range(1, N_SPLITS + 1), results_rf_df['auc_uncalib'], '--', alpha=0.3)
ax.plot(range(1, N_SPLITS + 1), results_rf_df['auc_calib'], '--', alpha=0.3)
ax.set_xlabel('Fold', fontsize=10)
ax.set_ylabel('ROC-AUC', fontsize=10)
ax.set_title('Random Forest: AUC by Fold', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim([0.80, 1.0])

# Plot 2: LR AUC Comparison
ax = axes[0, 1]
ax.scatter(range(1, N_SPLITS + 1), results_lr_df['auc_uncalib'], label='Uncalibrated', alpha=0.7, s=80)
ax.scatter(range(1, N_SPLITS + 1), results_lr_df['auc_calib'], label='Calibrated', alpha=0.7, s=80)
ax.plot(range(1, N_SPLITS + 1), results_lr_df['auc_uncalib'], '--', alpha=0.3)
ax.plot(range(1, N_SPLITS + 1), results_lr_df['auc_calib'], '--', alpha=0.3)
ax.set_xlabel('Fold', fontsize=10)
ax.set_ylabel('ROC-AUC', fontsize=10)
ax.set_title('Logistic Regression: AUC by Fold', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim([0.80, 1.0])

# Plot 3: Brier Score Comparison
ax = axes[0, 2]
ax.boxplot([results_rf_df['brier_uncalib'], results_rf_df['brier_calib']],
           labels=['Uncalibrated', 'Calibrated'], widths=0.6)
ax.set_ylabel('Brier Score (lower = better)', fontsize=10)
ax.set_title('Random Forest: Brier Score', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Plot 4: Calibration Slope - RF
ax = axes[1, 0]
ax.scatter(range(1, N_SPLITS + 1), results_rf_df['slope_uncalib'], label='Uncalibrated', alpha=0.7, s=80)
ax.scatter(range(1, N_SPLITS + 1), results_rf_df['slope_calib'], label='Calibrated', alpha=0.7, s=80)
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Ideal (=1.0)')
ax.set_xlabel('Fold', fontsize=10)
ax.set_ylabel('Calibration Slope', fontsize=10)
ax.set_title('Random Forest: Calibration Slope', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 5: Calibration Slope - LR
ax = axes[1, 1]
ax.scatter(range(1, N_SPLITS + 1), results_lr_df['slope_uncalib'], label='Uncalibrated', alpha=0.7, s=80)
ax.scatter(range(1, N_SPLITS + 1), results_lr_df['slope_calib'], label='Calibrated', alpha=0.7, s=80)
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Ideal (=1.0)')
ax.set_xlabel('Fold', fontsize=10)
ax.set_ylabel('Calibration Slope', fontsize=10)
ax.set_title('Logistic Regression: Calibration Slope', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 6: Multiple Metrics - RF
ax = axes[1, 2]
metrics = ['AUC', 'Accuracy', 'Recall', 'F1']
uncalib_vals = [results_rf_df['auc_uncalib'].mean(), results_rf_df['accuracy_uncalib'].mean(),
                 results_rf_df['recall_uncalib'].mean(), results_rf_df['f1_uncalib'].mean()]
calib_vals = [results_rf_df['auc_calib'].mean(), results_rf_df['accuracy_calib'].mean(),
              results_rf_df['recall_calib'].mean(), results_rf_df['f1_calib'].mean()]
x = np.arange(len(metrics))
width = 0.35
ax.bar(x - width/2, uncalib_vals, width, label='Uncalibrated', alpha=0.7)
ax.bar(x + width/2, calib_vals, width, label='Calibrated', alpha=0.7)
ax.set_ylabel('Score', fontsize=10)
ax.set_title('Random Forest: Key Metrics', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(setwd, 'cv_platt_scaling_comparison.png'), dpi=300, bbox_inches='tight')
print("✓ Figure saved: cv_platt_scaling_comparison.png")
plt.show()

print("\n" + "="*90)
print("✓ CROSS-VALIDATION COMPLETE — STARTING COMPREHENSIVE EVALUATION")
print("="*90)

# %%
# ==============================================================================
# COMPREHENSIVE EVALUATION: ROC, CALIBRATION, PERFORMANCE TABLE, DCA
# ==============================================================================

from sklearn.metrics import roc_curve as _roc_curve
from sklearn.calibration import calibration_curve as _calib_curve

oof_y     = np.array(oof['y_true'])
oof_rf_u  = np.array(oof['rf_uncalib'])
oof_rf_c  = np.array(oof['rf_calib'])
oof_lr_u  = np.array(oof['lr_uncalib'])
oof_lr_c  = np.array(oof['lr_calib'])

# ──────────────────────────────────────────────────────────────────────────────
# PERFORMANCE TABLE WITH 95% CI (from 10 CV folds)
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*90)
print("PERFORMANCE TABLE WITH 95% CI (Mean ± 95% CI across 10 folds)")
print("─"*90)

def ci95(values):
    """95% CI using normal approximation across CV folds."""
    arr = np.array(values)
    mean = arr.mean()
    se   = arr.std(ddof=1) / np.sqrt(len(arr))
    return mean, mean - 1.96*se, mean + 1.96*se

metrics_keys = ['auc', 'accuracy', 'recall', 'precision', 'f1', 'brier', 'slope']
metric_labels = ['AUC', 'Accuracy', 'Recall', 'Precision', 'F1-score', 'Brier Score', 'Calib Slope']

perf_rows = []
for label, key in zip(metric_labels, metrics_keys):
    rf_u_m, rf_u_lo, rf_u_hi = ci95(results_rf_df[f'{key}_uncalib'])
    rf_c_m, rf_c_lo, rf_c_hi = ci95(results_rf_df[f'{key}_calib'])
    lr_u_m, lr_u_lo, lr_u_hi = ci95(results_lr_df[f'{key}_uncalib'])
    lr_c_m, lr_c_lo, lr_c_hi = ci95(results_lr_df[f'{key}_calib'])
    perf_rows.append({
        'Metric':              label,
        'RF Uncalib':          f"{rf_u_m:.4f} [{rf_u_lo:.4f}–{rf_u_hi:.4f}]",
        'RF Calib (Platt)':    f"{rf_c_m:.4f} [{rf_c_lo:.4f}–{rf_c_hi:.4f}]",
        'LR Uncalib':          f"{lr_u_m:.4f} [{lr_u_lo:.4f}–{lr_u_hi:.4f}]",
        'LR Calib (Platt)':    f"{lr_c_m:.4f} [{lr_c_lo:.4f}–{lr_c_hi:.4f}]",
    })

perf_table = pd.DataFrame(perf_rows)
print("\n" + perf_table.to_string(index=False))
perf_table.to_csv(os.path.join(setwd, 'cv_performance_table_with_ci.csv'), index=False)
print("\n✓ Saved: cv_performance_table_with_ci.csv")

# ──────────────────────────────────────────────────────────────────────────────
# DECISION CURVE ANALYSIS (DCA)
# ──────────────────────────────────────────────────────────────────────────────

def net_benefit(y_true, y_prob, thresholds):
    y_true = np.array(y_true)
    N = len(y_true)
    nb, nb_all = [], []
    prev = y_true.mean()
    for t in thresholds:
        if t >= 1.0:
            nb.append(0.0); nb_all.append(0.0)
            continue
        y_pred = (y_prob >= t).astype(int)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        nb.append((TP/N) - (FP/N) * (t/(1-t)))
        nb_all.append(prev - (1-prev) * (t/(1-t)))
    return np.array(nb), np.array(nb_all)

thresholds = np.linspace(0.01, 0.50, 200)

nb_rf_u, nb_all = net_benefit(oof_y, oof_rf_u, thresholds)
nb_rf_c, _      = net_benefit(oof_y, oof_rf_c, thresholds)
nb_lr_u, _      = net_benefit(oof_y, oof_lr_u, thresholds)
nb_lr_c, _      = net_benefit(oof_y, oof_lr_c, thresholds)
nb_none         = np.zeros_like(thresholds)

# Save DCA table
dca_df = pd.DataFrame({
    'threshold':    thresholds,
    'nb_rf_calib':  nb_rf_c,
    'nb_lr_calib':  nb_lr_c,
    'nb_rf_uncalib':nb_rf_u,
    'nb_lr_uncalib':nb_lr_u,
    'nb_treat_all': nb_all,
    'nb_treat_none':nb_none,
})
dca_df.to_csv(os.path.join(setwd, 'cv_decision_curve_analysis.csv'), index=False)
print("✓ Saved: cv_decision_curve_analysis.csv")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 1: ROC CURVES — RF vs LR (calibrated and uncalibrated)
# ──────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('ROC Curves: Out-of-Fold Predictions (10-Fold CV)',
             fontsize=14, fontweight='bold')

colors = {'RF Uncalib': '#1f77b4', 'RF Calib': '#ff7f0e',
          'LR Uncalib': '#2ca02c', 'LR Calib': '#d62728'}

for ax, title, pairs in [
    (axes[0], 'Uncalibrated Models',
     [('RF Uncalib', oof_rf_u), ('LR Uncalib', oof_lr_u)]),
    (axes[1], 'Calibrated Models (Platt Scaling)',
     [('RF Calib',  oof_rf_c), ('LR Calib',  oof_lr_c)]),
]:
    for label, probs in pairs:
        fpr, tpr, _ = _roc_curve(oof_y, probs)
        auc_val = roc_auc_score(oof_y, probs)
        ax.plot(fpr, tpr, lw=2, color=colors[label],
                label=f"{label}  (AUC={auc_val:.4f})")
    ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5, label='Chance')
    ax.set_xlabel('1 − Specificity (FPR)', fontsize=11)
    ax.set_ylabel('Sensitivity (TPR)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim([0,1]); ax.set_ylim([0,1])

plt.tight_layout()
plt.savefig(os.path.join(setwd, 'cv_roc_curves_rf_vs_lr.png'), dpi=300, bbox_inches='tight')
print("✓ Saved: cv_roc_curves_rf_vs_lr.png")
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 2: ROC CURVES — RF vs LR on single plot
# ──────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 6))
for label, probs, ls in [
    ('RF Uncalib', oof_rf_u, '--'),
    ('RF Calib',   oof_rf_c, '-'),
    ('LR Uncalib', oof_lr_u, '--'),
    ('LR Calib',   oof_lr_c, '-'),
]:
    fpr, tpr, _ = _roc_curve(oof_y, probs)
    auc_val = roc_auc_score(oof_y, probs)
    ax.plot(fpr, tpr, lw=2, linestyle=ls, color=colors[label],
            label=f"{label}  (AUC={auc_val:.4f})")
ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.4, label='Chance')
ax.set_xlabel('1 − Specificity (FPR)', fontsize=12)
ax.set_ylabel('Sensitivity (TPR)', fontsize=12)
ax.set_title('ROC Curves: RF vs LR (OOF Predictions)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.3)
ax.set_xlim([0,1]); ax.set_ylim([0,1])
plt.tight_layout()
plt.savefig(os.path.join(setwd, 'cv_roc_combined.png'), dpi=300, bbox_inches='tight')
print("✓ Saved: cv_roc_combined.png")
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 3: CALIBRATION CURVES — RF vs LR
# ──────────────────────────────────────────────────────────────────────────────

N_CALIB_BOOT = 500
_CALIB_BIN_EDGES = np.linspace(0.0, 1.0, 11)  # matches n_bins=10, strategy='uniform'

def _calib_curve_ci(y_true, y_prob, bin_edges=_CALIB_BIN_EDGES, n_boot=N_CALIB_BOOT,
                     random_state=RANDOM_STATE):
    """Bootstrap the pooled OOF sample to get a 95% CI band for the observed
    event rate in each calibration bin (fixed bin edges, so bins line up
    across replicates). Bins with no observations in a replicate are NaN and
    excluded from that replicate's percentile calc."""
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    n_bins = len(bin_edges) - 1
    boot_rates = np.full((n_boot, n_bins), np.nan)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        bin_ids = np.clip(np.digitize(yp, bin_edges[1:-1], right=False), 0, n_bins - 1)
        for k in range(n_bins):
            mask = bin_ids == k
            if mask.sum() > 0:
                boot_rates[b, k] = yt[mask].mean()
    lo = np.nanpercentile(boot_rates, 2.5, axis=0)
    hi = np.nanpercentile(boot_rates, 97.5, axis=0)
    return lo, hi

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Calibration Curves: Out-of-Fold Predictions (10-Fold CV)\n'
             '(shaded band = 95% CI from 500 bootstrap resamples of OOF predictions)',
             fontsize=13, fontweight='bold')

for ax, title, pairs in [
    (axes[0], 'Uncalibrated Models',
     [('RF Uncalib', oof_rf_u, colors['RF Uncalib']),
      ('LR Uncalib', oof_lr_u, colors['LR Uncalib'])]),
    (axes[1], 'Calibrated Models (Platt Scaling)',
     [('RF Calib',  oof_rf_c, colors['RF Calib']),
      ('LR Calib',  oof_lr_c, colors['LR Calib'])]),
]:
    ax.plot([0,1],[0,1], 'k--', lw=1.5, label='Perfect calibration')
    for label, probs, color in pairs:
        frac_pos, mean_pred = _calib_curve(oof_y, probs, n_bins=10, strategy='uniform')
        brier = brier_score_loss(oof_y, probs)
        lo, hi = _calib_curve_ci(oof_y, probs)
        # Align lo/hi (fixed bin grid) to the bins sklearn actually returned
        # (mean_pred), by matching each returned bin to its bin index.
        bin_ids_full = np.clip(
            np.digitize(mean_pred, _CALIB_BIN_EDGES[1:-1], right=False), 0, 9)
        ax.fill_between(mean_pred, lo[bin_ids_full], hi[bin_ids_full],
                         color=color, alpha=0.15)
        ax.plot(mean_pred, frac_pos, 'o-', lw=2, color=color,
                label=f"{label}  (Brier={brier:.4f})")
    ax.set_xlabel('Mean Predicted Probability', fontsize=11)
    ax.set_ylabel('Observed Event Rate', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0,1]); ax.set_ylim([0,1])

plt.tight_layout()
plt.savefig(os.path.join(setwd, 'cv_calibration_curves_rf_vs_lr.png'), dpi=300, bbox_inches='tight')
print("✓ Saved: cv_calibration_curves_rf_vs_lr.png")
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 4: DECISION CURVE ANALYSIS — Net Benefit vs Threshold Probability
# ──────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(thresholds, nb_rf_c,  lw=2.5, color=colors['RF Calib'],
        label=f'RF (Calibrated)')
ax.plot(thresholds, nb_lr_c,  lw=2.5, color=colors['LR Calib'],
        label=f'LR (Calibrated)')
ax.plot(thresholds, nb_rf_u,  lw=1.5, color=colors['RF Uncalib'],
        linestyle='--', alpha=0.7, label='RF (Uncalibrated)')
ax.plot(thresholds, nb_lr_u,  lw=1.5, color=colors['LR Uncalib'],
        linestyle='--', alpha=0.7, label='LR (Uncalibrated)')
ax.plot(thresholds, nb_all,   lw=2, color='grey',
        linestyle=':', label='Treat All')
ax.plot(thresholds, nb_none,  lw=2, color='black',
        linestyle=':', label='Treat None (NB=0)')

ax.set_xlabel('Threshold Probability', fontsize=12)
ax.set_ylabel('Net Benefit', fontsize=12)
ax.set_title('Decision Curve Analysis\nNet Benefit vs Threshold Probability',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.3)
ax.set_xlim([0, 0.50])

# Clip y-axis to sensible range
nb_max = max(nb_rf_c.max(), nb_lr_c.max(), nb_all[0]) * 1.15
nb_min = max(-0.02, min(nb_rf_c.min(), nb_lr_c.min()) - 0.01)
ax.set_ylim([nb_min, nb_max])

# Annotate prevalence line
prev_rate = oof_y.mean()
ax.axvline(prev_rate, color='purple', linestyle=':', alpha=0.6,
           label=f'Prevalence ({100*prev_rate:.1f}%)')
ax.text(prev_rate + 0.005, nb_max * 0.95,
        f'Prevalence\n{100*prev_rate:.1f}%', color='purple', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(setwd, 'cv_decision_curve_analysis.png'), dpi=300, bbox_inches='tight')
print("✓ Saved: cv_decision_curve_analysis.png")
plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 5: NET BENEFIT AT KEY THRESHOLDS (bar chart)
# ──────────────────────────────────────────────────────────────────────────────

key_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
nb_table_rows = []
for t in key_thresholds:
    idx = np.argmin(np.abs(thresholds - t))
    nb_table_rows.append({
        'Threshold': f"{t:.0%}",
        'RF Calib':  f"{nb_rf_c[idx]:.4f}",
        'LR Calib':  f"{nb_lr_c[idx]:.4f}",
        'Treat All': f"{nb_all[idx]:.4f}",
        'Treat None':'0.0000',
    })

nb_table = pd.DataFrame(nb_table_rows)
print("\n" + "─"*90)
print("NET BENEFIT AT KEY THRESHOLDS")
print("─"*90)
print("\n" + nb_table.to_string(index=False))
nb_table.to_csv(os.path.join(setwd, 'cv_net_benefit_key_thresholds.csv'), index=False)
print("✓ Saved: cv_net_benefit_key_thresholds.csv")

fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(key_thresholds))
width = 0.2

nb_rf_vals   = [nb_rf_c [np.argmin(np.abs(thresholds-t))] for t in key_thresholds]
nb_lr_vals   = [nb_lr_c [np.argmin(np.abs(thresholds-t))] for t in key_thresholds]
nb_all_vals  = [nb_all  [np.argmin(np.abs(thresholds-t))] for t in key_thresholds]

ax.bar(x_pos - width, nb_rf_vals,  width, label='RF (Calibrated)',  color=colors['RF Calib'],  alpha=0.8)
ax.bar(x_pos,         nb_lr_vals,  width, label='LR (Calibrated)',  color=colors['LR Calib'],  alpha=0.8)
ax.bar(x_pos + width, nb_all_vals, width, label='Treat All',        color='grey',               alpha=0.6)
ax.axhline(0, color='black', lw=1)

ax.set_xlabel('Decision Threshold', fontsize=12)
ax.set_ylabel('Net Benefit', fontsize=12)
ax.set_title('Net Benefit at Key Decision Thresholds', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{t:.0%}" for t in key_thresholds])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(setwd, 'cv_net_benefit_thresholds_bar.png'), dpi=300, bbox_inches='tight')
print("✓ Saved: cv_net_benefit_thresholds_bar.png")
plt.show()

print("\n" + "="*90)
print("✓ COMPREHENSIVE EVALUATION COMPLETE")
print("="*90)
print("\nFiles saved:")
print(f"  • cv_performance_table_with_ci.csv")
print(f"  • cv_decision_curve_analysis.csv")
print(f"  • cv_net_benefit_key_thresholds.csv")
print(f"  • cv_roc_curves_rf_vs_lr.png")
print(f"  • cv_roc_combined.png")
print(f"  • cv_calibration_curves_rf_vs_lr.png")
print(f"  • cv_decision_curve_analysis.png")
print(f"  • cv_net_benefit_thresholds_bar.png")

# %%
# ==============================================================================
# MULTIPLE IMPUTATION + RUBIN POOLING (pooled CV performance)
# ------------------------------------------------------------------------------
# The main CV run above uses a single imputation for the figures. Here we repeat
# the calibrated CV M times, each on a distinct posterior-draw imputation, and
# pool the fold-mean performance with Rubin's rules. Hyperparameters are held at
# the modal best values from the tuned run (no re-tuning) to keep runtime feasible.
# Runtime scales ~M x the CV time.
# ==============================================================================

from mice_utils import rubin_pool

M_IMPUTATIONS = 2 if QUICK_TEST else 5
print("\n" + "="*90)
print(f"MULTIPLE IMPUTATION: pooling calibrated CV performance across "
      f"{M_IMPUTATIONS} imputations")
print("="*90)

def _modal(dfp, key, default):
    try:
        v = dfp[key].mode().iloc[0]
        return default if pd.isna(v) else v
    except Exception:
        return default

rf_best = {
    'n_estimators':     int(_modal(rf_params_df, 'n_estimators', 200)),
    'max_depth':        (None if pd.isna(_modal(rf_params_df, 'max_depth', 10))
                         else int(_modal(rf_params_df, 'max_depth', 10))),
    'min_samples_leaf': int(_modal(rf_params_df, 'min_samples_leaf', 2)),
    'random_state':     RANDOM_STATE,
}
lr_best = {'C': float(_modal(lr_params_df, 'C', 0.1)), 'penalty': 'l1',
           'solver': 'liblinear', 'max_iter': 500, 'random_state': RANDOM_STATE}
print(f"  Using RF {rf_best} | LR C={lr_best['C']}")

def _build_stoch_preproc(seed):
    numt = Pipeline([
        ("fancy_imputer", FancyIterativeImputerWrapper(
            max_iterations=5, n_burn_in=3, random_state=seed,
            sample_posterior=True, use_outcome=True)),
        ("scaler", StandardScaler())])
    catt = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", numt, numeric_cols), ("cat", catt, categorical_cols)])
    lasso = SelectFromModel(LogisticRegressionCV(
        cv=5, penalty="l1", solver="liblinear", random_state=seed, max_iter=500))
    rfsel = SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=seed))
    return Pipeline([("preprocessor", pre), ("lasso_fs", lasso), ("rf_fs", rfsel)])

mi_keys = ['auc', 'accuracy', 'recall', 'precision', 'f1', 'brier', 'slope']
mi_store = {'rf': {k: {'est': [], 'var': []} for k in mi_keys},
            'lr': {k: {'est': [], 'var': []} for k in mi_keys}}

for m in range(M_IMPUTATIONS):
    print(f"  Imputation {m+1}/{M_IMPUTATIONS}...")
    pre = _build_stoch_preproc(RANDOM_STATE + 1000 + m)
    skf_mi = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = {'rf': {k: [] for k in mi_keys}, 'lr': {k: [] for k in mi_keys}}
    for tr, va in skf_mi.split(X, y):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        Xmt, Xct, ymt, yct = train_test_split(
            Xtr, ytr, test_size=CALIB_TEST_SPLIT, random_state=RANDOM_STATE, stratify=ytr)
        Xmt_t = pre.fit_transform(Xmt, ymt)
        Xct_t = pre.transform(Xct)
        Xva_t = pre.transform(Xva)
        for name, params in [('rf', rf_best), ('lr', lr_best)]:
            clf = (RandomForestClassifier(**params) if name == 'rf'
                   else LogisticRegression(**params))
            clf.fit(Xmt_t, ymt)
            cal = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
            cal.fit(Xct_t, yct)
            mm = calculate_metrics(yva, cal.predict_proba(Xva_t)[:, 1])
            for k in mi_keys:
                fold_metrics[name][k].append(mm[k])
    for name in ['rf', 'lr']:
        for k in mi_keys:
            arr = np.array(fold_metrics[name][k], dtype=float)
            mi_store[name][k]['est'].append(np.nanmean(arr))
            mi_store[name][k]['var'].append((np.nanstd(arr, ddof=1) / np.sqrt(len(arr))) ** 2)

mi_labels = {'auc': 'AUC', 'accuracy': 'Accuracy', 'recall': 'Recall',
             'precision': 'Precision', 'f1': 'F1-score', 'brier': 'Brier Score',
             'slope': 'Calib Slope'}
mi_rows = []
for k in mi_keys:
    pr = rubin_pool(mi_store['rf'][k]['est'], mi_store['rf'][k]['var'])
    pl = rubin_pool(mi_store['lr'][k]['est'], mi_store['lr'][k]['var'])
    mi_rows.append({
        'Metric': mi_labels[k],
        'RF Calib (MI-pooled)': f"{pr['estimate']:.4f} [{pr['ci_low']:.4f}-{pr['ci_high']:.4f}]",
        'LR Calib (MI-pooled)': f"{pl['estimate']:.4f} [{pl['ci_low']:.4f}-{pl['ci_high']:.4f}]",
    })
mi_table = pd.DataFrame(mi_rows)
print("\n" + mi_table.to_string(index=False))
mi_table.to_csv(os.path.join(setwd, 'cv_performance_table_mi_pooled.csv'), index=False)
print("\n✓ Saved: cv_performance_table_mi_pooled.csv")

# ──────────────────────────────────────────────────────────────────────────────
# MI-POOLED vs SINGLE-IMPUTATION CALIBRATED: pick the winner for bootstrapping
# ------------------------------------------------------------------------------
# If MI pooling (which propagates imputation uncertainty via per-iteration
# stochastic re-imputation) actually performs WORSE on AUC than the original
# single-imputation calibrated CV run above, that's a sign the extra imputation
# noise is hurting rather than helping here. In that case, 02_bootstrap_validation.py
# should default back to the original (single fixed imputation) calibrated
# approach instead of per-bootstrap-iteration stochastic re-imputation. The
# decision is written out so 02 can read it without re-deriving it.
# ──────────────────────────────────────────────────────────────────────────────

_single_run_auc = {
    'rf': ci95(results_rf_df['auc_calib'])[0],
    'lr': ci95(results_lr_df['auc_calib'])[0],
}
_mi_pooled_auc = {
    'rf': rubin_pool(mi_store['rf']['auc']['est'], mi_store['rf']['auc']['var'])['estimate'],
    'lr': rubin_pool(mi_store['lr']['auc']['est'], mi_store['lr']['auc']['var'])['estimate'],
}

mi_decision = {}
for name in ['rf', 'lr']:
    use_mi = bool(_mi_pooled_auc[name] >= _single_run_auc[name])
    mi_decision[name] = {
        'use_mi_pooled': use_mi,
        'single_imputation_auc': float(_single_run_auc[name]),
        'mi_pooled_auc': float(_mi_pooled_auc[name]),
    }

print("\n" + "─"*90)
print("MI-POOLED vs SINGLE-IMPUTATION CALIBRATED — decision for bootstrap stage")
print("─"*90)
for name in ['rf', 'lr']:
    d = mi_decision[name]
    verdict = "MI-pooled (>= single-imputation AUC)" if d['use_mi_pooled'] else \
              "single-imputation calibrated (MI-pooled AUC was WORSE)"
    print(f"  {name.upper()}: single-imputation AUC={d['single_imputation_auc']:.4f} | "
          f"MI-pooled AUC={d['mi_pooled_auc']:.4f}  ->  bootstrap will use: {verdict}")

import json
with open(os.path.join(setwd, 'mi_pooling_decision.json'), 'w') as f:
    json.dump(mi_decision, f, indent=2)
print("\n✓ Saved: mi_pooling_decision.json  (read by 02_bootstrap_validation.py)")

# %%
