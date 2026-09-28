# %%
"""
SHAP FEATURE IMPORTANCE — RANDOM FOREST & LOGISTIC REGRESSION
==============================================================
Computes SHAP values for both final models and produces publication feature-
importance plots:
  • beeswarm summary (direction + magnitude of each feature's effect)
  • mean |SHAP| bar plot (global importance ranking)
  • mean |SHAP| CSV table

Uses TreeExplainer for the random forest and LinearExplainer for logistic
regression. A random subsample is used for the beeswarm plot for speed.

Run AFTER 02_bootstrap_validation.py (uses the same best hyperparameters).
Requires:  pip install shap

Author: [Your name]
Date: July 2026
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

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

try:
    import shap
except ImportError:
    raise SystemExit("shap is not installed. Install with:  pip install shap")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

RANDOM_STATE = 42
setwd = str(OUTPUT_DIR)
output_dir = setwd
SHAP_SAMPLE = 2000          # rows used for beeswarm (subsample for speed)
BACKGROUND = 200            # background rows for LinearExplainer

FEATURE_LABELS = {
    "age_patient": "Age",
    "x4at_score": "4AT score",
    "total_acb_score": "Anticholinergic burden score",
    "unique_drug_count": "Anticholinergic drug count",
    "boc_risk_nominal_Yes": "Behaviours of concern",
    "dementia_1": "Comorbid dementia",
    "nurse_verified_risk_of_cogn_impairment_Yes": "Cognitive impairment risk",
    "sex_patient_M": "Male sex",
    "hb_status_Normal": "Normal haemoglobin status",
}
def label(f):
    return FEATURE_LABELS.get(str(f), str(f))

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
# LOAD + PREPROCESS + FIT
# ==============================================================================

print("Loading data...")
df = pd.read_csv(CLEANED_CSV)
df.drop(columns=["disch_disposition"], inplace=True, errors="ignore")
df = df.dropna(subset=["delirium"])
X = df[numeric_cols + categorical_cols].copy()
y = df["delirium"].values

print("Preprocessing (MICE + feature selection)...")
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
preproc = Pipeline([("preprocessor", preprocessor),
                    ("lasso_fs", lasso_selector), ("rf_fs", rf_selector)])

X_pp = preproc.fit_transform(X, y)
names = [label(n.replace("num__", "").replace("cat__", ""))
         for n in preproc.get_feature_names_out()]
print(f"Features ({len(names)}): {names}")

rf = RandomForestClassifier(**RF_BEST_PARAMS).fit(X_pp, y)
lr = LogisticRegression(**LR_BEST_PARAMS).fit(X_pp, y)

# Subsample for plotting
rng = np.random.RandomState(RANDOM_STATE)
sample_idx = rng.choice(len(X_pp), size=min(SHAP_SAMPLE, len(X_pp)), replace=False)
X_sample = X_pp[sample_idx]
X_df = pd.DataFrame(X_sample, columns=names)

# ==============================================================================
# SHAP — RANDOM FOREST (TreeExplainer)
# ==============================================================================

print("\nComputing SHAP values for Random Forest (TreeExplainer)...")
rf_explainer = shap.TreeExplainer(rf)
rf_sv = rf_explainer.shap_values(X_sample)
# Binary classifier: take positive-class contributions
if isinstance(rf_sv, list):
    rf_sv = rf_sv[1]
elif rf_sv.ndim == 3:           # (n, features, classes)
    rf_sv = rf_sv[:, :, 1]

# ==============================================================================
# SHAP — LOGISTIC REGRESSION (LinearExplainer)
# ==============================================================================

print("Computing SHAP values for Logistic Regression (LinearExplainer)...")
bg_idx = rng.choice(len(X_pp), size=min(BACKGROUND, len(X_pp)), replace=False)
lr_explainer = shap.LinearExplainer(lr, X_pp[bg_idx])
lr_sv = lr_explainer.shap_values(X_sample)
if isinstance(lr_sv, list):
    lr_sv = lr_sv[1] if len(lr_sv) > 1 else lr_sv[0]

# ==============================================================================
# PLOTS
# ==============================================================================

# Consistent model colours (match the ROC / calibration / DCA figures)
RF_COLOR = "#1f77b4"   # blue
LR_COLOR = "#d62728"   # red

# Fix a common feature order (by RF importance) so both bar charts line up
# for side-by-side placement.
order = np.argsort(np.abs(rf_sv).mean(axis=0))[::-1]
names   = [names[i] for i in order]
X_df    = X_df.iloc[:, order]
rf_sv   = rf_sv[:, order]
lr_sv   = lr_sv[:, order]

def save_summary(sv, title, prefix, bar_color):
    # Beeswarm (colour encodes feature value, so keep the default colormap)
    plt.figure()
    shap.summary_plot(sv, X_df, feature_names=names, show=False, plot_size=(8, 5), sort=False)
    plt.title(f"SHAP summary — {title}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    p1 = os.path.join(output_dir, f"shap_beeswarm_{prefix}.png")
    plt.savefig(p1, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[OK] Saved: shap_beeswarm_{prefix}.png")

    # Bar (mean |SHAP|), NORMALISED to this model's maximum so the top feature
    # = 1.0 in both models — makes RF and LR panels comparable side by side.
    mean_abs = np.abs(sv).mean(axis=0)
    norm = mean_abs / mean_abs.max() if mean_abs.max() > 0 else mean_abs
    fig, ax = plt.subplots(figsize=(8, 5))
    ypos = np.arange(len(names))[::-1]        # first feature at top
    ax.barh(ypos, norm, color=bar_color, alpha=0.9)
    for yp, val in zip(ypos, norm):
        ax.text(val + 0.01, yp, f"{val:.2f}", va='center', fontsize=9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Relative importance (mean |SHAP|, normalised to model max)", fontsize=11)
    ax.set_title(f"SHAP importance — {title}", fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    p2 = os.path.join(output_dir, f"shap_bar_{prefix}.png")
    plt.savefig(p2, dpi=300, bbox_inches='tight'); plt.close()
    print(f"[OK] Saved: shap_bar_{prefix}.png")

save_summary(rf_sv, "Random Forest", "rf", RF_COLOR)
save_summary(lr_sv, "Logistic Regression", "lr", LR_COLOR)

# ==============================================================================
# MEAN |SHAP| TABLE (both models)
# ==============================================================================

rf_mean = np.abs(rf_sv).mean(axis=0)
lr_mean = np.abs(lr_sv).mean(axis=0)
shap_table = pd.DataFrame({
    "Feature": names,
    "RF mean|SHAP|": np.round(rf_mean, 4),
    "LR mean|SHAP|": np.round(lr_mean, 4),
}).sort_values("RF mean|SHAP|", ascending=False)
shap_table.to_csv(os.path.join(output_dir, "shap_importance_table.csv"), index=False)
print("[OK] Saved: shap_importance_table.csv")

print("\n" + "="*60)
print("SHAP MEAN |VALUE| BY FEATURE")
print("="*60)
print(shap_table.to_string(index=False))
print("\nDone.")
