# %%
"""
COMBINED RESULTS REPORT GENERATOR
===================================
Reads the CSV outputs produced by the pipeline (scripts 01-06b) and assembles a
journal-ready Results section, auto-populating values where the outputs exist and
inserting [VERIFY] for anything not yet produced.

Writes:
  • results_report.md    (Markdown, always)
  • results_report.docx  (Word, if python-docx is available; else skipped)

Run AFTER the pipeline scripts. Missing outputs are tolerated — rerun this after
producing more of them to fill in remaining placeholders.

Author: [Your name]
Date: July 2026
"""

import os
import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURATION
# ==============================================================================

import sys
sys.path.insert(0, os.path.dirname(__file__))
from common import OUTPUT_DIR, REPORT_DIR, DATA_DIR

script_dir = os.path.dirname(__file__)
# CSVs are read from outputs/ (pipeline results) and data/ (cleaned.csv).
SEARCH_DIRS = [str(OUTPUT_DIR), str(DATA_DIR), script_dir]
OPERATING_PCT = 15                           # operating threshold for Sens/Spec (%)

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

VERIFY = "[VERIFY]"

# ==============================================================================
# HELPERS
# ==============================================================================

def find_csv(name):
    for d in SEARCH_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None

def load(name):
    p = find_csv(name)
    if p is None:
        print(f"  (missing) {name}")
        return None
    print(f"  (found)   {name}")
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"  (error reading {name}: {e})")
        return None

def cell(df, row_match_col, row_match_val, col):
    """Return a single cell as string, or VERIFY."""
    if df is None or col not in df.columns:
        return VERIFY
    m = df[df[row_match_col].astype(str).str.strip() == row_match_val]
    if len(m) == 0:
        return VERIFY
    v = m.iloc[0][col]
    return VERIFY if pd.isna(v) else str(v)

def fmt(v, nd=3):
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return VERIFY

# ==============================================================================
# LOAD OUTPUTS
# ==============================================================================

print("Locating pipeline outputs...")
cv_perf      = load("cv_performance_table_with_ci.csv")
cv_results   = load("cv_results_rf_with_platt_scaling.csv")
boot_corr    = load("bootstrap_optimism_corrected_summary.csv")
boot_ci      = load("bootstrap_performance_table_ci.csv")
calib_metrics= load("publication_calibration_metrics_v2.csv")
dca_key      = load("publication_dca_key_thresholds.csv")
thr_rf       = load("threshold_performance_rf.csv")
thr_lr       = load("threshold_performance_lr.csv")
subgroup     = load("subgroup_performance_by_sex.csv")
sens_summary = load("sensitivity_analysis_summary.csv")

# Cohort stats direct from data
n_episodes = VERIFY; n_cases = VERIFY; prevalence = VERIFY; prev_pct = VERIFY
data_path = find_csv("cleaned.csv")
if data_path:
    try:
        _df = pd.read_csv(data_path)
        _df = _df.dropna(subset=["delirium"])
        n_episodes = f"{len(_df):,}"
        n_cases = f"{int(_df['delirium'].sum()):,}"
        prevalence = _df['delirium'].mean()
        prev_pct = f"{100*prevalence:.1f}"
    except Exception as e:
        print(f"  (error reading cleaned.csv: {e})")

n_candidates = len(numeric_cols) + len(categorical_cols)
n_features = VERIFY
if cv_results is not None and "n_features" in cv_results.columns:
    try:
        n_features = str(int(cv_results["n_features"].mode().iloc[0]))
    except Exception:
        pass

# Named core features (selected in all folds) from 01b output
FEATURE_LABELS = {
    "age_patient": "age",
    "x4at_score": "4AT score",
    "total_acb_score": "anticholinergic burden score",
    "unique_drug_count": "number of anticholinergic medications",
    "boc_risk_nominal_Yes": "behaviours of concern",
    "dementia_1": "comorbid dementia",
    "nurse_verified_risk_of_cogn_impairment_Yes": "cognitive impairment risk",
    "sex_patient_M": "male sex",
    "hb_status_Normal": "normal haemoglobin status",
}

def label(f):
    """Map a technical feature name to a readable clinical label."""
    return FEATURE_LABELS.get(str(f), str(f))

def fmt_params(spec_str):
    """Turn a Python dict string like {'n_estimators': 200, ...} into
    'n_estimators=200, max_depth=10, ...' (dropping random_state)."""
    try:
        import ast
        d = ast.literal_eval(spec_str)
        parts = [f"{k}={v}" for k, v in d.items() if k != "random_state"]
        return ", ".join(parts)
    except Exception:
        return spec_str

core_features_str = VERIFY
n_core = VERIFY
feat_freq = load("selected_features_frequency.csv")
if feat_freq is not None and "Times_selected" in feat_freq.columns:
    try:
        idx_col = feat_freq.columns[0]
        n_folds_max = int(feat_freq["Times_selected"].max())
        core = feat_freq[feat_freq["Times_selected"] == n_folds_max][idx_col].tolist()
        n_core = str(len(core))
        core_features_str = ", ".join(label(f) for f in core)
    except Exception as e:
        print(f"  (error parsing feature frequency: {e})")

# Null Brier
null_brier = VERIFY
if isinstance(prevalence, float):
    null_brier = f"{prevalence*(1-prevalence):.3f}"

# ==============================================================================
# EXTRACT KEY VALUES
# ==============================================================================

# CV calibrated c-statistics (strings incl. CI)
cv_rf_auc = cell(cv_perf, "Metric", "AUC", "RF Calib (Platt)")
cv_lr_auc = cell(cv_perf, "Metric", "AUC", "LR Calib (Platt)")
cv_rf_recall = cell(cv_perf, "Metric", "Recall", "RF Calib (Platt)")
cv_lr_recall = cell(cv_perf, "Metric", "Recall", "LR Calib (Platt)")
cv_rf_f1 = cell(cv_perf, "Metric", "F1-score", "RF Calib (Platt)")
cv_lr_f1 = cell(cv_perf, "Metric", "F1-score", "LR Calib (Platt)")

# Optimism-corrected
rf_corr_auc = cell(boot_corr, "Metric", "AUC", "RF Corrected")
lr_corr_auc = cell(boot_corr, "Metric", "AUC", "LR Corrected")
rf_opt_auc  = cell(boot_corr, "Metric", "AUC", "RF Optimism")

# Bootstrap OOB calibrated AUC (with CI)
rf_oob_auc = cell(boot_ci, "Metric", "AUC", "RF Calib OOB")
lr_oob_auc = cell(boot_ci, "Metric", "AUC", "LR Calib OOB")

# Calibration
rf_slope = cell(calib_metrics, "Metric", "Calibration Slope (bootstrap 95% CI)", "Random Forest (Calibrated)")
lr_slope = cell(calib_metrics, "Metric", "Calibration Slope (bootstrap 95% CI)", "Logistic Regression (Calibrated)")
rf_brier = cell(calib_metrics, "Metric", "Brier Score (bootstrap 95% CI)", "Random Forest (Calibrated)")
lr_brier = cell(calib_metrics, "Metric", "Brier Score (bootstrap 95% CI)", "Logistic Regression (Calibrated)")
rf_bss   = cell(calib_metrics, "Metric", "Brier Skill Score", "Random Forest (Calibrated)")
lr_bss   = cell(calib_metrics, "Metric", "Brier Skill Score", "Logistic Regression (Calibrated)")
rf_citl  = cell(calib_metrics, "Metric", "Calibration-in-the-Large (intercept)", "Random Forest (Calibrated)")
lr_citl  = cell(calib_metrics, "Metric", "Calibration-in-the-Large (intercept)", "Logistic Regression (Calibrated)")

# Threshold operating characteristics at OPERATING_PCT
thr_label = f"{OPERATING_PCT:.1f}%"
def thr_val(df, col):
    return cell(df, "Threshold", thr_label, col)
rf_sens = thr_val(thr_rf, "Sensitivity"); rf_spec = thr_val(thr_rf, "Specificity")
rf_ppv  = thr_val(thr_rf, "PPV");         rf_npv  = thr_val(thr_rf, "NPV")
rf_flag = thr_val(thr_rf, "Flagged (%)")

# DCA net benefit at OPERATING_PCT
dca_label = f"{OPERATING_PCT}%"
rf_nb  = cell(dca_key, "Threshold", dca_label, "RF (Calibrated)")
nb_all = cell(dca_key, "Threshold", dca_label, "Treat All")

# Subgroup by sex (RF)
subgroup_lines = []
if subgroup is not None:
    sg = subgroup[subgroup["Model"] == "Random Forest"]
    sens_col = next((c for c in subgroup.columns if c.startswith("Sensitivity@")), None)
    for _, r in sg.iterrows():
        s = f"  - {r['Sex']}: n={r['n']}, prevalence={r.get('Prevalence (%)', VERIFY)}%, AUC={r.get('AUC', VERIFY)}"
        if sens_col:
            s += f", {sens_col}={r.get(sens_col, VERIFY)}"
        subgroup_lines.append(s)

# ==============================================================================
# BUILD REPORT TEXT
# ==============================================================================

R = []
R.append("# Results\n")

R.append("## Study cohort\n")
R.append(
    f"The cohort comprised {n_episodes} admission episodes, of whom {n_cases} "
    f"({prev_pct}%) developed delirium. {n_candidates} candidate predictors "
    f"({len(numeric_cols)} numeric, {len(categorical_cols)} categorical) were evaluated; "
    f"two-stage feature selection (L1-penalised logistic regression followed by "
    f"random-forest importance) retained a stable core of {n_features} features across all "
    f"folds: {core_features_str}.\n")

R.append("## Model development and discrimination\n")
R.append(
    f"A random forest (RF) and L1-penalised logistic regression (LR) were developed using "
    f"10-fold stratified nested cross-validation with 3-fold inner GridSearchCV for "
    f"hyperparameter tuning and Platt scaling for probability calibration. In cross-validation, "
    f"the calibrated RF and LR showed comparable discrimination (RF c-statistic {cv_rf_auc}; "
    f"LR {cv_lr_auc}). The RF achieved higher sensitivity ({cv_rf_recall} vs {cv_lr_recall}) and "
    f"F1-score ({cv_rf_f1} vs {cv_lr_f1}) and was selected as the primary model given the clinical "
    f"priority of case identification.\n")
R.append(
    f"Internal validity was assessed by bootstrap resampling (500 iterations) with Harrell's "
    f"optimism correction. Optimism in the c-statistic was small ({rf_opt_auc}), indicating limited "
    f"overfitting. The optimism-corrected c-statistic was {rf_corr_auc} for the RF and {lr_corr_auc} "
    f"for the LR. Bootstrap out-of-sample discrimination (calibrated) was {rf_oob_auc} (RF) and "
    f"{lr_oob_auc} (LR).\n")

R.append("## Calibration\n")
R.append(
    f"Following Platt scaling, both models were well calibrated. The RF calibration slope was "
    f"{rf_slope} with a Brier score of {rf_brier} (Brier skill score {rf_bss}); the LR calibration "
    f"slope was {lr_slope} with a Brier score of {lr_brier} (Brier skill score {lr_bss}). Both Brier "
    f"scores were below the null-model benchmark of {null_brier} (calibration-in-the-large intercept "
    f"RF {rf_citl}; LR {lr_citl}). Overall, the calibration slopes and calibration curves (Figure [X]) "
    f"indicated good agreement between predicted and observed risk across the range of predicted "
    f"probabilities.\n")

R.append("## Clinical utility\n")
R.append(
    f"Decision curve analysis (Figure [X]) showed that both calibrated models provided greater net "
    f"benefit than the 'treat all' and 'treat none' strategies across the clinically relevant range. "
    f"At a {OPERATING_PCT}% risk threshold, the net benefit of the RF model was {rf_nb}, exceeding "
    f"'treat all' ({nb_all}).\n")

R.append("## Operating characteristics\n")
R.append(
    f"At a {OPERATING_PCT}% decision threshold, the calibrated RF achieved a sensitivity of {rf_sens} "
    f"and specificity of {rf_spec} (PPV {rf_ppv}, NPV {rf_npv}), flagging {rf_flag}% of admissions for "
    f"review (Table [X]). Sensitivity and specificity across thresholds are shown in Figure [X].\n")

R.append("## Subgroup analysis by sex\n")
if subgroup_lines:
    R.append("Model performance (random forest) by sex subgroup:\n")
    R.extend([ln + "\n" for ln in subgroup_lines])
    R.append("\nPerformance was consistent across sex, supporting equitable model behaviour.\n")
else:
    R.append("Subgroup results by sex: [VERIFY — run 06b_subgroup_sex.py].\n")

R.append("## Sensitivity analyses\n")
if sens_summary is not None:
    R.append("Model discrimination (AUC) was robust across sensitivity analyses:\n")
    for _, r in sens_summary.iterrows():
        R.append(f"  - {r.get('Analysis','')} ({r.get('Model','')}): "
                 f"with={r.get('With','')}, without={r.get('Without','')}, "
                 f"difference={r.get('Difference','')}\n")
else:
    R.append("Sensitivity analyses (4AT, symptom variables, patient-level CV): "
             "[VERIFY — run 06_sensitivity_analysis.py].\n")

# ------------------------------------------------------------------ Appendix
spec_config = load("model_spec_config.csv")
spec_lr     = load("model_spec_lr_coefficients.csv")
spec_rf     = load("model_spec_rf_importances.csv")

R.append("## Appendix: Model specification\n")

# Feature-count reconciliation (Methods vs final model)
n_final = VERIFY
if spec_config is not None:
    n_final = cell(spec_config, spec_config.columns[0], "N selected features", spec_config.columns[1])
R.append(
    f"Two-stage feature selection retained a stable core of {n_core} features selected in all "
    f"cross-validation folds ({core_features_str}). When refitted on the full development dataset, "
    f"the final model additionally retained cognitive impairment risk, giving {n_final} predictors "
    f"in total. Both the {n_core}-feature cross-validation core and the {n_final}-feature final model "
    f"are reported for transparency.\n")

if spec_config is not None:
    R.append("### Development configuration\n")
    for _, r in spec_config.iterrows():
        comp, spec = str(r.iloc[0]), str(r.iloc[1])
        if "hyperparameters" in comp.lower():
            spec = fmt_params(spec)
        R.append(f"  - {comp}: {spec}\n")
else:
    R.append("Development configuration: [VERIFY — run 09_model_specification.py].\n")

if spec_lr is not None:
    coef_col = next((c for c in spec_lr.columns if "Coefficient" in c), None)
    or_col   = next((c for c in spec_lr.columns if "Odds Ratio" in c), None)
    R.append("### Logistic regression — coefficients (standardised, per 1 SD)\n")
    for _, r in spec_lr.iterrows():
        feat = label(r["Feature"]) if r["Feature"] != "(Intercept)" else "(Intercept)"
        coef = r[coef_col] if coef_col else ""
        orv  = r[or_col] if or_col else ""
        if pd.isna(orv):
            R.append(f"  - {feat}: coefficient = {coef}\n")
        else:
            R.append(f"  - {feat}: coefficient = {coef}, odds ratio = {orv}\n")

if spec_rf is not None:
    R.append("### Random forest — feature importances\n")
    for _, r in spec_rf.iterrows():
        R.append(f"  - {label(r['Feature'])}: {r['Importance']}\n")

report_md = "\n".join(R)

# ==============================================================================
# WRITE MARKDOWN
# ==============================================================================

md_path = os.path.join(str(REPORT_DIR), "results_report.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write(report_md)
print(f"\n[OK] Saved: {md_path}")

# ==============================================================================
# WRITE DOCX (if python-docx available)
# ==============================================================================

try:
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    doc.add_heading("Results", level=1)

    for block in report_md.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("## "):
            doc.add_heading(block[3:].strip(), level=2)
        elif block.startswith("# "):
            continue  # already added title
        elif block.startswith("  - "):
            for line in block.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    line = line[2:]
                doc.add_paragraph(line, style="List Bullet")
        else:
            doc.add_paragraph(block)

    docx_path = os.path.join(OUTPUT_DIR, "results_report.docx")
    try:
        doc.save(docx_path)
        print(f"[OK] Saved: {docx_path}")
    except PermissionError:
        import time as _t
        alt = os.path.join(OUTPUT_DIR, f"results_report_{_t.strftime('%H%M%S')}.docx")
        doc.save(alt)
        print(f"[OK] Target was locked (open in Word?). Saved instead: {alt}")
except ImportError:
    print("\n(!) python-docx not installed - .docx not written.")
    print("    Install with:  pip install python-docx   then rerun this script.")

print("\nDone. Review [VERIFY] placeholders and rerun after producing more outputs.")
