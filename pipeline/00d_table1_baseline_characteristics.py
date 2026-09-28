"""
00d_table1_baseline_characteristics.py — manuscript Table 1 (baseline characteristics).

Builds the standard "Table 1" stratified by the outcome (delirium vs no
delirium), with an Overall column:
  • continuous variables: mean (SD) and median [IQR], compared with Mann-
    Whitney U (non-parametric; several predictors here are skewed, e.g.
    anticholinergic burden / drug counts)
  • categorical variables: n (%), compared with chi-square (Fisher's exact
    falls back automatically for any 2x2 table with small expected counts)

Only true BASELINE variables are included (i.e. present-at/near-admission
predictors). Post-admission / outcome-adjacent fields (discharge disposition,
etc.) are deliberately excluded — Table 1 characterises the cohort at
baseline, not what happened to them afterwards.

Run on data/cleaned.csv (pre-imputation — missingness is reported as-is,
consistent with 00b_missingness_analysis.py).
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from common import CLEANED_CSV, OUTPUT_DIR

print("Loading data...")
df = pd.read_csv(CLEANED_CSV)
df["delirium_label"] = df["delirium"].map({0: "No delirium", 1: "Delirium"})

# ------------------------------------------------------------------------------
# Variable definitions (baseline only)
# ------------------------------------------------------------------------------
CONTINUOUS = [
    ("age_patient", "Age, years"),
    ("x4at_score", "4AT score"),
    ("total_acb_score", "Anticholinergic burden (ACB) score"),
    ("unique_drug_count", "Number of anticholinergic medications"),
    ("mean_cr_result_value", "Mean creatinine (µmol/L)"),
    ("mean_hb_result_val", "Mean haemoglobin (g/L)"),
    ("mean_wcc_result_val", "Mean white cell count (x10⁹/L)"),
]

BINARY_YESNO_NUMERIC = [
    ("dementia", "Dementia"),
    ("atrial_fibrillation", "Atrial fibrillation"),
    ("copd", "COPD"),
    ("psychiatric_disorders", "Psychiatric disorder"),
    ("hip_fracture", "Hip fracture"),
    ("is_hearing_impaired", "Hearing impairment"),
    ("aha_visual_impairment", "Visual impairment"),
]

# Variables whose denominator should be the FULL group N instead of the
# default "% of those assessed" (currently none -- reverted; see git history /
# session notes for the full-N variant that was tried for hearing/visual
# impairment and then reverted back to % of assessed).
DENOM_TOTAL_COLS = set()

CATEGORICAL = [
    ("sex_patient", "Sex", ["M", "F"]),  # 'O' group excluded (n=6, too small to report a %)
    ("race_grouped", "Aboriginal and/or Torres Strait Islander status", None),
    ("admission_type_pharmacy", "Admission type", None),
    ("boc_risk_nominal", "Behaviours of concern flagged", None),
    ("nurse_verified_risk_of_cogn_impairment", "Nurse-verified cognitive impairment risk", None),
    ("x4at_score_categorical", "4AT risk category", None),
]

# Collapse race into a standard Indigenous-status grouping (raw categories are
# highly imbalanced and fragmented -- see docstring).
_indig = {"Aboriginal but not Torres Strait Islande", "Torres Strait Islander but not Aborigina",
          "Aboriginal and Torres Strait Islander Or"}
df["race_grouped"] = df["race"].apply(
    lambda v: np.nan if pd.isna(v)
    else ("Aboriginal and/or Torres Strait Islander" if v in _indig
          else ("Not stated" if v == "Not Stated/Inadequately Described" else "Not Indigenous"))
)

N_TOTAL = len(df)
groups = {
    "Overall": df,
    "No delirium": df[df["delirium"] == 0],
    "Delirium": df[df["delirium"] == 1],
}


def fmt_n_pct(n, denom):
    pct = 100 * n / denom if denom else float("nan")
    return f"{n:,} ({pct:.1f}%)"


def continuous_row(col, label):
    row = {"Characteristic": label}
    vals = {}
    for gname, gdf in groups.items():
        s = gdf[col].dropna()
        vals[gname] = s
        mean, sd = s.mean(), s.std()
        med, q1, q3 = s.median(), s.quantile(0.25), s.quantile(0.75)
        n_missing = gdf[col].isna().sum()
        pct_missing = 100 * n_missing / len(gdf) if len(gdf) else float("nan")
        row[gname] = (
            f"Mean (SD): {mean:.1f} ({sd:.1f})\n"
            f"Median [IQR]: {med:.1f} [{q1:.1f}–{q3:.1f}]\n"
            f"(missing={pct_missing:.1f}%)"
        )
    # Mann-Whitney U: No delirium vs Delirium
    a, b = vals["No delirium"], vals["Delirium"]
    if len(a) > 0 and len(b) > 0:
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        row["p-value"] = "<0.001" if p < 0.001 else f"{p:.3f}"
    else:
        row["p-value"] = "—"
    return row


def binary_row(col, label):
    row = {"Characteristic": label}
    counts = {}
    use_total_denom = col in DENOM_TOTAL_COLS
    for gname, gdf in groups.items():
        present = gdf[col].dropna()
        n_yes = int((present == 1).sum())
        n_denom = len(gdf) if use_total_denom else len(present)
        counts[gname] = (n_yes, n_denom)
        row[gname] = fmt_n_pct(n_yes, n_denom)
    # Chi-square: No delirium vs Delirium
    (yes0, n0), (yes1, n1) = counts["No delirium"], counts["Delirium"]
    table = [[yes0, n0 - yes0], [yes1, n1 - yes1]]
    try:
        _, p, _, _ = stats.chi2_contingency(table)
        row["p-value"] = "<0.001" if p < 0.001 else f"{p:.3f}"
    except Exception:
        row["p-value"] = "—"
    return row


def categorical_row(col, label, restrict_categories):
    rows = []
    cats = restrict_categories or [c for c in df[col].dropna().unique()]
    header_row = {"Characteristic": label}
    for gname in groups:
        header_row[gname] = ""
    header_row["p-value"] = ""
    rows.append(header_row)

    # Chi-square across all categories, No delirium vs Delirium
    ct = pd.crosstab(df["delirium_label"], df[col])
    try:
        _, p, _, _ = stats.chi2_contingency(ct)
        p_str = "<0.001" if p < 0.001 else f"{p:.3f}"
    except Exception:
        p_str = "—"

    for i, cat in enumerate(cats):
        row = {"Characteristic": f"    {cat}"}
        for gname, gdf in groups.items():
            n_cat = int((gdf[col] == cat).sum())
            n_assessed = gdf[col].notna().sum()
            row[gname] = fmt_n_pct(n_cat, n_assessed)
        row["p-value"] = p_str if i == 0 else ""
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------
# Build table
# ------------------------------------------------------------------------------
table_rows = []

table_rows.append({"Characteristic": "N (episodes)", "Overall": f"{N_TOTAL:,}",
                    "No delirium": f"{len(groups['No delirium']):,}",
                    "Delirium": f"{len(groups['Delirium']):,}", "p-value": ""})

table_rows.append({"Characteristic": "-- Continuous variables: mean (SD); median [IQR] --",
                    "Overall": "", "No delirium": "", "Delirium": "", "p-value": ""})
for col, label in CONTINUOUS:
    table_rows.append(continuous_row(col, label))

table_rows.append({"Characteristic": "-- Binary variables: n (%) of those assessed --",
                    "Overall": "", "No delirium": "", "Delirium": "", "p-value": ""})
for col, label in BINARY_YESNO_NUMERIC:
    table_rows.append(binary_row(col, label))

table_rows.append({"Characteristic": "-- Categorical variables: n (%) --",
                    "Overall": "", "No delirium": "", "Delirium": "", "p-value": ""})
for col, label, restrict in CATEGORICAL:
    table_rows.extend(categorical_row(col, label, restrict))

table1 = pd.DataFrame(table_rows)[["Characteristic", "Overall", "No delirium", "Delirium", "p-value"]]

out_path = os.path.join(str(OUTPUT_DIR), "table1_baseline_characteristics.csv")
table1.to_csv(out_path, index=False)
print(f"\n[OK] Saved: {out_path}\n")
print(table1.to_string(index=False))

print(
    "\nNOTE: 'sex_patient' group 'O' (n=6) and any variable's missing/unassessed "
    "records are excluded from that variable's percentage base -- percentages are "
    "of those ASSESSED for that variable, not of N total, except where noted."
)
_pct_missing_hearing = 100 * df["is_hearing_impaired"].isna().mean()
_pct_missing_visual = 100 * df["aha_visual_impairment"].isna().mean()
print(
    f"\nCAUTION: 'Hearing impairment' and especially 'Visual impairment' have very "
    f"high missingness ({_pct_missing_hearing:.1f}% and {_pct_missing_visual:.1f}% of "
    f"episodes respectively have no charted value) and are likely charted "
    "selectively (i.e. only when a deficit is clinically observed), not as a routine "
    "screen with a documented negative result. The % among those ASSESSED therefore "
    "substantially overstates true prevalence in the full cohort -- report with this "
    "caveat, or consider reporting only 'documented as impaired / total N' rather "
    "than a rate among the assessed subset."
)
print("Done.")
