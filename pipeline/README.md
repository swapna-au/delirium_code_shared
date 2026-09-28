# Delirium Risk Prediction — Analysis Pipeline

Ordered, self-contained pipeline. The easiest way to run everything is from the
repository root:

```bash
python run_all.py
```

To run a single stage, run it directly (it finds `common.py` automatically):

```bash
python pipeline/09_model_specification.py
```

## Shared configuration — `common.py`

All stages import their shared setup from `common.py`:

- **Paths** (repo-relative, so it runs on any machine): input `data/cleaned.csv`;
  generated figures/tables/models → `outputs/`; result docs → `reports/`.
- **Hyperparameters** `RF_BEST_PARAMS` / `LR_BEST_PARAMS` — defined **once** here
  (selected by the stage-01 nested CV) and imported everywhere, so they can't drift.
- **`FancyIterativeImputerWrapper`** — the MICE imputation transformer.

## Run order

| # | Script | Purpose | Key outputs (→ `outputs/`, docs → `reports/`) |
|---|--------|---------|-------------|
| 00 | 00_prepare_data.py | Runs **only** if `data/source_file_bronze.csv` is present. De-identifies + cleans bronze → `data/cleaned.csv` (Python port of the retired `logistic_regression.rmd`). Also derives `patient_cluster_id`, a one-way salted hash of Urn, so patient-level grouped CV (stage 06) actually works without retaining the raw identifier. | data/cleaned.csv |
| 00b | 00b_missingness_analysis.py | Missingness summary (predictors + outcome), heatmap, and Little's MCAR test on numeric predictors. | missingness_summary.csv, missingness_heatmap.png, missingness_barchart.png, missingness_mcar_test.csv |
| 00c | 00c_mice_convergence.py | MICE convergence diagnostic (multiple chains, same max_iterations/n_burn_in as the modelling stages). | mice_convergence.csv, mice_convergence.png |
| 01 | 01_cross_validation.py | 10-fold nested CV + GridSearchCV + Platt. | cv_*.csv, cv_*.png, cv_best_rf/lr_params_per_fold.csv |
| 01b | 01b_selected_features_per_fold.py | Features selected in each fold (stable core). | selected_features_per_fold.csv, selected_features_frequency.csv |
| 02 | 02_bootstrap_validation.py | 500-iteration bootstrap optimism correction. | bootstrap_*.csv/png |
| 03 | 03_plot_roc.py | Publication ROC figure (RF vs LR, calibrated) + 95% CI. | publication_roc_calibrated.png |
| 04 | 04_plot_calibration.py | Publication calibration figure + metrics table. | publication_calibration_plot.png, publication_calibration_metrics_v2.csv |
| 05 | 05_plot_dca.py | Decision curve analysis + net-benefit tables. | publication_dca.png, publication_dca*.csv |
| 05b | 05b_threshold_performance.py | Sens/Spec/PPV/NPV across thresholds + plot. | threshold_performance_rf/lr.csv, threshold_sensitivity_specificity.png |
| 06 | 06_sensitivity_analysis.py | Sensitivity analyses (without-4AT, symptom vars, patient-level CV). | sensitivity_analysis_summary.csv, sensitivity_analysis_report.txt |
| 06b | 06b_subgroup_sex.py | Subgroup / fairness analysis: sex, age band, dementia, cognitive impairment risk, hip fracture, admission type (medical/surgical). Reports AUC, Brier, calibration slope, Sens/Spec/PPV/NPV per group. ICU and RACF status are reported as unavailable (not collected). | subgroup_performance_extended.csv, subgroup_performance_by_sex.csv (backward-compat), subgroup_auc_by_sex.png, subgroup_auc_extended.png |
| 07 | 07_figure1_pipeline_diagram.py | (Optional) Manuscript Figure 1 diagram. | figure_1_pipeline_*.png |
| 09 | 09_model_specification.py | Model spec (config, LR coefficients/ORs, RF importances, Platt) + serialised models. | model_spec_*.csv, **reports/model_specification.md**, final_model_rf/lr.pkl |
| 10 | 10_shap_importance.py | SHAP feature importance (bar + beeswarm, both models). | shap_*.png, shap_importance_table.csv |
| 08 | 08_results_report.py | Assembles a journal-ready Results section from every stage above. | **reports/results_report.md**, results_report.docx |

`run_all.py` runs them in this order (08 last, as it summarises the rest). If a
stage fails it is reported but the run continues, so one bad stage never hides
the others.

## Notes

- **Data**: the pipeline starts from the committed, de-identified `data/cleaned.csv`.
  If `data/source_file_bronze.csv` is present locally (not in the repo — it holds
  identifiers), `run_all.py` runs stage `00_prepare_data.py` first to rebuild
  `cleaned.csv` from it.
- **Feature count**: 01b reports the 6-feature stable core selected in all CV folds;
  09 refits on the full dataset and may retain one more (7). Both are reported.
- **Subgroup (06b)**: `OPERATING_THRESHOLD` and `EXCLUDE_SEX_GROUPS` / `MIN_GROUP_N`
  are set near the top; the n=6 "O" sex group is excluded by default. Groupings
  are configured via `EXTENDED_GROUPINGS`; add an entry there to cover another
  available column. ICU and RACF status cannot be reported — not present in
  the current extract.
- **Patient-level CV (06)**: requires `patient_cluster_id` in `cleaned.csv` (added
  by stage 00). If absent, the patient-level sensitivity check falls back to a
  row-index placeholder and prints a warning rather than silently reporting a
  misleading "grouped" result.
- **MICE burn-in**: `FancyIterativeImputerWrapper(max_iterations=5, n_burn_in=3)`
  runs `n_burn_in + max_iterations` total chained-equation cycles and returns
  the final one — this is how "burn-in" maps onto scikit-learn's IterativeImputer,
  which has no native concept of discarding early draws.
- **Serialised models (09)**: `outputs/final_model_rf.pkl` / `final_model_lr.pkl` hold
  the complete pipeline (preprocess + select + model + Platt). Load with joblib and
  call `.predict_proba(X_raw)[:, 1]`.
- Stages are Jupyter-style (`# %%` cells) and also run as plain scripts.
- **EDA**: `eda.py` (repo root) reproduces the cohort's descriptive charts (ported from
  the old R analysis) into `outputs/eda/`. It is exploratory and not part of `run_all.py`.
