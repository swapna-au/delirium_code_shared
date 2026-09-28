#!/usr/bin/env python
"""
run_all.py — run the whole delirium risk-prediction pipeline, end to end.

    python run_all.py

Outputs (figures, tables, models) are written to  outputs/  and the
human-readable result docs to  reports/ .

Data:
  • By default the pipeline starts from the committed, de-identified
    data/cleaned.csv.
  • If you have the raw join data/source_file_bronze.csv locally (it is NOT in
    the repo — it contains identifiers), stage 00 runs first and rebuilds
    cleaned.csv from it.

Each stage is run as its own process; a failing stage is reported but does not
stop the rest, so you always see the full picture in one run.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPE = HERE / "pipeline"
BRONZE = HERE / "data" / "source_file_bronze.csv"

# Ordered stages. 00 is included only when the (local-only) bronze file exists.
STAGES = []
if BRONZE.exists() and (PIPE / "00_prepare_data.py").exists():
    STAGES.append("00_prepare_data.py")
STAGES += [
    "00b_missingness_analysis.py",
    "00c_mice_convergence.py",
    "00d_table1_baseline_characteristics.py",
    "01_cross_validation.py",
    "01b_selected_features_per_fold.py",
    "02_bootstrap_validation.py",
    "03_plot_roc.py",
    "04_plot_calibration.py",
    "05_plot_dca.py",
    "05b_threshold_performance.py",
    "06_sensitivity_analysis.py",
    "06b_subgroup_sex.py",
    "07_figure1_pipeline_diagram.py",
    "07c_consort_diagram.py",
    "09_model_specification.py",
    "10_shap_importance.py",
    "08_results_report.py",          # last — it summarises every other stage
]


def main() -> int:
    env = dict(os.environ, MPLBACKEND="Agg")   # headless plotting
    failed = []
    for stage in STAGES:
        path = PIPE / stage
        if not path.exists():
            print(f"SKIP  {stage} (not found)")
            continue
        print(f"\n{'=' * 72}\n▶  {stage}\n{'=' * 72}", flush=True)
        result = subprocess.run([sys.executable, str(path)], env=env)
        if result.returncode != 0:
            print(f"✗  {stage} FAILED (exit {result.returncode})")
            failed.append(stage)
        else:
            print(f"✓  {stage}")

    print(f"\n{'=' * 72}")
    if failed:
        print(f"Completed with {len(failed)} failure(s): {', '.join(failed)}")
        return 1
    print("✅  Pipeline complete.  Results in  outputs/  and  reports/ .")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
