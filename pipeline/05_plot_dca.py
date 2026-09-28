"""Publication-ready decision curve analysis (DCA) figure.

Restores the previous (richer) styling -- prevalence reference line, larger
title/legend, matched y-axis padding, clean spines -- while reading the
Harrell-corrected net benefit already computed by 02_bootstrap_validation.py
(bootstrap_decision_curve_analysis.csv). No CI band is drawn: the new
pipeline reports optimism_mcse (Monte Carlo standard error of the bootstrap
optimism estimate) as a convergence diagnostic only, not a valid confidence
interval for the corrected estimate itself -- consistent with
02_bootstrap_validation.py's documented position that a valid CI requires a
separate two-stage bootstrap.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CLEANED_CSV, OUTPUT_DIR


def main():
    output_dir = Path(OUTPUT_DIR)
    data = pd.read_csv(output_dir / "bootstrap_decision_curve_analysis.csv")

    df = pd.read_csv(CLEANED_CSV)
    df = df.drop(columns=["disch_disposition"], errors="ignore")
    df = df.dropna(subset=["delirium"])
    prevalence = df["delirium"].mean()

    thresholds = data.threshold.to_numpy()
    nb_rf = data.nb_rf_corrected.to_numpy()
    nb_lr = data.nb_lr_corrected.to_numpy()
    nb_all = data.nb_treat_all.to_numpy()
    nb_none = data.nb_treat_none.to_numpy()

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(thresholds, nb_rf, lw=2.5, color='#1f77b4', label='Random Forest (Corrected)')
    ax.plot(thresholds, nb_lr, lw=2.5, color='#d62728', label='Logistic Regression (Corrected)')
    ax.plot(thresholds, nb_all, lw=1.8, color='grey', linestyle='--', label='Treat All')
    ax.plot(thresholds, nb_none, lw=1.8, color='black', linestyle=':', label='Treat None')

    ax.axvline(prevalence, color='purple', linestyle=':', alpha=0.6)
    ax.text(prevalence + 0.005, ax.get_ylim()[1] * 0.9,
            f'Prevalence {100 * prevalence:.1f}%', color='purple', fontsize=9)

    ax.set_xlabel('Threshold Probability', fontsize=13)
    ax.set_ylabel('Net Benefit', fontsize=13)
    ax.set_title('Decision Curve Analysis\n(Bootstrap Optimism-Corrected, Platt Calibrated)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(thresholds.min(), thresholds.max())
    nb_max = max(nb_rf.max(), nb_lr.max(), nb_all[0]) * 1.15
    nb_min = max(-0.02, min(nb_rf.min(), nb_lr.min()) - 0.01)
    ax.set_ylim([nb_min, nb_max])
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.3)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "publication_dca.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: publication_dca.png")

    key_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    rows = []
    for threshold in key_thresholds:
        row = data.iloc[(data.threshold - threshold).abs().argmin()]
        rows.append({
            'Threshold': f"{threshold:.0%}",
            'RF (Corrected)': f"{row.nb_rf_corrected:.4f}",
            'LR (Corrected)': f"{row.nb_lr_corrected:.4f}",
            'Treat All': f"{row.nb_treat_all:.4f}",
            'Treat None': '0.0000',
        })
    key_table = pd.DataFrame(rows)
    print("\n" + key_table.to_string(index=False))
    key_table.to_csv(output_dir / "publication_dca_key_thresholds.csv", index=False)
    print("✓ Saved: publication_dca_key_thresholds.csv")


if __name__ == "__main__":
    main()
