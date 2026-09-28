"""Publication-ready ROC curve figure.

Restores the previous (richer) figure styling -- shaded bootstrap CI band
around the curve itself, annotated legend, reference line, clean spines --
while sourcing values from the new validation pipeline
(apparent_predictions.csv, bootstrap_performance_summary.csv). The curve and
CI band are apparent (fixed-prediction resampling, no refit); the
Harrell-corrected AUC is shown alongside as a point estimate, consistent with
02_bootstrap_validation.py's documented position that no CI is attached to an
optimism-corrected estimate.

Two resampling schemes appear on this figure and must not be conflated. The
shaded band and the legend's AUC interval are the *same* scheme -- resampling
the fixed apparent predictions, no refitting -- applied to two quantities
(sensitivity at each FPR, and the AUC scalar), so both use the same resample
count, read from bootstrap_performance_summary.csv. The corrected AUC is a
different scheme entirely (Harrell optimism correction, full refit per
repetition); its repetition count is read from the same file.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from common import OUTPUT_DIR


RANDOM_STATE = 42
FPR_GRID = np.linspace(0, 1, 101)


def roc_band(y_true, y_prob, n_boot, random_state=RANDOM_STATE):
    """Bootstrap 95% CI band for the ROC curve (fixed predictions, no refit):
    resample (y, y_prob) pairs with replacement, recompute the ROC curve each
    time, interpolate TPR onto a fixed FPR grid, then take the 2.5/97.5
    percentile of TPR at each grid point across replicates."""
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    tprs = np.full((n_boot, len(FPR_GRID)), np.nan)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(yt, yp)
        tprs[b] = np.interp(FPR_GRID, fpr_b, tpr_b)
    lo = np.nanpercentile(tprs, 2.5, axis=0)
    hi = np.nanpercentile(tprs, 97.5, axis=0)
    return lo, hi


def main():
    output_dir = Path(OUTPUT_DIR)
    predictions = pd.read_csv(output_dir / "apparent_predictions.csv")
    summary = pd.read_csv(output_dir / "bootstrap_performance_summary.csv")
    y = predictions["outcome"].to_numpy()

    # The band reuses the apparent-CI resample count recorded by
    # 02_bootstrap_validation.py, because it is the same resampling scheme
    # applied to a different quantity; taking it from the data keeps the two
    # in step if that setting ever changes.
    n_apparent_resamples = int(summary["apparent_ci_resamples"].iloc[0])

    styles = {
        "RF": ("rf_calibrated", "#1f77b4", 42, "Random Forest"),
        "LR": ("lr_calibrated", "#d62728", 142, "Logistic Regression"),
    }

    fig, ax = plt.subplots(figsize=(6, 6))

    curves = {}
    for model, (column, colour, seed, label) in styles.items():
        p = predictions[column].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        lo, hi = roc_band(y, p, n_boot=n_apparent_resamples, random_state=seed)
        curves[model] = (fpr, tpr, lo, hi, colour)
        ax.fill_between(FPR_GRID, lo, hi, color=colour, alpha=0.15, lw=0)

    for model, (column, colour, seed, label) in styles.items():
        fpr, tpr, lo, hi, colour = curves[model]
        row = summary.query(
            "model == @model and basis == 'calibrated' and metric == 'auc'"
        ).iloc[0]
        ax.plot(
            fpr, tpr, lw=2.5, color=colour,
            # Two short lines rather than one long one: a single-line entry is
            # wide enough that the legend box reaches into the curves.
            label=(
                f"{label}\n"
                f"AUC {row.apparent:.3f} "
                f"({row.apparent_ci_low:.3f}–{row.apparent_ci_high:.3f}); "
                f"corrected {row.corrected:.3f}"
            ),
        )

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Reference line (AUC = 0.50)')

    ax.set_xlabel('1 − Specificity', fontsize=13)
    ax.set_ylabel('Sensitivity', fontsize=13)
    ax.set_title('Receiver Operating Characteristic Curve\n'
                 '(Apparent Performance, Platt Calibrated)',
                 fontsize=13, fontweight='bold')
    # Lower right is the empty region of an ROC plot, so the legend sits there
    # clear of the curves, which hug the upper left.
    ax.legend(fontsize=8.5, loc='lower right', framealpha=0.9,
              labelspacing=0.8, borderpad=0.7)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "publication_roc_calibrated.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: publication_roc_calibrated.png")


if __name__ == "__main__":
    main()
