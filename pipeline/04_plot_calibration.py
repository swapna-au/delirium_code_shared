"""Publication-ready calibration figure.

Restores the previous (richer) styling -- calibration curve with a shaded
bootstrap CI band, prediction histogram along the bottom, Brier Skill Score
-- while sourcing values from the new validation pipeline
(apparent_predictions.csv, bootstrap_performance_summary.csv). Curve and band
are computed on apparent (fixed) predictions; Harrell-corrected Brier/slope
are annotated as point estimates (no CI attached, consistent with
02_bootstrap_validation.py's documented position).

Calibration is summarised by slope, intercept and Brier/Brier skill score
only. Large-sample goodness-of-fit tests are deliberately not reported: at
this sample size they reject essentially regardless of practical calibration
quality, so they carry no useful information here.

The band's resample count is read from bootstrap_performance_summary.csv so
that every fixed-prediction interval in the paper -- this band, the ROC band,
and the apparent AUC CI -- uses one count from a single source.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from common import OUTPUT_DIR


RANDOM_STATE = 42
N_BINS = 10


def metric(summary, model, name):
    return summary.query(
        "model == @model and basis == 'calibrated' and metric == @name"
    ).iloc[0]


def fixed_quantile_edges(y_prob, n_bins=N_BINS):
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(y_prob, qs)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def calib_band(y_true, y_prob, edges, n_boot, random_state=RANDOM_STATE):
    """Bootstrap 95% CI band for the calibration curve (fixed predictions, no
    refit), using bin edges fixed from the ORIGINAL predictions' quantiles so
    every replicate is binned identically and lines up with the plotted curve."""
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    n_bins_ = len(edges) - 1
    rates = np.full((n_boot, n_bins_), np.nan)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        bin_ids = np.clip(np.digitize(yp, edges[1:-1], right=False), 0, n_bins_ - 1)
        for k in range(n_bins_):
            mask = bin_ids == k
            if mask.sum() > 0:
                rates[b, k] = yt[mask].mean()
    lo = np.nanpercentile(rates, 2.5, axis=0)
    hi = np.nanpercentile(rates, 97.5, axis=0)
    return lo, hi


def main():
    output_dir = Path(OUTPUT_DIR)
    predictions = pd.read_csv(output_dir / "apparent_predictions.csv")
    summary = pd.read_csv(output_dir / "bootstrap_performance_summary.csv")
    y = predictions["outcome"].to_numpy()
    y_rf = predictions["rf_calibrated"].to_numpy()
    y_lr = predictions["lr_calibrated"].to_numpy()

    # Same resampling scheme as the ROC band and the apparent AUC CI, so it
    # takes the same count from the run that produced those values.
    n_apparent_resamples = int(summary["apparent_ci_resamples"].iloc[0])

    prevalence = y.mean()
    null_brier = prevalence * (1 - prevalence)

    # ------------------------------------------------------------------------
    # Calibration metrics table (apparent + 95% CI, and Harrell corrected)
    # ------------------------------------------------------------------------
    rf_brier = metric(summary, "RF", "brier")
    lr_brier = metric(summary, "LR", "brier")
    rf_slope = metric(summary, "RF", "calibration_slope")
    lr_slope = metric(summary, "LR", "calibration_slope")
    rf_intercept = metric(summary, "RF", "calibration_intercept")
    lr_intercept = metric(summary, "LR", "calibration_intercept")

    rf_bss = 1 - (brier_score_loss(y, y_rf) / null_brier)
    lr_bss = 1 - (brier_score_loss(y, y_lr) / null_brier)

    print("\n" + "=" * 70)
    print("CALIBRATION METRICS (Apparent 95% CI; Harrell-Corrected point estimate)")
    print("=" * 70)

    metrics_table = pd.DataFrame({
        'Metric': [
            'Brier Score (apparent, 95% CI)',
            'Brier Score (corrected)',
            'Brier Skill Score',
            'Null Brier Score (prevalence x (1-prevalence))',
            'Calibration Slope (apparent, 95% CI)',
            'Calibration Slope (corrected)',
            'Calibration Intercept (apparent, 95% CI)',
            'Calibration Intercept (corrected)',
        ],
        'Random Forest (Calibrated)': [
            f"{rf_brier.apparent:.4f} [{rf_brier.apparent_ci_low:.4f}–{rf_brier.apparent_ci_high:.4f}]",
            f"{rf_brier.corrected:.4f}",
            f"{rf_bss:.4f}",
            f"{null_brier:.4f}",
            f"{rf_slope.apparent:.4f} [{rf_slope.apparent_ci_low:.4f}–{rf_slope.apparent_ci_high:.4f}]",
            f"{rf_slope.corrected:.4f}",
            f"{rf_intercept.apparent:.4f} [{rf_intercept.apparent_ci_low:.4f}–{rf_intercept.apparent_ci_high:.4f}]",
            f"{rf_intercept.corrected:.4f}",
        ],
        'Logistic Regression (Calibrated)': [
            f"{lr_brier.apparent:.4f} [{lr_brier.apparent_ci_low:.4f}–{lr_brier.apparent_ci_high:.4f}]",
            f"{lr_brier.corrected:.4f}",
            f"{lr_bss:.4f}",
            f"{null_brier:.4f}",
            f"{lr_slope.apparent:.4f} [{lr_slope.apparent_ci_low:.4f}–{lr_slope.apparent_ci_high:.4f}]",
            f"{lr_slope.corrected:.4f}",
            f"{lr_intercept.apparent:.4f} [{lr_intercept.apparent_ci_low:.4f}–{lr_intercept.apparent_ci_high:.4f}]",
            f"{lr_intercept.corrected:.4f}",
        ],
    })
    print("\n" + metrics_table.to_string(index=False))
    metrics_table.to_csv(output_dir / "publication_calibration_metrics_v2.csv", index=False)
    print("\n✓ Saved: publication_calibration_metrics_v2.csv")

    # ------------------------------------------------------------------------
    # Figure: calibration curve (top) + prediction histogram (bottom)
    # ------------------------------------------------------------------------
    fig = plt.figure(figsize=(7, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.08)
    ax_cal = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])

    n_bins = N_BINS
    rf_frac, rf_mean = calibration_curve(y, y_rf, n_bins=n_bins, strategy='quantile')
    lr_frac, lr_mean = calibration_curve(y, y_lr, n_bins=n_bins, strategy='quantile')

    rf_edges = fixed_quantile_edges(y_rf)
    lr_edges = fixed_quantile_edges(y_lr)
    rf_frac_lo, rf_frac_hi = calib_band(
        y, y_rf, rf_edges, n_boot=n_apparent_resamples, random_state=RANDOM_STATE
    )
    lr_frac_lo, lr_frac_hi = calib_band(
        y, y_lr, lr_edges, n_boot=n_apparent_resamples, random_state=RANDOM_STATE + 100
    )

    ax_cal.plot([0, 0.50], [0, 0.50], 'k--', lw=1.2, alpha=0.6, label='Perfect calibration')
    ax_cal.fill_between(rf_mean, rf_frac_lo, rf_frac_hi, color='#1f77b4', alpha=0.15, lw=0)
    ax_cal.fill_between(lr_mean, lr_frac_lo, lr_frac_hi, color='#d62728', alpha=0.15, lw=0)

    # Short stacked lines rather than one long one, matching 03_plot_roc.py: a
    # single-line entry is wide enough that the legend box reaches the curves.
    ax_cal.plot(
        rf_mean, rf_frac, 'o-', lw=2.5, color='#1f77b4', markersize=7,
        label=(f"Random Forest\n"
               f"Brier {rf_brier.apparent:.3f} "
               f"({rf_brier.apparent_ci_low:.3f}–{rf_brier.apparent_ci_high:.3f}); "
               f"corrected {rf_brier.corrected:.3f}\n"
               f"Slope {rf_slope.apparent:.3f}; corrected {rf_slope.corrected:.3f}"),
    )
    ax_cal.plot(
        lr_mean, lr_frac, 's-', lw=2.5, color='#d62728', markersize=7,
        label=(f"Logistic Regression\n"
               f"Brier {lr_brier.apparent:.3f} "
               f"({lr_brier.apparent_ci_low:.3f}–{lr_brier.apparent_ci_high:.3f}); "
               f"corrected {lr_brier.corrected:.3f}\n"
               f"Slope {lr_slope.apparent:.3f}; corrected {lr_slope.corrected:.3f}"),
    )

    ax_cal.set_ylabel('Observed Event Rate', fontsize=12)
    ax_cal.set_xlim([0, 0.50])
    ax_cal.set_ylim([0, 0.70])
    # Upper left is the empty region here, since the calibration curves run
    # from lower left to upper right.
    ax_cal.legend(fontsize=8.5, loc='upper left', framealpha=0.9,
                  labelspacing=0.8, borderpad=0.7)
    ax_cal.set_title('Calibration Plot\n(Apparent Performance, Platt Calibrated)',
                      fontsize=13, fontweight='bold')
    ax_cal.grid(alpha=0.3)
    ax_cal.tick_params(labelbottom=False)
    for spine in ('top', 'right'):
        ax_cal.spines[spine].set_visible(False)

    ax_cal.axhline(prevalence, color='grey', linestyle=':', lw=1.2, alpha=0.7)
    ax_cal.text(0.01, prevalence + 0.015, f'Prevalence {100 * prevalence:.1f}%',
                color='grey', fontsize=8)

    ax_hist.hist(y_rf, bins=40, color='#1f77b4', alpha=0.5, label='RF')
    ax_hist.hist(y_lr, bins=40, color='#d62728', alpha=0.5, label='LR')
    ax_hist.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax_hist.set_ylabel('Count', fontsize=10)
    ax_hist.set_xlim([0, 0.50])
    ax_hist.legend(fontsize=9, loc='upper right')
    ax_hist.grid(alpha=0.3)
    for spine in ('top', 'right'):
        ax_hist.spines[spine].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "publication_calibration_plot.png", dpi=300, bbox_inches="tight")
    print("✓ Saved: publication_calibration_plot.png")

    # Retain the compact machine-readable CSV from the new pipeline too.
    rows = []
    for model in ("RF", "LR"):
        for name in ("brier", "calibration_slope", "calibration_intercept", "citl"):
            row = metric(summary, model, name)
            rows.append({
                "model": model,
                "metric": name,
                "apparent": row.apparent,
                "apparent_ci_low": row.apparent_ci_low,
                "apparent_ci_high": row.apparent_ci_high,
                "optimism": row.optimism,
                "corrected": row.corrected,
            })
    pd.DataFrame(rows).to_csv(output_dir / "publication_calibration_metrics.csv", index=False)
    print("✓ Saved: publication_calibration_metrics.csv")


if __name__ == "__main__":
    main()
