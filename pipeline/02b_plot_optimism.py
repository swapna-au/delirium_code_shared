"""Bootstrap optimism-correction figure.

Replaces an orphaned figure (outputs/bootstrap_optimism_correction.png) left
over from the superseded pipeline, which reported stale values and framed the
correction in terms of out-of-bag performance. Out-of-bag performance is not
the optimism-corrected estimate, and that framing was removed from the
manuscript; this figure reports the Harrell quantities directly.

Panels:
  A  Per-iteration AUC in the bootstrap sample vs. in the original data (RF)
  B  Same for LR -- the gap between the two is the optimism
  C  Distribution of per-iteration optimism, both models
  D  Apparent vs. optimism-corrected AUC (apparent CI shown; the corrected
     estimate carries no interval, consistent with 02_bootstrap_validation.py)
  E  Convergence of the corrected estimate with increasing repetitions

All values are read from the current run's outputs; nothing is recomputed, so
this script is cheap to re-run after 02_bootstrap_validation.py.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUTPUT_DIR

RF_COLOUR = "#1f77b4"
LR_COLOUR = "#d62728"
BASIS = "calibrated"


def main():
    output_dir = Path(OUTPUT_DIR)
    summary = pd.read_csv(output_dir / "bootstrap_performance_summary.csv")
    iters = pd.read_csv(output_dir / "bootstrap_iteration_metrics.csv")
    convergence = pd.read_csv(output_dir / "bootstrap_convergence_diagnostic.csv")

    auc_iters = iters.query("metric == 'auc' and basis == @BASIS")
    auc_summary = summary.query("metric == 'auc' and basis == @BASIS")
    n_reps = int(auc_iters["iteration"].nunique())

    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.28)

    # ---- A/B: in-sample vs original-data AUC, per iteration -----------------
    for col, (model, colour, name) in enumerate(
        [("RF", RF_COLOUR, "Random Forest"), ("LR", LR_COLOUR, "Logistic Regression")]
    ):
        ax = fig.add_subplot(gs[0, col])
        d = auc_iters.query("model == @model")
        ax.scatter(d["test_on_original"], d["bootstrap_sample"], s=14,
                   color=colour, alpha=0.55, edgecolors="none")
        lims = [
            min(d["test_on_original"].min(), d["bootstrap_sample"].min()) - 0.004,
            max(d["test_on_original"].max(), d["bootstrap_sample"].max()) + 0.004,
        ]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("AUC in original data", fontsize=10)
        ax.set_ylabel("AUC in bootstrap sample", fontsize=10)
        ax.set_title(f"{name}: per-iteration performance\n"
                     "(above the line = optimism)", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # ---- C: distribution of optimism ---------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    for model, colour, name in [("RF", RF_COLOUR, "Random Forest"),
                                 ("LR", LR_COLOUR, "Logistic Regression")]:
        d = auc_iters.query("model == @model")["optimism"]
        ax.hist(d, bins=30, color=colour, alpha=0.55,
                label=f"{name} (mean {d.mean():+.4f})")
    ax.axvline(0, color="black", lw=1, alpha=0.7)
    ax.set_xlabel("Optimism in AUC", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title(f"Distribution of optimism\n({n_reps} repetitions)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # ---- D: apparent vs corrected AUC --------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    models = ["RF", "LR"]
    labels = ["Random Forest", "Logistic Regression"]
    x = np.arange(len(models))
    width = 0.36
    apparent = [auc_summary.query("model == @m")["apparent"].iloc[0] for m in models]
    corrected = [auc_summary.query("model == @m")["corrected"].iloc[0] for m in models]
    err = [
        [apparent[i] - auc_summary.query("model == @m")["apparent_ci_low"].iloc[0]
         for i, m in enumerate(models)],
        [auc_summary.query("model == @m")["apparent_ci_high"].iloc[0] - apparent[i]
         for i, m in enumerate(models)],
    ]
    ax.bar(x - width / 2, apparent, width, yerr=err, capsize=4,
           color="#f4a582", label="Apparent (95% CI)")
    ax.bar(x + width / 2, corrected, width, color="#77aadd",
           label="Optimism-corrected")
    for xi, (a, c) in enumerate(zip(apparent, corrected)):
        ax.text(xi - width / 2, a + 0.004, f"{a:.4f}", ha="center", fontsize=8.5)
        ax.text(xi + width / 2, c + 0.004, f"{c:.4f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("AUC", fontsize=10)
    ax.set_ylim(0.85, 0.91)
    ax.set_title("Apparent vs optimism-corrected AUC", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # ---- E: convergence of the corrected estimate --------------------------
    ax = fig.add_subplot(gs[1, 1:])
    conv = convergence[convergence["metric"] == "auc"]
    for model, colour, name in [("RF", RF_COLOUR, "Random Forest"),
                                 ("LR", LR_COLOUR, "Logistic Regression")]:
        d = conv[conv["model"] == model].sort_values("n_iterations")
        ax.errorbar(d["n_iterations"], d["corrected"], yerr=d["optimism_mcse"],
                    marker="o", lw=1.8, capsize=3, color=colour, label=name)
    ax.set_xlabel("Bootstrap repetitions", fontsize=10)
    ax.set_ylabel("Optimism-corrected AUC", fontsize=10)
    ax.set_title("Stability of the corrected estimate with increasing repetitions\n"
                 "(error bars = Monte Carlo SE of the optimism)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.suptitle("Bootstrap optimism correction following internal validation: "
                 "Random Forest and Logistic Regression",
                 fontsize=13, fontweight="bold")

    out = output_dir / "bootstrap_optimism_correction.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out.name}")

    print(f"\nRepetitions: {n_reps}")
    print(auc_summary[["model", "apparent", "apparent_ci_low", "apparent_ci_high",
                       "optimism", "optimism_mcse", "corrected"]].to_string(index=False))


if __name__ == "__main__":
    main()
