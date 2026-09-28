"""Threshold performance table and figure (Sensitivity/Specificity vs threshold).

Restores the previous (richer) styling -- single combined panel for both
models, prevalence reference line, larger fonts, clean spines -- while
reading the Harrell-corrected threshold performance already computed by
02_bootstrap_validation.py (threshold_performance_rf.csv /
threshold_performance_lr.csv).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import CLEANED_CSV, OUTPUT_DIR


def main():
    output_dir = Path(OUTPUT_DIR)
    rf = pd.read_csv(output_dir / "threshold_performance_rf.csv")
    lr = pd.read_csv(output_dir / "threshold_performance_lr.csv")

    df = pd.read_csv(CLEANED_CSV)
    df = df.drop(columns=["disch_disposition"], errors="ignore")
    df = df.dropna(subset=["delirium"])
    prevalence = df["delirium"].mean()

    print("\n" + "=" * 80)
    print("RANDOM FOREST (Calibrated) — Threshold Performance (Optimism-Corrected)")
    print("=" * 80)
    print(rf.round(4).to_string(index=False))

    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION (Calibrated) — Threshold Performance (Optimism-Corrected)")
    print("=" * 80)
    print(lr.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(rf.threshold, rf.sensitivity, lw=2.2, color='#1f77b4', label='RF Sensitivity')
    ax.plot(rf.threshold, rf.specificity, lw=2.2, color='#1f77b4', linestyle='--', label='RF Specificity')
    ax.plot(lr.threshold, lr.sensitivity, lw=2.2, color='#d62728', label='LR Sensitivity')
    ax.plot(lr.threshold, lr.specificity, lw=2.2, color='#d62728', linestyle='--', label='LR Specificity')

    ax.axvline(prevalence, color='purple', linestyle=':', alpha=0.6)
    ax.text(prevalence + 0.005, 0.05, f'Prevalence {100 * prevalence:.1f}%',
            color='purple', fontsize=9)

    ax.set_xlabel('Decision Threshold', fontsize=13)
    ax.set_ylabel('Sensitivity / Specificity', fontsize=13)
    ax.set_title('Sensitivity and Specificity vs Threshold\n(Optimism-Corrected, Platt-Calibrated Models)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim([rf.threshold.min(), rf.threshold.max()])
    ax.set_ylim([0, 1])
    ax.legend(fontsize=9, loc='center right', framealpha=0.9)
    ax.grid(alpha=0.3)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    fig.savefig(
        output_dir / "threshold_sensitivity_specificity.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("\n✓ Saved: threshold_sensitivity_specificity.png")


if __name__ == "__main__":
    main()
