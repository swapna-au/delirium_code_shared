"""Subgroup (fairness) tables and figures from the primary bootstrap output.

Restores the previous (richer) styling -- bar-value labels, fixed y-axis
range, per-grouping panel grid with a shared legend, clean spines -- while
reading the Harrell-corrected subgroup performance already computed by
02_bootstrap_validation.py (subgroup_performance_extended.csv). Groupings
plotted are whichever are actually present in that file (currently Sex and
Age; groupings only get produced there when a mask exists for them --
see subgroup_masks() in 02_bootstrap_validation.py).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUTPUT_DIR


def auc_bar_panel(ax, sub_table, group_order, title):
    x = np.arange(len(group_order))
    width = 0.35
    rf_auc = [
        sub_table[(sub_table.model == "RF") & (sub_table["_label"] == g)].auc.values
        for g in group_order
    ]
    lr_auc = [
        sub_table[(sub_table.model == "LR") & (sub_table["_label"] == g)].auc.values
        for g in group_order
    ]
    rf_auc = [v[0] if len(v) else np.nan for v in rf_auc]
    lr_auc = [v[0] if len(v) else np.nan for v in lr_auc]

    b1 = ax.bar(x - width / 2, rf_auc, width, label='Random Forest', color='#1f77b4', alpha=0.85)
    b2 = ax.bar(x + width / 2, lr_auc, width, label='Logistic Regression', color='#d62728', alpha=0.85)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f'{h:.3f}',
                        ha='center', va='bottom', fontsize=7)
    ax.set_ylabel('Optimism-Corrected AUC', fontsize=10)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(group_order, rotation=30, ha='right', fontsize=8)
    ax.set_ylim([0.5, 1.0])
    ax.grid(axis='y', alpha=0.3)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)


def main():
    output_dir = Path(OUTPUT_DIR)
    data = pd.read_csv(output_dir / "subgroup_performance_extended.csv")

    # Backward-compatible sex-only view (Sex groups + Overall, same as before).
    sex_table = data.query("grouping in ['Sex', 'Overall']").copy()
    sex_table.to_csv(output_dir / "subgroup_performance_by_sex.csv", index=False)
    print("✓ Saved: subgroup_performance_by_sex.csv")

    # ── Figure 1: sex only (backward-compatible filename) ──────────────────
    sex_only = data.query("grouping == 'Sex'").copy()
    sex_only["_label"] = sex_only["group"]
    if not sex_only.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sex_group_order = list(dict.fromkeys(sex_only["group"]))
        auc_bar_panel(ax, sex_only, sex_group_order,
                      'Discrimination (AUC) by Sex Subgroup\n(Optimism-Corrected)')
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(output_dir / "subgroup_auc_by_sex.png", dpi=300, bbox_inches='tight')
        print("✓ Saved: subgroup_auc_by_sex.png")

    # ── Figure 2: all available groupings (one panel per grouping variable) ─
    groupings = [g for g in data.grouping.unique() if g != "Overall"]
    if not groupings:
        return
    ncols = min(3, len(groupings))
    nrows = int(np.ceil(len(groupings) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)
    table_ext = data.query("grouping != 'Overall'").copy()
    table_ext["_label"] = table_ext["group"]
    for i, grouping in enumerate(groupings):
        ax = axes[i // ncols][i % ncols]
        sub = table_ext[table_ext["grouping"] == grouping]
        if sub.empty:
            ax.axis('off')
            continue
        order = list(dict.fromkeys(sub["_label"]))
        auc_bar_panel(ax, sub, order, grouping)
    for j in range(len(groupings), nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Discrimination (AUC) by Subgroup — Extended Analysis\n(Optimism-Corrected)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(output_dir / "subgroup_auc_extended.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: subgroup_auc_extended.png")


if __name__ == "__main__":
    main()
