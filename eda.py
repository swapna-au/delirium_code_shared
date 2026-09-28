"""
eda.py — exploratory data analysis for the delirium cohort.

Python port of the descriptive charts from the retired logistic_regression.rmd.
This is NOT part of run_all.py (it is exploratory, not part of the modelling
pipeline); run it on its own:

    python eda.py

Reads data/cleaned.csv and writes figures to outputs/eda/.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline"))
from common import CLEANED_CSV, OUTPUT_DIR

EDA_DIR = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
PALETTE = "viridis"


def save(fig, name):
    path = EDA_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


def main() -> None:
    df = pd.read_csv(CLEANED_CSV).dropna(subset=["delirium"])
    df["delirium"] = df["delirium"].astype(int)
    print(f"Loaded {len(df):,} episodes; writing EDA figures to {EDA_DIR}")

    # 1. Delirium class balance
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.countplot(data=df, y="delirium", hue="delirium", palette=PALETTE, legend=False, ax=ax)
    ax.set(title="Distribution of delirium cases", ylabel="Delirium", xlabel="Count")
    save(fig, "01_delirium_distribution.png")

    # 2. Sex distribution
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.countplot(data=df, y="sex_patient", hue="sex_patient", palette=PALETTE, legend=False, ax=ax)
    ax.set(title="Sex", ylabel=None, xlabel="Count")
    save(fig, "02_sex_distribution.png")

    # 3. 4AT score by delirium (histogram, free y per facet)
    g = sns.displot(data=df, x="x4at_score", col="delirium", hue="delirium",
                    palette=PALETTE, legend=False, facet_kws={"sharey": False}, height=4)
    g.figure.suptitle("4AT score by delirium status", y=1.02)
    save(g.figure, "03_4at_by_delirium.png")

    # 4. Mean tympanic temperature by delirium (physiological range only)
    temp = df[df["temperature_tympanic_mean"] >= 30]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=temp, x="delirium", y="temperature_tympanic_mean",
                hue="delirium", palette=PALETTE, legend=False, ax=ax)
    ax.set(title="Mean tympanic temperature", xlabel="Delirium")
    save(fig, "04_temperature_by_delirium.png")

    # 5. Mean respiratory rate by delirium
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="delirium", y="respiratory_rate_mean",
                hue="delirium", palette=PALETTE, legend=False, ax=ax)
    ax.set(title="Mean respiratory rate", xlabel="Delirium")
    save(fig, "05_respiratory_by_delirium.png")

    # 6. Dementia x delirium counts (heatmap)
    ct = pd.crosstab(df["dementia"], df["delirium"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(ct, annot=True, fmt=",d", cmap=PALETTE, ax=ax)
    ax.set(title="Dementia vs delirium (counts)", xlabel="Delirium", ylabel="Dementia")
    save(fig, "06_dementia_delirium.png")

    # 7. Pulse rate vs temperature, coloured by delirium
    pt = df[df["temperature_tympanic_mean"] >= 30]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=pt, x="peripheral_pulse_rate_mean", y="temperature_tympanic_mean",
                    hue="delirium", palette=PALETTE, s=8, alpha=0.4, ax=ax)
    ax.set(title="Pulse rate vs temperature", xlabel="Mean peripheral pulse rate",
           ylabel="Mean tympanic temperature")
    save(fig, "07_pulse_temp_scatter.png")

    # 8. Delirium rate by BOC risk (ordinal)
    if "boc_risk_ordinal" in df.columns:
        boc = df.dropna(subset=["boc_risk_ordinal"])
        pct = (boc.groupby("boc_risk_ordinal")["delirium"].mean() * 100).reset_index()
        order = [c for c in ["High", "Moderate", "Low"] if c in pct["boc_risk_ordinal"].values]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=pct, x="boc_risk_ordinal", y="delirium", order=order,
                    hue="boc_risk_ordinal", palette=PALETTE, legend=False, ax=ax)
        ax.set(title="Delirium rate by BOC risk", xlabel="BOC risk", ylabel="Delirium (%)")
        for c in ax.containers:
            ax.bar_label(c, fmt="%.1f%%")
        save(fig, "08_boc_delirium.png")

    # 9. Nurse-verified risk measures by delirium
    nurse_cols = [c for c in df.columns if c.startswith("nurse_verified_")]
    long = df.melt(id_vars="delirium", value_vars=nurse_cols, var_name="measure", value_name="val")
    long = long.dropna(subset=["val"])
    long["measure"] = long["measure"].str.replace("nurse_verified_", "", regex=False)
    g = sns.catplot(data=long, x="val", hue="delirium", col="measure", col_wrap=3,
                    kind="count", palette=PALETTE, height=3, sharey=False)
    g.figure.suptitle("Delirium by nurse-verified measures", y=1.02)
    save(g.figure, "09_nurse_verified.png")

    print("EDA complete.")


if __name__ == "__main__":
    main()
