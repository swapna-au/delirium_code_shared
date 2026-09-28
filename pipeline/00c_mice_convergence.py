"""
00c_mice_convergence.py — MICE convergence diagnostic.

Checks convergence/mixing of the chained-equation imputation for the numeric
predictors, using the same estimator settings as the modelling stages
(sample_posterior=True, n_burn_in=3; see common.FancyIterativeImputerWrapper).
Runs several independent imputation chains (different random_state) and traces
each variable's mean imputed (previously-missing) value across successive
chained-equation cycles WITHIN each chain.

Tracing a continuing chain
--------------------------
scikit-learn's IterativeImputer only returns the state after its final cycle,
so a per-cycle trace cannot be read off a single fit_transform, and refitting
from scratch at each cycle length would yield independent runs rather than one
chain (successive points would not be states of the same chain, and the trace
could not show mixing in the usual sense).

Instead, each chain is fitted ONCE with max_iter = n_burn_in + n_recorded, and
its stored ``imputation_sequence_`` -- one fitted estimator per (cycle,
feature) -- is then replayed in order, exactly as ``transform`` does, taking a
snapshot after each complete round-robin cycle. Successive snapshots are
therefore successive states of one continuing chain, and burn-in cycles are
discarded from the recorded trace.

Well-behaved MICE shows the traced means fluctuating around a stable level
without drift, and the chains overlapping (mixing) rather than separating.
This is the convergence diagnostic referenced in the Methods.

Outputs: mice_convergence.png, mice_convergence.csv
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

sys.path.insert(0, os.path.dirname(__file__))
from common import CLEANED_CSV, OUTPUT_DIR, RANDOM_STATE

N_CHAINS = 5
N_RECORDED = 10           # recorded cycles per chain, after burn-in
N_BURN_IN = 3             # matches the modelling-stage setting

numeric_cols = ["age_patient", "x4at_score", "total_acb_score", "unique_drug_count"]


def trace_chain(X, miss_mask, seed, n_burn_in=N_BURN_IN, n_recorded=N_RECORDED):
    """Fit one imputation chain and return the mean imputed value per variable
    after each recorded cycle.

    Relies on IterativeImputer's ``imputation_sequence_`` and the private
    ``_initial_imputation`` / ``_impute_one_feature`` helpers to replay the
    fitted per-cycle estimators and snapshot intermediate states, which the
    public API does not expose. Verified against scikit-learn 1.6.1.
    """
    imputer = IterativeImputer(
        max_iter=n_burn_in + n_recorded,
        sample_posterior=True,
        random_state=seed,
    )
    imputer.fit(X)

    for attr in ("imputation_sequence_", "n_iter_"):
        if not hasattr(imputer, attr):
            raise RuntimeError(
                f"IterativeImputer is missing '{attr}'; this scikit-learn "
                "version does not support replaying the imputation sequence."
            )

    n_cycles = imputer.n_iter_
    per_cycle = len(imputer.imputation_sequence_) // n_cycles

    Xt_init, Xt, mask_missing_values, _ = imputer._initial_imputation(X, in_fit=False)

    records = []
    for step, triplet in enumerate(imputer.imputation_sequence_):
        Xt, _ = imputer._impute_one_feature(
            Xt,
            mask_missing_values,
            triplet.feat_idx,
            triplet.neighbor_feat_idx,
            estimator=triplet.estimator,
            fit_mode=False,
        )
        finished_cycle = (step + 1) % per_cycle == 0
        if not finished_cycle:
            continue
        cycle = (step + 1) // per_cycle
        if cycle <= n_burn_in:          # discard burn-in
            continue
        for j, col in enumerate(numeric_cols):
            m = miss_mask[:, j]
            records.append({
                "iteration": cycle - n_burn_in,
                "variable": col,
                "mean_imputed": Xt[m, j].mean() if m.sum() else np.nan,
            })
    return records


def main() -> None:
    print("Loading data...")
    df = pd.read_csv(CLEANED_CSV)
    df = df.dropna(subset=["delirium"])
    X = df[numeric_cols].apply(pd.to_numeric, errors="coerce").values
    miss_mask = np.isnan(X)

    rows = []
    for chain in range(N_CHAINS):
        print(f"Tracing chain {chain + 1}/{N_CHAINS}...")
        for rec in trace_chain(X, miss_mask, seed=RANDOM_STATE + chain):
            rows.append({"chain": chain + 1, **rec})

    trace = pd.DataFrame(rows)
    trace.to_csv(OUTPUT_DIR / "mice_convergence.csv", index=False)
    print("[OK] Saved: mice_convergence.csv")

    vars_with_missing = [c for j, c in enumerate(numeric_cols) if miss_mask[:, j].sum() > 0]
    n = len(vars_with_missing) or 1
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, col in zip(axes[0], vars_with_missing):
        sub = trace[trace["variable"] == col]
        for chain in range(1, N_CHAINS + 1):
            cc = sub[sub["chain"] == chain]
            ax.plot(cc["iteration"], cc["mean_imputed"], marker="o", lw=1.5,
                    alpha=0.8, label=f"Chain {chain}")
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Chained-equation cycle (after {N_BURN_IN} burn-in cycles)")
        ax.set_ylabel("Mean of imputed values")
        ax.grid(alpha=0.3)
    if len(axes[0]):
        axes[0][0].legend(fontsize=8)
    fig.suptitle("MICE convergence: mean of imputed values vs iteration (by chain)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mice_convergence.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] Saved: mice_convergence.png")

    last_cycle = trace["iteration"].max()
    final = trace[trace["iteration"] == last_cycle].groupby("variable")["mean_imputed"].agg(
        ["mean", "std", "min", "max"]).reset_index()
    print("\nFinal-iteration mean of imputed values across chains (mixing check):")
    print(final.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
