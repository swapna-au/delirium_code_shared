"""
verify_outputs.py — golden-baseline regression check for the refactor.

Compares two folders of pipeline outputs and confirms the numbers are
unchanged. This is the safety net: the published results must not move when
we refactor the code.

Protocol
--------
1. BEFORE refactoring, run the pipeline once in the pinned environment and
   keep its outputs as the "baseline" (or use the committed outputs at the
   pre-refactor commit).
2. AFTER refactoring, run the pipeline again into a fresh folder.
3. Diff them:

       python verify_outputs.py <baseline_dir> <current_dir>

   Exit code 0  → every compared file matches (numbers unchanged).
   Exit code 1  → at least one difference (details printed).

Comparison rules
----------------
  *.csv          numeric cells compared within tolerance (default rtol=1e-4,
                 atol=1e-6); non-numeric cells compared exactly; NaN == NaN.
  *.md / *.txt   compared line-by-line (exact).
  *.png etc.     not compared byte-for-byte (figures are not byte-stable);
                 listed for a visual check instead.

Adjust tolerance with --rtol / --atol if a run legitimately differs only in
floating-point noise (e.g. a different BLAS build).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TEXT_EXTS = {".md", ".txt"}
CSV_EXTS = {".csv"}
# Files we do not diff byte-for-byte (regenerated, not byte-stable).
SKIP_EXTS = {".png", ".jpg", ".jpeg", ".pdf", ".docx", ".pkl", ".xlsx", ".html"}


def compare_csv(base: Path, cur: Path, rtol: float, atol: float) -> list[str]:
    """Return a list of human-readable differences (empty means identical)."""
    diffs: list[str] = []
    try:
        a = pd.read_csv(base)
        b = pd.read_csv(cur)
    except Exception as e:  # noqa: BLE001 - report, don't crash the whole run
        return [f"could not read as CSV ({e})"]

    if list(a.columns) != list(b.columns):
        return [f"columns differ:\n    baseline: {list(a.columns)}\n    current:  {list(b.columns)}"]
    if a.shape != b.shape:
        return [f"shape differs: baseline {a.shape} vs current {b.shape}"]

    for col in a.columns:
        s_a, s_b = a[col], b[col]
        if pd.api.types.is_numeric_dtype(s_a) and pd.api.types.is_numeric_dtype(s_b):
            va, vb = s_a.to_numpy(dtype=float), s_b.to_numpy(dtype=float)
            close = np.isclose(va, vb, rtol=rtol, atol=atol, equal_nan=True)
            if not close.all():
                i = int(np.argmax(~close))
                diffs.append(
                    f"column '{col}': {int((~close).sum())} value(s) differ "
                    f"(first at row {i}: baseline={va[i]!r} vs current={vb[i]!r})"
                )
        else:
            # Non-numeric: exact match, treating NaN as equal.
            neq = ~((s_a == s_b) | (s_a.isna() & s_b.isna()))
            if neq.any():
                i = int(neq.to_numpy().argmax())
                diffs.append(
                    f"column '{col}': {int(neq.sum())} value(s) differ "
                    f"(first at row {i}: baseline={s_a.iloc[i]!r} vs current={s_b.iloc[i]!r})"
                )
    return diffs


def compare_text(base: Path, cur: Path) -> list[str]:
    a = base.read_text(encoding="utf-8", errors="replace").splitlines()
    b = cur.read_text(encoding="utf-8", errors="replace").splitlines()
    if a == b:
        return []
    diffs = []
    for n, (la, lb) in enumerate(zip(a, b), 1):
        if la != lb:
            diffs.append(f"line {n}: baseline={la!r} vs current={lb!r}")
            if len(diffs) >= 5:
                diffs.append("... (further line differences suppressed)")
                break
    if len(a) != len(b):
        diffs.append(f"line count differs: baseline {len(a)} vs current {len(b)}")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description="Diff two pipeline-output folders within tolerance.")
    ap.add_argument("baseline", type=Path, help="folder of reference (pre-refactor) outputs")
    ap.add_argument("current", type=Path, help="folder of new (post-refactor) outputs")
    ap.add_argument("--rtol", type=float, default=1e-4, help="relative tolerance for numeric cells")
    ap.add_argument("--atol", type=float, default=1e-6, help="absolute tolerance for numeric cells")
    args = ap.parse_args()

    if not args.baseline.is_dir() or not args.current.is_dir():
        print("error: both arguments must be existing directories", file=sys.stderr)
        return 2

    base_files = {p.relative_to(args.baseline) for p in args.baseline.rglob("*") if p.is_file()}
    cur_files = {p.relative_to(args.current) for p in args.current.rglob("*") if p.is_file()}

    n_ok = n_diff = n_skip = 0
    visual: list[str] = []
    missing = sorted(base_files - cur_files)

    for rel in sorted(base_files & cur_files):
        ext = rel.suffix.lower()
        base, cur = args.baseline / rel, args.current / rel
        if ext in CSV_EXTS:
            diffs = compare_csv(base, cur, args.rtol, args.atol)
        elif ext in TEXT_EXTS:
            diffs = compare_text(base, cur)
        elif ext in SKIP_EXTS:
            n_skip += 1
            visual.append(str(rel))
            continue
        else:
            n_skip += 1
            continue

        if diffs:
            n_diff += 1
            print(f"✗ DIFF  {rel}")
            for d in diffs:
                print(f"        {d}")
        else:
            n_ok += 1
            print(f"✓ ok    {rel}")

    print("\n" + "─" * 60)
    print(f"matched: {n_ok}   differing: {n_diff}   not-compared: {n_skip}")
    if visual:
        print(f"\nBinary/figure files to eyeball ({len(visual)}): " + ", ".join(visual))
    if missing:
        print(f"\nIn baseline but MISSING from current ({len(missing)}):")
        for m in missing:
            print(f"  - {m}")
    extra = sorted(cur_files - base_files)
    if extra:
        print(f"\nNew in current, not in baseline ({len(extra)}): " + ", ".join(str(e) for e in extra))

    if n_diff or missing:
        print("\nRESULT: outputs DIFFER — investigate before trusting the refactor.")
        return 1
    print("\nRESULT: numbers unchanged ✔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
