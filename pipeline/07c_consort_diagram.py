"""
07c_consort_diagram.py — CONSORT-style cohort flow diagram.

Standard CONSORT flowchart conventions: a single vertical flow of retained
cohort boxes down the centre, with exclusion boxes branching off to the side
at each step (connected by a short horizontal arrow), rather than folded into
the main flow. Counts are taken from the funnel instrumentation added to
ingest/join_processed.py ([COHORT FUNNEL] print lines / exclusion print
statements) and data/cleaned.csv for the final analysis cohort.

Outputs: consort_diagram.png (300 DPI), consort_diagram.pdf (vector)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(__file__))
from common import OUTPUT_DIR

MAIN_X0, MAIN_X1 = 2.2, 7.8      # main flow column (x-span)
MAIN_CX = (MAIN_X0 + MAIN_X1) / 2
EXCL_X0, EXCL_X1 = 8.6, 13.4     # exclusion side-box column (x-span)

MAIN_FACE = "#E8F4F8"
EXCL_FACE = "#FDEDEC"
MAIN_EDGE = "black"
EXCL_EDGE = "#C0392B"


def box(ax, x0, x1, y0, y1, lines, face, edge, fontsizes=None, bold_first=True):
    rect = FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.08",
        edgecolor=edge, facecolor=face, linewidth=1.6,
    )
    ax.add_patch(rect)
    cx = (x0 + x1) / 2
    n = len(lines)
    if fontsizes is None:
        fontsizes = [10] + [9] * (n - 1)
    for i, (line, fs) in enumerate(zip(lines, fontsizes)):
        frac = (i + 1) / (n + 1)
        y = y1 - frac * (y1 - y0)
        ax.text(cx, y, line, fontsize=fs, ha="center", va="center",
                fontweight="bold" if (i == 0 and bold_first) else "normal")
    return cx, y0, y1


def down_arrow(ax, x, y_from, y_to):
    ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))


def side_arrow(ax, x_from, x_to, y):
    ax.annotate("", xy=(x_to, y), xytext=(x_from, y),
                arrowprops=dict(arrowstyle="-|>", color=EXCL_EDGE, lw=1.4))


def main():
    fig, ax = plt.subplots(figsize=(13, 12.5), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(5.5, 27)
    ax.axis("off")
    ax.set_title("Cohort Flow Diagram", fontsize=15, fontweight="bold", pad=10)

    # ---- Step 1: raw admissions --------------------------------------------
    y0, y1 = 24.5, 26.2
    box(ax, MAIN_X0, MAIN_X1, y0, y1,
        ["Admissions extracted from EMR", "(Eastern Health, 2023-01-01 to 2024-11-01)",
         "N = 34,632"],
        MAIN_FACE, MAIN_EDGE)
    down_arrow(ax, MAIN_CX, y0, 22.9)

    # ---- Exclusion 1: ward / cancelled admit --------------------------------
    ex_y0, ex_y1 = 23.6, 25.4
    box(ax, EXCL_X0, EXCL_X1, ex_y0, ex_y1,
        ["Excluded (n = 5,665)",
         "• Cancelled admission: 1,896",
         "• BALFRED BHH ward: 2,040",
         "• B1E BHH ward: 298",
         "• HITH BHH (virtual/outreach care): 1,507"],
        EXCL_FACE, EXCL_EDGE, fontsizes=[10, 9, 9, 9, 9])
    side_arrow(ax, MAIN_X1, EXCL_X0, 24.5)

    # ---- Step 2: post ward/cancelled filter ---------------------------------
    y0, y1 = 21.0, 22.5
    box(ax, MAIN_X0, MAIN_X1, y0, y1,
        ["Admissions after cancelled-admission /", "excluded-ward filtering",
         "N = 28,967", "(delirium-positive, any timing: n = 2,334)"],
        MAIN_FACE, MAIN_EDGE)
    down_arrow(ax, MAIN_CX, y0, 19.3)

    # ---- Exclusion 2: missing outcome ---------------------------------------
    ex_y0, ex_y1 = 20.0, 21.4
    box(ax, EXCL_X0, EXCL_X1, ex_y0, ex_y1,
        ["Excluded (n = 2)",
         "• Missing delirium outcome",
         "  (zero matching diagnosis records for",
         "  the encounter — outcome undeterminable)"],
        EXCL_FACE, EXCL_EDGE, fontsizes=[10, 9, 8.5, 8.5])
    side_arrow(ax, MAIN_X1, EXCL_X0, 21.0)

    # ---- Step 3: post missing-outcome exclusion -----------------------------
    y0, y1 = 17.5, 19.0
    box(ax, MAIN_X0, MAIN_X1, y0, y1,
        ["Admissions with a determinable delirium outcome",
         "N = 28,965", "(delirium-positive, any timing: n = 2,334)"],
        MAIN_FACE, MAIN_EDGE, fontsizes=[10, 11, 9])
    down_arrow(ax, MAIN_CX, y0, 15.8)

    # ---- Exclusion 3: substance use disorder --------------------------------
    ex_y0, ex_y1 = 16.5, 17.9
    box(ax, EXCL_X0, EXCL_X1, ex_y0, ex_y1,
        ["Excluded (n = 400)",
         "• Acute substance use disorder diagnosis",
         "  (alcohol / opioid / sedative / other-drug",
         "  intoxication, withdrawal, or dependence)",
         "  of which 50 were also delirium-positive"],
        EXCL_FACE, EXCL_EDGE, fontsizes=[10, 9, 8.5, 8.5, 8.5])
    side_arrow(ax, MAIN_X1, EXCL_X0, 17.5)

    # ---- Step 4: final analysis cohort --------------------------------------
    y0, y1 = 14.0, 15.8
    box(ax, MAIN_X0, MAIN_X1, y0, y1,
        ["Final analysis cohort",
         "N = 28,565 admission episodes",
         "(20,136 unique patients; 4,946 with >1 admission)",
         "Delirium: n = 2,284 (8.0%)"],
        MAIN_FACE, MAIN_EDGE, fontsizes=[11, 11, 9, 10])

    # ---- Footnote ------------------------------------------------------------
    ax.text(
        MAIN_CX, 12.9,
        "Delirium outcome: any delirium diagnosis coded during the encounter (any timing).\n"
        "An admission-timing (incident vs. prevalent) definition was investigated and not used\n"
        "(see manuscript Limitations) — diagnosis timestamps could not be reliably distinguished\n"
        "from discharge-coding lag for the diagnosis types carrying most delirium codes.\n\n"
        "Encounters excluded for a missing outcome (n = 2) had zero matching rows in the\n"
        "diagnosis extract at all, so delirium status could not be determined from this data\n"
        "(not the same as a negative/assessed result) and were excluded rather than imputed.",
        fontsize=8.5, ha="center", va="top", style="italic", color="#444444",
    )

    plt.tight_layout()
    png_path = os.path.join(str(OUTPUT_DIR), "consort_diagram.png")
    pdf_path = os.path.join(str(OUTPUT_DIR), "consort_diagram.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")


if __name__ == "__main__":
    main()
