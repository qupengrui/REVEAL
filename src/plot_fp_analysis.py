#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot the function-level false-positive (FP) / line-level behaviour analysis.

Inputs (produced by the FP analysis script):
    Summary.csv        metric, value
    Distribution.csv   bucket, count, ratio
    FP_Samples.csv     one row per function-level FP sample

Outputs:
    FP_line_distribution.pdf  (vector, for the paper)
    FP_line_distribution.png  (raster preview)

Usage:
    python plot_fp_analysis.py --dir <FP_Analysis_csv_folder>
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- style ----
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

C_ZERO = "#b8bdc4"      # neutral grey for the "0 line" bucket
C_POS = "#2f6db5"       # single accent colour for buckets with >0 lines
C_ECDF = "#1f4e79"
C_MARK = "#c0392b"

# Chinese bucket labels -> English labels used in the paper
BUCKET_MAP = {
    "0 行": "0", "1 行": "1", "2 行": "2", "3 行": "3",
    "4-5 行": "4--5", "6-10 行": "6--10",
    "11-20 行": "11--20", ">20 行": ">20",
}
# order used in the paper figure (rows missing from the CSV are kept as 0)
BUCKET_ORDER = ["0", "1", "2", "3", "4--5", "6--10", "11--20", ">20"]


def load(input_dir):
    summary = pd.read_csv(os.path.join(input_dir, "Summary.csv"),
                          encoding="utf-8-sig", header=None, names=["k", "v"])
    summary = summary.dropna(subset=["k"])
    summary["k"] = summary["k"].astype(str).str.strip()
    summary["v"] = summary["v"].astype(str).str.strip()
    s = dict(zip(summary["k"], summary["v"]))

    dist = pd.read_csv(os.path.join(input_dir, "Distribution.csv"),
                       encoding="utf-8-sig")
    dist.columns = ["bucket", "count", "ratio"]
    dist["bucket"] = dist["bucket"].astype(str).str.strip().map(
        lambda x: BUCKET_MAP.get(x, x))

    fp = pd.read_csv(os.path.join(input_dir, "FP_Samples.csv"),
                     encoding="utf-8-sig")
    fp.columns = [c.strip() for c in fp.columns]
    return s, dist, fp


def panel_a(ax, dist, n_fp):
    """Bar chart of the per-function falsely-flagged line count buckets."""
    counts = {b: 0 for b in BUCKET_ORDER}
    for b, c in zip(dist["bucket"], dist["count"]):
        if b in counts:
            counts[b] = int(c)

    x = np.arange(len(BUCKET_ORDER))
    vals = [counts[b] for b in BUCKET_ORDER]
    colors = [C_ZERO if b == "0" else C_POS for b in BUCKET_ORDER]

    bars = ax.bar(x, vals, width=0.68, color=colors,
                  edgecolor="black", linewidth=0.4, zorder=3)

    ymax = max(vals) * 1.22
    for xi, v in zip(x, vals):
        if v == 0:
            continue
        ax.text(xi, v + ymax * 0.02, f"{v} ({v / n_fp * 100:.1f}%)",
                ha="center", va="bottom", fontsize=6.4, linespacing=1.1)

    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_ORDER)
    ax.set_xlabel("No. of falsely flagged lines per FP function")
    ax.set_ylabel("No. of function-level FP samples")
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("(a) Distribution of line-level over-reporting", loc="left",
                 fontweight="bold", pad=4)
    return bars


def panel_b(ax, fp, n_fp, stats):
    """ECDF of the per-function falsely-flagged line count."""
    v = np.sort(fp["n_line_fp"].astype(float).values)
    y = np.arange(1, len(v) + 1) / len(v) * 100.0

    # step plot, starting from (0, 0)
    ax.step(np.concatenate(([0], v)), np.concatenate(([0], y)),
            where="post", color=C_ECDF, linewidth=1.1, zorder=3)

    # annotate the cumulative share at a few integer cut-offs
    cuts = [0, 4, 9]
    for xv in cuts:
        yv = float(np.mean(v <= xv) * 100)
        ax.plot([xv], [yv], marker="o", ms=3.0, mfc="white", mec=C_MARK,
                mew=0.8, zorder=4)
        ax.annotate(f"{yv:.1f}% ($\\leq${xv})", xy=(xv, yv),
                    xytext=(6, -10), textcoords="offset points",
                    fontsize=6.2, color=C_MARK)

    ax.set_xlabel("No. of falsely flagged lines per FP function")
    ax.set_ylabel("Cumulative proportion of FP samples (%)")
    ax.set_xlim(-0.5, max(v) + 1.0)
    ax.set_ylim(0, 105)
    ax.grid(linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("(b) Cumulative distribution", loc="left",
                 fontweight="bold", pad=4)

    info = ("median = {:g}\nmean = {:.2f}\nP90 = {:g}, P95 = {:g}\nmax = {:g}"
            .format(stats["median"], stats["mean"], stats["p90"],
                    stats["p95"], stats["max"]))
    ax.text(0.97, 0.10, info, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.2, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999999",
                      lw=0.4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        default=r"D:\Python_Line-level_Vulnerability-detection\save_file\FP_Analysis"
                r"\func_fp_line_analysis_epoch_042__f1_0.4623_20261030_104052_csv")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--scatter", action="store_true",
                    help="also draw the optional P(y=1) vs. line-FP scatter plot")
    args = ap.parse_args()

    summary, dist, fp = load(args.dir)
    n_fp = len(fp)

    v = fp["n_line_fp"].astype(float).values
    stats = {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "p95": float(np.percentile(v, 95)),
        "max": float(np.max(v)),
    }

    print("FP samples           :", n_fp)
    print("with >=1 line FP     :", int((v > 0).sum()),
          "({:.2f}%)".format((v > 0).mean() * 100))
    print("total line FP        :", int(v.sum()))
    print("stats                :", stats)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    panel_a(axes[0], dist, n_fp)
    panel_b(axes[1], fp, n_fp, stats)
    fig.subplots_adjust(wspace=0.32, left=0.09, right=0.99,
                        top=0.90, bottom=0.17)

    for ext in ("pdf", "png"):
        path = os.path.join(args.outdir, "FP_line_distribution." + ext)
        fig.savefig(path)
        print("saved:", path)
    plt.close(fig)

    if args.scatter:
        fig2, ax2 = plt.subplots(figsize=(3.4, 2.7))
        prob = fp["func_prob"].astype(float).values
        jitter = (np.random.RandomState(0).rand(len(v)) - 0.5) * 0.35
        ax2.scatter(prob, v + jitter, s=12, c=C_POS, alpha=0.75,
                    edgecolors="black", linewidths=0.3, zorder=3)
        ax2.set_xlabel("Function-level predicted probability $P(y=1)$")
        ax2.set_ylabel("No. of falsely flagged lines")
        ax2.set_xlim(0.49, 1.01)
        ax2.grid(linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)
        ax2.set_axisbelow(True)
        for side in ("top", "right"):
            ax2.spines[side].set_visible(False)
        r = np.corrcoef(prob, v)[0, 1]
        ax2.text(0.97, 0.94, f"Pearson $r$ = {r:.2f}", transform=ax2.transAxes,
                 ha="right", va="top", fontsize=6.4)
        fig2.tight_layout()
        for ext in ("pdf", "png"):
            path = os.path.join(args.outdir, "FP_prob_vs_linefp." + ext)
            fig2.savefig(path)
            print("saved:", path)
        plt.close(fig2)


if __name__ == "__main__":
    main()
