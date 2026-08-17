"""
Generate the dataset-construction and descriptive figures used in the
methods section of both papers: the 11-step cleaning funnel, the
post-cleaning property distributions, and the chemistry-cluster size
distribution that motivates repeated grouped CV. Reads the cleaned CSV
written by src/data_cleaning.py; run that first.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.plotting_style import (
    COLORBLIND_PALETTE,
    add_panel_label,
    apply_style,
    get_figsize,
    save_figure,
)

FIGURES_DIR = Path("figures")

# Actual row counts from the run_cleaning_pipeline() run logged against
# the 2026-08-15 ThermoelectricMaterials pull (data/processed/cleaned_
# ThermoelectricMaterials_2026-08-15.csv), re-run 2026-08-17 after fixing
# steps 8/9 to drop rows (not NaN-out values) and the sigma lower bound
# (1 -> 10 S/m) to match the thesis, see CLAUDE.md's Data Cleaning
# Pipeline section. Step 1 is an expansion (each raw curve digitizes into
# many temperature-property points), so it is annotated separately
# rather than folded into the monotonic funnel.
RAW_CURVES = 155_758
CLEANING_STEPS = [
    ("1. Property extraction\n& range filtering", 1_992_138),
    ("2. Integration &\nconsolidation", 1_992_138),
    ("3. Temperature filtering\n300-800K", 1_093_377),
    ("4. Pivot long→wide", 397_791),
    ("5. Formula cleaning", 392_927),
    ("6. zT self-consistency\ncheck", 388_221),
    ("7. DFT data removal", 387_235),
    ("8. Multi-source\nconsistency filtering", 308_656),
    ("9. MAD outlier filter", 289_318),
    ("10. Min. temperature\ncoverage", 284_671),
    ("11. Smoothness filter", 280_348),
]


def load_cleaned_dataset(processed_data_dir="data/processed", project="ThermoelectricMaterials"):
    """Load the most recently written cleaned_<project>_<date>.csv."""
    candidates = sorted(Path(processed_data_dir).glob(f"cleaned_{project}_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No cleaned_{project}_*.csv found in {processed_data_dir}; run src/data_cleaning.py first"
        )
    return pd.read_csv(candidates[-1]), candidates[-1]


def make_cleaning_funnel(out_path):
    """
    Horizontal funnel chart of row count after each of the 11 cleaning
    steps, annotated with n and step-to-step retention. Step 1's
    point-per-curve expansion is called out separately since a literal
    funnel (monotonic decrease) only starts from step 1 onward.
    """
    labels = [label for label, _ in CLEANING_STEPS]
    counts = [count for _, count in CLEANING_STEPS]

    fig, ax = plt.subplots(figsize=get_figsize(1, 1, panel_width=7.0, panel_height=6.0))

    y_pos = np.arange(len(labels))[::-1]
    max_count = max(counts)
    bar_color = COLORBLIND_PALETTE[5]

    ax.barh(y_pos, counts, color=bar_color, height=0.6)
    for i, (y, count) in enumerate(zip(y_pos, counts)):
        if i == 0:
            # step 1 expands curves into points -- not a same-unit
            # reduction, so a "% kept" figure here would be nonsensical.
            label = f"n={count:,}"
        else:
            retention = counts[i] / counts[i - 1]
            label = f"n={count:,}  ({retention:.0%} kept)"
        ax.text(count + max_count * 0.015, y, label, va="center", ha="left", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Rows (temperature-property points)")
    ax.set_xlim(0, max_count * 1.32)

    fig.tight_layout()
    fig.text(
        0.01,
        -0.02,
        f"Note: step 1 input is {RAW_CURVES:,} raw digitized curves (curve-level), "
        f"not point-level rows -- each curve expands into multiple\n"
        f"temperature-property points, so step 1's {counts[0]:,} is an expansion, not a reduction.",
        ha="left",
        va="top",
        fontsize=8,
        style="italic",
    )
    save_figure(fig, out_path)
    plt.close(fig)


def make_property_distributions(df, out_path):
    """
    2x2 panel of histograms for S, sigma, kappa, zT after cleaning.
    sigma and kappa use log-scale x-axes (consistent with the log-space
    MAD filtering and noise-floor treatment used elsewhere in the
    pipeline); S and zT use linear axes. Each panel is annotated with
    its sample size n.
    """
    panels = [
        ("S", "Seebeck coefficient (μV/K)", False),
        ("sigma", "Electrical conductivity (S/m)", True),
        ("kappa", "Thermal conductivity (W/m·K)", True),
        ("zT", "Figure of merit zT", False),
    ]
    panel_letters = ["a", "b", "c", "d"]

    fig, axes = plt.subplots(2, 2, figsize=get_figsize(2, 2))
    hist_color = COLORBLIND_PALETTE[5]

    for ax, (col, xlabel, log_scale), letter in zip(axes.flat, panels, panel_letters):
        values = df[col].dropna()
        n = len(values)

        if log_scale:
            bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 40)
            ax.set_xscale("log")
        else:
            bins = 40

        ax.hist(values, bins=bins, color=hist_color, edgecolor="white", linewidth=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.text(
            0.97, 0.95, f"n={n:,}", transform=ax.transAxes, ha="right", va="top", fontsize=10,
        )
        add_panel_label(ax, letter)

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def make_cluster_size_distribution(df, out_path):
    """
    Histogram of chemistry-cluster sizes: number of distinct sample_id
    values grouped under each chemistry_cluster_id, log-scale x-axis
    given the expected heavy right skew (many singleton/small clusters,
    a handful of large ones). Motivates repeated grouped CV over a
    single grouped split.
    """
    cluster_sizes = df.groupby("chemistry_cluster_id")["sample_id"].nunique()

    fig, ax = plt.subplots(figsize=get_figsize(1, 1, panel_width=5.0, panel_height=3.8))
    bins = np.logspace(0, np.log10(cluster_sizes.max()), 30)
    ax.hist(cluster_sizes, bins=bins, color=COLORBLIND_PALETTE[3], edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("Samples per chemistry cluster")
    ax.set_ylabel("Number of clusters")

    stats_text = (
        f"n clusters = {len(cluster_sizes):,}\n"
        f"median = {cluster_sizes.median():.0f}\n"
        f"max = {cluster_sizes.max():,}\n"
        f"clusters with 1 sample = {(cluster_sizes == 1).sum():,} "
        f"({(cluster_sizes == 1).mean():.0%})"
    )
    ax.text(
        0.97, 0.95, stats_text, transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", linewidth=0.5),
    )

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def main():
    apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df, source_path = load_cleaned_dataset()
    print(f"Loaded {len(df):,} rows from {source_path}")

    make_cleaning_funnel(FIGURES_DIR / "cleaning_funnel")
    print("Saved cleaning_funnel.png / .pdf")

    make_property_distributions(df, FIGURES_DIR / "property_distributions")
    print("Saved property_distributions.png / .pdf")

    make_cluster_size_distribution(df, FIGURES_DIR / "cluster_size_distribution")
    print("Saved cluster_size_distribution.png / .pdf")


if __name__ == "__main__":
    main()
