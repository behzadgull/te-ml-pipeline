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
# Pooled out-of-fold R^2 under chemistry-cluster grouped CV, frozen
# hyperparameters per (target, model_type) via nested_cv.py's tune_once,
# see CLAUDE.md "Confirmed Results -- Five-Way Ladder" and Paper A item
# 1's model-comparison note. sigma/kappa are log10-space R^2 (frozen
# LOG_TRANSFORM_TARGETS decision); S/zT are linear-space. XGBoost is the
# highest-scoring model for every target here, so its chemistry-cluster
# number doubles as the reported "honest ceiling" this project anchors
# to -- the reference line below is that same XGBoost value per property,
# not the independent noise-floor ceiling src/noise_floor.py computes.
MODEL_COMPARISON_PROPERTIES = ["S", "sigma", "kappa", "zT"]
MODEL_COMPARISON_PROPERTY_LABELS = [
    "S", "$\\sigma$ (log$_{10}$)", "$\\kappa$ (log$_{10}$)", "zT",
]
MODEL_COMPARISON_RESULTS = {
    "XGBoost":       {"S": 0.8083, "sigma": 0.7522, "kappa": 0.8226, "zT": 0.7965},
    "LightGBM":      {"S": 0.8054, "sigma": 0.7468, "kappa": 0.8199, "zT": 0.7926},
    "Random Forest": {"S": 0.7802, "sigma": 0.7014, "kappa": 0.7936, "zT": 0.7574},
    "Ridge":         {"S": 0.4316, "sigma": 0.4173, "kappa": 0.5542, "zT": 0.3368},
}
MODEL_COMPARISON_CEILING_MODEL = "XGBoost"

# Five-way validation-inflation ladder, pooled out-of-fold R^2, frozen
# XGBoost hyperparameters reused unchanged across all five rungs -- see
# CLAUDE.md "Confirmed Results -- Five-Way Ladder" (Paper A item 1).
# sigma/kappa in log10 space, S/zT linear, same as MODEL_COMPARISON_RESULTS.
LADDER_PROPERTIES = ["S", "sigma", "kappa", "zT"]
LADDER_PROPERTY_LABELS = [
    "S", "$\\sigma$ (log$_{10}$)", "$\\kappa$ (log$_{10}$)", "zT",
]
LADDER_STRATEGIES = ["Random 80/20", "5-Fold CV", "10-Fold CV", "Composition CV", "Chemistry-Cluster CV"]
LADDER_RESULTS = {
    "Random 80/20":         {"S": 0.9586, "sigma": 0.9539, "kappa": 0.9615, "zT": 0.9138},
    "5-Fold CV":            {"S": 0.9585, "sigma": 0.9533, "kappa": 0.9611, "zT": 0.9132},
    "10-Fold CV":           {"S": 0.9594, "sigma": 0.9550, "kappa": 0.9625, "zT": 0.9148},
    "Composition CV":       {"S": 0.8314, "sigma": 0.7791, "kappa": 0.8380, "zT": 0.8164},
    "Chemistry-Cluster CV": {"S": 0.8083, "sigma": 0.7522, "kappa": 0.8226, "zT": 0.7965},
}
LADDER_LEAKY_STRATEGY = "Random 80/20"
LADDER_HONEST_STRATEGY = "Chemistry-Cluster CV"

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
    sigma and kappa use log10-scale x-axes and log-spaced bins,
    consistent with the space nested_cv.py actually trains and scores
    those two targets in (LOG_TRANSFORM_TARGETS, see CLAUDE.md's noise-
    floor decision); S and zT stay linear, matching their training
    space too. Each panel is annotated with its non-null sample size n
    and arithmetic mean, both in the panel's native (raw, not log)
    units, matching the Reference final-dataset per-property statistics
    table in CLAUDE.md.
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
        mean = values.mean()
        if col == "sigma":
            # Axis is log10-scale here, so annotate log10(mean) rather
            # than the raw-unit mean -- a raw-unit number next to a log
            # axis reads as if it were the log-space value.
            mean_label = f"mean={np.log10(mean):.2f} (log$_{{10}}$ scale)"
        else:
            mean_label = f"mean={mean:,.0f}" if mean >= 100 else f"mean={mean:.2f}"

        if log_scale:
            bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 40)
            ax.set_xscale("log")
        else:
            bins = 40

        ax.hist(values, bins=bins, color=hist_color, edgecolor="white", linewidth=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.text(
            0.97, 0.95, f"n={n:,}\n{mean_label}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
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


def make_model_comparison(out_path):
    """
    Figure 1: grouped bar chart comparing four model families (XGBoost,
    LightGBM, Random Forest, Ridge) on pooled out-of-fold R^2 under
    chemistry-cluster grouped CV, one group per target (S, sigma, kappa,
    zT). A short horizontal reference line spans each group at that
    property's honest chemistry-cluster ceiling (XGBoost's own score --
    the highest-capacity, best-performing model here; see the module-
    level MODEL_COMPARISON_* comment). Ridge's much lower bars visually
    justify choosing a GBDT over a linear baseline, motivating XGBoost
    empirically rather than asserting it.
    """
    model_names = list(MODEL_COMPARISON_RESULTS.keys())
    n_models = len(model_names)
    n_props = len(MODEL_COMPARISON_PROPERTIES)

    # Colorblind-safe, but skip COLORBLIND_PALETTE[0] (black) for the
    # bars themselves -- black is reserved for the dashed ceiling line,
    # and a black bar reads poorly against it. Deep blue / orange /
    # sky blue / bluish green stay distinguishable from each other and
    # from the black reference line.
    bar_colors = [
        COLORBLIND_PALETTE[5],  # deep blue
        COLORBLIND_PALETTE[1],  # orange
        COLORBLIND_PALETTE[2],  # sky blue
        COLORBLIND_PALETTE[3],  # bluish green
    ]

    fig, ax = plt.subplots(figsize=get_figsize(1, 1, panel_width=9.0, panel_height=5.2))

    group_centers = np.arange(n_props)
    bar_width = 0.8 / n_models
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * bar_width

    for i, model in enumerate(model_names):
        heights = [MODEL_COMPARISON_RESULTS[model][prop] for prop in MODEL_COMPARISON_PROPERTIES]
        x = group_centers + offsets[i]
        bars = ax.bar(
            x, heights, width=bar_width * 0.92,
            color=bar_colors[i], label=model, edgecolor="white", linewidth=0.5,
        )
        for rect, h in zip(bars, heights):
            ax.text(
                rect.get_x() + rect.get_width() / 2, h + 0.02, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8, rotation=0,
            )

    # Span the ceiling line only across the XGBoost bar itself, not the
    # full group -- spanning the whole group put the dashed line right
    # through the Random Forest/Ridge value labels for properties where
    # those bars sit close in height to the ceiling.
    xgboost_offset = offsets[model_names.index(MODEL_COMPARISON_CEILING_MODEL)]
    ceiling_label_added = False
    for j, prop in enumerate(MODEL_COMPARISON_PROPERTIES):
        ceiling = MODEL_COMPARISON_RESULTS[MODEL_COMPARISON_CEILING_MODEL][prop]
        line_center = group_centers[j] + xgboost_offset
        ax.plot(
            [line_center - bar_width * 0.6, line_center + bar_width * 0.6], [ceiling, ceiling],
            color="black", linestyle="--", linewidth=1.4, zorder=5,
            label="Honest chemistry-cluster ceiling (XGBoost)" if not ceiling_label_added else None,
        )
        ceiling_label_added = True

    ax.set_xticks(group_centers)
    ax.set_xticklabels(MODEL_COMPARISON_PROPERTY_LABELS)
    ax.set_ylabel("Pooled out-of-fold R$^2$\n(chemistry-cluster CV)")
    ax.set_ylim(0, 1.08)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=4, framealpha=0.9,
        columnspacing=1.2, handletextpad=0.5,
    )

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def make_validation_ladder(out_path):
    """
    Figure 2: five-way validation-inflation ladder, grouped bar chart,
    one group per target (S, sigma, kappa, zT), five validation
    strategies per group (random 80/20, 5-fold, 10-fold, composition,
    chemistry-cluster), pooled out-of-fold R^2, frozen hyperparameters
    reused unchanged across every rung -- see CLAUDE.md "Confirmed
    Results -- Five-Way Ladder". The three ungrouped rungs (random/
    5-fold/10-fold) are shaded with similar grays to visually read as
    "the same, indistinguishable number" (they differ by <0.002);
    composition and chemistry-cluster get distinct, increasingly bold
    colors to mark the two real drops. A bracket to the right of each
    group annotates the inflation gap: random 80/20 minus chemistry-
    cluster, the headline number this ladder exists to report.
    """
    n_props = len(LADDER_PROPERTIES)
    n_strategies = len(LADDER_STRATEGIES)

    # Grays for the three statistically-indistinguishable ungrouped
    # rungs, then a distinct orange for composition, then blue for
    # chemistry-cluster (same blue MODEL_COMPARISON_RESULTS uses for
    # XGBoost's honest number, tying the two figures together).
    bar_colors = ["#c7c7c7", "#999999", "#636363", "#E69F00", "#0072B2"]

    fig, ax = plt.subplots(figsize=get_figsize(1, 1, panel_width=11.5, panel_height=5.5))

    group_centers = np.arange(n_props)
    bar_width = 0.8 / n_strategies
    offsets = (np.arange(n_strategies) - (n_strategies - 1) / 2) * bar_width

    for i, strategy in enumerate(LADDER_STRATEGIES):
        heights = [LADDER_RESULTS[strategy][prop] for prop in LADDER_PROPERTIES]
        x = group_centers + offsets[i]
        bars = ax.bar(
            x, heights, width=bar_width * 0.92,
            color=bar_colors[i], label=strategy, edgecolor="white", linewidth=0.5,
        )
        for rect, h in zip(bars, heights):
            ax.text(
                rect.get_x() + rect.get_width() / 2, h + 0.015, f"{h:.2f}",
                ha="center", va="bottom", fontsize=7.5, rotation=0,
            )

    # Inflation-gap bracket: placed clear of the bars, in the gap before
    # the next group, so it never collides with the tightly-packed
    # value labels above (random/5-fold/10-fold differ by <0.002 from
    # each other, so a line spanning the group would strike through
    # multiple labels at once, same failure mode fixed in Figure 1).
    bracket_x_gap = 0.42
    for j, prop in enumerate(LADDER_PROPERTIES):
        y_leaky = LADDER_RESULTS[LADDER_LEAKY_STRATEGY][prop]
        y_honest = LADDER_RESULTS[LADDER_HONEST_STRATEGY][prop]
        gap = y_leaky - y_honest
        x_bracket = group_centers[j] + bracket_x_gap

        ax.annotate(
            "", xy=(x_bracket, y_honest), xytext=(x_bracket, y_leaky),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.1, shrinkA=0, shrinkB=0),
        )
        # short horizontal tick marks at each end so the bracket reads
        # as spanning exactly the leaky and honest heights
        for y in (y_leaky, y_honest):
            ax.plot(
                [x_bracket - 0.035, x_bracket + 0.035], [y, y],
                color="black", lw=1.1, solid_capstyle="butt",
            )
        ax.text(
            x_bracket + 0.06, (y_leaky + y_honest) / 2, f"$\\Delta$={gap:.3f}",
            ha="left", va="center", fontsize=8.5, fontweight="bold",
        )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(LADDER_PROPERTY_LABELS)
    ax.set_xlim(-0.5, n_props - 1 + 0.62)
    ax.set_ylabel("Pooled out-of-fold R$^2$")
    ax.set_ylim(0, 1.08)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=5, framealpha=0.9,
        columnspacing=1.2, handletextpad=0.5,
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

    make_property_distributions(df, FIGURES_DIR / "fig_property_distributions")
    print("Saved fig_property_distributions.png / .pdf")

    make_cluster_size_distribution(df, FIGURES_DIR / "cluster_size_distribution")
    print("Saved cluster_size_distribution.png / .pdf")

    make_model_comparison(FIGURES_DIR / "fig1_model_comparison")
    print("Saved fig1_model_comparison.png / .pdf")

    make_validation_ladder(FIGURES_DIR / "fig2_validation_ladder")
    print("Saved fig2_validation_ladder.png / .pdf")


if __name__ == "__main__":
    main()
