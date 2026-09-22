"""
Generate the dataset-construction and descriptive figures used in the
methods section of both papers: the 11-step cleaning funnel, the
post-cleaning property distributions, and the chemistry-cluster size
distribution that motivates repeated grouped CV. Reads the cleaned CSV
written by src/data_cleaning.py; run that first.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.plotting_style import (
    COLORBLIND_PALETTE,
    add_panel_label,
    apply_style,
    get_figsize,
    save_figure,
)

FIGURES_DIR = Path("figures")

# Figure 3: actual-vs-predicted scatter, chemistry-cluster CV, XGBoost.
# Reads pooled out-of-fold predictions checkpointed from the real Kaggle
# production run (device=cuda, frozen hyperparameters, 5 repeats x 5
# outer folds) -- these are NOT regenerable locally (no GPU, see
# CLAUDE.md's Local dev environment note), so this figure only runs if
# that checkpoint tree has been copied in under FIG3_CHECKPOINT_DIR.
FIG3_CHECKPOINT_DIR = Path("checkpoints") / "saved_predictions" / "checkpoints"
FIG3_TARGETS = ["S", "sigma", "kappa", "zT"]
FIG3_PANEL_LETTERS = ["a", "b", "c", "d"]
# sigma/kappa checkpoints hold log10-space y_true/y_pred (target_scale
# in run_config.json), matching nested_cv.py's LOG_TRANSFORM_TARGETS --
# axis labels reflect that; S/zT are linear/raw, same as their training
# scale.
FIG3_AXIS_LABELS = {
    "S": "S (μV/K)",
    "sigma": "log$_{10}(\\sigma$, S/m)",
    "kappa": "log$_{10}(\\kappa$, W/m·K)",
    "zT": "zT",
}

# Row counts for File A (the canonical dataset, 2026-08-22 pull -- see
# CLAUDE.md's Canonical Dataset section), from
# results/cleaning_funnel/20260914T100914/funnel_counts.json, produced by
# running src/data_cleaning.py's 11 step functions individually against
# the raw files at checkpoints/saved_predictions/te-ml-pipeline/data/raw/
# (File A's own raw pull, confirmed via its extraction_metadata.json:
# upstream_db_snapshot 2026-08-22, 9,494/55,261/156,101 papers/samples/
# curves). Verified 2026-09-14: step 11's output (280,664) matches File
# A's existing cleaned CSV on disk exactly (checkpoints/saved_predictions/
# te-ml-pipeline/data/processed/cleaned_ThermoelectricMaterials_2026-08-15.csv),
# confirming the pipeline is deterministic and reproduces File A's actual
# cleaned dataset. Step 1 is an expansion (each raw curve digitizes into
# many temperature-property points), so it is annotated separately
# rather than folded into the monotonic funnel.
#
# The previous values here (see FILE B (SUPERSEDED) block below) were
# File B's (the 2026-08-15 pull) 2026-08-17 re-run numbers, mistakenly
# left in place after File A became canonical on 2026-09-11 -- this is
# also what checkpoints/saved_predictions/te-ml-pipeline/figures/
# cleaning_funnel.png (mislabeled: sits in File A's directory tree but
# renders File B's funnel) was generated from. File A's own funnel had
# never been computed until 2026-09-14.
# STALE (flagged 2026-09-19, values NOT changed): this entire table
# predates BOTH grouping fixes (SNAP(0.05) chemistry_cluster_id, S5
# randomized_group_kfold, commit c1c6873) and the reproducible 2026-08-22
# checkpoint set -- every cell, including the XGBoost row, is from the
# orphaned pre-2026-08-22 run CLAUDE.md's Five-Way Ladder section
# describes as "none of which could be verified or reproduced" (XGBoost
# row here == that section's quoted orphaned values: S=0.8083,
# sigma=0.7522, kappa=0.8226, zT=0.7965). It is therefore two
# regenerations behind the current snapfix ladder, not one. Regenerating
# this figure for real needs LightGBM/Random Forest/Ridge chemistry-
# cluster refits against the snapfix CSV (tune_once + 5 repeats x 5 outer
# folds each per model, via nested_cv.py's --model flag and the
# six-command workflow in CLAUDE.md's Paper A item 1) -- only XGBoost's
# cell could be swapped in today (from results/ladder_regen_snapfix/
# 20260917T150000/), so nothing below is touched rather than partially
# updated. See the make_figures.py task report for the compute-cost
# estimate of the full rerun.
#
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
#
# Composition and Chemistry-Cluster CV rows: results/ladder_regen_snapfix/
# 20260917T150000/{S,sigma,kappa,zT}_{composition,chemistry}_full/,
# tabulated in reports/regen_snapfix/20260917T150000/ladder_metrics.json
# -- regenerated against the fixed grouping (SNAP(0.05)
# chemistry_cluster_id, S5 randomized_group_kfold, commit c1c6873) and the
# snapfix featurized CSV. Point estimates only (bar heights below); each
# also carries an across-5-repeat SD in CLAUDE.md's Five-Way Ladder table
# (composition: S +/-0.0044, sigma +/-0.0015, kappa +/-0.0013, zT
# +/-0.0009; chemistry: S +/-0.0050, sigma +/-0.0020, kappa +/-0.0021, zT
# +/-0.0045) not plotted here, since this figure draws single bar heights
# with no error-bar mechanism.
#
# Random 80/20, 5-Fold, and 10-Fold rows: checkpoints/ladder_regen_dl/
# {target}_{composition,kfold,random}_f{5,10,20}/ (folder-name suffix is
# outer-fold count, e.g. sigma_random_f20, kappa_kfold_f10), each with a
# run_config.json confirming target_scale=log10 (sigma/kappa), seed=0,
# and the canonical frozen-hyperparameters path. Pooled R^2 recomputed
# directly from these checkpoints' *_predictions.npz files matches
# CLAUDE.md's confirmed ungrouped row to 4 decimals on every target.
# These rungs are unaffected by the SNAP(0.05)/S5 grouping fix for a
# structural reason, not just a code-path one: the snapfix featurized CSV
# differs from File A only in the chemistry_cluster_id column (see
# CLAUDE.md's Grouping Fixes section), and the ungrouped split code path
# never reads that column, so re-running it against the snapfix CSV
# cannot change its output.
#
# RESOLVED 2026-09-22: the previous values here (S=0.9586, sigma=0.9539,
# kappa=0.9615, zT=0.9138 for random 80/20, etc.) were NOT a log10-vs-
# linear scale mismatch -- both this row and the current one use log10
# for sigma/kappa. They were the orphaned pre-2026-08-22 run's ungrouped
# cells (found verbatim, all three rows, in checkpoints/saved_predictions/
# te-ml-pipeline/CLAUDE.md, a stale in-tree snapshot of an earlier
# CLAUDE.md whose own Five-Way Ladder table -- itself explicitly log10-
# space -- matches this file's old LADDER_RESULTS on all five rungs) --
# the same run current CLAUDE.md already calls out as superseded and
# unreproducible for its composition/chemistry cells. No run_config.json
# for that orphaned run exists anywhere on disk. See the PRE-FIX
# (SUPERSEDED) block below for the old values, kept for audit.
LADDER_PROPERTIES = ["S", "sigma", "kappa", "zT"]
LADDER_PROPERTY_LABELS = [
    "S", "$\\sigma$ (log$_{10}$)", "$\\kappa$ (log$_{10}$)", "zT",
]
LADDER_STRATEGIES = ["Random 80/20", "5-Fold CV", "10-Fold CV", "Composition CV", "Chemistry-Cluster CV"]

# PRE-FIX (SUPERSEDED 2026-09-19, ungrouped rows added 2026-09-22) -- all
# five rows below are the orphaned pre-2026-08-22 run (same origin as
# MODEL_COMPARISON_RESULTS's stale table above). Composition/chemistry
# were superseded twice over (the reproducible 2026-08-22 checkpoint set,
# then the SNAP(0.05)/S5 grouping fix); random/5-fold/10-fold are
# superseded once (by the reproducible 2026-08-22 checkpoint set only --
# the grouping fix does not touch them, see the comment above). Kept for
# audit, not deleted -- see CLAUDE.md's Five-Way Ladder section,
# "SUPERSEDED 2026-09-19" entries.
# "Random 80/20":         {"S": 0.9586, "sigma": 0.9539, "kappa": 0.9615, "zT": 0.9138},
# "5-Fold CV":            {"S": 0.9585, "sigma": 0.9533, "kappa": 0.9611, "zT": 0.9132},
# "10-Fold CV":           {"S": 0.9594, "sigma": 0.9550, "kappa": 0.9625, "zT": 0.9148},
# "Composition CV":       {"S": 0.8314, "sigma": 0.7791, "kappa": 0.8380, "zT": 0.8164},
# "Chemistry-Cluster CV": {"S": 0.8083, "sigma": 0.7522, "kappa": 0.8226, "zT": 0.7965},

LADDER_RESULTS = {
    "Random 80/20":         {"S": 0.9588, "sigma": 0.9152, "kappa": 0.9442, "zT": 0.9186},
    "5-Fold CV":            {"S": 0.9588, "sigma": 0.9150, "kappa": 0.9444, "zT": 0.9184},
    "10-Fold CV":           {"S": 0.9595, "sigma": 0.9175, "kappa": 0.9459, "zT": 0.9196},
    "Composition CV":       {"S": 0.8322, "sigma": 0.7762, "kappa": 0.8565, "zT": 0.8178},
    "Chemistry-Cluster CV": {"S": 0.7528, "sigma": 0.7020, "kappa": 0.8092, "zT": 0.7456},
}
LADDER_LEAKY_STRATEGY = "Random 80/20"
LADDER_HONEST_STRATEGY = "Chemistry-Cluster CV"

# FILE B (SUPERSEDED) -- the 2026-08-15 pull's 2026-08-17 re-run funnel,
# kept for audit/comparison, not deleted. File A is canonical for every
# Paper A result as of 2026-09-11 (CLAUDE.md Canonical Dataset section);
# do not use these values for the methodology figure.
# RAW_CURVES = 155_758
# CLEANING_STEPS = [
#     ("1. Property extraction\n& range filtering", 1_992_138),
#     ("2. Integration &\nconsolidation", 1_992_138),
#     ("3. Temperature filtering\n300-800K", 1_093_377),
#     ("4. Pivot long→wide", 397_791),
#     ("5. Formula cleaning", 392_927),
#     ("6. zT self-consistency\ncheck", 388_221),
#     ("7. DFT data removal", 387_235),
#     ("8. Multi-source\nconsistency filtering", 308_656),
#     ("9. MAD outlier filter", 289_318),
#     ("10. Min. temperature\ncoverage", 284_671),
#     ("11. Smoothness filter", 280_348),
# ]

RAW_CURVES = 156_101
CLEANING_STEPS = [
    ("1. Property extraction\n& range filtering", 1_996_047),
    ("2. Integration &\nconsolidation", 1_996_047),
    ("3. Temperature filtering\n300-800K", 1_096_324),
    ("4. Pivot long→wide", 398_763),
    ("5. Formula cleaning", 393_381),
    ("6. zT self-consistency\ncheck", 388_669),
    ("7. DFT data removal", 387_683),
    ("8. Multi-source\nconsistency filtering", 308_998),
    ("9. MAD outlier filter", 289_637),
    ("10. Min. temperature\ncoverage", 284_987),
    ("11. Smoothness filter", 280_664),
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


# Figure 5: headroom decomposition. Reads both source artifacts directly at
# call time -- no hardcoded R^2, SD, or ceiling values -- per the task
# instruction not to hardcode numbers.
#   - grouped (chemistry-cluster) R^2 and across-repeat SD:
#     reports/regen_snapfix/20260917T150000/ladder_metrics.json
#     ("<target>_chemistry_full" runs, per_repeat_r2_mean/_std)
#   - ungrouped random 80/20 R^2: LADDER_RESULTS above (already the
#     confirmed CLAUDE.md value, corrected 2026-09-22)
#   - measurement-only ceiling (R^2_max) and combined (measurement +
#     digitization) ceiling: results/noise_floor/20260917T172251/
#     noise_floor_inputs.json ("item3_combined_ceiling_new")
# Combined ceiling uses the LOWER bound (r2_comb_lower), which gives the
# smallest headroom and is therefore the conservative choice for the
# paper's claim -- the upper bound would overstate how much of the gap to
# perfect prediction is theoretically closeable. For zT, the
# "zT (vs ZT_author_declared)" row is used, not the recomputed-TEP row.
HEADROOM_LADDER_METRICS_PATH = Path("reports/regen_snapfix/20260917T150000/ladder_metrics.json")
HEADROOM_NOISE_FLOOR_INPUTS_PATH = Path("results/noise_floor/20260917T172251/noise_floor_inputs.json")
HEADROOM_PROPERTIES = ["S", "sigma", "kappa", "zT"]
HEADROOM_PROPERTY_LABELS = [
    "S", "$\\sigma$ (log$_{10}$)", "$\\kappa$ (log$_{10}$)", "zT",
]
HEADROOM_ZT_COMBINED_LABEL = "zT (vs ZT_author_declared)"

HEADROOM_SEGMENT_ORDER = ["achieved", "headroom", "digitization", "measurement"]
HEADROOM_SEGMENT_DISPLAY_NAMES = {
    "achieved": "Achieved (chemistry-cluster grouped R$^2$)",
    "headroom": "Headroom to combined ceiling",
    "digitization": "Digitization noise",
    "measurement": "Measurement noise",
}
# Greyscale, hatched fills -- distinguishable without color, per the print
# constraint -- darkest for the real, achieved signal, lightening toward
# the unreachable measurement-noise segment.
HEADROOM_SEGMENT_COLORS = {
    "achieved": "#4d4d4d",
    "headroom": "#999999",
    "digitization": "#cccccc",
    "measurement": "#eaeaea",
}
HEADROOM_SEGMENT_HATCHES = {
    "achieved": None,
    "headroom": "//",
    "digitization": "xx",
    "measurement": "..",
}


def load_headroom_data(
    ladder_metrics_path=HEADROOM_LADDER_METRICS_PATH,
    noise_floor_inputs_path=HEADROOM_NOISE_FLOOR_INPUTS_PATH,
):
    """
    Assemble the four segment boundaries (achieved / headroom /
    digitization noise / measurement noise) per property from the two
    committed artifacts named in the module-level HEADROOM_* comment
    above. Returns {property: {grouped_r2, grouped_sd, ungrouped_r2,
    r2_comb_lower, r2_meas, headroom_width, digitization_width,
    measurement_width}}.

    The achieved/headroom boundary and headroom_width are read directly
    from noise_floor_inputs.json's own confirmed_new/headroom_lower
    fields rather than recomputed as r2_comb_lower - grouped_r2 here:
    that JSON's headroom_lower was itself computed against the rounded
    4-decimal confirmed R^2 (CLAUDE.md's own convention), and re-deriving
    it from ladder_metrics.json's full-precision grouped_r2 instead lands
    S on the opposite side of a rounding boundary (0.2085 raw -> 0.208
    displayed, vs 0.209 in CLAUDE.md's COMBINED CEILING table) purely
    from a precision mismatch between the two source files, not a real
    data difference. A sanity check below confirms the two files agree
    to 4 decimals before trusting either.
    """
    with open(ladder_metrics_path, encoding="utf-8") as f:
        ladder_metrics = json.load(f)
    with open(noise_floor_inputs_path, encoding="utf-8") as f:
        noise_floor = json.load(f)

    combined_by_label = {
        entry["label"]: entry for entry in noise_floor["item3_combined_ceiling_new"]
    }

    data = {}
    for prop in HEADROOM_PROPERTIES:
        run = ladder_metrics["runs"][f"{prop}_chemistry_full"]
        grouped_r2 = run["per_repeat_r2_mean"]
        grouped_sd = run["per_repeat_r2_std"]

        ungrouped_r2 = LADDER_RESULTS["Random 80/20"][prop]

        combined_label = HEADROOM_ZT_COMBINED_LABEL if prop == "zT" else prop
        combined_entry = combined_by_label[combined_label]
        r2_comb_lower = combined_entry["r2_comb_lower"]
        r2_meas = combined_entry["r2_meas"]
        confirmed_r2 = combined_entry["confirmed_new"]

        if round(grouped_r2, 4) != confirmed_r2:
            raise ValueError(
                f"{prop}: ladder_metrics.json grouped R^2 ({grouped_r2:.4f}) does not "
                f"match noise_floor_inputs.json's confirmed_new ({confirmed_r2}) -- "
                f"the two source artifacts have diverged, do not trust either silently."
            )

        data[prop] = {
            "grouped_r2": grouped_r2,
            "grouped_sd": grouped_sd,
            "ungrouped_r2": ungrouped_r2,
            "r2_comb_lower": r2_comb_lower,
            "r2_meas": r2_meas,
            "achieved_width": confirmed_r2,
            "headroom_width": combined_entry["headroom_lower"],
            "digitization_width": r2_meas - r2_comb_lower,
            "measurement_width": 1.0 - r2_meas,
        }
    return data


def make_headroom_decomposition(
    out_path,
    ladder_metrics_path=HEADROOM_LADDER_METRICS_PATH,
    noise_floor_inputs_path=HEADROOM_NOISE_FLOOR_INPUTS_PATH,
):
    """
    Figure 5: one horizontal stacked bar per property (S, sigma, kappa,
    zT), spanning R^2 = 0 to 1.0, decomposed left to right into achieved
    (chemistry-cluster grouped R^2), headroom to the combined label-noise
    ceiling, digitization noise, and measurement noise. An error bar on
    the achieved/headroom boundary shows the grouped R^2's across-repeat
    SD; a dashed marker shows where the ungrouped random 80/20 R^2 falls,
    so the reader sees how much apparent performance is validation
    artefact rather than real headroom closed. See the module-level
    HEADROOM_* comment for data provenance and the ceiling-bound choice.
    """
    data = load_headroom_data(ladder_metrics_path, noise_floor_inputs_path)

    fig, ax = plt.subplots(figsize=get_figsize(1, 1, panel_width=10.0, panel_height=5.1))
    y_pos = np.arange(len(HEADROOM_PROPERTIES))[::-1]
    bar_height = 0.55

    segment_labeled = set()
    marker_labeled = False
    for y, prop in zip(y_pos, HEADROOM_PROPERTIES):
        d = data[prop]
        achieved_end = d["achieved_width"]
        headroom_end = achieved_end + d["headroom_width"]
        digitization_end = headroom_end + d["digitization_width"]
        segments = [
            ("achieved", 0.0, achieved_end),
            ("headroom", achieved_end, headroom_end),
            ("digitization", headroom_end, digitization_end),
            ("measurement", digitization_end, 1.0),
        ]
        for name, left, right in segments:
            label = HEADROOM_SEGMENT_DISPLAY_NAMES[name] if name not in segment_labeled else None
            ax.barh(
                y, right - left, left=left, height=bar_height,
                color=HEADROOM_SEGMENT_COLORS[name], hatch=HEADROOM_SEGMENT_HATCHES[name],
                edgecolor="black", linewidth=0.6, label=label, zorder=2,
            )
            segment_labeled.add(name)

        # Headroom segment width label, placed above the bar so it never
        # collides with the ungrouped-R^2 marker line, which falls inside
        # the headroom segment for every property here.
        headroom_center = (achieved_end + headroom_end) / 2
        ax.text(
            headroom_center, y + bar_height / 2 + 0.10, f"$\\Delta$={d['headroom_width']:.3f}",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold", zorder=4,
        )

        ax.errorbar(
            achieved_end, y, xerr=d["grouped_sd"], color="black",
            capsize=3, elinewidth=1.1, capthick=1.1, zorder=5,
        )

        marker_label = "Ungrouped random 80/20 R$^2$" if not marker_labeled else None
        ax.plot(
            [d["ungrouped_r2"], d["ungrouped_r2"]],
            [y - bar_height / 2 - 0.04, y + bar_height / 2 + 0.04],
            color="black", linestyle="--", linewidth=1.8, zorder=6, label=marker_label,
        )
        marker_labeled = True

    ax.set_yticks(y_pos)
    ax.set_yticklabels(HEADROOM_PROPERTY_LABELS, fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(y_pos.min() - 0.55, y_pos.max() + 0.55)
    ax.set_xlabel("R$^2$ (per-property matched space)")

    # Legend placed on the FIGURE (not the axes), at a fixed
    # figure-fraction position, with subplots_adjust reserving room below
    # the axes for both the xlabel and the legend -- ax.legend() with a
    # large negative bbox_to_anchor plus tight_layout(rect=...) fought
    # each other and put the legend box on top of the xlabel text.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
        bbox_transform=fig.transFigure, ncol=2, framealpha=0.9,
        columnspacing=1.2, handletextpad=0.6,
    )
    fig.subplots_adjust(bottom=0.32)
    save_figure(fig, out_path)
    plt.close(fig)


def load_fig3_predictions(target, checkpoint_dir=FIG3_CHECKPOINT_DIR):
    """
    Load and pool every repeatN_foldM_predictions.npz for `target` under
    checkpoint_dir/<target>_chemistry/ -- one array of all held-out
    (y_true, y_pred) pairs across every outer fold and repeat, exactly
    what nested_cv.py's own pooled out-of-fold R^2 is computed from.
    Returns (y_true, y_pred, run_config dict).
    """
    target_dir = checkpoint_dir / f"{target}_chemistry"
    run_config_path = target_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(
            f"No run_config.json in {target_dir} -- expected the Kaggle chemistry-cluster "
            f"checkpoint tree for target={target!r}"
        )
    with open(run_config_path, encoding="utf-8") as f:
        run_config = json.load(f)

    npz_paths = sorted(target_dir.glob("repeat*_fold*_predictions.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No repeat*_fold*_predictions.npz files in {target_dir}")

    y_true_parts, y_pred_parts = [], []
    for npz_path in npz_paths:
        data = np.load(npz_path)
        y_true_parts.append(data["y_true"])
        y_pred_parts.append(data["y_pred"])

    return np.concatenate(y_true_parts), np.concatenate(y_pred_parts), run_config


def make_actual_vs_predicted(out_path, checkpoint_dir=FIG3_CHECKPOINT_DIR):
    """
    Figure 3: 2x2 panel of actual-vs-predicted scatter plots, one per
    target (S, sigma, kappa, zT), chemistry-cluster grouped CV, XGBoost,
    frozen hyperparameters -- pooled out-of-fold predictions loaded from
    the real Kaggle run's checkpoints (see load_fig3_predictions).

    sigma/kappa panels plot the checkpointed values directly (already
    log10-space, per run_config's target_scale -- matches
    nested_cv.py's LOG_TRANSFORM_TARGETS); S/zT plot raw/linear values.
    Each panel gets a y=x reference line (perfect prediction) and is
    annotated with the pooled R^2 (recomputed here from the loaded
    arrays, not copied from CLAUDE.md, so this figure is self-verifying)
    and n. No temperature_bin color-coding: the local featurized dataset
    is off by ~200-260 rows per target from whatever the Kaggle run
    actually trained on (confirmed by attempting to regenerate the exact
    same chemistry-cluster splits locally and finding the reconstructed
    y_true did not match the checkpointed y_true for any fold), so a
    per-point temperature color would not be trustworthy -- flagged and
    dropped rather than plotted anyway.
    """
    fig, axes = plt.subplots(
        2, 2, figsize=get_figsize(2, 2, panel_width=4.4, panel_height=4.2),
        constrained_layout=True,
    )
    point_color = COLORBLIND_PALETTE[5]  # blue -- same "honest model" color as Figures 1-2

    for ax, target, letter in zip(axes.flat, FIG3_TARGETS, FIG3_PANEL_LETTERS):
        y_true, y_pred, run_config = load_fig3_predictions(target, checkpoint_dir)
        pooled_r2 = r2_score(y_true, y_pred)
        n = len(y_true)

        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        pad = (hi - lo) * 0.03
        lo, hi = lo - pad, hi + pad

        ax.scatter(
            y_true, y_pred, s=2, alpha=0.12, color=point_color,
            edgecolors="none", rasterized=True,
        )
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.2, zorder=5)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        axis_label = FIG3_AXIS_LABELS[target]
        ax.set_xlabel(f"Actual {axis_label}")
        ax.set_ylabel(f"Predicted {axis_label}")

        ax.text(
            0.04, 0.96, f"$R^2$={pooled_r2:.4f}\nn={n:,}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", linewidth=0.5),
        )
        add_panel_label(ax, letter)

    # constrained_layout (set in plt.subplots above) handles spacing for
    # this figure -- calling tight_layout() here would fight it and
    # reproduce the overlapping-label bug this replaced.
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

    make_headroom_decomposition(FIGURES_DIR / "fig5_headroom")
    print("Saved fig5_headroom.png / .pdf")

    try:
        make_actual_vs_predicted(FIGURES_DIR / "fig3_actual_vs_predicted")
        print("Saved fig3_actual_vs_predicted.png / .pdf")
    except FileNotFoundError as e:
        print(f"Skipped fig3_actual_vs_predicted: {e}")


if __name__ == "__main__":
    main()
