"""
Generate the dataset-construction and descriptive figures used in the
methods section of both papers: the 11-step cleaning funnel, the
post-cleaning property distributions, and the chemistry-cluster size
distribution that motivates repeated grouped CV. Reads the cleaned CSV
written by src/data_cleaning.py; run that first.
"""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.noise_floor import load_aligned_target_values
from src.plotting_style import (
    COLORBLIND_PALETTE,
    add_panel_label,
    apply_style,
    get_figsize,
    save_figure,
)

FIGURES_DIR = Path("figures")

# The snapfix featurized CSV's chemistry_cluster_id, NOT File A's cleaned
# CSV. File A's cleaned CSV predates the 2026-09-19 SNAP(0.05)/S5
# grouping fixes (commit c1c6873) and its own chemistry_cluster_id column
# is still the pre-fix, buggy one (12,036 clusters, decimal-coefficient
# formulas like "Ca1Mn0.9O3" left unsnapped -- see CLAUDE.md's Grouping
# Fixes section, BUG 1); src.noise_floor.load_cleaned_dataset() drops
# that column entirely for exactly this reason (see its docstring). Only
# the snapfix featurized CSV has the corrected column (8,908 clusters).
# make_cluster_size_distribution() must load cluster membership from here.
SNAPFIX_FEATURIZED_CSV = Path("data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv")
SNAPFIX_FEATURIZED_CSV_SHA256 = "d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489"

# Shared provenance paths, referenced by more than one figure below.
LADDER_METRICS_PATH = Path("reports/regen_snapfix/20260917T150000/ladder_metrics.json")
NOISE_FLOOR_INPUTS_PATH = Path("results/noise_floor/20260923T202312/noise_floor_inputs.json")
DESCRIPTOR_ABLATION_METRICS_PATH = Path("reports/ablation_snapfix/20260918T000111/ablation_metrics.json")
SHAP_ATTRIBUTION_DIR = Path("results/shap_attribution/20260917T134930")
UNGROUPED_SNAPFIX_DIR = Path("results/ungrouped_snapfix/20260922T093243")
ZT_COMBINED_CEILING_LABEL = "zT (vs ZT_author_declared)"
ZT_HEADROOM_FRACTION_LABEL = "zT (declared)"

# Shared colour-blind-safe palette (Okabe-Ito, the same set as
# COLORBLIND_PALETTE in src/plotting_style.py) and hatch patterns, one
# assignment per concept, used identically across every figure in this
# file so the same concept always reads as the same colour+hatch
# wherever it appears. Only 7 non-black Okabe-Ito colours are usable as
# fills (black is reserved project-wide for reference lines/markers, per
# the convention already set by MODEL_COMPARISON's ceiling line and this
# figure set's other markers) -- 13 concepts need colours, so 6 concepts
# below deliberately reuse another concept's colour+hatch pair wholesale.
# Every reuse is annotated with why it is safe: either the two concepts
# never appear in the same panel, or (internal/chemistry_cluster) they
# are literally the same underlying quantity and sharing a colour is the
# correct read, not a collision.
PALETTE = {
    # Primary assignments -- one Okabe-Ito hue each.
    "random_split": "#56B4E9",        # sky blue
    "kfold": "#F0E442",               # yellow
    "composition": "#E69F00",         # orange
    "chemistry_cluster": "#0072B2",   # blue
    "magpie": "#CC79A7",              # reddish purple
    "cbfv": "#009E73",                # bluish green
    "full_feature_set": "#D55E00",    # vermillion
    # Reused assignments -- safe because the reused-from concept never
    # appears in the same panel as this one (or, for "internal", because
    # it IS chemistry_cluster's quantity under another name).
    "measurement_noise": "#009E73",   # = cbfv (never co-plotted)
    "digitization_noise": "#CC79A7",  # = magpie (never co-plotted)
    "headroom": "#F0E442",            # = kfold (never co-plotted)
    "internal": "#0072B2",            # = chemistry_cluster (same quantity)
    "external_fullset": "#E69F00",    # = composition (never co-plotted)
    "external_insupport": "#D55E00",  # = full_feature_set (never co-plotted)
}
HATCH = {
    "random_split": None,
    "kfold": "..",
    "composition": "//",
    "chemistry_cluster": "xx",
    "magpie": "\\\\",
    "cbfv": "||",
    "full_feature_set": "++",
    "measurement_noise": "||",        # = cbfv's hatch, paired with its colour reuse
    "digitization_noise": "\\\\",     # = magpie's hatch, paired with its colour reuse
    "headroom": "..",                 # = kfold's hatch, paired with its colour reuse
    "internal": "xx",                 # = chemistry_cluster's hatch (same quantity)
    "external_fullset": "//",         # = composition's hatch, paired with its colour reuse
    "external_insupport": "++",       # = full_feature_set's hatch, paired with its colour reuse
}

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
# Random 80/20, 5-Fold, and 10-Fold rows: results/ungrouped_snapfix/
# 20260922T093243/{target}_{random_f20,kfold_f5,kfold_f10}/, tabulated in
# reports/ungrouped_snapfix/20260922T093243/metrics.json and table1.md.
#
# CORRECTED 2026-09-22: the claim that previously stood here -- that
# checkpoints/ladder_regen_dl/ was File A's rows, unaffected by the
# grouping fix "for a structural reason" -- was FALSE for the row-set
# claim specifically (the structural argument about chemistry_cluster_id
# not being read by the ungrouped split path remains true, but it does
# not establish which dataset was used, only that the grouping fix
# wouldn't change the answer within whatever dataset was used).
# checkpoints/ladder_regen_dl/ has no recorded CSV path or SHA256
# anywhere in it, and its row counts (S=185,844, sigma=183,246,
# kappa=121,535, zT=129,851) do not match File A/snapfix (S=185,064,
# sigma=182,755, kappa=121,110, zT=129,419) on any target -- a fourth,
# unidentified dataset snapshot. Found 2026-09-22; see CLAUDE.md's
# Five-Way Ladder and Canonical Dataset sections for the full
# correction. Rerun on the verified snapfix CSV
# (results/ungrouped_snapfix/20260922T093243/, row counts confirmed
# against load_target_data): deltas from the old ladder_regen_dl values
# are small (|delta| <= 0.0008 per cell) and do not change any
# qualitative conclusion, but the provenance claim was wrong and is
# corrected here rather than left standing.
#
# The random rung is 20 independent, OVERLAPPING 80/20 holdout draws via
# sklearn ShuffleSplit, pooled -- not a single split (see
# load_ladder_ungrouped_sd below, and CLAUDE.md's Grouping Key section).
#
# RESOLVED 2026-09-22 (unrelated to the above): the previous values here
# (S=0.9586, sigma=0.9539, kappa=0.9615, zT=0.9138 for random 80/20,
# etc.) were NOT a log10-vs-linear scale mismatch -- both this row and
# the current one use log10 for sigma/kappa. They were the orphaned
# pre-2026-08-22 run's ungrouped cells (found verbatim, all three rows,
# in checkpoints/saved_predictions/te-ml-pipeline/CLAUDE.md, a stale
# in-tree snapshot of an earlier CLAUDE.md whose own Five-Way Ladder
# table -- itself explicitly log10-space -- matches this file's old
# LADDER_RESULTS on all five rungs) -- the same run current CLAUDE.md
# already calls out as superseded and unreproducible for its
# composition/chemistry cells. No run_config.json for that orphaned run
# exists anywhere on disk. See the PRE-FIX (SUPERSEDED) block below for
# the old values, kept for audit.
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

# UNIDENTIFIED-DATASET (SUPERSEDED 2026-09-22) -- ladder_regen_dl values,
# correct relative to the orphaned run above but sourced from a dataset
# that was never identified (see the correction comment above). Kept for
# audit, not deleted.
# "Random 80/20": {"S": 0.9588, "sigma": 0.9152, "kappa": 0.9442, "zT": 0.9186},
# "5-Fold CV":    {"S": 0.9588, "sigma": 0.9150, "kappa": 0.9444, "zT": 0.9184},
# "10-Fold CV":   {"S": 0.9595, "sigma": 0.9175, "kappa": 0.9459, "zT": 0.9196},

LADDER_RESULTS = {
    "Random 80/20":         {"S": 0.9582, "sigma": 0.9150, "kappa": 0.9434, "zT": 0.9180},
    "5-Fold CV":            {"S": 0.9585, "sigma": 0.9152, "kappa": 0.9436, "zT": 0.9181},
    "10-Fold CV":           {"S": 0.9595, "sigma": 0.9174, "kappa": 0.9455, "zT": 0.9193},
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


def load_snapfix_cluster_columns(
    path=SNAPFIX_FEATURIZED_CSV, expected_sha256=SNAPFIX_FEATURIZED_CSV_SHA256
):
    """
    Load only sample_id/chemistry_cluster_id from the snapfix featurized
    CSV, verifying its SHA256 first. Used by make_cluster_size_distribution
    -- see the SNAPFIX_FEATURIZED_CSV module comment for why File A's
    cleaned CSV is not a substitute for cluster membership.
    """
    path = Path(path)
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{path} has SHA256 {actual_sha256}, expected {expected_sha256} (snapfix). "
            "Refusing to compute cluster sizes against an unverified dataset."
        )
    return pd.read_csv(path, usecols=["sample_id", "chemistry_cluster_id"])


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


def make_property_distributions(out_path):
    """
    2x2 panel of histograms for S, sigma, kappa, zT after cleaning.
    sigma and kappa use log10-scale x-axes and log-spaced bins,
    consistent with the space nested_cv.py actually trains and scores
    those two targets in (LOG_TRANSFORM_TARGETS, see CLAUDE.md's noise-
    floor decision); S and zT stay linear, matching their training
    space too. Each panel is annotated with its non-null sample size n
    and arithmetic mean, both in the panel's native (raw, not log)
    units.

    Each property's values come from load_aligned_target_values() (the
    same ladder-aligned, featurized-CSV loader src.noise_floor.compute_all
    uses), not a single shared cleaned-CSV df -- so n in each panel is
    185,064 / 182,755 / 121,110 / 129,419 (S/sigma/kappa/zT), matching
    both LADDER_N_ROWS and paper.md's own stated per-target row counts,
    not the cleaned CSV's own slightly larger notna count.
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
        values = load_aligned_target_values(col)
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


def load_ladder_grouped_sd(ladder_metrics_path=LADDER_METRICS_PATH):
    """
    Across-repeat SD for the composition and chemistry-cluster rungs.
    Read from LADDER_METRICS_PATH's own per_repeat_r2_std field, not
    hardcoded. Returns {"Composition CV": {prop: sd},
    "Chemistry-Cluster CV": {prop: sd}}.
    """
    with open(ladder_metrics_path, encoding="utf-8") as f:
        ladder_metrics = json.load(f)
    sd = {"Composition CV": {}, "Chemistry-Cluster CV": {}}
    for prop in LADDER_PROPERTIES:
        sd["Composition CV"][prop] = ladder_metrics["runs"][f"{prop}_composition_full"]["per_repeat_r2_std"]
        sd["Chemistry-Cluster CV"][prop] = ladder_metrics["runs"][f"{prop}_chemistry_full"]["per_repeat_r2_std"]
    return sd


LADDER_UNGROUPED_STRATEGY_DIRS = {
    "Random 80/20": "random_f20", "5-Fold CV": "kfold_f5", "10-Fold CV": "kfold_f10",
}


def load_ladder_ungrouped_sd(ungrouped_dir=UNGROUPED_SNAPFIX_DIR):
    """
    Per-draw (random 80/20, 20 draws) or per-fold (5-fold/10-fold) SD for
    the three ungrouped rungs, computed directly from
    results/ungrouped_snapfix/20260922T093243/'s saved predictions --
    not hardcoded, and not read from a precomputed field, since none of
    these three rungs write one. These runs replaced the
    unidentified-dataset checkpoints/ladder_regen_dl/ runs on 2026-09-22
    (see CLAUDE.md's Five-Way Ladder and Canonical Dataset corrections).
    Returns {"Random 80/20": {prop: sd}, "5-Fold CV": {prop: sd},
    "10-Fold CV": {prop: sd}}.
    """
    sd = {strategy: {} for strategy in LADDER_UNGROUPED_STRATEGY_DIRS}
    for strategy, dir_suffix in LADDER_UNGROUPED_STRATEGY_DIRS.items():
        for prop in LADDER_PROPERTIES:
            run_dir = Path(ungrouped_dir) / f"{prop}_{dir_suffix}"
            npz_paths = sorted(run_dir.glob("repeat0_fold*_predictions.npz"))
            per_fold_r2 = []
            for npz_path in npz_paths:
                data = np.load(npz_path)
                per_fold_r2.append(r2_score(data["y_true"], data["y_pred"]))
            sd[strategy][prop] = float(np.std(per_fold_r2, ddof=1))
    return sd


def make_validation_ladder(
    out_path, ladder_metrics_path=LADDER_METRICS_PATH, ungrouped_dir=UNGROUPED_SNAPFIX_DIR,
):
    """
    Figure 2: five-way validation-inflation ladder, grouped bar chart,
    one group per target (S, sigma, kappa, zT), five validation
    strategies per group (random 80/20, 5-fold, 10-fold, composition,
    chemistry-cluster), pooled out-of-fold R^2, frozen hyperparameters
    reused unchanged across every rung -- see CLAUDE.md "Confirmed
    Results -- Five-Way Ladder". Colours and hatches come from the
    shared PALETTE/HATCH dicts: random 80/20 gets its own concept
    colour, 5-fold and 10-fold both use the "kfold" concept (identical
    colour+hatch), so the three ungrouped rungs still read as "the same,
    indistinguishable number" (they differ by <0.003) while random 80/20
    is individually addressable (it reappears as a marker in the
    headroom figure). All five rungs now carry error bars: composition/
    chemistry-cluster get across-repeat SD from load_ladder_grouped_sd,
    random 80/20/5-fold/10-fold get per-draw/per-fold SD from
    load_ladder_ungrouped_sd (added 2026-09-22 alongside the
    unidentified-dataset correction -- these three rungs never had error
    bars before, not because there was no spread to show, but because
    the per-draw/per-fold structure that produces it wasn't being read).
    A bracket to the right of each group annotates the inflation gap:
    random 80/20 minus chemistry-cluster, the headline number this
    ladder exists to report.
    """
    n_props = len(LADDER_PROPERTIES)
    n_strategies = len(LADDER_STRATEGIES)
    all_sd = {**load_ladder_grouped_sd(ladder_metrics_path), **load_ladder_ungrouped_sd(ungrouped_dir)}

    bar_colors = [
        PALETTE["random_split"], PALETTE["kfold"], PALETTE["kfold"],
        PALETTE["composition"], PALETTE["chemistry_cluster"],
    ]
    bar_hatches = [
        HATCH["random_split"], HATCH["kfold"], HATCH["kfold"],
        HATCH["composition"], HATCH["chemistry_cluster"],
    ]

    fig, ax = plt.subplots(figsize=get_figsize(1, 1, panel_width=11.5, panel_height=5.5))

    group_centers = np.arange(n_props)
    bar_width = 0.8 / n_strategies
    offsets = (np.arange(n_strategies) - (n_strategies - 1) / 2) * bar_width

    for i, strategy in enumerate(LADDER_STRATEGIES):
        heights = [LADDER_RESULTS[strategy][prop] for prop in LADDER_PROPERTIES]
        x = group_centers + offsets[i]
        bars = ax.bar(
            x, heights, width=bar_width * 0.92,
            color=bar_colors[i], hatch=bar_hatches[i], label=strategy,
            edgecolor="black", linewidth=0.5, zorder=2,
        )
        for rect, h in zip(bars, heights):
            ax.text(
                rect.get_x() + rect.get_width() / 2, h + 0.015, f"{h:.2f}",
                ha="center", va="bottom", fontsize=7.5, rotation=0,
            )
        errs = [all_sd[strategy][prop] for prop in LADDER_PROPERTIES]
        ax.errorbar(
            x, heights, yerr=errs, fmt="none", color="black",
            capsize=3, elinewidth=1.1, capthick=1.1, zorder=5,
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
#     LADDER_METRICS_PATH ("<target>_chemistry_full" runs,
#     per_repeat_r2_mean/_std)
#   - ungrouped random 80/20 R^2: LADDER_RESULTS above (already the
#     confirmed CLAUDE.md value, corrected 2026-09-22)
#   - measurement-only ceiling (R^2_max) and combined (measurement +
#     digitization) ceiling: NOISE_FLOOR_INPUTS_PATH
#     ("item3_combined_ceiling_new")
# Combined ceiling uses the LOWER bound (r2_comb_lower), which gives the
# smallest headroom and is therefore the conservative choice for the
# paper's claim -- the upper bound would overstate how much of the gap to
# perfect prediction is theoretically closeable. For zT, the
# "zT (vs ZT_author_declared)" row is used, not the recomputed-TEP row.
# Colours/hatches: "achieved" reuses the "internal" palette concept (it IS
# the internal chemistry-cluster R^2), "headroom"/"digitization_noise"/
# "measurement_noise" use their own PALETTE/HATCH entries -- see the
# module-level PALETTE comment for why these are safe colour reuses.
HEADROOM_PROPERTIES = ["S", "sigma", "kappa", "zT"]
HEADROOM_PROPERTY_LABELS = [
    "S", "$\\sigma$ (log$_{10}$)", "$\\kappa$ (log$_{10}$)", "zT",
]

HEADROOM_SEGMENT_ORDER = ["achieved", "headroom", "digitization", "measurement"]
HEADROOM_SEGMENT_DISPLAY_NAMES = {
    "achieved": "Achieved (chemistry-cluster grouped R$^2$)",
    "headroom": "Headroom to combined ceiling",
    "digitization": "Digitization noise",
    "measurement": "Measurement noise",
}
HEADROOM_SEGMENT_PALETTE_KEY = {
    "achieved": "internal",
    "headroom": "headroom",
    "digitization": "digitization_noise",
    "measurement": "measurement_noise",
}
HEADROOM_SEGMENT_COLORS = {name: PALETTE[key] for name, key in HEADROOM_SEGMENT_PALETTE_KEY.items()}
HEADROOM_SEGMENT_HATCHES = {name: HATCH[key] for name, key in HEADROOM_SEGMENT_PALETTE_KEY.items()}
# headroom-only emphasis: denser hatch and a bolder edge than every other
# segment (including the shared HATCH["kfold"] value, left untouched for
# the ladder figure) -- see the drawing loop's comment for why.
HEADROOM_SEGMENT_HATCH_OVERRIDE = {"headroom": "..."}
HEADROOM_SEGMENT_EDGE_LINEWIDTH = {"headroom": 2.2}


def load_headroom_data(
    ladder_metrics_path=LADDER_METRICS_PATH,
    noise_floor_inputs_path=NOISE_FLOOR_INPUTS_PATH,
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

        combined_label = ZT_COMBINED_CEILING_LABEL if prop == "zT" else prop
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
    ladder_metrics_path=LADDER_METRICS_PATH,
    noise_floor_inputs_path=NOISE_FLOOR_INPUTS_PATH,
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
            # Okabe-Ito yellow ("headroom") is the lowest-contrast colour
            # in the palette on white -- without extra emphasis it reads
            # as flatter than the achieved segment's dense dark-blue
            # crosshatch, even though headroom is this figure's main
            # point. Denser hatching and a bolder edge, on this segment
            # only, fix that without changing its colour or touching the
            # shared HATCH["kfold"] entry the ladder figure also uses.
            hatch = HEADROOM_SEGMENT_HATCH_OVERRIDE.get(name, HEADROOM_SEGMENT_HATCHES[name])
            linewidth = HEADROOM_SEGMENT_EDGE_LINEWIDTH.get(name, 0.6)
            ax.barh(
                y, right - left, left=left, height=bar_height,
                color=HEADROOM_SEGMENT_COLORS[name], hatch=hatch,
                edgecolor="black", linewidth=linewidth, label=label, zorder=2,
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
            color=PALETTE["random_split"], linestyle="--", linewidth=2.2, zorder=6,
            path_effects=[pe.withStroke(linewidth=3.6, foreground="black")], label=marker_label,
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


# SHAP attribution shares: random vs chemistry-cluster split, coarse (3)
# and fine (10) descriptor-semantic groups. Source: SHAP_ATTRIBUTION_DIR
# ("results/shap_attribution/20260917T134930/summary.json"), the 25-fold
# (5 repeats x 5 outer folds per arm) regeneration -- see CLAUDE.md's SHAP
# Attribution Comparison section. Reads summary.json's own
# coarse_group_comparison/fine_group_comparison lists directly (mean
# share, across-fold SD, and delta_in_pooled_sd_units per group); nothing
# here is hardcoded. random_split/chemistry_cluster colours from PALETTE,
# used identically to the ladder and headroom figures.
SHAP_COARSE_ORDER = ["cbfv", "magpie", "temperature"]
SHAP_COARSE_LABELS = {"cbfv": "CBFV", "magpie": "MAGPIE", "temperature": "Temperature"}


def load_shap_group_data(shap_dir=SHAP_ATTRIBUTION_DIR):
    """
    Load coarse and fine SHAP attribution-share group comparisons from
    summary.json. Returns (coarse_rows, fine_rows, max_delta_row), each
    row a dict with group/random_mean_share/random_sd/
    chemistry_mean_share/chemistry_sd/delta_in_pooled_sd_units. fine_rows
    is sorted by descending mean of the two arms' shares. max_delta_row
    is the fine-group row with the largest |delta_in_pooled_sd_units|.
    """
    with open(shap_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    coarse_by_group = {row["group"]: row for row in summary["coarse_group_comparison"]}
    coarse_rows = [coarse_by_group[g] for g in SHAP_COARSE_ORDER]

    fine_rows = sorted(
        summary["fine_group_comparison"],
        key=lambda row: (row["random_mean_share"] + row["chemistry_mean_share"]) / 2,
        reverse=True,
    )
    max_delta_row = max(fine_rows, key=lambda row: abs(row["delta_in_pooled_sd_units"]))
    return coarse_rows, fine_rows, max_delta_row


def _plot_shap_paired_bars(ax, rows, group_labels, horizontal):
    """Shared paired-bar drawing for both SHAP panels (random vs chemistry)."""
    n = len(rows)
    positions = np.arange(n)
    bar_width = 0.36
    random_vals = [row["random_mean_share"] for row in rows]
    random_errs = [row["random_sd"] for row in rows]
    chemistry_vals = [row["chemistry_mean_share"] for row in rows]
    chemistry_errs = [row["chemistry_sd"] for row in rows]

    if horizontal:
        ax.barh(
            positions + bar_width / 2, random_vals, height=bar_width, xerr=random_errs,
            color=PALETTE["random_split"], hatch=HATCH["random_split"], edgecolor="black",
            linewidth=0.6, error_kw=dict(capsize=2.5, elinewidth=1.0, capthick=1.0), label="Random 80/20",
        )
        ax.barh(
            positions - bar_width / 2, chemistry_vals, height=bar_width, xerr=chemistry_errs,
            color=PALETTE["chemistry_cluster"], hatch=HATCH["chemistry_cluster"], edgecolor="black",
            linewidth=0.6, error_kw=dict(capsize=2.5, elinewidth=1.0, capthick=1.0), label="Chemistry-Cluster CV",
        )
        ax.set_yticks(positions)
        ax.set_yticklabels(group_labels)
        ax.set_xlabel("Mean |SHAP| share")
    else:
        ax.bar(
            positions - bar_width / 2, random_vals, width=bar_width, yerr=random_errs,
            color=PALETTE["random_split"], hatch=HATCH["random_split"], edgecolor="black",
            linewidth=0.6, error_kw=dict(capsize=2.5, elinewidth=1.0, capthick=1.0), label="Random 80/20",
        )
        ax.bar(
            positions + bar_width / 2, chemistry_vals, width=bar_width, yerr=chemistry_errs,
            color=PALETTE["chemistry_cluster"], hatch=HATCH["chemistry_cluster"], edgecolor="black",
            linewidth=0.6, error_kw=dict(capsize=2.5, elinewidth=1.0, capthick=1.0), label="Chemistry-Cluster CV",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(group_labels)
        ax.set_ylabel("Mean |SHAP| share")


def make_shap_attribution(out_path, shap_dir=SHAP_ATTRIBUTION_DIR):
    """
    SHAP attribution shares, random vs chemistry-cluster split. Left
    panel: three coarse families (CBFV, MAGPIE, temperature), paired
    vertical bars. Right panel: ten fine semantic groups, paired
    horizontal bars sorted by mean share. Error bars are across-fold SD
    (25 folds per arm). No significance stars -- the visual point is
    that every pair overlaps within its error bars, i.e. attribution
    shares are indistinguishable between split strategies despite the
    large R^2 gap the ladder figure shows. The largest fine-group delta
    (valence_electron_config) is annotated in pooled fold-SD units.
    """
    coarse_rows, fine_rows, max_delta_row = load_shap_group_data(shap_dir)

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=get_figsize(1, 2, panel_width=6.0, panel_height=5.2),
    )

    coarse_labels = [SHAP_COARSE_LABELS[row["group"]] for row in coarse_rows]
    _plot_shap_paired_bars(ax_left, coarse_rows, coarse_labels, horizontal=False)
    add_panel_label(ax_left, "a")

    fine_labels = [row["group"].replace("_", " ") for row in fine_rows]
    _plot_shap_paired_bars(ax_right, fine_rows, fine_labels, horizontal=True)
    ax_right.invert_yaxis()
    add_panel_label(ax_right, "b")

    max_delta_idx = fine_rows.index(max_delta_row)
    y_pos = max_delta_idx
    ax_right.annotate(
        f"Largest group delta: {max_delta_row['group'].replace('_', ' ')},\n"
        f"{abs(max_delta_row['delta_in_pooled_sd_units']):.2f} pooled fold-SD",
        xy=(max(max_delta_row["random_mean_share"], max_delta_row["chemistry_mean_share"]), y_pos),
        xytext=(0.60, y_pos - 2.0 if y_pos < len(fine_rows) - 2 else y_pos + 1.5),
        textcoords="data", fontsize=8, ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
    )

    handles, labels = ax_left.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
        bbox_transform=fig.transFigure, ncol=2, framealpha=0.9,
        columnspacing=1.2, handletextpad=0.6,
    )
    fig.subplots_adjust(bottom=0.16, wspace=0.35)
    save_figure(fig, out_path)
    plt.close(fig)


# Descriptor ablation against the label-noise ceiling. 2x2 panels, one per
# target (S, sigma, kappa, zT). Sources:
#   - per-feature-set R^2 (magpie/cbfv/full) and across-repeat SD:
#     DESCRIPTOR_ABLATION_METRICS_PATH
#     ("<target>.<magpie|cbfv|full>.per_repeat_r2_mean/_std")
#   - combined ceiling band and the full-minus-magpie headroom fraction:
#     NOISE_FLOOR_INPUTS_PATH ("item3_combined_ceiling_new" for the
#     band, "item5_descriptor_ablation_headroom_fractions_new" for the
#     delta and fraction-of-headroom-closed annotation -- both read
#     directly rather than recomputed here, since the latter already
#     encodes the same rounded-confirmed-R^2 convention the headroom
#     figure's load_headroom_data had to reconcile by hand). zT uses the
#     "zT (vs ZT_author_declared)" combined-ceiling row and the
#     "zT (declared)" fraction row, matching the headroom figure's choice.
DESCRIPTOR_ABLATION_PROPERTIES = ["S", "sigma", "kappa", "zT"]
DESCRIPTOR_ABLATION_PROPERTY_LABELS = {
    "S": "S", "sigma": "$\\sigma$ (log$_{10}$)", "kappa": "$\\kappa$ (log$_{10}$)", "zT": "zT",
}
DESCRIPTOR_ABLATION_FEATURE_SETS = ["magpie", "cbfv", "full"]
DESCRIPTOR_ABLATION_FEATURE_COUNTS = {"magpie": 133, "cbfv": 265, "full": 397}


def load_descriptor_ablation_data(
    ablation_metrics_path=DESCRIPTOR_ABLATION_METRICS_PATH,
    noise_floor_inputs_path=NOISE_FLOOR_INPUTS_PATH,
):
    """
    Assemble per-target feature-count/R^2/SD triples, the combined-
    ceiling band, and the full-minus-magpie delta/headroom-fraction
    annotation, from the two committed artifacts named in the
    module-level comment above. Returns {property: {feature_counts,
    r2_values, r2_sd, band_lower, band_upper, delta, fraction_pct}}.

    delta and fraction_pct are NOT read from noise_floor_inputs.json's
    own item5_descriptor_ablation_headroom_fractions_new field, even
    though that field is now full-precision-correct as of the
    2026-09-24 row-alignment rerun (it previously stored the 4-decimal-
    rounded ablation delta for S/sigma/kappa, e.g. S=0.0064 instead of
    the full-precision 0.006472, which shifted the displayed fraction
    range across a rounding boundary -- CLAUDE.md's published S/kappa
    fractions were never actually wrong, only this field's own delta
    was). Kept reading delta from ablation_metrics.json directly and
    recomputing fraction_pct against item3's headroom bounds regardless,
    since that path is correct independent of which noise-floor artifact
    is loaded and needs no future audit to confirm it still agrees with
    item5.
    """
    with open(ablation_metrics_path, encoding="utf-8") as f:
        ablation = json.load(f)
    with open(noise_floor_inputs_path, encoding="utf-8") as f:
        noise_floor = json.load(f)

    combined_by_label = {
        entry["label"]: entry for entry in noise_floor["item3_combined_ceiling_new"]
    }
    deltas_full_minus_magpie = ablation["deltas_full_minus_magpie"]

    data = {}
    for prop in DESCRIPTOR_ABLATION_PROPERTIES:
        entry = ablation[prop]
        r2_values = [entry[fs]["per_repeat_r2_mean"] for fs in DESCRIPTOR_ABLATION_FEATURE_SETS]
        r2_sd = [entry[fs]["per_repeat_r2_std"] for fs in DESCRIPTOR_ABLATION_FEATURE_SETS]

        combined_label = ZT_COMBINED_CEILING_LABEL if prop == "zT" else prop
        combined_entry = combined_by_label[combined_label]
        delta = deltas_full_minus_magpie[prop]
        headroom_lower = combined_entry["headroom_lower"]
        headroom_upper = combined_entry["headroom_upper"]

        data[prop] = {
            "feature_counts": [DESCRIPTOR_ABLATION_FEATURE_COUNTS[fs] for fs in DESCRIPTOR_ABLATION_FEATURE_SETS],
            "r2_values": r2_values,
            "r2_sd": r2_sd,
            "band_lower": combined_entry["r2_comb_lower"],
            "band_upper": combined_entry["r2_comb_upper"],
            "delta": delta,
            "fraction_pct": (100 * delta / headroom_upper, 100 * delta / headroom_lower),
        }
    return data


def make_descriptor_ablation(
    out_path,
    ablation_metrics_path=DESCRIPTOR_ABLATION_METRICS_PATH,
    noise_floor_inputs_path=NOISE_FLOOR_INPUTS_PATH,
):
    """
    2x2 panels, one per target. Each panel: three points (magpie=133,
    cbfv=265, full=397 features) with across-repeat SD error bars,
    connected by a line (chemistry_cluster/"internal" colour, since
    every point here is a chemistry-cluster grouped R^2), plus a shaded
    horizontal band at the combined label-noise ceiling range
    ("headroom" colour). Y axis spans from just below the lowest point
    to 1.0, so the gap between the points and the band -- the dominant
    visual -- is what the reader sees. Each panel is annotated with the
    full-minus-magpie delta and the fraction of headroom it closes.
    """
    data = load_descriptor_ablation_data(ablation_metrics_path, noise_floor_inputs_path)

    fig, axes = plt.subplots(2, 2, figsize=get_figsize(2, 2, panel_width=4.6, panel_height=4.0))
    panel_letters = ["a", "b", "c", "d"]

    for ax, prop, letter in zip(axes.flat, DESCRIPTOR_ABLATION_PROPERTIES, panel_letters):
        d = data[prop]
        x = d["feature_counts"]

        ax.axhspan(
            d["band_lower"], d["band_upper"], color=PALETTE["headroom"], alpha=0.35,
            zorder=1, label="Combined ceiling range" if prop == "S" else None,
        )
        ax.errorbar(
            x, d["r2_values"], yerr=d["r2_sd"], color=PALETTE["chemistry_cluster"],
            marker="o", markersize=5, linewidth=1.6, capsize=3, elinewidth=1.1, capthick=1.1,
            zorder=3, label="Chemistry-cluster R$^2$" if prop == "S" else None,
        )

        y_low = min(d["r2_values"]) - max(d["r2_sd"]) - 0.02
        ax.set_ylim(y_low, 1.0)
        ax.set_xlim(min(x) - 40, max(x) + 40)
        ax.set_xticks(x)
        ax.set_xlabel("Number of features")
        ax.set_ylabel(f"R$^2$ ({DESCRIPTOR_ABLATION_PROPERTY_LABELS[prop]})")

        fraction_lo, fraction_hi = d["fraction_pct"]
        annotation_y = (max(d["r2_values"]) + d["band_lower"]) / 2
        ax.text(
            sum(x) / len(x), annotation_y,
            f"full - magpie = {d['delta']:.4f}\ncloses {fraction_lo:.1f}-{fraction_hi:.1f}% of headroom",
            ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", linewidth=0.5, alpha=0.9),
            zorder=4,
        )
        add_panel_label(ax, letter)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
        bbox_transform=fig.transFigure, ncol=2, framealpha=0.9,
        columnspacing=1.2, handletextpad=0.6,
    )
    fig.subplots_adjust(bottom=0.13, hspace=0.38, wspace=0.35)
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


# zT parity, random vs chemistry-cluster grouping, shown directly rather
# than only as a summary R^2. Per-row out-of-fold predictions:
#   - random 80/20: results/ungrouped_snapfix/20260922T093243/
#     zT_random_f20/ (20 outer folds, the snapfix-dataset rerun -- see
#     CLAUDE.md's Five-Way Ladder correction. Previously pointed at
#     checkpoints/ladder_regen_dl/zT_random_f20/, an unidentified,
#     non-File-A dataset snapshot; corrected 2026-09-22.)
#   - chemistry-cluster (snapfix grouping): results/ladder_regen_snapfix/
#     20260917T150000/zT_chemistry_full/ (25 outer folds = 5 repeats x 5
#     folds, the checkpoint directory behind the confirmed 0.7456
#     chemistry-cluster zT value after the SNAP(0.05)/S5 grouping fix)
# Both R^2 values are recomputed here from the raw predictions (not
# copied from CLAUDE.md), so this figure is self-verifying; see the
# printed report in the task response for the exact match confirmation.
ZT_PARITY_RANDOM_CHECKPOINT_DIR = Path("results/ungrouped_snapfix/20260922T093243/zT_random_f20")
ZT_PARITY_CHEMISTRY_CHECKPOINT_DIR = Path("results/ladder_regen_snapfix/20260917T150000/zT_chemistry_full")


def load_zt_parity_predictions(checkpoint_dir):
    """Pool every repeat*_fold*_predictions.npz under checkpoint_dir into one (y_true, y_pred) pair."""
    npz_paths = sorted(Path(checkpoint_dir).glob("repeat*_fold*_predictions.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No repeat*_fold*_predictions.npz files in {checkpoint_dir}")
    y_true_parts, y_pred_parts = [], []
    for npz_path in npz_paths:
        data = np.load(npz_path)
        y_true_parts.append(data["y_true"])
        y_pred_parts.append(data["y_pred"])
    return np.concatenate(y_true_parts), np.concatenate(y_pred_parts), len(npz_paths)


def make_zt_parity(
    out_path,
    random_checkpoint_dir=ZT_PARITY_RANDOM_CHECKPOINT_DIR,
    chemistry_checkpoint_dir=ZT_PARITY_CHEMISTRY_CHECKPOINT_DIR,
):
    """
    Two hexbin parity panels for zT, random 80/20 (left) vs
    chemistry-cluster (right), identical axis limits and identical
    (shared, log-scaled) colour scale on both, so the difference in
    prediction spread -- not an artifact of differing color normalization
    -- is the dominant visual. Identity line on both. Each panel is
    labelled with its split-type name in that split's PALETTE colour
    (random_split / chemistry_cluster) rather than a generic a/b letter,
    and annotated with its own recomputed pooled R^2 and n.
    """
    yt_r, yp_r, n_files_r = load_zt_parity_predictions(random_checkpoint_dir)
    yt_c, yp_c, n_files_c = load_zt_parity_predictions(chemistry_checkpoint_dir)
    r2_r = r2_score(yt_r, yp_r)
    r2_c = r2_score(yt_c, yp_c)

    lo = min(yt_r.min(), yp_r.min(), yt_c.min(), yp_c.min())
    hi = max(yt_r.max(), yp_r.max(), yt_c.max(), yp_c.max())
    pad = (hi - lo) * 0.03
    lo, hi = lo - pad, hi + pad

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=get_figsize(1, 2, panel_width=5.2, panel_height=5.0), constrained_layout=True,
    )

    hb_left = ax_left.hexbin(yt_r, yp_r, gridsize=60, extent=(lo, hi, lo, hi), cmap="viridis", norm=LogNorm(), mincnt=1)
    hb_right = ax_right.hexbin(yt_c, yp_c, gridsize=60, extent=(lo, hi, lo, hi), cmap="viridis", norm=LogNorm(), mincnt=1)
    vmax = max(hb_left.get_array().max(), hb_right.get_array().max())
    shared_norm = LogNorm(vmin=1, vmax=vmax)
    hb_left.set_norm(shared_norm)
    hb_right.set_norm(shared_norm)

    panels = [
        (ax_left, hb_left, "Random 80/20", PALETTE["random_split"], r2_r, len(yt_r)),
        (ax_right, hb_right, "Chemistry-Cluster CV", PALETTE["chemistry_cluster"], r2_c, len(yt_c)),
    ]
    for ax, hb, split_name, color, r2, n in panels:
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.2, zorder=5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Actual zT")
        ax.set_ylabel("Predicted zT")
        ax.text(
            0.04, 0.96, split_name, transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color=color,
        )
        ax.text(
            0.04, 0.87, f"$R^2$={r2:.4f}\nn={n:,}", transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", linewidth=0.5),
        )

    fig.colorbar(hb_right, ax=[ax_left, ax_right], label="Rows per hexbin (log scale)", shrink=0.85)
    save_figure(fig, out_path)
    plt.close(fig)


# =====================================================================
# External transfer: full-set vs in-support R^2, ESTM (2 dedup passes)
# and teMatDb (DOI-disjoint only). Sources:
#   - internal chemistry-cluster R^2: LADDER_METRICS_PATH
#     ("<target>_chemistry_full" runs, per_repeat_r2_mean) -- same
#     numbers as every other figure's "honest ceiling".
#   - ESTM full-set R^2: EXTERNAL_SNAPFIX_DIR/estm_results.json
#     (dedup_a_source_doi = DOI-disjoint, dedup_b_chemistry_cluster =
#     cluster-disjoint).
#   - ESTM in-support R^2 and OOD fractions: recomputed here directly
#     from EXTERNAL_SNAPFIX_DIR/estm_predictions_pass{a,b}.npz, using
#     the joint S/sigma/kappa/temperature bounds CLAUDE.md documents as
#     reconstructed from the snapfix training set's own min/max
#     (ESTM_TRAINING_BOUNDS below) -- no committed script produces these
#     numbers (CLAUDE.md's own "RESOLVED 2026-09-19" note says so), but
#     the bounds and resulting n/R^2 are fully documented and were
#     validated there against an exact n=2,709 row-count match; recomputing
#     here reproduces that table exactly (verified before rendering).
#   - teMatDb: EXTERNAL_SNAPFIX_DIR/tematdb_D_scoring_snapfix.json,
#     stratum "a" (61 samples / 903 rows / 49 clusters) -- confirmed to
#     be the DOI-disjoint stratum by matching its R^2 values to CLAUDE.md's
#     documented teMatDb DOI-disjoint figures (S=0.699, sigma=0.175,
#     kappa=0.653, zT=0.496). Stratum "b" (cluster-disjoint) is NOT
#     plotted: 27 samples across 27 distinct clusters cannot support a
#     meaningful R^2 (one row per cluster on average), and CLAUDE.md's
#     own caveats already warn against quoting stratum-b R^2 precisely.
#     teMatDb has no in-support/full-set split here: its OOD tail is
#     0.12% (see CLAUDE.md), too small to be worth splitting on.
# zT_derived is excluded from every panel: CLAUDE.md documents it as
# numerically ill-conditioned (a 0.7% smear-factor change swung ESTM
# pass-b zT_derived R^2 from -0.019 to +0.101), so only the direct-vs-
# derived ORDERING is stable enough to report, not this pathway's
# point value sitting alongside direct zT as if equally trustworthy.
EXTERNAL_SNAPFIX_DIR = Path("results/external_snapfix/20260917T160553")
EXTERNAL_TRANSFER_PROPERTIES = ["S", "sigma", "kappa", "zT_direct"]
EXTERNAL_TRANSFER_PROPERTY_LABELS = {
    "S": "S", "sigma": "$\\sigma$ (log$_{10}$)", "kappa": "$\\kappa$ (log$_{10}$)", "zT_direct": "zT",
}
# Reconstructed from the snapfix training CSV's own per-property min/max
# (300-800K rows only), joint across S/sigma/kappa -- see CLAUDE.md's
# External Validation section, "RESOLVED 2026-09-19".
ESTM_TRAINING_BOUNDS = {
    "S": (-461.0258, 562.5),
    "sigma": (958.7831, 1656678.2772),
    "kappa": (0.2830364, 13.7753),
}


def load_internal_chemistry_r2(ladder_metrics_path=LADDER_METRICS_PATH):
    """Internal chemistry-cluster R^2 per property, keyed to match EXTERNAL_TRANSFER_PROPERTIES."""
    with open(ladder_metrics_path, encoding="utf-8") as f:
        ladder_metrics = json.load(f)
    return {
        "S": ladder_metrics["runs"]["S_chemistry_full"]["per_repeat_r2_mean"],
        "sigma": ladder_metrics["runs"]["sigma_chemistry_full"]["per_repeat_r2_mean"],
        "kappa": ladder_metrics["runs"]["kappa_chemistry_full"]["per_repeat_r2_mean"],
        "zT_direct": ladder_metrics["runs"]["zT_chemistry_full"]["per_repeat_r2_mean"],
    }


def load_estm_pass_transfer(pass_letter, external_dir=EXTERNAL_SNAPFIX_DIR):
    """
    Full-set and in-support R^2, plus marginal and joint OOD fractions,
    for one ESTM dedup pass ("a" = DOI-disjoint, "b" = cluster-disjoint).
    Computed directly from estm_predictions_pass<letter>.npz using
    ESTM_TRAINING_BOUNDS; not read from any precomputed field, since none
    exists on disk for this quantity.
    """
    data = np.load(external_dir / f"estm_predictions_pass{pass_letter}.npz", allow_pickle=True)
    in_support = np.ones(len(data["S_true"]), dtype=bool)
    marginal_ood_pct = {}
    for prop in ["S", "sigma", "kappa"]:
        values = data[f"{prop}_true"]
        lo, hi = ESTM_TRAINING_BOUNDS[prop]
        prop_in_support = (values >= lo) & (values <= hi)
        marginal_ood_pct[prop] = 100 * (1 - prop_in_support.mean())
        in_support &= prop_in_support
    in_support &= (data["temperature"] >= 300) & (data["temperature"] <= 800)

    pred_keys = {
        "S": ("S_true", "S_pred"),
        "sigma": ("sigma_log_true", "sigma_log_pred"),
        "kappa": ("kappa_log_true", "kappa_log_pred"),
        "zT_direct": ("zT_true", "zT_direct_pred"),
    }
    result = {"joint_ood_pct": 100 * (1 - in_support.mean()), "marginal_ood_pct": marginal_ood_pct}
    for prop, (true_key, pred_key) in pred_keys.items():
        yt, yp = data[true_key], data[pred_key]
        result[prop] = {
            "full_r2": r2_score(yt, yp), "full_n": len(yt),
            "insupport_r2": r2_score(yt[in_support], yp[in_support]), "insupport_n": int(in_support.sum()),
        }
    return result


def load_tematdb_doi_disjoint(scoring_path=EXTERNAL_SNAPFIX_DIR / "tematdb_D_scoring_snapfix.json"):
    """teMatDb stratum 'a' (DOI-disjoint) R^2 per property; stratum 'b' (cluster-disjoint, 27 clusters) is not loaded here -- see module comment."""
    with open(scoring_path, encoding="utf-8") as f:
        scoring = json.load(f)
    metrics = scoring["strata"]["a"]["metrics"]
    return {
        "S": metrics["S"]["r2"], "sigma": metrics["sigma_log10"]["r2"],
        "kappa": metrics["kappa_log10"]["r2"], "zT_direct": metrics["zT_direct_vs_declared"]["r2"],
    }


def make_external_transfer(out_path, ladder_metrics_path=LADDER_METRICS_PATH, external_dir=EXTERNAL_SNAPFIX_DIR):
    """
    Three panels: ESTM DOI-disjoint, ESTM cluster-disjoint, teMatDb
    DOI-disjoint. Each shows grouped bars per property (S, sigma, kappa,
    zT direct): internal chemistry-cluster R^2, external full-set R^2,
    and (ESTM only) external in-support R^2. OOD fraction annotated
    above each property group -- the property's own marginal OOD % for
    S/sigma/kappa, and the joint OOD % for zT (which has no bound of its
    own; its in-support subset is the same joint one used for S/sigma/
    kappa together). teMatDb has no in-support bar (0.12% OOD, judged
    negligible in CLAUDE.md) and is not annotated.
    """
    internal = load_internal_chemistry_r2(ladder_metrics_path)
    estm_a = load_estm_pass_transfer("a", external_dir)
    estm_b = load_estm_pass_transfer("b", external_dir)
    tematdb = load_tematdb_doi_disjoint()

    fig, axes = plt.subplots(1, 3, figsize=get_figsize(1, 3, panel_width=5.3, panel_height=5.0))
    props = EXTERNAL_TRANSFER_PROPERTIES
    labels = [EXTERNAL_TRANSFER_PROPERTY_LABELS[p] for p in props]
    n_props = len(props)
    group_centers = np.arange(n_props)

    def draw_panel(ax, bars, ood_pct=None):
        n_bars = len(bars)
        bar_width = 0.8 / n_bars
        offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_width
        for i, (bar_label, color, hatch, values) in enumerate(bars):
            heights = [values[p] for p in props]
            x = group_centers + offsets[i]
            rects = ax.bar(
                x, heights, width=bar_width * 0.92, color=color, hatch=hatch,
                edgecolor="black", linewidth=0.5, label=bar_label, zorder=2,
            )
            for rect, h in zip(rects, heights):
                ax.text(rect.get_x() + rect.get_width() / 2, h + 0.02, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=7, rotation=90 if h < 0.3 else 0)
        if ood_pct is not None:
            for j, p in enumerate(props):
                ax.text(group_centers[j], 1.05, f"OOD {ood_pct[p]:.1f}%", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(group_centers)
        ax.set_xticklabels(labels)
        ax.set_ylim(-0.15, 1.18)
        ax.axhline(0, color="black", linewidth=0.6, zorder=1)

    ood_a = {**estm_a["marginal_ood_pct"], "zT_direct": estm_a["joint_ood_pct"]}
    draw_panel(axes[0], [
        ("Internal (chemistry-cluster)", PALETTE["internal"], HATCH["internal"], internal),
        ("External, full-set", PALETTE["external_fullset"], HATCH["external_fullset"],
         {p: estm_a[p]["full_r2"] for p in props}),
        ("External, in-support", PALETTE["external_insupport"], HATCH["external_insupport"],
         {p: estm_a[p]["insupport_r2"] for p in props}),
    ], ood_pct=ood_a)
    axes[0].set_ylabel("R$^2$")
    add_panel_label(axes[0], "a")

    ood_b = {**estm_b["marginal_ood_pct"], "zT_direct": estm_b["joint_ood_pct"]}
    draw_panel(axes[1], [
        ("Internal (chemistry-cluster)", PALETTE["internal"], HATCH["internal"], internal),
        ("External, full-set", PALETTE["external_fullset"], HATCH["external_fullset"],
         {p: estm_b[p]["full_r2"] for p in props}),
        ("External, in-support", PALETTE["external_insupport"], HATCH["external_insupport"],
         {p: estm_b[p]["insupport_r2"] for p in props}),
    ], ood_pct=ood_b)
    add_panel_label(axes[1], "b")

    draw_panel(axes[2], [
        ("Internal (chemistry-cluster)", PALETTE["internal"], HATCH["internal"], internal),
        ("External, full-set", PALETTE["external_fullset"], HATCH["external_fullset"], tematdb),
    ])
    axes[2].text(
        0.5, 1.12, "OOD tail 0.12% (not split)", transform=axes[2].transAxes,
        ha="center", va="bottom", fontsize=7.5, style="italic",
    )
    add_panel_label(axes[2], "c")

    handles, hlabels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, hlabels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
        bbox_transform=fig.transFigure, ncol=3, framealpha=0.9, columnspacing=1.2, handletextpad=0.6,
    )
    fig.subplots_adjust(bottom=0.22, top=0.85, wspace=0.3)
    save_figure(fig, out_path)
    plt.close(fig)


# =====================================================================
# Electrical conductivity extrapolation: training vs ESTM cluster-
# disjoint (pass b) log10(sigma) distributions, with training's cleaned
# floor and the ESTM mass extrapolating below it. Sources:
#   - training sigma: the snapfix featurized CSV's own "sigma" column
#     (non-null rows) -- SIGMA_EXTRAPOLATION_TRAINING_CSV.
#   - ESTM sigma: EXTERNAL_SNAPFIX_DIR/estm_predictions_passb.npz's
#     sigma_true (pass b = cluster-disjoint, chosen because its sigma
#     OOD fraction, ~15.5%, is the one CLAUDE.md reports and cross-checks
#     against; pass a's is ~12.0%, a different number).
# Floor = ESTM_TRAINING_BOUNDS["sigma"][0] (958.7831 S/m), the same
# bound load_estm_pass_transfer uses for the in-support split above.
SIGMA_EXTRAPOLATION_TRAINING_CSV = Path("data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv")


def load_sigma_extrapolation_data(
    training_csv=SIGMA_EXTRAPOLATION_TRAINING_CSV, external_dir=EXTERNAL_SNAPFIX_DIR, n_bins=50,
):
    """
    Returns log10(sigma) arrays for training and ESTM pass-b, shared bin
    edges spanning both, per-array histogram counts, the training floor
    in log10 space, and the fraction of ESTM mass below it.
    """
    train_df = pd.read_csv(training_csv, usecols=["sigma"])
    train_log_sigma = np.log10(train_df["sigma"].dropna().to_numpy())
    train_log_sigma = train_log_sigma[np.isfinite(train_log_sigma)]

    estm_data = np.load(external_dir / "estm_predictions_passb.npz", allow_pickle=True)
    estm_sigma = estm_data["sigma_true"]
    estm_log_sigma = np.log10(estm_sigma[estm_sigma > 0])

    floor = ESTM_TRAINING_BOUNDS["sigma"][0]
    log_floor = np.log10(floor)

    lo = min(train_log_sigma.min(), estm_log_sigma.min())
    hi = max(train_log_sigma.max(), estm_log_sigma.max())
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    train_counts, _ = np.histogram(train_log_sigma, bins=bin_edges)
    estm_counts, _ = np.histogram(estm_log_sigma, bins=bin_edges)

    frac_below_floor = float((estm_log_sigma < log_floor).mean())
    return {
        "train_log_sigma": train_log_sigma, "estm_log_sigma": estm_log_sigma,
        "bin_edges": bin_edges, "train_counts": train_counts, "estm_counts": estm_counts,
        "floor": floor, "log_floor": log_floor, "frac_below_floor": frac_below_floor,
    }


def make_sigma_extrapolation(out_path, training_csv=SIGMA_EXTRAPOLATION_TRAINING_CSV, external_dir=EXTERNAL_SNAPFIX_DIR):
    """
    Overlaid density histograms of log10(sigma): training (snapfix CSV)
    vs ESTM cluster-disjoint rows. Vertical line at training's cleaned
    conductivity floor; the ESTM mass below it (extrapolation, not
    interpolation, per CLAUDE.md's External Validation section) is
    shaded and its fraction annotated.
    """
    d = load_sigma_extrapolation_data(training_csv, external_dir)

    fig, ax = plt.subplots(figsize=get_figsize(1, 1, panel_width=8.0, panel_height=5.0))
    ax.hist(
        d["train_log_sigma"], bins=d["bin_edges"], density=True, color=PALETTE["internal"],
        alpha=0.55, label="Training (snapfix)", zorder=2,
    )
    ax.hist(
        d["estm_log_sigma"], bins=d["bin_edges"], density=True, color=PALETTE["external_fullset"],
        alpha=0.55, label="ESTM, cluster-disjoint", zorder=3,
    )
    ax.axvspan(
        d["bin_edges"][0], d["log_floor"], color="black", alpha=0.08, zorder=1,
        label=f"ESTM below training floor ({100 * d['frac_below_floor']:.1f}%)",
    )
    ax.axvline(d["log_floor"], color="black", linestyle="--", linewidth=1.4, zorder=4)
    ax.set_xlabel("log$_{10}(\\sigma$, S/m)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9.5)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


# =====================================================================
# Direct-vs-derived zT parity, both pathways from the identical
# all-four-properties-present subset (56,088 rows, chemistry-cluster
# grouped CV, each model on its own target's canonical frozen
# hyperparameters -- see CLAUDE.md's Direct-vs-Derived zT section).
# Source: results/direct_vs_derived_snapfix/20260924T124138_per_target_cuda/
# repeat*_fold*.npz (5 repeats x 5 outer folds = 25 files). Superseded
# source (zT's set shared by all four models, device not recorded):
# checkpoints/direct_vs_derived_zt_snapfix_canonical/ (direct=0.7262,
# derived=0.5244). R^2 is pooled across ALL 25 files (not per-repeat-
# then-averaged) and is checked below against the folder's own
# results.json -- a different convention from the Five-Way Ladder's
# per-repeat-mean reporting, which is a different result computed a
# different way.
DIRECT_VS_DERIVED_CHECKPOINT_DIR = Path("results/direct_vs_derived_snapfix/20260924T124138_per_target_cuda")


def load_direct_vs_derived_predictions(checkpoint_dir=DIRECT_VS_DERIVED_CHECKPOINT_DIR):
    """Pool every repeat*_fold*.npz's direct and derived zT true/pred arrays."""
    npz_paths = sorted(Path(checkpoint_dir).glob("repeat*_fold*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No repeat*_fold*.npz files in {checkpoint_dir}")
    direct_true, direct_pred, derived_true, derived_pred = [], [], [], []
    for npz_path in npz_paths:
        data = np.load(npz_path)
        direct_true.append(data["zT_direct_true"]); direct_pred.append(data["zT_direct_pred"])
        derived_true.append(data["zT_derived_true"]); derived_pred.append(data["zT_derived_pred"])
    return {
        "direct_true": np.concatenate(direct_true), "direct_pred": np.concatenate(direct_pred),
        "derived_true": np.concatenate(derived_true), "derived_pred": np.concatenate(derived_pred),
        "n_files": len(npz_paths),
    }


def make_zt_direct_vs_derived(out_path, checkpoint_dir=DIRECT_VS_DERIVED_CHECKPOINT_DIR):
    """
    Two hexbin parity panels for zT: direct prediction (left) vs derived
    S^2*sigma*T/kappa (right), identical axis limits and a shared
    log-scaled colour scale, identity line on both. Panel labels use
    plain bold text with two existing PALETTE hues for contrast (direct/
    derived is not one of the shared PALETTE concepts; direct reuses
    "chemistry_cluster" -- the pathway this project treats as the honest
    number everywhere else -- and derived reuses "full_feature_set",
    chosen for visual contrast rather than shared semantics).
    """
    d = load_direct_vs_derived_predictions(checkpoint_dir)
    r2_direct = r2_score(d["direct_true"], d["direct_pred"])
    r2_derived = r2_score(d["derived_true"], d["derived_pred"])

    with open(Path(checkpoint_dir) / "results.json", encoding="utf-8") as f:
        recorded = json.load(f)
    for got, key in ((r2_direct, "zT_direct"), (r2_derived, "zT_derived")):
        if abs(got - recorded[key]["pooled_r2"]) > 1e-9:
            raise ValueError(
                f"{key}: pooled R^2 from the per-fold predictions ({got:.10f}) does not match "
                f"{checkpoint_dir}/results.json ({recorded[key]['pooled_r2']:.10f})"
            )

    lo = min(d["direct_true"].min(), d["direct_pred"].min(), d["derived_true"].min(), d["derived_pred"].min())
    hi = max(d["direct_true"].max(), d["direct_pred"].max(), d["derived_true"].max(), d["derived_pred"].max())
    pad = (hi - lo) * 0.03
    lo, hi = lo - pad, hi + pad

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=get_figsize(1, 2, panel_width=5.2, panel_height=5.0), constrained_layout=True,
    )
    hb_left = ax_left.hexbin(d["direct_true"], d["direct_pred"], gridsize=60, extent=(lo, hi, lo, hi),
                              cmap="viridis", norm=LogNorm(), mincnt=1)
    hb_right = ax_right.hexbin(d["derived_true"], d["derived_pred"], gridsize=60, extent=(lo, hi, lo, hi),
                                cmap="viridis", norm=LogNorm(), mincnt=1)
    vmax = max(hb_left.get_array().max(), hb_right.get_array().max())
    shared_norm = LogNorm(vmin=1, vmax=vmax)
    hb_left.set_norm(shared_norm)
    hb_right.set_norm(shared_norm)

    panels = [
        (ax_left, "Direct", PALETTE["chemistry_cluster"], r2_direct, len(d["direct_true"])),
        (ax_right, "Derived (S$^2\\sigma T/\\kappa$)", PALETTE["full_feature_set"], r2_derived, len(d["derived_true"])),
    ]
    for ax, label, color, r2, n in panels:
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.2, zorder=5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Actual zT")
        ax.set_ylabel("Predicted zT")
        ax.text(0.04, 0.96, label, transform=ax.transAxes, ha="left", va="top",
                fontsize=12, fontweight="bold", color=color)
        ax.text(0.04, 0.87, f"$R^2$={r2:.4f}\nn={n:,}", transform=ax.transAxes, ha="left", va="top",
                fontsize=9.5, bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", linewidth=0.5))

    fig.colorbar(hb_right, ax=[ax_left, ax_right], label="Rows per hexbin (log scale)", shrink=0.85)
    save_figure(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 and Figure 3 of the paper: two schematics, no data files read.
# ---------------------------------------------------------------------------

SCHEMATIC_TRAIN_COLOR = "#0072B2"  # Okabe-Ito blue
SCHEMATIC_TEST_COLOR = "#D55E00"   # Okabe-Ito vermillion
SCHEMATIC_GREY = "0.45"
SCHEMATIC_NOTE = "Schematic; values illustrative"


def _formula_mathtext(formula):
    """Render a formula string with numeric subscripts, e.g. 'Bi2Te2.7Se0.3' -> 'Bi$_{2}$Te$_{2.7}$Se$_{0.3}$'."""
    import re

    return re.sub(r"(?<=[A-Za-z\)])(\d+(?:\.\d+)?)", r"$_{\1}$", formula)


def _schematic_curve(x, level, rise, curvature=1.3, hump=0.0):
    """Smooth illustrative zT(T) shape on x in [0, 1]: a rising power law, optionally with a roll-over hump."""
    y = level + rise * x**curvature
    if hump:
        y = level + hump * np.sin(np.pi * np.clip(x / 0.9, 0, 1)) ** 1.2
    return y


def make_leakage_schematic(out_path):
    """
    Figure 1: three panels illustrating why random row splits leak.
    (a) one zT(T) curve with scattered test rows; (b) near-duplicate curves
    of one host lattice on both sides of a split; (c) the chemistry-cluster
    split, where the whole cluster is test. Illustrative values, no numeric
    tick labels.
    """
    from matplotlib.lines import Line2D

    T = np.linspace(300, 800, 21)          # 300-800 K at 25 K
    x = (T - 300) / 500.0                  # normalised; tick labels are suppressed
    train_kw = dict(marker="o", ms=5.0, mfc=SCHEMATIC_TRAIN_COLOR, mec="white", mew=0.5, ls="none", zorder=3)
    test_kw = dict(marker="D", ms=5.0, mfc=SCHEMATIC_TEST_COLOR, mec="white", mew=0.5, ls="none", zorder=4)
    x_right = {"a": 1.12, "b": 2.05, "c": 2.05}   # right x-limit: b and c leave room for curve labels

    fig, axes = plt.subplots(
        1, 3, figsize=(15.5, 3.9),
        gridspec_kw={"wspace": 0.12, "width_ratios": [x_right["a"] + 0.05, x_right["b"] + 0.05, x_right["c"] + 0.05]},
    )

    def base(ax, title, letter):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("T")
        ax.set_ylabel("zT")
        ax.set_xlim(-0.05, x_right[letter])
        ax.set_ylim(-0.02, 1.62)
        ax.set_title(title, fontsize=12, pad=8)
        add_panel_label(ax, letter, x=-0.02, y=1.02)

    def draw_curve(ax, y, is_test, label=None, label_y=None):
        """Draw a whole curve that is entirely train or entirely test, with an optional right-hand label."""
        color = SCHEMATIC_TEST_COLOR if is_test else SCHEMATIC_TRAIN_COLOR
        ax.plot(x, y, color=color, lw=1.1, alpha=0.5, zorder=2)
        ax.plot(x, y, **(test_kw if is_test else train_kw))
        if label is None:
            return None
        return ax.annotate(
            label, xy=(x[-1] + 0.02, y[-1]), xytext=(1.16, label_y), textcoords="data",
            fontsize=9.5, va="center", ha="left", color="0.15",
            arrowprops=dict(arrowstyle="-", color="0.65", lw=0.7, shrinkA=2, shrinkB=2),
        )

    note_kw = dict(fontsize=9.5, va="top", ha="left", color="0.15", linespacing=1.3)

    # (a) random row split: one curve, 4 of 21 points (19%) are test, scattered
    ax = axes[0]
    base(ax, "Random row split", "a")
    y = 0.25 + 0.95 * x**1.3
    test_idx = [3, 8, 13, 18]
    tr = [i for i in range(len(x)) if i not in test_idx]
    ax.plot(x, y, color=SCHEMATIC_GREY, lw=1.1, alpha=0.55, zorder=2)
    ax.plot(x[tr], y[tr], **train_kw)
    ax.plot(x[test_idx], y[test_idx], **test_kw)
    ax.text(0.0, 1.56, "test rows lie between\ntraining rows of the\nsame curve", **note_kw)
    ax.annotate("", xy=(x[8], y[8] + 0.05), xytext=(0.36, 1.05),
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8, shrinkA=0, shrinkB=2))

    # (b) near-duplicate curves across dopant labels: two train, two test, interleaved
    near = [
        # label, level, rise, curvature, is_test, label slot (b), label slot (c)
        ("PbTe", 0.36, 0.94, 1.30, False, 1.30, 1.42),
        ("Pb$_{0.99}$Na$_{0.01}$Te", 0.32, 0.90, 1.28, True, 1.14, 1.26),
        ("Pb$_{0.98}$Na$_{0.02}$Te", 0.28, 0.86, 1.32, False, 0.98, 1.10),
        ("(PbTe)$_{0.98}$(SrTe)$_{0.02}$", 0.24, 0.82, 1.30, True, 0.82, 0.94),
    ]
    ax = axes[1]
    base(ax, "Near-duplicates across dopant labels", "b")
    for name, level, rise, curv, is_test, slot_b, _slot_c in near:
        draw_curve(ax, level + rise * x**curv, is_test, label=name, label_y=slot_b)
    ax.text(0.0, 1.56, "same host lattice on both\nsides of the split", **note_kw)

    # (c) chemistry-cluster split: the same four curves all test, two differently shaped train curves
    ax = axes[2]
    base(ax, "Chemistry-cluster split", "c")
    cluster_labels = []
    for name, level, rise, curv, _is_test, _slot_b, slot_c in near:
        cluster_labels.append(draw_curve(ax, level + rise * x**curv, True, label=name, label_y=slot_c))
    bi = 0.10 + 0.34 * np.sin(np.pi * x**0.8)          # rises, peaks, rolls over
    se = 0.08 + 0.85 * x**3.2                           # low, then steep rise
    draw_curve(ax, se, False, label="SnSe", label_y=0.66)
    draw_curve(ax, bi, False, label="Bi$_{2}$Te$_{3}$", label_y=0.28)
    ax.text(0.0, 1.56, "whole cluster held out", **note_kw)
    # bracket around the four cluster labels, placed from their rendered extents
    fig.canvas.draw()
    inv = ax.transData.inverted()
    xs = [inv.transform((t.get_window_extent().x1, 0))[0] for t in cluster_labels]
    bx = max(xs) + 0.04
    ax.plot([bx - 0.03, bx, bx, bx - 0.03], [1.42 + 0.07, 1.42 + 0.07, 0.94 - 0.07, 0.94 - 0.07],
            color="0.3", lw=1.0, clip_on=False)

    handles = [
        Line2D([], [], marker="o", ms=6, mfc=SCHEMATIC_TRAIN_COLOR, mec="white", ls="none", label="Training row"),
        Line2D([], [], marker="D", ms=6, mfc=SCHEMATIC_TEST_COLOR, mec="white", ls="none", label="Test row"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.text(0.995, -0.055, SCHEMATIC_NOTE, fontsize=8.5, color="0.5", ha="right", va="center")
    save_figure(fig, out_path)
    plt.close(fig)


def chemistry_cluster_trace(formula):
    """
    Step-by-step trace of src.canonicalization.chemistry_cluster_id for one
    formula, for the grouping-rule schematic. The final id is the pipeline
    function's own return value; the intermediates re-apply that function's
    steps with the module's own constants (dopant threshold, snap
    tolerance), and the trace asserts that its last step reproduces the
    function's output, so a drift between the two fails loudly.
    """
    from src.canonicalization import (
        DEFAULT_DOPANT_THRESHOLD_FRAC,
        DEFAULT_SNAP_TOLERANCE,
        chemistry_cluster_id,
        parse_formula,
    )
    from pymatgen.core import Composition

    comp, err = parse_formula(formula)
    assert comp is not None, (formula, err)
    total = comp.num_atoms
    amounts = comp.get_el_amt_dict()
    kept = {el: a for el, a in amounts.items() if a / total >= DEFAULT_DOPANT_THRESHOLD_FRAC}
    removed = {el: a for el, a in amounts.items() if el not in kept}
    snap = {}
    for el, a in kept.items():
        nearest = round(a)
        rel = abs(a - nearest) / nearest if nearest >= 1 else None
        snapped = nearest >= 1 and rel <= DEFAULT_SNAP_TOLERANCE
        snap[el] = dict(before=a, after=float(nearest) if snapped else a, snapped=snapped, nearest=nearest, rel=rel)
    snapped_formula = "".join(f"{el}{v['after']:g}" for el, v in snap.items())
    reduced = Composition({el: v["after"] for el, v in snap.items()}).reduced_formula
    cluster_id = chemistry_cluster_id(comp)
    assert reduced == cluster_id, (formula, reduced, cluster_id)
    # readable form of the same id: the reduced composition written in the input's element order
    rc = Composition({el: v["after"] for el, v in snap.items()}).reduced_composition
    readable = "".join(f"{el}{'' if rc[el] == 1 else format(rc[el], 'g')}" for el in snap)
    assert Composition(readable).reduced_formula == cluster_id, (readable, cluster_id)
    return dict(
        formula=formula, at_pct={el: 100.0 * a / total for el, a in amounts.items()},
        kept=list(kept), removed=list(removed), snap=snap,
        snapped_formula=snapped_formula, cluster_id=cluster_id, readable=readable,
    )


GROUPING_RULE_FORMULAS = [
    "Pb0.97Te",
    "(PbTe)0.97(SrTe)0.02(Na2Te)0.01",
    "Bi2Te2.7Se0.3",
    "Bi2Te3",
]


def make_grouping_rule_schematic(out_path):
    """
    Figure 3: the chemistry-cluster grouping rule as a flow diagram over
    four formulas. Every intermediate and the final id come from
    `chemistry_cluster_trace`, which wraps the pipeline's
    chemistry_cluster_id. Fails (draws nothing) if rows 1 and 2 do not
    share PbTe's id, or if rows 3 and 4 share an id.
    """
    from matplotlib.patches import FancyBboxPatch
    from src.canonicalization import chemistry_cluster_id, parse_formula

    traces = [chemistry_cluster_trace(f) for f in GROUPING_RULE_FORMULAS]
    pbte_id = chemistry_cluster_id(parse_formula("PbTe")[0])
    ids = [t["cluster_id"] for t in traces]
    assert ids[0] == ids[1] == pbte_id, ("rows 1 and 2 must share PbTe's id", ids, pbte_id)
    assert ids[2] != ids[3], ("rows 3 and 4 must not share an id", ids)
    assert traces[0]["readable"] == traces[1]["readable"] == "PbTe", [t["readable"] for t in traces[:2]]
    assert traces[2]["readable"] == ids[2] and traces[3]["readable"] == ids[3], [t["readable"] for t in traces[2:]]
    t2 = traces[1]
    assert set(t2["removed"]) == {"Sr", "Na"} and all(round(t2["at_pct"][e], 1) == 1.0 for e in ("Sr", "Na")), t2
    assert "Se" in traces[2]["kept"] and round(traces[2]["at_pct"]["Se"], 1) == 6.0, traces[2]

    col_w = [4.0, 4.4, 3.7, 2.9, 3.1]
    col_x = [0.0]
    for w_ in col_w[:-1]:
        col_x.append(col_x[-1] + w_ + 0.35)
    headers = ["Input formula", "Remove elements < 5 at%\n(dopants)", "Snap amounts within 5%\nof an integer",
               "Reduce", "Cluster ID"]
    row_h, gap, top = 1.15, 0.3, 4.9
    fig, ax = plt.subplots(figsize=(17.2, 6.2))
    ax.set_xlim(-0.3, 22.3)
    ax.set_ylim(-0.75, 5.95)
    ax.axis("off")

    def box(i, y_top, h, lines, face="white", edge="0.35", lw=0.9, fontsize=9.5, weight="normal"):
        x0, w = col_x[i], col_w[i]
        ax.add_patch(FancyBboxPatch((x0, y_top - h), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=face, ec=edge, lw=lw))
        n = len(lines)
        for k, (txt, color) in enumerate(lines):
            yy = y_top - h / 2 + (n - 1) * 0.16 - k * 0.32
            ax.text(x0 + w / 2, yy, txt, ha="center", va="center", fontsize=fontsize, color=color, fontweight=weight)

    for i, htxt in enumerate(headers):
        ax.text(col_x[i] + col_w[i] / 2, 5.6, htxt, ha="center", va="center", fontsize=10.5, fontweight="bold", color="0.15")

    for r, (formula, t) in enumerate(zip(GROUPING_RULE_FORMULAS, traces)):
        y_top = top - r * (row_h + gap)
        mid = y_top - row_h / 2
        # 1: input
        box(0, y_top, row_h, [(_formula_mathtext(formula), "0.1")], fontsize=10)
        # 2: dopant removal, with the deciding at%
        kept_txt = ", ".join(f"{e} {t['at_pct'][e]:.1f}%" for e in t["kept"])
        lines = [(f"kept: {kept_txt}", SCHEMATIC_TRAIN_COLOR)]
        if t["removed"]:
            lines.append(("removed: " + ", ".join(f"{e} {t['at_pct'][e]:.1f}%" for e in t["removed"]), SCHEMATIC_TEST_COLOR))
        else:
            lines.append(("removed: none", "0.45"))
        box(1, y_top, row_h, lines)
        # 3: snap, per element
        snap_lines = []
        for e, v in t["snap"].items():
            if v["snapped"] and abs(v["before"] - v["after"]) > 1e-12:
                snap_lines.append((f"{e} {v['before']:g} → {v['after']:g}", "0.1"))
            elif v["snapped"]:
                snap_lines.append((f"{e} {v['before']:g} (integer)", "0.45"))
            elif v["nearest"] >= 1:
                snap_lines.append((f"{e} {v['before']:g}: {100 * v['rel']:.0f}% from {v['nearest']}, kept", "0.1"))
            else:
                snap_lines.append((f"{e} {v['before']:g}: kept", "0.1"))
        box(2, y_top, row_h, snap_lines[:3])
        # 4: reduce
        box(3, y_top, row_h, [(f"{_formula_mathtext(t['snapped_formula'])} →", "0.3"),
                              (_formula_mathtext(t["cluster_id"]), "0.1")])
        # 5: cluster id
        if t["readable"] == t["cluster_id"]:
            box(4, y_top, row_h, [(_formula_mathtext(t["cluster_id"]), "0.1")], face="0.94", edge="0.2", lw=1.4,
                fontsize=11.5, weight="bold")
        else:
            box(4, y_top, row_h, [], face="0.94", edge="0.2", lw=1.4)
            cx = col_x[4] + col_w[4] / 2
            ax.text(cx, mid + 0.16, _formula_mathtext(t["readable"]), ha="center", va="center",
                    fontsize=11.5, fontweight="bold", color="0.1")
            ax.text(cx, mid - 0.27, f"(ID: {t['cluster_id']})", ha="center", va="center", fontsize=8.5, color="0.5")
        for i in range(4):
            ax.annotate("", xy=(col_x[i + 1] - 0.03, mid), xytext=(col_x[i] + col_w[i] + 0.03, mid),
                        arrowprops=dict(arrowstyle="->", color="0.6", lw=0.9), annotation_clip=False)

    # bracket rows 1-2: same cluster
    y1 = top - 0.05
    y2 = top - (row_h + gap) - row_h + 0.05
    bx = col_x[4] + col_w[4] + 0.25
    ax.plot([bx - 0.12, bx, bx, bx - 0.12], [y1, y1, y2, y2], color="0.2", lw=1.3, clip_on=False)
    ax.text(bx + 0.18, (y1 + y2) / 2, "same\ncluster", ha="left", va="center", fontsize=10.5, fontweight="bold", color="0.15")

    save_figure(fig, out_path)
    plt.close(fig)
    return traces


def main():
    apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    make_cleaning_funnel(FIGURES_DIR / "cleaning_funnel")
    print("Saved cleaning_funnel.png / .pdf")

    make_property_distributions(FIGURES_DIR / "fig_property_distributions")
    print("Saved fig_property_distributions.png / .pdf")

    cluster_df = load_snapfix_cluster_columns()
    print(f"Loaded {len(cluster_df):,} rows from {SNAPFIX_FEATURIZED_CSV} for cluster sizes")
    make_cluster_size_distribution(cluster_df, FIGURES_DIR / "cluster_size_distribution")
    print("Saved cluster_size_distribution.png / .pdf")

    make_model_comparison(FIGURES_DIR / "fig1_model_comparison")
    print("Saved fig1_model_comparison.png / .pdf")

    make_validation_ladder(FIGURES_DIR / "fig2_validation_ladder")
    print("Saved fig2_validation_ladder.png / .pdf")

    make_headroom_decomposition(FIGURES_DIR / "fig5_headroom")
    print("Saved fig5_headroom.png / .pdf")

    make_shap_attribution(FIGURES_DIR / "shap_attribution")
    print("Saved shap_attribution.png / .pdf")

    make_descriptor_ablation(FIGURES_DIR / "descriptor_ablation")
    print("Saved descriptor_ablation.png / .pdf")

    make_zt_parity(FIGURES_DIR / "zt_parity_random_vs_grouped")
    print("Saved zt_parity_random_vs_grouped.png / .pdf")

    make_external_transfer(FIGURES_DIR / "external_transfer")
    print("Saved external_transfer.png / .pdf")

    make_sigma_extrapolation(FIGURES_DIR / "sigma_extrapolation")
    print("Saved sigma_extrapolation.png / .pdf")

    make_zt_direct_vs_derived(FIGURES_DIR / "zt_direct_vs_derived")
    print("Saved zt_direct_vs_derived.png / .pdf")

    make_leakage_schematic(FIGURES_DIR / "fig1_leakage_schematic")
    print("Saved fig1_leakage_schematic.png / .pdf")

    make_grouping_rule_schematic(FIGURES_DIR / "fig3_grouping_rule_schematic")
    print("Saved fig3_grouping_rule_schematic.png / .pdf")

    try:
        make_actual_vs_predicted(FIGURES_DIR / "fig3_actual_vs_predicted")
        print("Saved fig3_actual_vs_predicted.png / .pdf")
    except FileNotFoundError as e:
        print(f"Skipped fig3_actual_vs_predicted: {e}")


if __name__ == "__main__":
    main()
