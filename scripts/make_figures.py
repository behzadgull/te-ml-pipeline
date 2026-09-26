"""
Generate the dataset-construction and descriptive figures used in the
methods section of both papers: the 11-step cleaning funnel, the
post-cleaning property distributions, and the chemistry-cluster size
distribution that motivates repeated grouped CV. Reads the cleaned CSV
written by src/data_cleaning.py; run that first.
"""

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle as fs  # noqa: E402

from src.noise_floor import load_aligned_target_values
from src.plotting_style import (
    COLORBLIND_PALETTE,
    add_panel_label,
    apply_style,
    get_figsize,
    save_figure,
)

FIGURES_DIR = Path("figures")
PAPER_MD_PATH = Path("paper/paper.md")

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


# The eleven step labels are the pipeline's step names (src/data_cleaning.py). Section 2.2 of paper.md describes
# the steps in prose and never names them, so a word-for-word match against the paper is impossible; this check
# is the strongest available one: each label's anchor term(s) must occur in section 2.2 (prose plus the Figure 3
# caption), so no label can name a step the paper does not describe.
CLEANING_LABEL_ANCHORS = {
    1: ("range filtering",), 2: ("conductivity",), 3: ("300 to 800 K",), 4: ("Pivoting",), 5: ("Formulas",),
    6: ("S²σT/κ",), 7: ("computational",), 8: ("coefficient of variation",), 9: ("median-absolute-deviation",),
    10: ("distinct temperature",), 11: ("smoothness filter",),
}


def check_cleaning_labels_against_paper(labels, paper_path=PAPER_MD_PATH):
    """Assert the eleven labels exist in order and that each step's anchor term(s) occur in paper.md section 2.2."""
    text = Path(paper_path).read_text(encoding="utf-8")
    m = re.search(r"^## 2\.2 .*?(?=^## 2\.3 )", text, flags=re.S | re.M)
    assert m, "section 2.2 not found in paper.md"
    section = m.group(0)
    assert len(labels) == 11 and [int(l.split(".")[0]) for l in labels] == list(range(1, 12)), labels
    for k, anchors in CLEANING_LABEL_ANCHORS.items():
        for a in anchors:
            assert a in section, f"step {k} ({labels[k - 1]!r}): {a!r} does not occur in paper.md section 2.2"


def make_cleaning_funnel(out_path):
    """
    Figure 3: horizontal funnel of the row count after each of the 11 cleaning steps, annotated with the count
    and the step-to-step retention. Step 1 expands raw curves into points, so it carries no retention figure
    and its input (raw curves) is stated under the axis.
    """
    labels = [label.replace("\n", " ") for label, _ in CLEANING_STEPS]
    counts = [count for _, count in CLEANING_STEPS]
    check_cleaning_labels_against_paper(labels)

    fig, ax = fs.new_figure("double", 7.4)
    y_pos = np.arange(len(labels))[::-1]
    max_count = max(counts)
    ax.barh(y_pos, counts, color=fs.OI["blue"], height=0.62)
    for i, (y, count) in enumerate(zip(y_pos, counts)):
        label = f"{count:,}" if i == 0 else f"{count:,} ({counts[i] / counts[i - 1]:.0%} kept)"
        ax.text(count + max_count * 0.012, y, label, va="center", ha="left")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max_count * 1.55)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xticks([0, 0.5e6, 1.0e6, 1.5e6, 2.0e6])
    ax.set_xticklabels(["0", "0.5", "1.0", "1.5", "2.0"])
    ax.set_xlabel(
        "Rows after the step (millions of temperature-property points)\n"
        f"Step 1 input: {RAW_CURVES:,} raw curves (expanded into points)"
    )
    fs.grid(ax, "x")
    fs.panel_title(ax, "Rows remaining after each of the eleven cleaning steps")
    fs.save(fig, out_path, expect_width_cm=16.0)
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


LADDER_LEGEND_LABELS = ["Random 80/20", "5-fold", "10-fold", "Composition", "Chemistry cluster"]


def make_validation_ladder(
    out_path, ladder_metrics_path=LADDER_METRICS_PATH, ungrouped_dir=UNGROUPED_SNAPFIX_DIR,
):
    """
    Figure 5: five-way validation-inflation ladder, one group of five bars per property, pooled out-of-fold R^2
    with the frozen hyperparameters reused across every rung. Ungrouped rungs (random 80/20, 5-fold, 10-fold)
    are three greys; grouped rungs (composition, chemistry cluster) are two blues. Error bars are each rung's
    own SD (grouped: across the 5 repeats; ungrouped: across the 20 draws or the folds). A bracket beside each
    group gives random 80/20 minus chemistry cluster.
    """
    from matplotlib.patches import Patch

    n_props, n_strategies = len(LADDER_PROPERTIES), len(LADDER_STRATEGIES)
    all_sd = {**load_ladder_grouped_sd(ladder_metrics_path), **load_ladder_ungrouped_sd(ungrouped_dir)}

    # the plotted heights must equal the committed ladder metrics (grouped rungs, 4 dp)
    with open(ladder_metrics_path, encoding="utf-8") as f:
        runs = json.load(f)["runs"]
    for prop in LADDER_PROPERTIES:
        for strategy, run in (("Composition CV", "composition"), ("Chemistry-Cluster CV", "chemistry")):
            assert round(runs[f"{prop}_{run}_full"]["per_repeat_r2_mean"], 4) == LADDER_RESULTS[strategy][prop], (prop, strategy)

    colors = [*fs.UNGROUPED_GREYS, *fs.GROUPED_BLUES]
    fig, ax = fs.new_figure("double", 7.0)
    group_centers = np.arange(n_props)
    bar_width = 0.098
    offsets = (np.arange(n_strategies) - (n_strategies - 1) / 2) * bar_width

    for i, strategy in enumerate(LADDER_STRATEGIES):
        heights = [LADDER_RESULTS[strategy][prop] for prop in LADDER_PROPERTIES]
        x = group_centers + offsets[i]
        ax.bar(x, heights, width=bar_width * 0.94, color=colors[i], edgecolor=fs.EDGE_GREY, linewidth=0.4, zorder=2)
        errs = [all_sd[strategy][prop] for prop in LADDER_PROPERTIES]
        ax.errorbar(x, heights, yerr=errs, fmt="none", color="black", capsize=1.5, elinewidth=0.7, capthick=0.7, zorder=5)

    # inflation bracket beside each group, clear of the bars and of the next group
    x_off = 0.5 * n_strategies * bar_width + 0.035
    for j, prop in enumerate(LADDER_PROPERTIES):
        y_leaky = LADDER_RESULTS[LADDER_LEAKY_STRATEGY][prop]
        y_honest = LADDER_RESULTS[LADDER_HONEST_STRATEGY][prop]
        xb = group_centers[j] + x_off
        ax.plot([xb - 0.02, xb, xb, xb - 0.02], [y_leaky, y_leaky, y_honest, y_honest], color="black", lw=0.7, solid_joinstyle="miter", zorder=6)
        ax.text(xb + 0.025, (y_leaky + y_honest) / 2, f"$\\Delta$={y_leaky - y_honest:.3f}", ha="left", va="center")

    ax.set_xticks(group_centers)
    ax.set_xticklabels(LADDER_PROPERTY_LABELS)
    ax.set_xlim(-0.5, n_props - 1 + 0.78)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Pooled out-of-fold R$^2$")
    fs.panel_title(ax, "Pooled out-of-fold R$^2$ under five validation protocols")
    handles = [Patch(facecolor=c, edgecolor=fs.EDGE_GREY, linewidth=0.4) for c in colors]
    fs.legend_row(fig, handles, LADDER_LEGEND_LABELS)
    fs.save(fig, out_path, expect_width_cm=16.0)
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
HEADROOM_SEGMENT_COLORS = {
    "achieved": fs.OI["blue"],
    "headroom": "#F3D19C",
    "digitization": "#E3B7CE",
    "measurement": "#A9D8C8",
}
# headroom-only emphasis: denser hatch and a bolder edge than every other
# segment (including the shared HATCH["kfold"] value, left untouched for
# the ladder figure) -- see the drawing loop's comment for why.


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
    Figure 8: one horizontal stacked bar per property spanning R^2 = 0 to 1, split left to right into achieved
    (chemistry-cluster grouped R^2), headroom to the combined label-noise ceiling, digitization noise and
    measurement noise. An error bar on the achieved/headroom boundary is the across-repeat SD of the grouped
    R^2; a dashed line marks the ungrouped random 80/20 R^2. Muted tones, no hatching.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    data = load_headroom_data(ladder_metrics_path, noise_floor_inputs_path)

    fig, ax = fs.new_figure("double", 6.2)
    y_pos = np.arange(len(HEADROOM_PROPERTIES))[::-1]
    bar_height = 0.5
    for y, prop in zip(y_pos, HEADROOM_PROPERTIES):
        d = data[prop]
        achieved_end = d["achieved_width"]
        headroom_end = achieved_end + d["headroom_width"]
        digitization_end = headroom_end + d["digitization_width"]
        for name, left, right in (
            ("achieved", 0.0, achieved_end), ("headroom", achieved_end, headroom_end),
            ("digitization", headroom_end, digitization_end), ("measurement", digitization_end, 1.0),
        ):
            ax.barh(y, right - left, left=left, height=bar_height, color=HEADROOM_SEGMENT_COLORS[name],
                    edgecolor=fs.EDGE_GREY, linewidth=0.4, zorder=2)
        ax.text((achieved_end + headroom_end) / 2, y + bar_height / 2 + 0.07, f"$\\Delta$ = {d['headroom_width']:.3f}",
                ha="center", va="bottom", zorder=4)
        ax.errorbar(achieved_end, y, xerr=d["grouped_sd"], color="black", capsize=1.5, elinewidth=0.7, capthick=0.7, zorder=5)
        ax.plot([d["ungrouped_r2"]] * 2, [y - bar_height / 2 - 0.03, y + bar_height / 2 + 0.03],
                color="black", linestyle=(0, (3, 2)), linewidth=1.0, zorder=6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(HEADROOM_PROPERTY_LABELS)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(y_pos.min() - 0.55, y_pos.max() + 0.75)
    ax.set_xlabel("R$^2$ (per-property matched space)")
    fs.grid(ax, "x")
    fs.panel_title(ax, "Where each property's R$^2$ range goes: achieved, headroom and label noise")
    handles = [Patch(facecolor=HEADROOM_SEGMENT_COLORS[n], edgecolor=fs.EDGE_GREY, linewidth=0.4) for n in HEADROOM_SEGMENT_ORDER]
    handles.append(Line2D([], [], color="black", linestyle=(0, (3, 2)), linewidth=1.0))
    labels = [HEADROOM_SEGMENT_DISPLAY_NAMES[n] for n in HEADROOM_SEGMENT_ORDER] + ["Ungrouped random 80/20 R$^2$"]
    fs.legend_row(fig, handles, labels, ncol=3)
    fs.save(fig, out_path, expect_width_cm=16.0)
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


SHAP_RANDOM_COLOR = "#9A9A9A"
SHAP_GROUPING_CODE = Path("scripts/shap_attribution_zT.py")
# group key -> readable label. The definition of each group is the member list in SHAP_GROUPING_CODE's
# FINE_GROUP_MEMBERS (read below with ast, not imported, since that script imports xgboost); the label says
# what the members are. "temperature" is the single temperature_bin column.
SHAP_FINE_LABELS = {
    "temperature": "Temperature",
    "valence_electron_config": "Valence-electron configuration",
    "atomic_radius": "Atomic and ionic radii",
    "thermodynamic_bulk": "Bulk thermodynamic properties",
    "periodic_position": "Periodic-table position",
    "electronegativity": "Electronegativity",
    "dft_groundstate": "DFT ground state",
    "metal_class": "Metal class",
    "melting_point": "Melting point",
    "atomic_mass": "Atomic mass",
}


def load_shap_fine_group_members(path=SHAP_GROUPING_CODE):
    """FINE_GROUP_MEMBERS ({group key: [raw MAGPIE/CBFV property names]}) read from the grouping code itself."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FINE_GROUP_MEMBERS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"FINE_GROUP_MEMBERS not found in {path}")


def load_shap_group_data(shap_dir=SHAP_ATTRIBUTION_DIR):
    """
    Load coarse and fine SHAP attribution-share group comparisons from summary.json. Returns
    (coarse_rows, fine_rows, max_delta_row), each row a dict with group/random_mean_share/random_sd/
    chemistry_mean_share/chemistry_sd/delta_in_pooled_sd_units. fine_rows is sorted by descending mean of the
    two arms' shares. max_delta_row is the fine-group row with the largest |delta_in_pooled_sd_units|.
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
    """Paired bars (random 80/20 grey, chemistry cluster blue) with across-fold SD error bars."""
    positions = np.arange(len(rows))
    w = 0.38
    series = (
        ("random", SHAP_RANDOM_COLOR, -w / 2),
        ("chemistry", fs.OI["blue"], +w / 2),
    )
    err_kw = dict(capsize=1.5, elinewidth=0.7, capthick=0.7, ecolor="black")
    for key, color, shift in series:
        vals = [row[f"{key}_mean_share"] for row in rows]
        errs = [row[f"{key}_sd"] for row in rows]
        if horizontal:
            ax.barh(positions + shift, vals, height=w * 0.94, xerr=errs, color=color, edgecolor=fs.EDGE_GREY,
                    linewidth=0.4, error_kw=err_kw, zorder=2)
        else:
            ax.bar(positions + shift, vals, width=w * 0.94, yerr=errs, color=color, edgecolor=fs.EDGE_GREY,
                   linewidth=0.4, error_kw=err_kw, zorder=2)
    if horizontal:
        ax.set_yticks(positions)
        ax.set_yticklabels(group_labels)
        ax.set_xlabel("Mean |SHAP| share")
        fs.grid(ax, "x")
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels(group_labels)
        ax.set_ylabel("Mean |SHAP| share")


def make_shap_attribution(out_path, shap_dir=SHAP_ATTRIBUTION_DIR):
    """
    Figure 7: SHAP attribution shares for zT, random 80/20 vs chemistry-cluster split. Two panels whose axes have
    equal width: (a) three coarse descriptor families, (b) ten fine semantic groups sorted by mean share. Error
    bars are across-fold SD (25 folds per arm). The largest fine-group delta is stated in the caption (and
    asserted here). The right panel's tick labels take more room than the left's, so the gridspec width ratio is
    iterated until the two axes are within 0.5% of each other in width.
    """
    import textwrap

    from matplotlib.patches import Patch

    coarse_rows, fine_rows, max_delta_row = load_shap_group_data(shap_dir)
    assert max_delta_row["group"] == "valence_electron_config" and round(abs(max_delta_row["delta_in_pooled_sd_units"]), 2) == 0.78, max_delta_row
    members = load_shap_fine_group_members()
    assert set(SHAP_FINE_LABELS) == set(members) | {"temperature"} == {r["group"] for r in fine_rows}, (sorted(SHAP_FINE_LABELS), sorted(members))
    fine_labels = ["\n".join(textwrap.wrap(SHAP_FINE_LABELS[r["group"]], 24, break_long_words=False)) for r in fine_rows]

    def build(ratio):
        fig, (ax_left, ax_right) = fs.new_figure("double", 8.2, 1, 2, width_ratios=[1.0, ratio])
        _plot_shap_paired_bars(ax_left, coarse_rows, [SHAP_COARSE_LABELS[r["group"]] for r in coarse_rows], horizontal=False)
        fs.panel_title(ax_left, "Three descriptor families", letter="a")
        _plot_shap_paired_bars(ax_right, fine_rows, fine_labels, horizontal=True)
        ax_right.invert_yaxis()
        fs.panel_title(ax_right, "Ten semantic groups", letter="b")
        fig.canvas.draw()
        return fig, ax_left.get_position().width, ax_right.get_position().width

    ratio = 1.0
    for _ in range(8):
        fig, w_left, w_right = build(ratio)
        if abs(w_left / w_right - 1) < 0.005:
            break
        plt.close(fig)
        ratio *= w_left / w_right          # right axes too wide -> shrink its gridspec cell
    else:
        raise AssertionError(f"SHAP panels did not reach equal axes width: {w_left:.4f} vs {w_right:.4f}")

    handles = [Patch(facecolor=SHAP_RANDOM_COLOR, edgecolor=fs.EDGE_GREY, linewidth=0.4),
               Patch(facecolor=fs.OI["blue"], edgecolor=fs.EDGE_GREY, linewidth=0.4)]
    fs.legend_row(fig, handles, ["Random 80/20", "Chemistry cluster"])
    fs.save(fig, out_path, expect_width_cm=16.0)
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


def load_descriptor_ablation_gain_data(
    ablation_metrics_path=DESCRIPTOR_ABLATION_METRICS_PATH,
    noise_floor_inputs_path=NOISE_FLOOR_INPUTS_PATH,
):
    """
    Gain in chemistry-cluster grouped R^2 over MAGPIE alone, as a percentage of the headroom to the combined
    label-noise ceiling, for each feature set and property. For repeat i the gain is R^2(feature set, i) minus
    R^2(MAGPIE, i) (the runs share seeds, so the repeats are paired); the percentage divides by the lower
    headroom bound (the larger fraction, i.e. the 'closes at most' reading). Returns {property: {feature_counts,
    gain_pct_mean, gain_pct_sd, headroom_lower, headroom_upper}}; the MAGPIE point is 0 by construction.

    Assertions (the numbers the paper quotes): the full-set mean gain equals ablation_metrics.json's
    deltas_full_minus_magpie to 1e-9, and the fraction ranges from load_descriptor_ablation_data span exactly
    2.2 to 4.2 percent over the four properties, as the abstract and Section 3.4 state.
    """
    with open(ablation_metrics_path, encoding="utf-8") as f:
        ablation = json.load(f)
    with open(noise_floor_inputs_path, encoding="utf-8") as f:
        noise_floor = json.load(f)
    combined_by_label = {e["label"]: e for e in noise_floor["item3_combined_ceiling_new"]}

    data = {}
    for prop in DESCRIPTOR_ABLATION_PROPERTIES:
        combined = combined_by_label[ZT_COMBINED_CEILING_LABEL if prop == "zT" else prop]
        head_lo, head_hi = combined["headroom_lower"], combined["headroom_upper"]
        per_repeat = {fs_: np.asarray(ablation[prop][fs_]["per_repeat_r2"], float) for fs_ in DESCRIPTOR_ABLATION_FEATURE_SETS}
        assert all(len(v) == 5 for v in per_repeat.values()), prop
        means, sds = [], []
        for fs_ in DESCRIPTOR_ABLATION_FEATURE_SETS:
            gain = per_repeat[fs_] - per_repeat["magpie"]
            means.append(100 * gain.mean() / head_lo)
            sds.append(100 * gain.std(ddof=1) / head_lo)
        assert abs(means[-1] / 100 * head_lo - ablation["deltas_full_minus_magpie"][prop]) < 1e-9, prop
        data[prop] = {
            "feature_counts": [DESCRIPTOR_ABLATION_FEATURE_COUNTS[s] for s in DESCRIPTOR_ABLATION_FEATURE_SETS],
            "gain_pct_mean": means, "gain_pct_sd": sds, "headroom_lower": head_lo, "headroom_upper": head_hi,
        }
    # the paper's range: full-minus-MAGPIE closes 2.2 to 4.2 percent of headroom, over both headroom bounds
    lows = [100 * ablation["deltas_full_minus_magpie"][p] / data[p]["headroom_upper"] for p in DESCRIPTOR_ABLATION_PROPERTIES]
    highs = [100 * ablation["deltas_full_minus_magpie"][p] / data[p]["headroom_lower"] for p in DESCRIPTOR_ABLATION_PROPERTIES]
    assert round(min(lows), 1) == 2.2 and round(max(highs), 1) == 4.2, (min(lows), max(highs))
    return data


ABLATION_LINE_STYLE = {
    "S": (fs.OI["blue"], "o"),
    "sigma": (fs.OI["vermillion"], "s"),
    "kappa": (fs.OI["green"], "^"),
    "zT": (fs.OI["purple"], "D"),
}


def make_descriptor_ablation(
    out_path,
    ablation_metrics_path=DESCRIPTOR_ABLATION_METRICS_PATH,
    noise_floor_inputs_path=NOISE_FLOOR_INPUTS_PATH,
):
    """
    Figure 9 (single column): per property (categorical x), the gain in chemistry-cluster grouped R^2 over MAGPIE
    alone as a percentage of the headroom to the combined label-noise ceiling. Filled marker: the full set (MAGPIE
    plus CBFV, 397 features) minus MAGPIE. Open marker: CBFV alone (265 features) minus MAGPIE. Error bars are the
    SD across the 5 repeats of the paired per-repeat gain; no connecting lines. The absolute-R^2 version is
    make_descriptor_ablation_absolute (supplementary). The 2.2-4.2 percent range the paper quotes is asserted in
    load_descriptor_ablation_gain_data.
    """
    data = load_descriptor_ablation_gain_data(ablation_metrics_path, noise_floor_inputs_path)
    props = DESCRIPTOR_ABLATION_PROPERTIES
    x = np.arange(len(props))
    dx = 0.17
    blue = fs.OI["blue"]
    full_m = [data[p]["gain_pct_mean"][2] for p in props]
    full_s = [data[p]["gain_pct_sd"][2] for p in props]
    cbfv_m = [data[p]["gain_pct_mean"][1] for p in props]
    cbfv_s = [data[p]["gain_pct_sd"][1] for p in props]

    fig, ax = fs.new_figure("single", 6.0)
    ax.axhline(0, color="black", linewidth=0.5, zorder=1)
    ax.errorbar(x + dx, full_m, yerr=full_s, fmt="o", color=blue, markersize=4.5, capsize=1.5, elinewidth=0.7,
                capthick=0.7, label="Full set (MAGPIE + CBFV)", zorder=3)
    ax.errorbar(x - dx, cbfv_m, yerr=cbfv_s, fmt="o", color=blue, markerfacecolor="white", markeredgewidth=0.9,
                markersize=4.5, capsize=1.5, elinewidth=0.7, capthick=0.7, label="CBFV instead of MAGPIE", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([DESCRIPTOR_ABLATION_PROPERTY_LABELS[p] for p in props])
    ax.set_xlim(-0.6, len(props) - 0.4)
    lows = [m - s for m, s in zip(full_m + cbfv_m, full_s + cbfv_s)]
    highs = [m + s for m, s in zip(full_m + cbfv_m, full_s + cbfv_s)]
    ax.set_ylim(min(lows) - 0.4, max(highs) + 1.9)          # headroom above the data for the legend
    ax.set_ylabel("Share of headroom closed (%)")
    ax.legend(loc="upper left", ncol=1)
    fs.panel_title(ax, "Gain over MAGPIE, as a share of headroom")
    fs.save(fig, out_path, expect_width_cm=8.0)
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


def make_descriptor_ablation_absolute(
    out_path,
    ablation_metrics_path=DESCRIPTOR_ABLATION_METRICS_PATH,
    noise_floor_inputs_path=NOISE_FLOOR_INPUTS_PATH,
):
    """
    Supplementary: the absolute-R^2 version of the descriptor ablation, one panel per property, with the
    combined label-noise ceiling band and across-repeat SD error bars. Not in the manuscript body.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    data = load_descriptor_ablation_data(ablation_metrics_path, noise_floor_inputs_path)
    fig, axes = fs.new_figure("double", 10.0, 2, 2)
    for ax, prop, letter in zip(axes.flat, DESCRIPTOR_ABLATION_PROPERTIES, "abcd"):
        d = data[prop]
        x = d["feature_counts"]
        ax.axhspan(d["band_lower"], d["band_upper"], color=fs.OI["orange"], alpha=0.28, linewidth=0, zorder=1)
        ax.errorbar(x, d["r2_values"], yerr=d["r2_sd"], color=fs.OI["blue"], marker="o", markersize=3.5, linewidth=1.0,
                    capsize=1.5, elinewidth=0.7, capthick=0.7, zorder=3)
        ax.set_ylim(min(d["r2_values"]) - max(d["r2_sd"]) - 0.02, 1.0)
        ax.set_xlim(min(x) - 40, max(x) + 40)
        ax.set_xticks(x)
        ax.set_xlabel("Number of features")
        ax.set_ylabel("R$^2$")
        fs.panel_title(ax, DESCRIPTOR_ABLATION_PROPERTY_LABELS[prop], letter=letter)
    handles = [Line2D([], [], color=fs.OI["blue"], marker="o", markersize=3.5, linewidth=1.0),
               Patch(facecolor=fs.OI["orange"], alpha=0.28, linewidth=0)]
    fs.legend_row(fig, handles, ["Chemistry-cluster grouped R$^2$", "Combined label-noise ceiling range"])
    fs.save(fig, out_path, expect_width_cm=16.0)
    plt.close(fig)


def _density_cmap():
    """Sequential blues for hexbin density: light enough to read on white, dark enough at the top; no yellow."""
    from matplotlib.colors import ListedColormap

    return ListedColormap(plt.cm.Blues(np.linspace(0.25, 1.0, 256)))


def _hexbin_parity_pair(fig, axes, panels, lo, hi, cbar_label="Rows per hexbin (log scale)"):
    """
    Draw two hexbin parity panels on a shared axis range and one shared log colour scale. `panels` is a list of
    (ax, actual, predicted, title, letter, r2). The sample sizes are given in the caption, not in the image.
    """
    cmap = _density_cmap()
    hbs = []
    for ax, actual, predicted, title, letter, r2 in panels:
        hb = ax.hexbin(actual, predicted, gridsize=50, extent=(lo, hi, lo, hi), cmap=cmap, norm=LogNorm(), mincnt=1, linewidths=0)
        hbs.append(hb)
    vmax = max(hb.get_array().max() for hb in hbs)
    norm = LogNorm(vmin=1, vmax=vmax)
    for hb in hbs:
        hb.set_norm(norm)
    for (ax, actual, predicted, title, letter, r2) in panels:
        ax.plot([lo, hi], [lo, hi], color=fs.EDGE_GREY, linestyle=(0, (3, 2)), linewidth=0.7, zorder=5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Actual zT")
        ax.set_ylabel("Predicted zT")
        ax.grid(False)
        ax.text(0.05, 0.95, f"$R^2$ = {r2:.4f}", transform=ax.transAxes, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", linewidth=0.4))
        fs.panel_title(ax, title, letter=letter)
    cbar = fig.colorbar(hbs[-1], ax=list(axes), label=cbar_label, shrink=0.8, aspect=22, pad=0.02)
    cbar.outline.set_linewidth(0.4)
    return hbs


def make_zt_parity(
    out_path,
    random_checkpoint_dir=ZT_PARITY_RANDOM_CHECKPOINT_DIR,
    chemistry_checkpoint_dir=ZT_PARITY_CHEMISTRY_CHECKPOINT_DIR,
):
    """
    Figure 6: two hexbin parity panels for zT, random 80/20 (a) and chemistry-cluster (b), identical axes and one
    shared log colour scale so the difference in spread is the dominant visual. R^2 is recomputed here from the
    raw predictions; n is given in the caption.
    """
    yt_r, yp_r, n_files_r = load_zt_parity_predictions(random_checkpoint_dir)
    yt_c, yp_c, n_files_c = load_zt_parity_predictions(chemistry_checkpoint_dir)
    r2_r, r2_c = r2_score(yt_r, yp_r), r2_score(yt_c, yp_c)
    # the values the paper's caption quotes (Section 3.1)
    assert round(r2_r, 4) == 0.9180 and round(r2_c, 4) == 0.7456 and len(yt_r) == 517_680 and len(yt_c) == 647_095, (r2_r, r2_c, len(yt_r), len(yt_c))

    lo = min(yt_r.min(), yp_r.min(), yt_c.min(), yp_c.min())
    hi = max(yt_r.max(), yp_r.max(), yt_c.max(), yp_c.max())
    pad = (hi - lo) * 0.03
    lo, hi = lo - pad, hi + pad

    fig, axes = fs.new_figure("double", 6.9, 1, 2)
    _hexbin_parity_pair(fig, axes, [
        (axes[0], yt_r, yp_r, "Random 80/20", "a", r2_r),
        (axes[1], yt_c, yp_c, "Chemistry-cluster CV", "b", r2_c),
    ], lo, hi)
    fs.save(fig, out_path, expect_width_cm=16.0)
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


EXTERNAL_PANELS = (
    ("a", "ESTM, DOI-disjoint", 3123),
    ("b", "ESTM, cluster-disjoint", 1448),
)


EXTERNAL_TICK_SYMBOL = {"S": "S", "sigma": "$\\sigma$", "kappa": "$\\kappa$", "zT_direct": "zT"}


def make_external_transfer(out_path, ladder_metrics_path=LADDER_METRICS_PATH, external_dir=EXTERNAL_SNAPFIX_DIR):
    """
    Figure 10: three panels (ESTM DOI-disjoint, ESTM cluster-disjoint, teMatDb DOI-disjoint) of grouped bars per
    property (S, sigma, kappa, zT direct): internal chemistry-cluster R^2, external full-set R^2 and (ESTM only)
    external in-support R^2. Under each property, the share of external rows outside the training range (the
    property's own marginal share for S, sigma and kappa; the joint share for zT). teMatDb has no in-support bar
    (0.12% out-of-range tail, given in the caption). The panel titles are checked against the row counts of the
    passes they are drawn from.
    """
    from matplotlib.patches import Patch

    internal = load_internal_chemistry_r2(ladder_metrics_path)
    estm = {letter: load_estm_pass_transfer(letter, external_dir) for letter, _, _ in EXTERNAL_PANELS}
    for letter, title, n_expected in EXTERNAL_PANELS:      # a = DOI-disjoint (3,123 rows), b = cluster-disjoint (1,448 rows)
        assert estm[letter]["S"]["full_n"] == n_expected, (letter, title, estm[letter]["S"]["full_n"], n_expected)
    tematdb = load_tematdb_doi_disjoint()

    fig, axes = fs.new_figure("double", 6.3, 1, 3, sharey=True)
    props = EXTERNAL_TRANSFER_PROPERTIES
    centers = np.arange(len(props))
    c_int, c_full, c_in = fs.OI["blue"], fs.OI["orange"], fs.OI["green"]

    def draw_panel(ax, bars, ood_pct, title, letter):
        n_bars = len(bars)
        bar_width = 0.8 / n_bars
        offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_width
        for i, (color, values) in enumerate(bars):
            ax.bar(centers + offsets[i], [values[p] for p in props], width=bar_width * 0.92, color=color,
                   edgecolor=fs.EDGE_GREY, linewidth=0.4, zorder=2)
        ax.axhline(0, color="black", linewidth=0.5, zorder=3)
        ticks = [EXTERNAL_TICK_SYMBOL[p] + ("\n" + f"{ood_pct[p]:.1f}%" if ood_pct else "\n ") for p in props]
        ax.set_xticks(centers)
        ax.set_xticklabels(ticks)
        ax.set_ylim(-0.1, 1.0)
        fs.panel_title(ax, title, letter=letter)

    for k, (letter, title, _n) in enumerate(EXTERNAL_PANELS):
        e = estm[letter]
        ood = {**e["marginal_ood_pct"], "zT_direct": e["joint_ood_pct"]}
        draw_panel(axes[k], [
            (c_int, internal),
            (c_full, {p: e[p]["full_r2"] for p in props}),
            (c_in, {p: e[p]["insupport_r2"] for p in props}),
        ], ood, title, "abc"[k])
    draw_panel(axes[2], [(c_int, internal), (c_full, tematdb)], None, "teMatDb, DOI-disjoint", "c")
    axes[0].set_ylabel("R$^2$")

    handles = [Patch(facecolor=c, edgecolor=fs.EDGE_GREY, linewidth=0.4) for c in (c_int, c_full, c_in)]
    fs.legend_row(fig, handles, ["Internal (chemistry-cluster CV)", "External, full set", "External, in-support rows"])
    fs.save(fig, out_path, expect_width_cm=16.0)
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
    Figure 11 (single column): overlaid density histograms of log10(sigma) for the training set (snapfix CSV) and
    ESTM's cluster-disjoint rows. A dashed vertical line marks training's cleaned conductivity floor; ESTM mass
    below it (extrapolation) is shaded and its share is given in the caption.
    """
    d = load_sigma_extrapolation_data(training_csv, external_dir)
    assert round(100 * d["frac_below_floor"], 1) == 15.5, d["frac_below_floor"]      # Section 3.5 / caption

    fig, ax = fs.new_figure("single", 5.6)
    ax.hist(d["train_log_sigma"], bins=d["bin_edges"], density=True, color=fs.OI["blue"], alpha=0.6, linewidth=0,
            label="Training", zorder=2)
    ax.hist(d["estm_log_sigma"], bins=d["bin_edges"], density=True, color=fs.OI["orange"], alpha=0.7, linewidth=0,
            label="ESTM, cluster-disjoint", zorder=3)
    ax.axvspan(d["bin_edges"][0], d["log_floor"], color="black", alpha=0.08, linewidth=0, zorder=1)
    ax.axvline(d["log_floor"], color="black", linestyle=(0, (3, 2)), linewidth=0.8, zorder=4, label="Training floor")
    ax.set_xlabel("log$_{10}$($\\sigma$ / S m$^{-1}$)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")
    fs.panel_title(ax, "Conductivity: training vs ESTM")
    fs.save(fig, out_path, expect_width_cm=8.0)
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
    Figure 12: two hexbin parity panels for zT, direct prediction (a) and derived S^2 sigma T / kappa (b), identical
    axes and one shared log colour scale. R^2 is recomputed from the per-fold predictions and checked against the
    folder's results.json (unchanged assertion); n is given in the caption.
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

    fig, axes = fs.new_figure("double", 6.9, 1, 2)
    _hexbin_parity_pair(fig, axes, [
        (axes[0], d["direct_true"], d["direct_pred"], "Direct prediction", "a", r2_direct),
        (axes[1], d["derived_true"], d["derived_pred"], "Derived: S$^2\\sigma T/\\kappa$", "b", r2_derived),
    ], lo, hi)
    fs.save(fig, out_path, expect_width_cm=16.0)
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
    Figure 1: three panels illustrating why random row splits leak. (a) one zT(T) curve with scattered test
    rows; (b) near-duplicate curves of one host lattice on both sides of a split; (c) the chemistry-cluster
    split, where the whole cluster is test. Illustrative values, no numeric tick labels (the caption says so).
    """
    from matplotlib.lines import Line2D

    T = np.linspace(300, 800, 21)          # 300-800 K at 25 K
    x = (T - 300) / 500.0                  # normalised; tick labels are suppressed
    blue, verm = fs.OI["blue"], fs.OI["vermillion"]
    train_kw = dict(marker="o", ms=3.2, mfc=blue, mec="white", mew=0.3, ls="none", zorder=3)
    test_kw = dict(marker="D", ms=3.0, mfc=verm, mec="white", mew=0.3, ls="none", zorder=4)
    x_right = {"a": 1.12, "b": 2.55, "c": 2.55}

    fig, axes = fs.new_figure("double", 6.2, 1, 3, width_ratios=[1.45, 2.4, 2.4])

    def base(ax, title, letter):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("T")
        ax.set_ylabel("zT")
        ax.set_xlim(-0.05, x_right[letter])
        ax.set_ylim(-0.02, 2.08)
        ax.grid(False)
        fs.panel_title(ax, title, letter=letter)

    def draw_curve(ax, y, is_test, label=None, label_y=None):
        color = verm if is_test else blue
        ax.plot(x, y, color=color, lw=0.8, alpha=0.5, zorder=2)
        ax.plot(x, y, **(test_kw if is_test else train_kw))
        if label is None:
            return None
        return ax.annotate(
            label, xy=(x[-1] + 0.02, y[-1]), xytext=(1.14, label_y), textcoords="data", va="center", ha="left", color="0.15",
            arrowprops=dict(arrowstyle="-", color="0.65", lw=0.5, shrinkA=1.5, shrinkB=1.5),
        )

    note_kw = dict(va="top", ha="left", color="0.15", linespacing=1.25)

    # (a) random row split: 4 of 21 points (19%) are test, scattered
    ax = axes[0]
    base(ax, "Random row split", "a")
    y = 0.25 + 0.95 * x**1.3
    test_idx = [3, 8, 13, 18]
    tr = [i for i in range(len(x)) if i not in test_idx]
    ax.plot(x, y, color=fs.OI["black"], lw=0.8, alpha=0.35, zorder=2)
    ax.plot(x[tr], y[tr], **train_kw)
    ax.plot(x[test_idx], y[test_idx], **test_kw)
    ax.text(0.0, 2.04, "test rows lie between\ntraining rows of the\nsame curve", **note_kw)
    ax.annotate("", xy=(x[8], y[8] + 0.05), xytext=(0.30, 1.42),
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.6, shrinkA=0, shrinkB=1.5))

    # (b) near-duplicate curves across dopant labels: two train, two test, interleaved
    near = [
        ("PbTe", 0.36, 0.94, 1.30, False, 1.44, 1.72),
        ("Pb$_{0.99}$Na$_{0.01}$Te", 0.32, 0.90, 1.28, True, 1.22, 1.50),
        ("Pb$_{0.98}$Na$_{0.02}$Te", 0.28, 0.86, 1.32, False, 1.00, 1.28),
        ("(PbTe)$_{0.98}$(SrTe)$_{0.02}$", 0.24, 0.82, 1.30, True, 0.78, 1.06),
    ]
    ax = axes[1]
    base(ax, "Near-duplicate dopant labels", "b")
    for name, level, rise, curv, is_test, slot_b, _slot_c in near:
        draw_curve(ax, level + rise * x**curv, is_test, label=name, label_y=slot_b)
    ax.text(0.0, 2.04, "same host lattice on both\nsides of the split", **note_kw)

    # (c) chemistry-cluster split: the same four curves all test, two differently shaped train curves
    ax = axes[2]
    base(ax, "Chemistry-cluster split", "c")
    cluster_labels = []
    for name, level, rise, curv, _is_test, _slot_b, slot_c in near:
        cluster_labels.append(draw_curve(ax, level + rise * x**curv, True, label=name, label_y=slot_c))
    bi = 0.10 + 0.34 * np.sin(np.pi * x**0.8)          # rises, peaks, rolls over
    se = 0.08 + 0.85 * x**3.2                           # low, then steep rise
    draw_curve(ax, se, False, label="SnSe", label_y=0.72)
    draw_curve(ax, bi, False, label="Bi$_{2}$Te$_{3}$", label_y=0.28)
    ax.text(0.0, 2.04, "whole cluster held out", **note_kw)
    fig.canvas.draw()                                   # bracket around the four cluster labels, from their extents
    inv = ax.transData.inverted()
    bx = max(inv.transform((t.get_window_extent().x1, 0))[0] for t in cluster_labels) + 0.04
    ax.plot([bx - 0.03, bx, bx, bx - 0.03], [1.72 + 0.1, 1.72 + 0.1, 1.06 - 0.1, 1.06 - 0.1], color="0.3", lw=0.7, clip_on=False)

    handles = [Line2D([], [], marker="o", ms=4, mfc=blue, mec="white", mew=0.3, ls="none"),
               Line2D([], [], marker="D", ms=4, mfc=verm, mec="white", mew=0.3, ls="none")]
    fs.legend_row(fig, handles, ["Training row", "Test row"])
    fs.save(fig, out_path, expect_width_cm=16.0)
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
    Figure 4: the chemistry-cluster grouping rule as a flow diagram over four formulas, drawn in cm-sized data
    units at 16 cm width with 8 pt text; a fit check fails the run if any text overflows its box. Every
    intermediate and the final id come from `chemistry_cluster_trace`, which wraps the pipeline's
    chemistry_cluster_id. Fails (draws nothing) if rows 1 and 2 do not share PbTe's id, or if rows 3 and 4 share
    an id.
    """
    import textwrap

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

    W = 16.0
    col_w = [3.3, 3.45, 3.15, 2.35, 2.25]
    gap = 0.2
    col_x = [0.0]
    for w_ in col_w[:-1]:
        col_x.append(col_x[-1] + w_ + gap)
    row_h, row_gap, head_h = 1.75, 0.22, 1.05
    H = head_h + 4 * row_h + 3 * row_gap + 0.1
    fig = plt.figure(figsize=(W * fs.CM, H * fs.CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    checks = []
    LS = 1.2

    def cell(i, y_top, lines, face="white", edge="0.35", lw=0.7, size=8, weight="normal", color="0.1"):
        x0, w = col_x[i], col_w[i]
        ax.add_patch(FancyBboxPatch((x0, y_top - row_h), w, row_h, boxstyle="round,pad=0.0,rounding_size=0.1", fc=face, ec=edge, lw=lw))
        text = ax.text(x0 + w / 2, y_top - row_h / 2, "\n".join(lines), ha="center", va="center", fontsize=size,
                       fontweight=weight, color=color, linespacing=LS)
        checks.append((text, (x0, y_top - row_h, x0 + w, y_top)))
        return text

    def wrap(s, k):
        return textwrap.wrap(s, k, break_long_words=False)

    headers = ["Input formula", "Remove elements\n< 5 at% (dopants)", "Snap amounts within\n5% of an integer", "Reduce", "Cluster ID"]
    for i, htxt in enumerate(headers):
        ax.text(col_x[i] + col_w[i] / 2, H - head_h / 2 + 0.02, htxt, ha="center", va="center", fontsize=8, fontweight="bold", color="0.1", linespacing=LS)

    first_input = ["Pb$_{0.97}$Te", "(PbTe)$_{0.97}$(SrTe)$_{0.02}$\n(Na$_{2}$Te)$_{0.01}$", "Bi$_{2}$Te$_{2.7}$Se$_{0.3}$", "Bi$_{2}$Te$_{3}$"]
    for r, (formula, t) in enumerate(zip(GROUPING_RULE_FORMULAS, traces)):
        y_top = H - head_h - r * (row_h + row_gap)
        mid = y_top - row_h / 2
        cell(0, y_top, first_input[r].split("\n"))
        # kept in blue, removed in vermillion: two text blocks stacked in one cell
        x0, w = col_x[1], col_w[1]
        ax.add_patch(FancyBboxPatch((x0, y_top - row_h), w, row_h, boxstyle="round,pad=0.0,rounding_size=0.1", fc="white", ec="0.35", lw=0.7))
        def two_lines(prefix, items):        # first item after the prefix, the rest on a second line
            head = f"{prefix} {items[0]}" + ("," if len(items) > 1 else "")
            return [head] + ([", ".join(items[1:])] if len(items) > 1 else [])

        kl = two_lines("kept:", [f"{e} {t['at_pct'][e]:.1f}%" for e in t["kept"]])
        rl = two_lines("removed:", [f"{e} {t['at_pct'][e]:.1f}%" for e in t["removed"]]) if t["removed"] else ["removed: none"]
        line_cm = 8 / 72 * 2.54 * LS
        block = (len(kl) + len(rl)) * line_cm
        y_k = y_top - (row_h - block) / 2
        tk = ax.text(x0 + w / 2, y_k, "\n".join(kl), ha="center", va="top", fontsize=8, color=fs.OI["blue"], linespacing=LS)
        tr_ = ax.text(x0 + w / 2, y_k - len(kl) * line_cm, "\n".join(rl), ha="center", va="top", fontsize=8,
                      color=fs.OI["vermillion"] if t["removed"] else "0.45", linespacing=LS)
        checks.extend([(tk, (x0, y_top - row_h, x0 + w, y_top)), (tr_, (x0, y_top - row_h, x0 + w, y_top))])
        # snap
        snap_lines = []
        for e, v in t["snap"].items():
            if v["snapped"] and abs(v["before"] - v["after"]) > 1e-12:
                snap_lines.append(f"{e} {v['before']:g} → {v['after']:g}")
            elif v["snapped"]:
                snap_lines.append(f"{e} {v['before']:g} (integer)")
            elif v["nearest"] >= 1:
                snap_lines += wrap(f"{e} {v['before']:g}: {100 * v['rel']:.0f}% from {v['nearest']}, kept", 20)
            else:
                snap_lines.append(f"{e} {v['before']:g}: kept")
        cell(2, y_top, snap_lines[:5])
        # reduce
        cell(3, y_top, [f"{_formula_mathtext(t['snapped_formula'])} →", _formula_mathtext(t["cluster_id"])])
        # cluster id: readable formula over the canonical ID in small grey text when they differ
        if t["readable"] == t["cluster_id"]:
            cell(4, y_top, [_formula_mathtext(t["cluster_id"])], face="0.94", edge="0.2", lw=1.0, weight="bold")
        else:
            x0, w = col_x[4], col_w[4]
            ax.add_patch(FancyBboxPatch((x0, y_top - row_h), w, row_h, boxstyle="round,pad=0.0,rounding_size=0.1", fc="0.94", ec="0.2", lw=1.0))
            ta = ax.text(x0 + w / 2, mid + 0.22, _formula_mathtext(t["readable"]), ha="center", va="center", fontsize=8, fontweight="bold", color="0.1")
            tb = ax.text(x0 + w / 2, mid - 0.22, f"(ID: {t['cluster_id']})", ha="center", va="center", fontsize=8, color="0.45")
            checks.extend([(ta, (x0, y_top - row_h, x0 + w, y_top)), (tb, (x0, y_top - row_h, x0 + w, y_top))])
        for i in range(4):
            ax.annotate("", xy=(col_x[i + 1] - 0.01, mid), xytext=(col_x[i] + col_w[i] + 0.01, mid),
                        arrowprops=dict(arrowstyle="->", color="0.55", lw=0.6, shrinkA=0, shrinkB=0, mutation_scale=6))

    # bracket rows 1-2: same cluster, label rotated beside it
    y1 = H - head_h - 0.05
    y2 = H - head_h - (row_h + row_gap) - row_h + 0.05
    bx = col_x[4] + col_w[4] + 0.12
    ax.plot([bx - 0.08, bx, bx, bx - 0.08], [y1, y1, y2, y2], color="0.2", lw=0.9)
    ax.text(bx + 0.2, (y1 + y2) / 2, "same cluster", ha="center", va="center", rotation=90, fontsize=8, fontweight="bold", color="0.15")

    fs.check_text_fits(fig, ax, checks)
    fs.save(fig, out_path, expect_width_cm=16.0)
    plt.close(fig)
    return traces


# ---------------------------------------------------------------------------
# Study-overview workflow figure (paper Figure 1). Every count is read from a
# committed artifact or the featurized CSV header and asserted before drawing;
# section numbers are read from paper.md's headings.
# ---------------------------------------------------------------------------

FUNNEL_COUNTS_PATH = Path("results/cleaning_funnel/20260914T100914/funnel_counts.json")
# committed copy of the raw pull record (README beside it: original path and SHA256)
RAW_PULL_METADATA_PATH = Path("results/raw_pull_metadata/extraction_metadata.json")
RAW_PULL_METADATA_SHA256 = "d101d6675ceb15190a435da19783b9bbdba616caebb46009275ca1429653c1e3"
OVERVIEW_TARGETS = ["S", "sigma", "kappa", "zT"]
OVERVIEW_TARGET_SYMBOLS = {"S": "S", "sigma": "σ", "kappa": "κ", "zT": "zT"}
# heading keyword -> analysis key; each must match exactly one "## 3.x" heading in paper.md
OVERVIEW_SECTION_KEYWORDS = {
    "ladder": "Grouped validation lowers",
    "attribution": "test-set composition",
    "ceiling": "measured ceiling",
    "ablation": "Tripling the descriptor count",
    "external": "Transfer to independent databases",
    "direct_vs_derived": "Predicting zT directly",
}


def _sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_study_overview_facts():
    """
    Gather and assert every number and section label the overview figure shows.

    Sources, all committed (no gitignored input is read, so this runs from a fresh clone): funnel_counts.json
    (papers, curves, cleaned rows, stage count, verification gate); the ladder metrics (per-target rows, feature
    count); the descriptor-ablation metrics (feature-set sizes; every set includes temperature_bin, see
    nested_cv.get_feature_columns, so MAGPIE = 133 - 1 and CBFV = 265 - 1); results/raw_pull_metadata (the raw
    pull's own record: snapshot date and raw counts, SHA256-checked); paper.md (section numbers from headings).
    Each value is also checked against the sentence in paper.md that states it, so the figure and the text
    cannot drift.
    """
    paper = PAPER_MD_PATH.read_text(encoding="utf-8")
    facts = {}

    funnel = json.loads(FUNNEL_COUNTS_PATH.read_text(encoding="utf-8"))
    assert len(funnel["funnel"]) == 11, len(funnel["funnel"])
    assert funnel["verification_gate"]["match"] is True
    facts["papers"] = funnel["raw_input_row_counts"]["papers"]
    facts["curves"] = funnel["raw_input_row_counts"]["curves"]
    facts["cleaned_rows"] = funnel["funnel"][-1]["rows"]
    assert funnel["funnel"][-1]["step"].startswith("11_"), funnel["funnel"][-1]["step"]
    assert facts["cleaned_rows"] == funnel["verification_gate"]["step11_computed_rows"]
    assert facts["cleaned_rows"] == CLEANING_STEPS[-1][1] and facts["curves"] == RAW_CURVES

    ladder = json.loads(LADDER_METRICS_PATH.read_text(encoding="utf-8"))["runs"]
    n_features = set()
    facts["target_rows"] = {}
    for t in OVERVIEW_TARGETS:
        runs = {k: v for k, v in ladder.items() if v["target"] == t}
        assert runs, t
        rows = {v["n_rows_header"] for v in runs.values()}
        assert len(rows) == 1, (t, rows)
        facts["target_rows"][t] = rows.pop()
        n_features |= {v["n_features"] for k, v in runs.items() if v["feature_set"] == "full"}
    assert len(n_features) == 1, n_features
    facts["n_features_ladder"] = n_features.pop()

    ablation = json.loads(DESCRIPTOR_ABLATION_METRICS_PATH.read_text(encoding="utf-8"))
    per_set = {}
    for name in ("magpie", "cbfv", "full"):
        vals = {ablation[t][name]["n_features"] for t in OVERVIEW_TARGETS}
        assert len(vals) == 1, (name, vals)
        per_set[name] = vals.pop()
    facts["n_temperature"] = 1                                   # temperature_bin is in every feature set
    facts["n_magpie"] = per_set["magpie"] - facts["n_temperature"]
    facts["n_cbfv"] = per_set["cbfv"] - facts["n_temperature"]
    assert per_set["full"] == facts["n_features_ladder"]
    assert facts["n_magpie"] + facts["n_cbfv"] + facts["n_temperature"] == facts["n_features_ladder"], facts

    # snapshot date: the raw pull's own record (committed copy), not a filename
    from datetime import datetime

    raw_meta_path = RAW_PULL_METADATA_PATH
    assert _sha256_file(raw_meta_path) == RAW_PULL_METADATA_SHA256, f"{raw_meta_path} does not match its recorded SHA256"
    meta = json.loads(raw_meta_path.read_text(encoding="utf-8"))
    assert meta["files"]["papers"]["counted_row_count"] == facts["papers"]
    assert meta["files"]["curves"]["counted_row_count"] == facts["curves"]
    pulled = datetime.fromisoformat(meta["extraction_timestamp_utc"])
    assert meta["upstream_db_snapshot"].startswith(pulled.strftime("%Y-%m-%d")), meta["upstream_db_snapshot"]
    facts["snapshot_date"] = f"{pulled.day} {pulled:%b %Y}"
    facts["snapshot_date_long"] = f"{pulled.day} {pulled:%B %Y}"
    facts["snapshot_source"] = f"{raw_meta_path} (extraction_timestamp_utc {meta['extraction_timestamp_utc']}; upstream_db_snapshot {meta['upstream_db_snapshot']})"

    dvd = json.loads((DIRECT_VS_DERIVED_CHECKPOINT_DIR / "results.json").read_text(encoding="utf-8"))
    dvd_cfg = json.loads((DIRECT_VS_DERIVED_CHECKPOINT_DIR / "run_config.json").read_text(encoding="utf-8"))
    assert dvd["hyperparams_mode"] == "per_target" and dvd["subset_n_rows"] == dvd_cfg["subset_n_rows"]
    facts["dvd_subset_rows"] = dvd["subset_n_rows"]

    headings = re.findall(r"^## (3\.\d+) (.*)$", paper, flags=re.M)
    facts["sections"] = {}
    for key, kw in OVERVIEW_SECTION_KEYWORDS.items():
        hits = [num for num, title in headings if kw in title]
        assert len(hits) == 1, (key, kw, hits)
        facts["sections"][key] = hits[0]

    # the same numbers, as written in paper.md
    def paper_has(*needles):
        for n in needles:
            assert n in paper, f"paper.md does not contain {n!r}"
    paper_has(f"{facts['papers']:,} publications", f"{facts['curves']:,} digitized property curves",
              f"{facts['cleaned_rows']:,} rows", f"{facts['n_features_ladder']} features",
              f"{facts['n_magpie']} MAGPIE", f"{facts['n_cbfv']} CBFV")
    paper_has(*[f"{v:,}" for v in facts["target_rows"].values()])
    paper_has(f"taken on {facts['snapshot_date_long']}", f"{facts['dvd_subset_rows']:,}")
    return facts


def make_study_overview(out_path):
    """
    Study-overview workflow figure: data pipeline on top, five analyses below, and the shared model box
    spanning them. Drawn at 16 cm width with all text at 8 pt or larger; no colour carries meaning
    (analyses are labelled A to E and carry their section number). After drawing, every text block is
    checked to lie inside its box, so an overflow fails the run instead of shipping.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    f = load_study_overview_facts()
    sec = f["sections"]
    tr = f["target_rows"]
    sym = OVERVIEW_TARGET_SYMBOLS

    W = 16.0                               # cm
    top_w, top_h, gap = 3.4, 2.75, 0.8
    a_h, a_y, b_h = 7.35, 1.85, 1.35
    H = a_y + a_h + 1.1 + top_h + 0.2
    fsz, fsz_title = 8.0, 8.0              # points; the minimum size is 8
    # save_figure crops with bbox_inches='tight' and its 0.1 in pad on each side (0.508 cm in total), so the canvas is
    # made that much narrower: the saved PNG is 16.0 cm wide at its natural size, and 8 pt text stays 8 pt at 16 cm
    fig = plt.figure(figsize=(W * fs.CM, H * fs.CM))                 # exactly 16 cm wide
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-0.1, W + 0.1)
    ax.set_ylim(0, H)
    ax.axis("off")
    checks = []                            # (text artist, (x0, y0, x1, y1) of its box)

    def box(x, y, w, h, head, body, face="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.12",
                                    fc=face, ec="0.15", lw=0.9))
        t1 = ax.text(x + w / 2, y + h - 0.15, head, ha="center", va="top", fontsize=fsz_title,
                     fontweight="bold", color="0.1", linespacing=1.3)
        fig.canvas.draw()                  # place the body just below the rendered heading, whatever its line count
        head_bottom = ax.transData.inverted().transform((0, t1.get_window_extent(fig.canvas.get_renderer()).y0))[1]
        t2 = ax.text(x + w / 2, head_bottom - 0.3, body, ha="center", va="top",
                     fontsize=fsz, color="0.1", linespacing=1.3)
        checks.extend([(t1, (x, y, x + w, y + h)), (t2, (x, y, x + w, y + h))])

    def arrow(p0, p1):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, lw=0.9, color="0.25",
                                     shrinkA=0, shrinkB=0))

    # ---- top row: data pipeline
    top_y = H - 0.2 - top_h
    xs = [i * (top_w + gap) for i in range(4)]
    box(xs[0], top_y, top_w, top_h, "Starrydata2\nsnapshot",
        f"{f['snapshot_date']}\n{f['papers']:,} papers\n{f['curves']:,} curves")
    box(xs[1], top_y, top_w, top_h, "Cleaning", f"11 stages\n{f['cleaned_rows']:,} rows")
    box(xs[2], top_y, top_w, top_h, "Featurization",
        f"{f['n_magpie']} MAGPIE\n+ {f['n_cbfv']} CBFV\n+ T\n= {f['n_features_ladder']} features")
    per_target = "\n".join(f"{sym[t]}  {tr[t]:,}" for t in OVERVIEW_TARGETS)
    box(xs[3], top_y, top_w, top_h, "Per-target\ndatasets", per_target)
    for i in range(3):
        arrow((xs[i] + top_w + 0.03, top_y + top_h / 2), (xs[i + 1] - 0.03, top_y + top_h / 2))

    # ---- five analysis boxes (line breaks are explicit so the maths string stays intact)
    n, a_gap = 5, 0.3
    a_w = (W - a_gap * (n - 1)) / n
    analyses = [
        (f"(A)  \u00a7{sec['ladder']}\nValidation\nladder",
         "ungrouped:\nrandom 80/20,\n5-fold, 10-fold\n\ngrouped\n(5 \u00d7 5 CV):\ncomposition,\nchemistry\ncluster\n\n+ attribution\ncontrol (\u00a7" + sec["attribution"]
         + "):\nTreeSHAP,\nrandom vs\ngrouped"),
        (f"(B)  \u00a7{sec['ceiling']}\nLabel-noise\nceiling",
         "round-robin\nmeasurement\nnoise\n\n+ teMatDb\ndigitization\nnoise"),
        (f"(C)  \u00a7{sec['ablation']}\nDescriptor\nablation", "feature sets:\nMAGPIE,\nCBFV,\nfull"),
        (f"(D)  \u00a7{sec['external']}\nExternal\ntransfer", "ESTM,\nteMatDb;\nin-support\nrestriction"),
        (f"(E)  \u00a7{sec['direct_vs_derived']}\nDirect vs\nderived zT",
         "zT predicted\ndirectly vs\n$S^2\\sigma T/\\kappa$\nfrom predicted\ncomponents\n\n" + f"{f['dvd_subset_rows']:,}-row\nsubset"),
    ]
    order = [float(x) for x in (sec["ladder"], sec["ceiling"], sec["ablation"], sec["external"], sec["direct_vs_derived"])]
    assert all(0 < (b_ - a_) < 1 for a_, b_ in zip([3.0] + order, order)) and order == sorted(order), order   # boxes run in section order
    x_centres = []
    for i, (head, body) in enumerate(analyses):
        x0 = i * (a_w + a_gap)
        x_centres.append(x0 + a_w / 2)
        box(x0, a_y, a_w, a_h, head, body, face="0.95")
    bus_y = top_y - 0.55
    x_src = x_centres[-1]                                   # the last box's centre lies under the last top box
    assert xs[3] < x_src < xs[3] + top_w
    ax.plot([x_src, x_src], [top_y, bus_y], color="0.25", lw=0.9)
    ax.plot([x_centres[0], x_src], [bus_y, bus_y], color="0.25", lw=0.9)
    for xc in x_centres:
        arrow((xc, bus_y), (xc, a_y + a_h + 0.03))

    # ---- bottom box spanning A-E
    ax.add_patch(FancyBboxPatch((0, 0.1), W, b_h, boxstyle="round,pad=0.0,rounding_size=0.12", fc="0.85", ec="0.15", lw=0.9))
    t = ax.text(W / 2, 0.1 + b_h / 2 + 0.25, "XGBoost, frozen per-target hyperparameters",
                ha="center", va="center", fontsize=fsz_title, fontweight="bold", color="0.1", linespacing=1.3)
    t2 = ax.text(W / 2, 0.1 + b_h / 2 - 0.3, "same model in every model-based analysis", ha="center", va="center",
                 fontsize=fsz, color="0.1", style="italic")
    checks.extend([(t, (0, 0.1, W, 0.1 + b_h)), (t2, (0, 0.1, W, 0.1 + b_h))])
    for i, xc in enumerate(x_centres):
        if i == 1:                                          # (B) label-noise ceiling trains no model
            continue
        ax.plot([xc, xc], [a_y, 0.1 + b_h], color="0.25", lw=0.9, ls=(0, (3, 2)))

    fs.check_text_fits(fig, ax, checks)                               # every text block inside its box, all text >= 8 pt
    fs.save(fig, out_path, expect_width_cm=16.0)
    plt.close(fig)
    return f


def make_contact_sheet(out_path, paper_path=PAPER_MD_PATH, scale=0.75):
    """
    One image with all 12 manuscript figures in paper order, for reviewing them together. The order and the
    file behind each figure number are read from paper.md's image lines, so they cannot drift from the
    manuscript. All figures share one scale, so a single-column (8 cm) figure appears half as wide as a
    double-column (16 cm) one; two columns of 16 cm slots.
    """
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import font_manager

    paper = Path(paper_path).read_text(encoding="utf-8")
    items = [(int(n), Path("figures") / f) for n, f in re.findall(r"^!\[Figure (\d+)\]\(figures/([^)]+)\)", paper, flags=re.M)]
    assert [n for n, _ in items] == list(range(1, 13)), items
    imgs = []
    for n, path in items:
        assert path.exists(), path
        imgs.append((n, Image.open(path).convert("RGB")))

    slot_w = round(fs.WIDTH_CM["double"] * fs.CM * fs.DPI)          # a 16 cm slot in pixels
    margin, label_h = 60, 70
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"), 46)
    rows = [imgs[i:i + 2] for i in range(0, len(imgs), 2)]
    row_heights = [max(im.size[1] for _, im in row) + label_h + margin for row in rows]
    sheet = Image.new("RGB", (2 * slot_w + 3 * margin, sum(row_heights) + margin), "white")
    draw = ImageDraw.Draw(sheet)
    y = margin
    for row, h in zip(rows, row_heights):
        for k, (n, im) in enumerate(row):
            x = margin + k * (slot_w + margin)
            draw.text((x, y), f"Figure {n}", fill="black", font=font)
            sheet.paste(im, (x, y + label_h))
            draw.rectangle([x - 1, y + label_h - 1, x + im.size[0], y + label_h + im.size[1]], outline=(210, 210, 210))
        y += h
    sheet = sheet.resize((round(sheet.size[0] * scale), round(sheet.size[1] * scale)), Image.LANCZOS)
    sheet.save(out_path, dpi=(fs.DPI * scale, fs.DPI * scale))
    return [n for n, _ in items]


def main():
    # legacy figures (not in the manuscript): their own older style, untouched by the overhaul
    apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    make_property_distributions(FIGURES_DIR / "fig_property_distributions")
    print("Saved fig_property_distributions.png / .pdf")

    cluster_df = load_snapfix_cluster_columns()
    print(f"Loaded {len(cluster_df):,} rows from {SNAPFIX_FEATURIZED_CSV} for cluster sizes")
    make_cluster_size_distribution(cluster_df, FIGURES_DIR / "cluster_size_distribution")
    print("Saved cluster_size_distribution.png / .pdf")

    make_model_comparison(FIGURES_DIR / "fig1_model_comparison")
    print("Saved fig1_model_comparison.png / .pdf")

    try:
        make_actual_vs_predicted(FIGURES_DIR / "fig3_actual_vs_predicted")
        print("Saved fig3_actual_vs_predicted.png / .pdf")
    except FileNotFoundError as e:
        print(f"Skipped fig3_actual_vs_predicted: {e}")

    # the twelve manuscript figures, in paper order, in scripts/figstyle.py's style
    fs.apply()
    paper_figures = [
        ("Figure 1", make_leakage_schematic, "fig1_leakage_schematic"),
        ("Figure 2", make_study_overview, "fig0_study_overview"),
        ("Figure 3", make_cleaning_funnel, "cleaning_funnel"),
        ("Figure 4", make_grouping_rule_schematic, "fig3_grouping_rule_schematic"),
        ("Figure 5", make_validation_ladder, "fig2_validation_ladder"),
        ("Figure 6", make_zt_parity, "zt_parity_random_vs_grouped"),
        ("Figure 7", make_shap_attribution, "shap_attribution"),
        ("Figure 8", make_headroom_decomposition, "fig5_headroom"),
        ("Figure 9", make_descriptor_ablation, "descriptor_ablation"),
        ("Figure 10", make_external_transfer, "external_transfer"),
        ("Figure 11", make_sigma_extrapolation, "sigma_extrapolation"),
        ("Figure 12", make_zt_direct_vs_derived, "zt_direct_vs_derived"),
    ]
    for label, fn, stem in paper_figures:
        fn(FIGURES_DIR / stem)
        print(f"Saved {stem}.png / .pdf ({label})")
    make_descriptor_ablation_absolute(FIGURES_DIR / "descriptor_ablation_absolute")
    print("Saved descriptor_ablation_absolute.png / .pdf (supplementary)")

    order = make_contact_sheet(FIGURES_DIR / "_contact_sheet.png")
    print(f"Saved _contact_sheet.png ({len(order)} figures in paper order)")


if __name__ == "__main__":
    main()
