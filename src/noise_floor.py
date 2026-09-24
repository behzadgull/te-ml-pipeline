"""
Noise-floor anchor, computed in both log and linear space.

Computes R^2_max = 1 - sigma^2_noise / sigma^2_total in log space (all
four properties) and in linear space (all four properties too, added so
S/zT can be compared apples-to-apples against their linear-space
confirmed R^2 -- see PAPER_SCALE), using the Alleno et al. 2015
round-robin uncertainties as the noise reference (S ~6%, sigma ~8%,
kappa ~11%, zT 19%) and this dataset's actual property variance
(log or raw, matching scale) for sigma_total. report() prints both
scales for every property, plus a "paper" column selecting whichever
scale matches how each property's confirmed chemistry-cluster R^2 was
actually scored.

sigma_total is computed on exactly the rows the chemistry-cluster ladder
rung scored for that target (load_aligned_target_values, via
src.nested_cv.load_target_data on the featurized CSV), not on the
cleaned CSV's own notna/positive column, which is a slightly larger row
set (featurization drops a small number of rows per target that fail
descriptor computation -- see CLAUDE.md's Data Cleaning Pipeline step 5
TODO). load_cleaned_dataset() is kept for the module's own row-count
banner and for callers that want the unaligned full cleaned CSV, and
now defaults to File A's cleaned CSV with a SHA256 check, not the
"data/processed" glob default that silently resolved to a different,
older cleaned CSV (see CLAUDE.md's Canonical Dataset section).
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nested_cv import load_target_data  # noqa: E402

PROCESSED_DATA_DIR = "data/processed"
PROJECT = "ThermoelectricMaterials"

# File A's cleaned CSV (CLAUDE.md's Canonical Dataset section) -- the
# "data/processed" glob default used to resolve to a different, older
# cleaned CSV that is neither File A nor File B (SHA256 2a1af8b8...,
# now renamed to cleaned_ThermoelectricMaterials_2026-08-15.STALE_2a1af8b8.csv
# so a stale glob hit raises FileNotFoundError instead of silently
# loading the wrong file).
FILE_A_CLEANED_CSV = Path(
    "checkpoints/saved_predictions/te-ml-pipeline/data/processed/"
    "cleaned_ThermoelectricMaterials_2026-08-15.csv"
)
FILE_A_CLEANED_CSV_SHA256 = "0275c5088521580a1156acf2078f77f0eb7d4a8c6b4e724864f0de0875d8a393"

# Per-target row count the chemistry-cluster ladder rung actually scored
# (n_rows_header, i.e. the single-pass row count BEFORE pooling across
# the 5 repeats/5 outer folds -- pooled_n = 5 * n_rows_header, verified
# directly for every target). Source:
# reports/regen_snapfix/20260917T150000/ladder_metrics.json, the same
# snapfix chemistry-cluster rung CLAUDE.md's Five-Way Ladder table cites.
LADDER_N_ROWS = {
    "S": 185064,
    "sigma": 182755,
    "kappa": 121110,
    "zT": 129419,
}

PROPERTIES = ["S", "sigma", "kappa", "zT"]

# Alleno et al. 2015, Rev. Sci. Instrum. 86:011301, DOI 10.1063/1.4905250.
# Round-robin relative measurement uncertainties from ONE skutterudite
# compound -- used here as an inference/lower-bound analogy for this
# dataset's noise floor, not a direct per-dataset measurement (see
# CLAUDE.md Paper A item 3). zT's value corrected 2026-09-23: Alleno et
# al. report a single temperature-averaged value of 19% for zT at 68%
# confidence over 300-700 K, not a 17-19% range -- that range had no
# support in the source and is removed.
RELATIVE_UNCERTAINTY = {
    "S": 0.06,
    "sigma": 0.08,
    "kappa": 0.11,
    "zT": 0.19,
}

# Frozen chemistry-cluster pooled out-of-fold R^2. Corrected 2026-09-24:
# this constant still held the PRE-grouping-fix values (S 0.8076, sigma
# 0.7600, kappa 0.8460, zT 0.7968) from the 2026-08-22 checkpoint set --
# superseded 2026-09-19 by the SNAP(0.05)/S5 grouping fixes (commit
# c1c6873) and the resulting snapfix featurized CSV, per CLAUDE.md's
# Five-Way Ladder table. Found auditing this module's row-alignment fix;
# every headroom this module ever printed after 2026-09-19 was computed
# against a stale confirmed-R^2 denominator. Now matches CLAUDE.md's
# current chemistry-cluster column exactly: results/ladder_regen_snapfix/
# 20260917T150000/{S,sigma,kappa,zT}_chemistry_full/.
CONFIRMED_CHEMISTRY_CLUSTER_R2 = {
    "S": 0.7528,
    "sigma": 0.7020,
    "kappa": 0.8092,
    "zT": 0.7456,
}

# Which R2_max scale is apples-to-apples with each property's confirmed
# R^2 above: sigma/kappa are scored in log10 space there
# (LOG_TRANSFORM_TARGETS), S/zT in linear space (excluded from that
# decision -- S can be negative, log10 undefined; zT isn't multiple
# orders of magnitude). report()'s "paper R2_max" / headroom columns use
# this to pick the matching scale per property.
PAPER_SCALE = {
    "S": "linear",
    "sigma": "log",
    "kappa": "log",
    "zT": "linear",
}


def sigma_noise_log(relative_uncertainty):
    """
    Log-space noise std for a fractional round-robin uncertainty, via
    the small-angle approximation std(log(x)) ~ eps for small relative
    error eps (since log(1+eps) ~ eps).
    """
    return relative_uncertainty


def sigma_total_log(values, ddof=1):
    """
    Std of log(|values|) for a property column, after dropping NaN and
    non-positive entries (log is undefined at and below zero).

    Uses natural log; R^2_max = 1 - sigma_noise_log^2/sigma_total_log^2
    is a ratio of two log-space variances computed in the same base, so
    it is invariant to the log base chosen -- log10 would give the
    identical R^2_max.

    Returns (std, n_used, n_dropped_nonpositive).
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    positive_mask = np.abs(values) > 0
    n_dropped = int((~positive_mask).sum())
    log_values = np.log(np.abs(values[positive_mask]))
    return float(np.std(log_values, ddof=ddof)), int(positive_mask.sum()), n_dropped


def sigma_noise_linear(relative_uncertainty, values):
    """
    Linear-space noise std: relative_uncertainty * median(|values|).
    Anchoring to the median (not mean) of magnitudes keeps this robust
    to the long right tail sigma/kappa have even in linear space.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return relative_uncertainty * float(np.median(np.abs(values)))


def sigma_total_linear(values, ddof=1):
    """
    Std of the raw property column (linear space), NaN dropped. Unlike
    log space, linear std is defined for negative and zero values, so
    nothing needs to be dropped beyond NaN -- S's negative/zero entries
    and zT's exact zeros are all valid here.

    Returns (std, n_used).
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return float(np.std(values, ddof=ddof)), int(len(values))


def compute_r2_max_log(property_name, values):
    """
    Log-space noise-floor ceiling for one property column.

    R^2_max = 1 - sigma_noise_log^2 / sigma_total_log^2, where
    sigma_noise_log comes from the Alleno et al. round-robin relative
    uncertainty (RELATIVE_UNCERTAINTY) and sigma_total_log from this
    dataset's actual log-property spread. Returns a dict of the ceiling
    and its components.

    property_name must be a key of RELATIVE_UNCERTAINTY ("S", "sigma",
    "kappa", or "zT"). For S (which can be negative or exactly zero,
    bounds -1000 to 1000 uV/K) and zT (bounded at exactly 0), rows with
    a non-positive value are dropped before taking the log -- see
    sigma_total_log's docstring; n_dropped_nonpositive reports how many.
    """
    if property_name not in RELATIVE_UNCERTAINTY:
        raise ValueError(f"Unknown property {property_name!r}; expected one of {PROPERTIES}")

    relative_uncertainty = RELATIVE_UNCERTAINTY[property_name]
    noise = sigma_noise_log(relative_uncertainty)
    total, n_used, n_dropped = sigma_total_log(values)
    r2_max = 1 - (noise ** 2) / (total ** 2)

    return {
        "property": property_name,
        "scale": "log",
        "relative_uncertainty": relative_uncertainty,
        "sigma_noise": noise,
        "sigma_total": total,
        "r2_max": r2_max,
        "n_used": n_used,
        "n_dropped_nonpositive": n_dropped,
    }


def compute_r2_max_linear(property_name, values):
    """
    Linear-space noise-floor ceiling for one property column.

    R^2_max = 1 - sigma_noise_linear^2 / sigma_total_linear^2, where
    sigma_noise_linear = relative_uncertainty * median(|values|) and
    sigma_total_linear is this dataset's actual raw-scale std for the
    property. Added so S and zT -- scored in linear space by the frozen
    LOG_TRANSFORM_TARGETS decision -- get a noise-floor ceiling in the
    same space as their confirmed chemistry-cluster R^2, instead of only
    the log-space ceiling compute_r2_max_log gives every property.
    Returns a dict of the ceiling and its components, same shape as
    compute_r2_max_log's (n_dropped_nonpositive is always 0 here: linear
    std doesn't require dropping non-positive values).
    """
    if property_name not in RELATIVE_UNCERTAINTY:
        raise ValueError(f"Unknown property {property_name!r}; expected one of {PROPERTIES}")

    relative_uncertainty = RELATIVE_UNCERTAINTY[property_name]
    noise = sigma_noise_linear(relative_uncertainty, values)
    total, n_used = sigma_total_linear(values)
    r2_max = 1 - (noise ** 2) / (total ** 2)

    return {
        "property": property_name,
        "scale": "linear",
        "relative_uncertainty": relative_uncertainty,
        "sigma_noise": noise,
        "sigma_total": total,
        "r2_max": r2_max,
        "n_used": n_used,
        "n_dropped_nonpositive": 0,
    }


def load_aligned_target_values(target, expected_n_rows=None):
    """
    Load one target's column from exactly the row set the chemistry-
    cluster ladder rung scored for it: src.nested_cv.load_target_data(target),
    which reads the featurized CSV and filters to df[target].notna() --
    the identical loader and filter the ladder itself uses.

    Asserts the resulting row count equals LADDER_N_ROWS[target] (or
    expected_n_rows, if given) so a silent row-set drift between this
    module and the ladder raises instead of producing a headroom number
    computed against the wrong denominator.
    """
    expected = LADDER_N_ROWS[target] if expected_n_rows is None else expected_n_rows
    df = load_target_data(target)
    n_rows = len(df)
    if n_rows != expected:
        raise ValueError(
            f"{target}: loaded {n_rows:,} rows via load_target_data, expected "
            f"{expected:,} (LADDER_N_ROWS[{target!r}], from the chemistry-cluster "
            "ladder rung's n_rows_header). Row sets have diverged -- do not "
            "compute sigma_total against this without finding out why."
        )
    return df[target]


def compute_all(properties=PROPERTIES):
    """
    Run both compute_r2_max_log and compute_r2_max_linear for every
    property in `properties`, each against its own row-aligned column
    (load_aligned_target_values) rather than a single shared cleaned-CSV
    df -- so sigma_total for each target is computed on exactly the rows
    the chemistry-cluster ladder rung scored for that target, not the
    slightly larger row set the cleaned CSV's own notna/positive filter
    gives (see module docstring). Returns {property: {"log": {...},
    "linear": {...}}}.
    """
    return {
        prop: {
            "log": compute_r2_max_log(prop, load_aligned_target_values(prop)),
            "linear": compute_r2_max_linear(prop, load_aligned_target_values(prop)),
        }
        for prop in properties
    }


def load_cleaned_dataset(path=FILE_A_CLEANED_CSV, expected_sha256=FILE_A_CLEANED_CSV_SHA256):
    """
    Load File A's cleaned CSV, verifying its SHA256 before returning it.

    Drops chemistry_cluster_id: File A's cleaned CSV predates the
    2026-09-19 SNAP(0.05)/S5 grouping fixes (commit c1c6873) and its
    column is still the pre-fix, buggy one (12,036 clusters, unsnapped
    decimal-coefficient formulas -- see CLAUDE.md's Grouping Fixes
    section, BUG 1). Dropping it here, at the source, means a caller
    cannot silently consume the stale column through this loader --
    the exact trap make_cluster_size_distribution (scripts/make_figures.py)
    would have fallen into otherwise; it now loads cluster membership
    from the snapfix featurized CSV instead.

    Not used by compute_all() any more (see load_aligned_target_values);
    kept for the module's own row-count banner in main() and for callers
    that want the unaligned full cleaned CSV rather than one target's
    aligned column.
    """
    path = Path(path)
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{path} has SHA256 {actual_sha256}, expected {expected_sha256} (File A). "
            "Refusing to compute the noise floor against an unverified dataset."
        )
    df = pd.read_csv(path)
    return df.drop(columns=["chemistry_cluster_id"]), path


def report(results, confirmed=CONFIRMED_CHEMISTRY_CLUSTER_R2, paper_scale=PAPER_SCALE):
    """
    Print a comparison table with both R2_max scales for every property,
    plus a "paper" R2_max/headroom pair that selects whichever scale
    (paper_scale) matches how each property's confirmed chemistry-
    cluster R^2 was actually scored -- log for sigma/kappa, linear for
    S/zT. That selection makes every row's headroom (paper R2_max minus
    confirmed) a genuinely like-for-like comparison, unlike a
    log-space-only ceiling would give for S/zT.
    """
    rows = []
    for prop in PROPERTIES:
        log_res = results[prop]["log"]
        lin_res = results[prop]["linear"]
        conf = confirmed[prop]
        scale = paper_scale[prop]
        paper_r2_max = log_res["r2_max"] if scale == "log" else lin_res["r2_max"]
        rows.append(
            {
                "property": prop,
                "relative_uncertainty": log_res["relative_uncertainty"],
                "r2_max_log": log_res["r2_max"],
                "r2_max_linear": lin_res["r2_max"],
                "paper_scale": scale,
                "paper_r2_max": paper_r2_max,
                "chemistry_cluster_r2": conf,
                "headroom": paper_r2_max - conf,
                "n_dropped_nonpositive_log": log_res["n_dropped_nonpositive"],
            }
        )

    header = (
        f"{'Property':<10}{'rel.unc.':>10}{'R2max(log)':>12}{'R2max(lin)':>12}"
        f"{'paper scale':>13}{'paper R2max':>13}{'chem-clust R2':>15}{'headroom':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['property']:<10}{row['relative_uncertainty']:>10.0%}"
            f"{row['r2_max_log']:>12.4f}{row['r2_max_linear']:>12.4f}"
            f"{row['paper_scale']:>13}{row['paper_r2_max']:>13.4f}"
            f"{row['chemistry_cluster_r2']:>15.4f}{row['headroom']:>10.4f}"
        )
    print()
    print(
        "'paper R2max' picks the scale matching how each property's confirmed "
        "chemistry-cluster R^2 was actually scored (sigma/kappa: log10, per the "
        "frozen LOG_TRANSFORM_TARGETS decision; S/zT: linear, since S can be "
        "negative -- log10 undefined -- and zT isn't multiple orders of magnitude) "
        "-- so 'headroom' is a like-for-like comparison for every property, not "
        "just sigma/kappa."
    )
    return rows


def main():
    df, source_path = load_cleaned_dataset()
    print(f"Loaded {len(df):,} rows from {source_path} (unaligned cleaned CSV, informational only)")
    print("sigma_total below is computed per-target from the ladder-aligned featurized CSV, not this file.\n")
    results = compute_all()
    report(results)


if __name__ == "__main__":
    main()
