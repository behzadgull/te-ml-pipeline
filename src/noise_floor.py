"""
Noise-floor anchor, computed in both log and linear space.

Computes R^2_max = 1 - sigma^2_noise / sigma^2_total in log space (all
four properties) and in linear space (all four properties too, added so
S/zT can be compared apples-to-apples against their linear-space
confirmed R^2 -- see PAPER_SCALE), using the Alleno et al. 2015
round-robin uncertainties as the noise reference (S ~6%, sigma ~8%,
kappa ~11%, zT ~17-19%) and this dataset's actual property variance
(log or raw, matching scale) for sigma_total. report() prints both
scales for every property, plus a "paper" column selecting whichever
scale matches how each property's confirmed chemistry-cluster R^2 was
actually scored.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DATA_DIR = "data/processed"
PROJECT = "ThermoelectricMaterials"

PROPERTIES = ["S", "sigma", "kappa", "zT"]

# Alleno et al. 2015, Rev. Sci. Instrum. 86:011301, DOI 10.1063/1.4905250.
# Round-robin relative measurement uncertainties from ONE skutterudite
# compound -- used here as an inference/lower-bound analogy for this
# dataset's noise floor, not a direct per-dataset measurement (see
# CLAUDE.md Paper A item 3). zT is reported as a 17-19% range; the
# midpoint 0.18 is used as the point estimate.
RELATIVE_UNCERTAINTY = {
    "S": 0.06,
    "sigma": 0.08,
    "kappa": 0.11,
    "zT": 0.18,
}

# Frozen chemistry-cluster pooled out-of-fold R^2 from CLAUDE.md's
# "Confirmed Results -- Five-Way Ladder" table (XGBoost, frozen
# hyperparameters).
CONFIRMED_CHEMISTRY_CLUSTER_R2 = {
    "S": 0.8083,
    "sigma": 0.7522,
    "kappa": 0.8226,
    "zT": 0.7965,
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


def compute_all(df, properties=PROPERTIES):
    """
    Run both compute_r2_max_log and compute_r2_max_linear for every
    property in `properties`. Returns {property: {"log": {...},
    "linear": {...}}}.
    """
    return {
        prop: {
            "log": compute_r2_max_log(prop, df[prop]),
            "linear": compute_r2_max_linear(prop, df[prop]),
        }
        for prop in properties
    }


def load_cleaned_dataset(processed_data_dir=PROCESSED_DATA_DIR, project=PROJECT):
    """Load the most recently written cleaned_<project>_<date>.csv."""
    candidates = sorted(Path(processed_data_dir).glob(f"cleaned_{project}_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No cleaned_{project}_*.csv found in {processed_data_dir}; run src/data_cleaning.py first"
        )
    return pd.read_csv(candidates[-1]), candidates[-1]


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
    print(f"Loaded {len(df):,} rows from {source_path}\n")
    results = compute_all(df)
    report(results)


if __name__ == "__main__":
    main()
