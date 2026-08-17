"""
Global data cleaning pipeline for the fresh Starrydata2 pull.

Implements the 11-step cleaning spec (CLAUDE.md, Data Cleaning Pipeline):
property extraction and range filtering, data integration and
consolidation, temperature filtering and binning, pivot long-to-wide,
formula cleaning via pymatgen, zT self-consistency check, DFT data
removal, multi-source consistency filtering (knee-method thresholds),
MAD outlier filtering, minimum temperature coverage, and the smoothness
filter.

Runs once, globally, before any train/test split (see Grouping Key,
"Global cleaning, not fold-local"). Each step is a standalone function
so callers (and the fold-local sensitivity check that comes later) can
re-run individual steps; run_cleaning_pipeline() chains all eleven and
prints a row-count line after each so the result can be sanity-checked
against CLAUDE.md's reference numbers.
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.canonicalization import DEFAULT_DOPANT_THRESHOLD_FRAC, add_canonical_columns

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
PROJECT = "ThermoelectricMaterials"

# Step 1 physical property bounds (Snyder & Toberer 2008).
PROPERTY_BOUNDS = {
    "S": (-1000.0, 1000.0),  # microV/K
    "sigma": (10.0, 1.0e7),  # S/m; lower bound = semiconductor-insulator boundary (confirmed against thesis 2026-08-17)
    "kappa": (0.05, 25.0),  # W/mK, Cahill-Pohl minimum
    "zT": (0.0, 4.0),
}

# Curve properties this pipeline extracts; "Electrical resistivity" is
# converted to sigma in step 2, everything else maps straight through.
TARGET_PROP_Y = {
    "Seebeck coefficient": "S",
    "Electrical conductivity": "sigma",
    "Electrical resistivity": "rho",
    "Thermal conductivity": "kappa",
    "ZT": "zT",
}
TEMP_PROP_X = {"Temperature", "T"}

DFT_KEYWORDS = ("DFT", "first principles", "first-principles", "ab initio", "VASP")

TEMP_MIN_K = 300
TEMP_MAX_K = 800
TEMP_BIN_WIDTH_K = 25

ZT_SELF_CONSISTENCY_MAX_REL_ERROR = 0.5
MAD_THRESHOLD = 3.5
MIN_TEMP_COVERAGE = 3
SMOOTHNESS_WINDOW = 3

# Step 8 starting values (CLAUDE.md: "use reasonable starting values").
# TODO(Phase 1): re-tune via find_knee_threshold() once a trained model
# on the chemistry-cluster split exists to evaluate held-out R^2 against.
MULTI_SOURCE_CV_THRESHOLDS = {"S": 0.5, "sigma": 0.8, "kappa": 0.5, "zT": 0.5}

WIDE_PROPERTIES = ("S", "sigma", "kappa", "zT")


def _report(label, df, key_cols=None):
    """Print a row-count sanity-check line, optionally with unique-key counts."""
    msg = f"[{label}] rows={len(df)}"
    if key_cols:
        for col in key_cols:
            if col in df.columns:
                msg += f", unique {col}={df[col].nunique()}"
    print(msg)


def load_raw_curves(raw_data_dir=RAW_DATA_DIR, project=PROJECT):
    """Load the raw curves table acquired by src/data_acquisition.py."""
    return pd.read_csv(Path(raw_data_dir) / f"{project}_curves.csv.gz", compression="gzip")


def load_raw_papers(raw_data_dir=RAW_DATA_DIR, project=PROJECT):
    """Load the raw papers table acquired by src/data_acquisition.py."""
    return pd.read_csv(Path(raw_data_dir) / f"{project}_papers.csv.gz", compression="gzip")


def step1_extract_and_filter_properties(curves_df):
    """
    Step 1: property extraction & range filtering (~1,114,628 rows on
    the reference pull).

    Explodes each curve row's digitized (x, y) point arrays into one row
    per (curve, point) restricted to prop_x in {Temperature, T} and
    prop_y in TARGET_PROP_Y, converts Seebeck from V/K to microV/K, and
    drops any point outside PROPERTY_BOUNDS (resistivity is checked
    against the inverted sigma bounds since it hasn't been converted
    yet -- that happens in step 2).
    """
    df = curves_df[curves_df["prop_x"].isin(TEMP_PROP_X) & curves_df["prop_y"].isin(TARGET_PROP_Y)]

    records = []
    for row in df.itertuples(index=False):
        try:
            x_vals = json.loads(row.x)
            y_vals = json.loads(row.y)
        except (json.JSONDecodeError, TypeError):
            continue
        if len(x_vals) != len(y_vals):
            continue
        prop = TARGET_PROP_Y[row.prop_y]
        for t, v in zip(x_vals, y_vals):
            if t is None or v is None:
                continue
            value = float(v)
            if prop == "S":
                value *= 1.0e6  # V/K -> microV/K
            records.append(
                {
                    "SID": row.SID,
                    "DOI": row.DOI,
                    "composition": row.composition,
                    "sample_id": row.sample_id,
                    "figure_id": row.figure_id,
                    "temperature_K": float(t),
                    "property": prop,
                    "value": value,
                }
            )

    long_df = pd.DataFrame.from_records(records)
    if long_df.empty:
        return long_df

    bounds = dict(PROPERTY_BOUNDS)
    lo, hi = PROPERTY_BOUNDS["sigma"]
    bounds["rho"] = (1.0 / hi, 1.0 / lo)  # bounds flip under inversion

    keep = pd.Series(False, index=long_df.index)
    for prop, (prop_lo, prop_hi) in bounds.items():
        is_prop = long_df["property"] == prop
        keep |= is_prop & long_df["value"].between(prop_lo, prop_hi)
    return long_df[keep].reset_index(drop=True)


def step2_integrate_and_consolidate(long_df):
    """
    Step 2: data integration & consolidation.

    Converts electrical resistivity (rho, ohm*m) to electrical
    conductivity (sigma, S/m) via sigma = 1/rho and merges it into the
    same "sigma" property as directly-reported conductivity, then
    re-applies the sigma bounds now that rho values have been inverted.
    """
    df = long_df.copy()
    is_rho = df["property"] == "rho"
    df.loc[is_rho, "value"] = 1.0 / df.loc[is_rho, "value"]
    df.loc[is_rho, "property"] = "sigma"

    lo, hi = PROPERTY_BOUNDS["sigma"]
    is_sigma = df["property"] == "sigma"
    out_of_bounds = is_sigma & ~df["value"].between(lo, hi)
    return df[~out_of_bounds].reset_index(drop=True)


def step3_filter_temperature(long_df):
    """
    Step 3: temperature filtering.

    Restricts to the 300-800K window and assigns each point to a 25K
    bin, labelled by the bin's lower edge (e.g. 300 covers [300,325)).
    """
    df = long_df[long_df["temperature_K"].between(TEMP_MIN_K, TEMP_MAX_K)].copy()
    bin_edges = np.arange(TEMP_MIN_K, TEMP_MAX_K + TEMP_BIN_WIDTH_K, TEMP_BIN_WIDTH_K)
    df["temperature_bin"] = pd.cut(
        df["temperature_K"], bins=bin_edges, right=False, labels=bin_edges[:-1]
    ).astype(float)
    return df.dropna(subset=["temperature_bin"]).reset_index(drop=True)


def step4_pivot_wide(long_df):
    """
    Step 4: pivot long to wide.

    One row per (SID, DOI, sample_id, composition, temperature_bin),
    with one column per property (S, sigma, kappa, zT). Where multiple
    digitized points from the same curve land in the same bin, takes
    their mean.
    """
    index_cols = ["SID", "DOI", "sample_id", "composition", "temperature_bin"]
    wide = (
        long_df.groupby(index_cols + ["property"])["value"]
        .mean()
        .unstack("property")
        .reset_index()
    )
    wide.columns.name = None
    for prop in WIDE_PROPERTIES:
        if prop not in wide.columns:
            wide[prop] = np.nan
    return wide


def step5_clean_formulas(wide_df, dopant_threshold_frac=DEFAULT_DOPANT_THRESHOLD_FRAC):
    """
    Step 5: formula cleaning.

    Parses `composition` with pymatgen via
    src/canonicalization.add_canonical_columns and drops rows whose
    formula pymatgen cannot parse. Also attaches composition_id and
    chemistry_cluster_id, needed by steps 8 and 10.
    """
    df = add_canonical_columns(
        wide_df, formula_col="composition", dopant_threshold_frac=dopant_threshold_frac
    )
    return df[df["parse_error"].isna()].drop(columns=["parse_error"]).reset_index(drop=True)


def step6_zt_self_consistency(wide_df, max_rel_error=ZT_SELF_CONSISTENCY_MAX_REL_ERROR):
    """
    Step 6: zT self-consistency check.

    For rows with a reported zT AND all three components (S, sigma,
    kappa) present, computes zT_calc = S^2 * sigma * T / kappa (S
    converted from microV/K back to V/K, T = the row's temperature bin)
    and drops the row if the relative error exceeds max_rel_error. Rows
    missing zT or any component are left untouched -- the check cannot
    be evaluated for them.
    """
    df = wide_df.copy()
    has_all = df[["S", "sigma", "kappa", "zT"]].notna().all(axis=1)
    s_volts = df["S"] / 1.0e6
    zt_calc = (s_volts**2) * df["sigma"] * df["temperature_bin"] / df["kappa"]
    rel_error = (df["zT"] - zt_calc).abs() / df["zT"].abs()
    fails = has_all & (rel_error > max_rel_error)
    return df[~fails].reset_index(drop=True)


def step7_remove_dft(wide_df, papers_df):
    """
    Step 7: DFT-derived data removal.

    Case-insensitive keyword search (DFT_KEYWORDS) over each paper's
    title and container_title via SID; drops all rows belonging to a
    matched paper.
    """
    papers = papers_df.copy()
    text = papers["title"].fillna("").astype(str) + " " + papers["container_title"].fillna("").astype(str)
    pattern = "|".join(re.escape(k) for k in DFT_KEYWORDS)
    is_dft = text.str.contains(pattern, case=False, regex=True, na=False)
    dft_sids = set(papers.loc[is_dft, "SID"])
    return wide_df[~wide_df["SID"].isin(dft_sids)].reset_index(drop=True)


def find_knee_threshold(wide_df, property_name, candidate_thresholds, evaluate_held_out_r2):
    """
    Data-driven knee-finding for the step 8 multi-source consistency CV
    threshold: tighten `candidate_thresholds` (evaluated descending)
    and pick the last threshold before further tightening stops
    improving held-out R^2 -- i.e. the point where it would only be
    removing data without buying anything.

    TODO(Phase 1): `evaluate_held_out_r2(filtered_df) -> float` needs a
    trained model on the chemistry-cluster split, which does not exist
    at this stage of the pipeline (Phase 0 runs before
    src/validation_ladder.py). Until then, callers should use
    MULTI_SOURCE_CV_THRESHOLDS (this module's frozen starting values)
    instead of calling this function, and re-run step 8 with the tuned
    thresholds once Phase 1 exists.
    """
    if evaluate_held_out_r2 is None:
        raise NotImplementedError(
            "find_knee_threshold requires evaluate_held_out_r2 from a trained "
            "model (Phase 1, src/validation_ladder.py) -- not available yet"
        )
    best_threshold = candidate_thresholds[0]
    best_r2 = -np.inf
    for threshold in sorted(candidate_thresholds, reverse=True):
        filtered = step8_multi_source_consistency(wide_df, cv_thresholds={property_name: threshold})
        r2 = evaluate_held_out_r2(filtered)
        if r2 <= best_r2 + 1e-6:  # tightening further stopped helping
            break
        best_threshold, best_r2 = threshold, r2
    return best_threshold


def step8_multi_source_consistency(wide_df, cv_thresholds=None):
    """
    Step 8: multi-source consistency filtering.

    Groups by (composition_id, temperature_bin) -- i.e. across
    different samples/papers reporting nominally the same formula at
    the same temperature -- and computes each property's coefficient of
    variation (std/mean) within the group. If ANY property's CV in a
    group exceeds cv_thresholds[property], every row in that group is
    dropped entirely (row-drop, matching the thesis mechanism -- see
    CLAUDE.md's 2026-08-17 TODO -- not a per-value NaN-out: multiple
    sources disagreeing on even one property means none of that group's
    rows can be trusted).

    Thresholds default to MULTI_SOURCE_CV_THRESHOLDS, the frozen
    starting values -- NOT the final knee-tuned thresholds, see
    find_knee_threshold's TODO.
    """
    cv_thresholds = cv_thresholds or MULTI_SOURCE_CV_THRESHOLDS
    if "composition_id" not in wide_df.columns:
        raise ValueError("run step5_clean_formulas before step8_multi_source_consistency")

    df = wide_df.copy()
    group_cols = ["composition_id", "temperature_bin"]
    exceeds_threshold = pd.Series(False, index=df.index)
    with warnings.catch_warnings():
        # groups that are entirely NaN for `prop` correctly produce a NaN
        # mean/std; numpy warns on that empty-slice reduction every time,
        # which is expected here and not worth surfacing per-group.
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0", category=RuntimeWarning)
        for prop, threshold in cv_thresholds.items():
            if prop not in df.columns:
                continue
            grouped = df.groupby(group_cols)[prop]
            mean = grouped.transform("mean")
            std = grouped.transform("std")
            count = grouped.transform("count")
            cv = (std / mean.abs()).where(count > 1, other=0.0)
            exceeds_threshold |= (cv > threshold).fillna(False)
    return df[~exceeds_threshold].reset_index(drop=True)


def step9_mad_outlier_filter(wide_df, mad_threshold=MAD_THRESHOLD):
    """
    Step 9: MAD outlier filter.

    For S (linear scale) and sigma/kappa (log10 scale), flags a value as
    an outlier if it deviates from the property's global median by more
    than mad_threshold * MAD: |x - median| > mad_threshold * MAD. A row
    with ANY flagged property is dropped entirely (row-drop, matching
    the thesis mechanism -- see CLAUDE.md's 2026-08-17 TODO -- not a
    per-value NaN-out). NOT applied to zT (step 1's [0,4] bound is used
    instead).
    """
    df = wide_df.copy()
    log_props = ("sigma", "kappa")
    linear_props = ("S",)

    is_outlier = pd.Series(False, index=df.index)
    for prop in linear_props + log_props:
        if prop not in df.columns:
            continue
        values = df[prop]
        transformed = np.log10(values) if prop in log_props else values
        valid = transformed.notna() & np.isfinite(transformed)
        if not valid.any():
            continue
        median = transformed[valid].median()
        mad = (transformed[valid] - median).abs().median()
        if mad == 0 or np.isnan(mad):
            continue
        deviation = (transformed - median).abs()
        is_outlier |= valid & (deviation > mad_threshold * mad)
    return df[~is_outlier].reset_index(drop=True)


def step10_min_temperature_coverage(wide_df, min_coverage=MIN_TEMP_COVERAGE):
    """
    Step 10: minimum temperature coverage.

    Drops all rows for a formula (composition_id) that has fewer than
    min_coverage distinct temperature_bin values with at least one
    non-null property remaining.
    """
    has_any_value = wide_df[list(WIDE_PROPERTIES)].notna().any(axis=1)
    valid = wide_df[has_any_value]
    coverage = valid.groupby("composition_id")["temperature_bin"].nunique()
    keep_formulas = coverage[coverage >= min_coverage].index
    return wide_df[wide_df["composition_id"].isin(keep_formulas)].reset_index(drop=True)


def step11_smoothness_filter(wide_df, window=SMOOTHNESS_WINDOW, mad_threshold=MAD_THRESHOLD):
    """
    Step 11: smoothness filter.

    Within each (sample_id, property) series ordered by temperature,
    computes a centered rolling median (window=3) and flags a point as
    a spike (set to NaN) if it deviates from the rolling median by more
    than mad_threshold * the series' own local MAD (reusing step 9's
    3.5xMAD rule, applied along the temperature axis instead of
    globally). Rows where every property has become NaN are dropped.
    """
    df = wide_df.sort_values(["sample_id", "temperature_bin"]).copy()

    def _flag_spikes(values):
        rolling_med = values.rolling(window, center=True, min_periods=1).median()
        deviation = (values - rolling_med).abs()
        mad = deviation.median()
        if mad == 0 or np.isnan(mad):
            return values
        spike = deviation > mad_threshold * mad
        return values.mask(spike)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        for prop in WIDE_PROPERTIES:
            if prop not in df.columns:
                continue
            df[prop] = df.groupby("sample_id", group_keys=False)[prop].apply(_flag_spikes)

    has_any_value = df[list(WIDE_PROPERTIES)].notna().any(axis=1)
    return df[has_any_value].reset_index(drop=True)


def run_cleaning_pipeline(
    raw_data_dir=RAW_DATA_DIR,
    processed_data_dir=PROCESSED_DATA_DIR,
    project=PROJECT,
    extraction_date=None,
    dopant_threshold_frac=DEFAULT_DOPANT_THRESHOLD_FRAC,
):
    """
    Run the full 11-step global cleaning pipeline once, on the raw pull
    acquired by src/data_acquisition.py, and write the cleaned dataset
    to data/processed/. Prints a row-count line after every step so the
    result can be sanity-checked against CLAUDE.md's reference numbers
    (~184,167 rows, ~13,605 unique formulas, ~2,834 parent chemical
    systems from the pipeline's original run).

    Cleaning is applied globally, once, before any train/test split --
    NOT fold-locally (CLAUDE.md Grouping Key, "Global cleaning, not
    fold-local"). The fold-local sensitivity check is a separate,
    later step and is not implemented here.
    """
    curves = load_raw_curves(raw_data_dir, project)
    papers = load_raw_papers(raw_data_dir, project)
    _report("raw curves", curves)

    df = step1_extract_and_filter_properties(curves)
    _report("step1 property extraction & range filtering", df)

    df = step2_integrate_and_consolidate(df)
    _report("step2 integration & consolidation (resistivity->conductivity)", df)

    df = step3_filter_temperature(df)
    _report("step3 temperature filtering 300-800K", df)

    df = step4_pivot_wide(df)
    _report("step4 pivot long->wide", df, key_cols=["sample_id"])

    df = step5_clean_formulas(df, dopant_threshold_frac)
    _report("step5 formula cleaning (pymatgen)", df, key_cols=["composition_id", "chemistry_cluster_id"])

    df = step6_zt_self_consistency(df)
    _report("step6 zT self-consistency check", df)

    df = step7_remove_dft(df, papers)
    _report("step7 DFT data removal", df)

    df = step8_multi_source_consistency(df)
    _report("step8 multi-source consistency filtering [TODO: knee-tune thresholds in Phase 1]", df)

    df = step9_mad_outlier_filter(df)
    _report("step9 MAD outlier filter", df)

    df = step10_min_temperature_coverage(df)
    _report("step10 minimum temperature coverage", df, key_cols=["composition_id"])

    df = step11_smoothness_filter(df)
    _report(
        "step11 smoothness filter",
        df,
        key_cols=["composition_id", "chemistry_cluster_id"],
    )

    processed_data_dir = Path(processed_data_dir)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    date_tag = extraction_date or "unknown-date"
    out_path = processed_data_dir / f"cleaned_{project}_{date_tag}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved cleaned dataset to {out_path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    import yaml

    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_cleaning_pipeline(
        extraction_date=config["extraction"]["starrydata2_date"],
        dopant_threshold_frac=config["clustering"]["dopant_threshold_at_pct"] / 100.0,
    )
