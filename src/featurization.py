"""
Composition-based descriptor featurization.

Computes descriptors once per unique formula, keyed by composition_id
from src/canonicalization.py, then merges back onto every row of the
cleaned dataset -- not once per row (CLAUDE.md's ~13x compute reduction
note; on the current cleaned pull it is closer to 19x, 340,831 rows vs
17,829 unique formulas). Two deliberately different composition-based
feature sets are computed per unique formula: matminer's MAGPIE
elemental-property statistics (Ward et al. 2016) and the CBFV package's
Oliynyk elemental-property statistics (Jha et al.) -- not two
computations of the same thing. CPU only, no GPU required.

CBFV silently drops formulas it cannot featurize instead of raising per
formula, so failures are detected by diffing input vs. output and then
explained by checking which elements are missing from CBFV's reference
property table. Formulas that fail either featurizer are excluded from
the final merged dataset (a partial feature vector isn't usable by most
downstream models) but are logged to a failures CSV and reported --
never silently dropped.
"""

import re
import time
from pathlib import Path

import pandas as pd
from CBFV import composition as cbfv_composition
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.periodic_table import DummySpecies

from src.canonicalization import parse_formula

# pymatgen's reduced_formula renders extremely small stoichiometric
# amounts (e.g. a ~1e-08 trace dopant) in scientific notation, e.g.
# "Ga1e-08Ge1". CBFV's regex-based formula tokenizer cannot parse the
# exponent and either raises outright or, worse, silently mis-tokenizes
# adjacent scientific-notation segments into one another (observed:
# "Si8e-05Bi0.5Sb1.5Te3C8e-05" crashed the whole batch call with
# "e-05e-05 is an invalid formula!"). matminer is unaffected since it
# featurizes the parsed pymatgen Composition object directly, never the
# string form. Pre-filter these out of CBFV's batch input so one
# malformed formula can't take down the whole run.
_SCIENTIFIC_NOTATION_RE = re.compile(r"\d[eE][+-]?\d")

# pymatgen's Composition parser accepts single/multi-letter tokens it
# doesn't recognize as real elements as a DummySpecies placeholder
# instead of raising -- e.g. "M0.125Ba0.125..." parses with element 'M'
# as DummySpecies rather than failing. These are template/generic-site
# notation from the source papers (e.g. "A2B2O7", "M" for a generic
# dopant site, "Ln" for "some lanthanide"), not real chemistry, and
# they crash CBFV's batch call outright ("'M' is not in list") since
# its reference tables only cover real elements. matminer raises a
# catchable KeyError for these instead (handled generically in
# featurize_magpie's except clause). These formulas already having a
# composition_id at all is a step5 (data_cleaning.py) formula-cleaning
# gap worth revisiting separately -- out of scope here, but flagged.

PROCESSED_DATA_DIR = Path("data/processed")
PROJECT = "ThermoelectricMaterials"

MAGPIE_FEATURIZER = ElementProperty.from_preset("magpie")
CBFV_PRESET = "oliynyk"
CBFV_ELEMENT_TABLE_PATH = Path(cbfv_composition.__file__).parent / "element_properties" / f"{CBFV_PRESET}.csv"


def _cbfv_supported_elements():
    """Element symbols covered by CBFV's reference property table for CBFV_PRESET."""
    table = pd.read_csv(CBFV_ELEMENT_TABLE_PATH)
    return set(table["element"])


def get_unique_formulas(df, formula_col="composition_id"):
    """Return the distinct formula identities to featurize, one row per composition_id."""
    return df[[formula_col]].drop_duplicates().reset_index(drop=True)


def featurize_magpie(unique_df, formula_col="composition_id"):
    """
    Compute matminer MAGPIE features for each unique formula.

    Returns (features_df, failures): features_df has one row per
    successfully featurized formula (formula_col + MagpieData columns);
    failures is a list of {"composition_id", "stage", "error"} dicts
    for formulas that failed to parse or to featurize.
    """
    records = []
    failures = []
    feature_labels = MAGPIE_FEATURIZER.feature_labels()

    for formula in unique_df[formula_col]:
        comp, parse_error = parse_formula(formula)
        if comp is None:
            failures.append({"composition_id": formula, "stage": "magpie_parse", "error": parse_error})
            continue
        try:
            values = MAGPIE_FEATURIZER.featurize(comp)
        except Exception as exc:  # matminer raises varied exception types for unsupported elements
            failures.append({"composition_id": formula, "stage": "magpie", "error": f"{type(exc).__name__}: {exc}"})
            continue
        records.append({formula_col: formula, **dict(zip(feature_labels, values))})

    return pd.DataFrame.from_records(records), failures


def _cbfv_failure_reason(comp, supported_elements):
    """Explain why CBFV silently dropped a formula, by checking element coverage."""
    missing_elements = sorted(str(el) for el in comp.elements if str(el) not in supported_elements)
    if missing_elements:
        return (
            f"element(s) {', '.join(missing_elements)} not in CBFV '{CBFV_PRESET}' "
            f"reference table ({len(supported_elements)} elements covered)"
        )
    return "unknown -- CBFV silently dropped this formula despite full element coverage"


def featurize_cbfv(unique_df, formula_col="composition_id", elem_prop=CBFV_PRESET):
    """
    Compute CBFV features (default: Oliynyk elemental-property
    statistics) for each unique formula via one batched call to
    CBFV.composition.generate_features.

    Returns (features_df, failures) in the same shape as
    featurize_magpie. CBFV columns are prefixed "CBFV_" to keep them
    visually distinct from matminer's "MagpieData ..." columns.
    """
    formulas = unique_df[formula_col].tolist()
    failures = []
    batchable = []

    for formula in formulas:
        if _SCIENTIFIC_NOTATION_RE.search(formula):
            failures.append(
                {
                    "composition_id": formula,
                    "stage": "cbfv_parse",
                    "error": (
                        "formula string contains scientific notation from a near-zero trace "
                        "amount (e.g. '8e-05'), which CBFV's formula tokenizer cannot parse"
                    ),
                }
            )
            continue

        comp, parse_error = parse_formula(formula)
        dummy_symbols = [str(el) for el in comp.elements if isinstance(el, DummySpecies)] if comp else []
        if dummy_symbols:
            failures.append(
                {
                    "composition_id": formula,
                    "stage": "cbfv_parse",
                    "error": (
                        f"formula contains placeholder/dummy species {dummy_symbols}, not real "
                        f"chemical elements (likely generic-site notation from the source paper, "
                        f"e.g. 'M', 'A', 'Ln'); CBFV has no reference data for these"
                    ),
                }
            )
            continue

        batchable.append(formula)

    input_df = pd.DataFrame({"formula": batchable, "target": 0.0})
    X, _y, returned_formulae, _skipped = cbfv_composition.generate_features(
        input_df, elem_prop=elem_prop, drop_duplicates=False, extend_features=False
    )
    features_df = X.add_prefix("CBFV_")
    features_df.insert(0, formula_col, returned_formulae.values)

    missing = sorted(set(batchable) - set(returned_formulae))
    supported_elements = _cbfv_supported_elements()
    for formula in missing:
        comp, parse_error = parse_formula(formula)
        if comp is None:
            failures.append({"composition_id": formula, "stage": "cbfv_parse", "error": parse_error})
            continue
        failures.append(
            {"composition_id": formula, "stage": "cbfv", "error": _cbfv_failure_reason(comp, supported_elements)}
        )

    return features_df, failures


def featurize_dataset(cleaned_df, formula_col="composition_id"):
    """
    Compute MAGPIE + CBFV features once per unique formula in
    cleaned_df, merge them onto every row of cleaned_df by formula_col,
    and return (featurized_df, failures_df, timing_dict).

    Rows whose formula failed either featurizer are excluded from
    featurized_df (an inner merge on formula_col does this
    automatically) -- but every failure is recorded in failures_df,
    including how many rows of cleaned_df that formula would have
    affected, so nothing is silently dropped.
    """
    timing = {}
    unique_df = get_unique_formulas(cleaned_df, formula_col)

    t0 = time.perf_counter()
    magpie_df, magpie_failures = featurize_magpie(unique_df, formula_col)
    timing["magpie_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cbfv_df, cbfv_failures = featurize_cbfv(unique_df, formula_col)
    timing["cbfv_seconds"] = time.perf_counter() - t0

    combined_features = magpie_df.merge(cbfv_df, on=formula_col, how="inner")

    all_failures = magpie_failures + cbfv_failures
    # a formula that succeeded magpie but failed cbfv (or vice versa) is
    # excluded by the inner merge above but only shows up in one of the
    # two failure lists -- make that "excluded overall" status explicit.
    succeeded_both = set(combined_features[formula_col])
    for formula in unique_df[formula_col]:
        if formula not in succeeded_both and not any(f["composition_id"] == formula for f in all_failures):
            all_failures.append({"composition_id": formula, "stage": "excluded_by_partner_stage", "error": None})

    row_counts = cleaned_df[formula_col].value_counts()
    failures_df = pd.DataFrame(all_failures)
    if not failures_df.empty:
        failures_df["n_rows_affected"] = failures_df["composition_id"].map(row_counts).fillna(0).astype(int)

    featurized_df = cleaned_df.merge(combined_features, on=formula_col, how="inner")

    timing["total_seconds"] = timing["magpie_seconds"] + timing["cbfv_seconds"]
    timing["n_unique_formulas"] = len(unique_df)
    timing["n_featurized_formulas"] = len(combined_features)
    timing["n_feature_columns"] = len(combined_features.columns) - 1  # exclude formula_col

    return featurized_df, failures_df, timing


def run_featurization(processed_data_dir=PROCESSED_DATA_DIR, project=PROJECT, extraction_date=None):
    """
    Load the most recent cleaned_<project>_*.csv, featurize it, save the
    featurized dataset and a failures log to data/processed/, and print
    a summary report.
    """
    processed_data_dir = Path(processed_data_dir)
    candidates = sorted(processed_data_dir.glob(f"cleaned_{project}_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No cleaned_{project}_*.csv in {processed_data_dir}; run src/data_cleaning.py first")
    cleaned_path = candidates[-1]
    cleaned_df = pd.read_csv(cleaned_path)
    print(f"Loaded {len(cleaned_df):,} rows from {cleaned_path}")

    featurized_df, failures_df, timing = featurize_dataset(cleaned_df)

    date_tag = extraction_date or "unknown-date"
    out_path = processed_data_dir / f"featurized_{project}_{date_tag}.csv"
    featurized_df.to_csv(out_path, index=False)

    failures_path = processed_data_dir / f"featurization_failures_{project}_{date_tag}.csv"
    failures_df.to_csv(failures_path, index=False)

    n_rows_dropped = len(cleaned_df) - len(featurized_df)
    n_unique_failed = timing["n_unique_formulas"] - timing["n_featurized_formulas"]
    n_failure_events = len(failures_df)
    print(f"\n=== Featurization report ===")
    print(f"Unique formulas: {timing['n_unique_formulas']:,}")
    print(f"Successfully featurized (both MAGPIE + CBFV): {timing['n_featurized_formulas']:,}")
    print(f"Failed formulas (unique, excluded from output): {n_unique_failed:,}")
    if n_failure_events != n_unique_failed:
        print(
            f"  ({n_failure_events:,} failure log entries -- some formulas fail more than one "
            f"stage, e.g. a placeholder/dummy-species formula fails both MAGPIE and CBFV)"
        )
    print(f"Feature columns: {timing['n_feature_columns']:,}")
    print(f"MAGPIE time: {timing['magpie_seconds']:.1f}s")
    print(f"CBFV time: {timing['cbfv_seconds']:.1f}s")
    print(f"Total featurization time: {timing['total_seconds']:.1f}s")
    print(f"Rows dropped due to failed formulas: {n_rows_dropped:,} of {len(cleaned_df):,}")
    print(f"Saved featurized dataset to {out_path} ({len(featurized_df):,} rows)")
    print(f"Saved failures log to {failures_path} ({n_failure_events:,} entries)")

    if not failures_df.empty:
        print("\nFailure log entries by stage:")
        print(failures_df["stage"].value_counts().to_string())
        n_both_stages = failures_df["composition_id"].duplicated().sum()
        if n_both_stages:
            print(f"({n_both_stages:,} formulas appear in more than one stage's failure log)")

    return featurized_df, failures_df, timing


if __name__ == "__main__":
    import yaml

    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_featurization(extraction_date=config["extraction"]["starrydata2_date"])
