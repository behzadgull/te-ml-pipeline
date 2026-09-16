"""
Composition canonicalization and chemistry-cluster definition.

Canonicalizes raw formulas to a comparable composition representation,
then derives the frozen chemistry-cluster grouping key: reduced
host-lattice stoichiometry with dopants below the 5 at% threshold
collapsed into the parent. Also supports the looser/stricter threshold
variants used for the 3-row sensitivity table.

Establishes the Chemistry cluster >= Composition >= Sample hierarchy
used by every grouped split in the project. All functions here are
pure and take/return pandas Series or pymatgen Composition objects;
this module has no I/O and is meant to be imported by data_cleaning.py
and later by validation_ladder.py / lofo_paperb.py etc.
"""

from pymatgen.core import Composition
from pymatgen.core.composition import CompositionError

# Fraction of total atoms (0-1), NOT a percentage -- 0.05 is the frozen
# 5 at% threshold. config.yaml's clustering.dopant_threshold_at_pct is
# stored as a percentage (5.0); divide by 100 before passing it here.
DEFAULT_DOPANT_THRESHOLD_FRAC = 0.05

# Relative tolerance for snapping a host amount to the nearest integer in
# chemistry_cluster_id() -- see SNAP(T) in that function's docstring.
# Selected by acceptance tests (2026-09): T=0.05 is the smallest tested
# value (0.01, 0.02, 0.03, 0.05, 0.08 were evaluated) that merges
# Pb0.97Te1 with PbTe and Co3.95Sb12 with CoSb3, while still leaving
# Bi2Te2.7Se0.3 distinct from Bi2Te3, elemental Te distinct from
# Sn0.24775Te1Pb0.24775-style SnPbTe alloys, elemental Fe distinct from
# V0.0693Cr0.0776Fe0.693C0.0966-style Fe-based alloys, and
# Hf0.167Zr0.167Ta0.167Ti0.167Nb0.167V0.167Co1Sb1 (a six-element
# high-entropy alloy) distinct from CoSb.
DEFAULT_SNAP_TOLERANCE = 0.05


def parse_formula(formula):
    """
    Parse a raw formula string with pymatgen's Composition class.

    Returns (Composition, None) on success, or (None, error_message) if
    the formula is empty, not a string, or unparseable by pymatgen.
    """
    if not isinstance(formula, str) or not formula.strip():
        return None, "empty or non-string formula"
    try:
        comp = Composition(formula)
    except (CompositionError, ValueError) as exc:
        return None, str(exc)
    if comp.num_atoms <= 0:
        return None, "zero-atom composition"
    return comp, None


def composition_id(comp):
    """
    Composition-level identity: the full canonicalized formula, reduced
    to its simplest integer ratio, with no dopant collapsing. Two
    formulas that differ only by a stoichiometric scale factor (e.g.
    "Pb1Te1" vs "Pb2Te2") map to the same composition_id.
    """
    return comp.reduced_formula


def chemistry_cluster_id(
    comp,
    dopant_threshold_frac=DEFAULT_DOPANT_THRESHOLD_FRAC,
    snap_tolerance=DEFAULT_SNAP_TOLERANCE,
):
    """
    Chemistry-cluster identity (FROZEN definition, CLAUDE.md Grouping
    Key): the reduced host-lattice stoichiometry with dopant elements
    below `dopant_threshold_frac` collapsed into the parent. Elements
    at or above the threshold are the host lattice; elements below it
    are dopants and do not split the cluster.

    Host elements are kept at their original (pre-dopant-removal)
    amounts, each SNAPPED to the nearest integer if it is within
    `snap_tolerance` (relative) of that integer -- see
    DEFAULT_SNAP_TOLERANCE -- and otherwise left as-is, before taking
    reduced_formula. This snap step is required: pymatgen's
    reduced_formula does NOT, on its own, reduce near-integer host
    amounts to an integer ratio -- an earlier version of this docstring
    claimed it did ("reduced to an integer ratio"), which was never
    true (Composition({"Pb": 0.97, "Te": 1.0}).reduced_formula returns
    "Te1Pb0.97", not "TePb") and is what let that gap go undetected:
    two rows differing only by measurement/digitization noise (e.g.
    Pb0.97Te1 vs PbTe) received different cluster ids instead of
    merging, fragmenting 10,222 of 12,036 clusters (79.3% of rows,
    File A) into decimal-suffixed near-duplicates of an integer
    neighbor. The explicit snap here is what actually merges them.

    Falls back to the full composition_id if every element in the
    formula falls below the threshold (e.g. a high-entropy composition
    with no single dominant element) -- this is a degenerate case for
    the 5 at% definition, not a silent default; callers doing the
    sensitivity-table analysis should check for this via
    `is_degenerate_cluster`.
    """
    total_atoms = comp.num_atoms
    host_amounts = {
        el: amt for el, amt in comp.get_el_amt_dict().items()
        if amt / total_atoms >= dopant_threshold_frac
    }
    if not host_amounts:
        return comp.reduced_formula
    snapped_amounts = {}
    for el, amt in host_amounts.items():
        nearest = round(amt)
        if nearest >= 1 and abs(amt - nearest) / nearest <= snap_tolerance:
            snapped_amounts[el] = float(nearest)
        else:
            snapped_amounts[el] = amt
    host_comp = Composition(snapped_amounts)
    return host_comp.reduced_formula


def is_degenerate_cluster(comp, dopant_threshold_frac=DEFAULT_DOPANT_THRESHOLD_FRAC):
    """True if no element in `comp` reaches the dopant threshold, i.e.
    chemistry_cluster_id fell back to the full composition instead of
    collapsing any dopants."""
    total_atoms = comp.num_atoms
    return not any(
        amt / total_atoms >= dopant_threshold_frac
        for amt in comp.get_el_amt_dict().values()
    )


def sample_id(raw_sample_id):
    """
    Sample-level identity: passthrough of the raw Starrydata2 sample_id
    (the finest level of the Chemistry cluster >= Composition >= Sample
    hierarchy -- one row per physically distinct measured sample).
    """
    return raw_sample_id


def add_canonical_columns(
    df,
    formula_col="composition",
    sample_id_col="sample_id",
    dopant_threshold_frac=DEFAULT_DOPANT_THRESHOLD_FRAC,
):
    """
    Add composition_id, chemistry_cluster_id, sample_id, and a
    parse_error column (None on success) to a copy of `df`, derived
    from `df[formula_col]`. Rows whose formula fails to parse get
    composition_id/chemistry_cluster_id set to None and parse_error set
    to pymatgen's error message; callers are responsible for dropping
    those rows (data_cleaning.py step 5).
    """
    out = df.copy()
    composition_ids = []
    cluster_ids = []
    parse_errors = []
    for formula in out[formula_col]:
        comp, error = parse_formula(formula)
        if comp is None:
            composition_ids.append(None)
            cluster_ids.append(None)
            parse_errors.append(error)
        else:
            composition_ids.append(composition_id(comp))
            cluster_ids.append(chemistry_cluster_id(comp, dopant_threshold_frac))
            parse_errors.append(None)

    out["composition_id"] = composition_ids
    out["chemistry_cluster_id"] = cluster_ids
    out["parse_error"] = parse_errors
    out["sample_id"] = out[sample_id_col].map(sample_id)
    return out
