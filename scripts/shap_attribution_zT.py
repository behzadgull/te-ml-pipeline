"""
Compare feature attributions between an XGBoost zT model fit under
split_strategy="random" and one fit under split_strategy="chemistry",
across all FIVE outer folds of repeat 0, seed 0 -- the same
fold-construction path src/nested_cv.py's run_nested_cv() uses for the
ladder itself (see _build_repeat0_folds below for exactly how its two
RNG draws are replayed).

Why five folds, not one: with a single fold there is no way to
distinguish a genuine attribution shift (the leakage-mechanism
hypothesis) from ordinary fold-to-fold variation in which rows happen to
land in train vs. test. Five folds gives an actual spread -- an
across-fold SD per group per model -- to read the random-vs-chemistry
delta against, the same reason CLAUDE.md's ladder itself reports
mean +/- across-repeat SD rather than a single split's number (see
Paper A item 1's per-repeat pooling note).

GOAL: test whether random-split R^2 inflation (CLAUDE.md's Five-Way
Ladder, Paper A item 1) is accompanied by a shift in feature credit
toward chemistry-fingerprint features (periodic-table position,
electronegativity, atomic radius -- the near-duplicate-identifying
descriptors) rather than toward the physically causal descriptors zT
should depend on. Supports Paper A Contribution #1: mechanism, not just
magnitude, behind the ladder's inflation gap.

BUILD-AND-VERIFY ONLY: this file has been syntax-checked (py_compile)
but never executed. No fit has been run.

Does NOT modify src/ -- imports src.nested_cv's existing
load_target_data, get_feature_columns, outer_splits, MODEL_REGISTRY,
_target_scale/_transform_target, _to_device/_to_host, and
_load_frozen_hyperparams unchanged. Fold construction is NOT
reimplemented: _build_repeat0_folds() below replays the exact two RNG
draws run_nested_cv() makes before its fold loop body executes for
repeat 0, then calls the SAME outer_splits() generator run_nested_cv()
calls, collecting all n_outer_folds folds it yields -- so the five
(train_idx, test_idx) pairs produced here are bit-identical to what a
real run_nested_cv(..., seed=seed) call would use for repeat 0's five
folds. See _build_repeat0_folds's docstring for the line-by-line
correspondence.

Does NOT use the `shap` package -- Kaggle's shap==0.51.0 requires
numpy>=2, but this project's requirements.txt pins numpy==1.26.4 (the
same numpy-2 incompatibility pattern noted for cupy in
src/nested_cv.py's _to_device docstring). Uses XGBoost's own native
exact TreeSHAP instead: Booster.predict(dmatrix, pred_contribs=True)
needs no shap import and works against the pinned numpy.

Device defaults to "cuda": the frozen zT hyperparameters
(checkpoints/saved_predictions/checkpoints/frozen_hyperparams/zT.json)
were tuned under device="cuda" on Kaggle, and every other confirmed
number in CLAUDE.md comes from cuda runs -- fitting these ten models on
CPU instead would introduce a gratuitous device difference on top of
the split_strategy difference actually under study. --device cpu
remains selectable for a local smoke test (this machine has no GPU, see
CLAUDE.md Local dev environment); TreeSHAP's pred_contribs computation
itself runs through xgboost's tree-traversal code regardless of device
and is not expected to be meaningfully faster or slower under cuda.

CAVEAT (recorded here, not only in CLAUDE.md's future Descriptor
Attribution section): the random-split model and the chemistry-cluster
model are trained on DIFFERENT training sets, by construction -- the
random-split model's training rows include near-duplicates (same or a
very similar chemistry_cluster_id) of its own test rows, which is
exactly the leakage mechanism under study. This script therefore
compares two different fitted models per fold, not an ablation of one
fixed model holding training data constant. Averaging over five folds
addresses fold-to-fold sampling variation within each split_strategy;
it does NOT eliminate this train-set-composition confound, which is
inherent to comparing the two split strategies at all.

Sign convention (D7's comparison table): delta = chemistry_mean_share -
random_mean_share. Positive delta = a group gets MORE attribution
credit under the honest chemistry-cluster split; negative = more credit
under the leaky random split (the direction the inflation-mechanism
hypothesis predicts for chemistry-fingerprint groups). "delta in units
of pooled SD" divides delta by the pooled across-fold SD of the two
five-fold samples being compared -- pooled_sd = sqrt((sd_random^2 +
sd_chemistry^2) / 2), the standard equal-n (n=5 each) two-sample pooled
SD -- so it reads as a Cohen's-d-style effect size on the SAME footing
CLAUDE.md's descriptor-ablation section already uses to caution that
"N times the SD" answers "distinguishable from fold noise", not "how
much does this matter" -- report the absolute delta as the primary
number, this ratio as a secondary distinguishability check.

Writes only under results/shap_attribution/<timestamp>/ -- does not
touch any ladder checkpoint, src/, or CLAUDE.md.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ is one level below the repo root
sys.path.insert(0, str(REPO_ROOT))

from src.nested_cv import (  # noqa: E402
    GROUP_COL,
    MODEL_REGISTRY,
    N_OUTER_FOLDS,
    TEMPERATURE_COL,
    _load_frozen_hyperparams,
    _target_scale,
    _to_device,
    _to_host,
    _transform_target,
    get_feature_columns,
    load_target_data,
    outer_splits,
)

TARGET = "zT"  # hard constraint: zT only
FEATURE_SET = "full"  # hard constraint: full 397-feature set
SPLIT_STRATEGIES_TO_COMPARE = ("random", "chemistry")
DEFAULT_FROZEN_HYPERPARAMS = (
    REPO_ROOT / "checkpoints" / "saved_predictions" / "checkpoints" / "frozen_hyperparams" / "zT.json"
)
DEFAULT_SEED = 0  # matches run_nested_cv's default seed and the ladder's own seed=0 runs
DEFAULT_SHAP_N = 20_000
DEFAULT_SHAP_SEED = 0


# ---------------------------------------------------------------------------
# D1: exact repeat-0, all-folds replication, reusing outer_splits unchanged.
# ---------------------------------------------------------------------------

def _build_repeat0_folds(split_strategy, n_rows, group_lookup, seed=DEFAULT_SEED, n_outer_folds=N_OUTER_FOLDS):
    """
    Reproduce run_nested_cv's exact (train_idx, test_idx) pairs for ALL
    n_outer_folds folds of repeat=0, given `seed`, under `split_strategy`
    -- WITHOUT calling run_nested_cv itself (which would also
    retune/fit/checkpoint).

    run_nested_cv's relevant body (src/nested_cv.py, ~line 1139-1145):
        rng_master = np.random.default_rng(seed)
        for repeat in range(n_repeats):
            repeat_rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            fold_iter = outer_splits(split_strategy, len(df), group_lookup, n_outer_folds, repeat_rng)
            for fold, (train_idx, test_idx) in enumerate(fold_iter):
                ...

    For repeat=0 this is exactly: draw ONE integer from rng_master to
    seed repeat_rng, build the SAME fold_iter generator via the SAME
    outer_splits() call, then take all n_outer_folds splits it yields,
    in order (fold=0..n_outer_folds-1). Nothing about how these folds
    are produced depends on any later repeat, so this is exact, not an
    approximation:
      - split_strategy in {"chemistry", "composition"}: outer_splits
        delegates to randomized_group_kfold, which computes all
        n_outer_folds fold assignments from ONE rng.permutation call
        up front, then yields them one at a time -- collecting all of
        them changes nothing about how they were computed.
      - split_strategy in {"random", "kfold"}: outer_splits draws ONE
        seed via rng.integers(...) and builds one sklearn splitter
        (ShuffleSplit / KFold) with that fixed random_state, then
        yields n_outer_folds splits from it -- collecting all of them
        is exactly what run_nested_cv's repeat-0 fold loop receives.

    Returns a list of (train_idx, test_idx) tuples, length n_outer_folds.
    """
    rng_master = np.random.default_rng(seed)
    repeat_rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))  # repeat=0
    fold_iter = outer_splits(split_strategy, n_rows, group_lookup, n_outer_folds, repeat_rng)
    return list(fold_iter)  # folds 0..n_outer_folds-1, in order


# ---------------------------------------------------------------------------
# D6: coarse (source) and fine (descriptor-semantic) feature grouping.
# ---------------------------------------------------------------------------

def _coarse_group_for_column(colname):
    """MagpieData* -> "magpie", CBFV_* -> "cbfv", temperature_bin -> "temperature"."""
    if colname == TEMPERATURE_COL:
        return "temperature"
    if colname.startswith("MagpieData"):
        return "magpie"
    if colname.startswith("CBFV_"):
        return "cbfv"
    raise ValueError(f"Unrecognized feature column {colname!r} for coarse grouping")


def _strip_stat_prefix(colname):
    """
    Split one MagpieData/CBFV_ feature column into (source, stat, raw
    property suffix). MAGPIE columns are "MagpieData <stat> <property>"
    (space-separated; e.g. "MagpieData avg_dev Number" -> stat="avg_dev",
    property="Number" -- "avg_dev" itself contains an underscore but no
    space, so splitting on the FIRST space is unambiguous). CBFV columns
    are "CBFV_<stat>_<property>" (underscore-separated; e.g.
    "CBFV_avg_1st_ionization_potential_(kJ/mol)" -> stat="avg",
    property="1st_ionization_potential_(kJ/mol)" -- none of CBFV's six
    stat tokens {avg, dev, range, max, min, mode} contain an underscore,
    so splitting on the FIRST underscore is unambiguous even though many
    property names do). Both stat vocabularies verified directly against
    the real featurized CSV header (2026-09-13): 22 MAGPIE properties x
    6 stats = 132 columns, 44 CBFV properties x 6 stats = 264 columns,
    132 + 264 + 1 (temperature_bin) = 397, matching the frozen full
    feature-set count exactly.
    """
    if colname.startswith("MagpieData "):
        rest = colname[len("MagpieData "):]
        stat, _, prop = rest.partition(" ")
        return "magpie", stat, prop
    if colname.startswith("CBFV_"):
        rest = colname[len("CBFV_"):]
        stat, _, prop = rest.partition("_")
        return "cbfv", stat, prop
    return None, None, colname  # temperature_bin: no stat prefix


def _normalize_property_name(name):
    """
    Collapse a raw property suffix (e.g. "Miracle_Radius_[pm]",
    "thermal_conductivity_(W/(m_K))_", "AtomicWeight") to a bare
    lowercase alphanumeric key, stripping any parenthetical or
    bracketed unit annotation (including the doubly-nested
    "(W/(m_K))" case) so magpie- and CBFV-style spellings of the same
    physical quantity can be looked up against one shared dict.
    """
    s = name
    while "(" in s:
        s = re.sub(r"\([^()]*\)", "", s)
    s = re.sub(r"\[[^\]]*\]", "", s)
    s = re.sub(r"[^A-Za-z0-9]", "", s).lower()
    return s


# Every raw property suffix that appears in the real featurized CSV's
# 22 MAGPIE + 44 CBFV base properties (enumerated directly from the
# header on 2026-09-13, not guessed), grouped by descriptor semantics.
# The mapping is exhaustive over both vocabularies: 6+2+7+23+2+10+3+10+3
# = 66 raw names = 22 + 44. Any column whose property does not resolve
# to one of these keys falls into "other" (see _fine_group_for_column);
# main() reports that count explicitly rather than assuming it is zero.
FINE_GROUP_MEMBERS = {
    "electronegativity": [
        "Electronegativity", "Pauling_Electronegativity", "MB_electonegativity",
        "Gordy_electonegativity", "Mulliken_EN", "Allred-Rockow_electronegativity",
    ],
    "atomic_mass": ["AtomicWeight", "Atomic_Weight"],
    "atomic_radius": [
        "CovalentRadius", "Atomic_Radius", "Miracle_Radius_[pm]", "Covalent_Radius",
        "Zunger_radii_sum", "ionic_radius", "crystal_radius",
    ],
    "valence_electron_config": [
        "NValence", "NdValence", "NfValence", "NpValence", "NsValence",
        "NUnfilled", "NdUnfilled", "NfUnfilled", "NpUnfilled", "NsUnfilled",
        "number_of_valence_electrons", "gilmor_number_of_valence_electron",
        "valence_s", "valence_p", "valence_d", "valence_f",
        "Number_of_unfilled_s_valence_electrons", "Number_of_unfilled_p_valence_electrons",
        "Number_of_unfilled_d_valence_electrons", "Number_of_unfilled_f_valence_electrons",
        "outer_shell_electrons", "metallic_valence", "l_quantum_number",
    ],
    "melting_point": ["MeltingT", "Melting_point_(K)"],
    "periodic_position": [
        "Column", "Row", "MendeleevNumber", "Number", "SpaceGroupNumber",
        "Period", "group", "families", "Mendeleev_Number", "Atomic_Number",
    ],
    "dft_groundstate": ["GSbandgap", "GSmagmom", "GSvolume_pa"],
    "thermodynamic_bulk": [
        "1st_ionization_potential_(kJ/mol)", "polarizability(A^3)", "Boiling_Point_(K)",
        "Density_(g/mL)", "specific_heat_(J/g_K)_", "heat_of_fusion_(kJ/mol)_",
        "heat_of_vaporization_(kJ/mol)_", "thermal_conductivity_(W/(m_K))_",
        "heat_atomization(kJ/mol)", "Cohesive_energy",
    ],
    "metal_class": ["Metal", "Nonmetal", "Metalliod"],
}

_PROPERTY_TO_FINE_GROUP = {
    _normalize_property_name(raw): group
    for group, raw_names in FINE_GROUP_MEMBERS.items()
    for raw in raw_names
}


def _fine_group_for_column(colname, unmapped_log):
    """
    Fine descriptor-semantic group for one feature column. Appends
    (colname, raw_property) to unmapped_log for anything that resolves
    to "other" so main() can report exactly which columns and how many.
    """
    if colname == TEMPERATURE_COL:
        return "temperature"
    _, _, prop = _strip_stat_prefix(colname)
    key = _normalize_property_name(prop)
    group = _PROPERTY_TO_FINE_GROUP.get(key)
    if group is None:
        unmapped_log.append((colname, prop))
        return "other"
    return group


# ---------------------------------------------------------------------------
# Provenance helper (same pattern as scripts/noise_floor_fileA.py).
# ---------------------------------------------------------------------------

def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# D1-D5, D8: fit and attribute ONE (split_strategy, fold) pair.
# ---------------------------------------------------------------------------

def _fit_and_attribute_one_fold(
    split_strategy, fold, train_idx, test_idx, X_full, y_full, feature_cols,
    best_params, device, shap_n, shap_seed,
):
    """
    D1: fit one frozen-hyperparameter XGBoost model on this fold's
    training rows (train_idx/test_idx already built by
    _build_repeat0_folds -- fold construction is not redone here).

    D2: outer R^2 on this fold's own test rows.

    D3: exact TreeSHAP (Booster.predict(..., pred_contribs=True)) on a
    fixed, seeded random subsample of THIS fold's test rows (default
    20,000 -- fewer only if the test fold itself is smaller). shap_seed
    is reused unchanged across all 10 (split_strategy, fold) pairs, so
    the subsample-selection RNG draw is not itself a source of
    fold-to-fold variation in the comparison.

    D4: per-feature mean(|SHAP|), raw and normalized to shares.

    D5: gain-based importance from the SAME booster, aligned to
    feature_cols by position ("f<i>" -> feature_cols[i]) -- X was fit as
    a plain numpy array (no column names), so the booster's own feature
    names are exactly "f0".."f396" in feature_cols order.

    Returns a dict with everything D8 needs to reconstruct a figure
    later without refitting, for this one (split_strategy, fold) pair.
    """
    n_features = len(feature_cols)

    X_train, y_train = X_full[train_idx], y_full[train_idx]
    X_test, y_test = X_full[test_idx], y_full[test_idx]

    model = MODEL_REGISTRY["xgboost"]["build"](best_params, device)
    t_fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t_fit_start

    y_pred_test = _to_host(model.predict(X_test))
    y_test_host = _to_host(y_test)
    outer_r2 = float(r2_score(y_test_host, y_pred_test))

    # D3: fixed, seeded subsample of THIS fold's test set.
    shap_rng = np.random.default_rng(shap_seed)
    n_shap_used = min(shap_n, len(test_idx))
    sub_pos = shap_rng.choice(len(test_idx), size=n_shap_used, replace=False)
    X_shap = _to_host(X_test)[sub_pos]

    booster = model.get_booster()
    t_shap_start = time.perf_counter()
    dmat = xgb.DMatrix(X_shap)
    contribs = booster.predict(dmat, pred_contribs=True)  # shape (n_shap_used, n_features + 1)
    shap_seconds = time.perf_counter() - t_shap_start

    shap_values = contribs[:, :-1]  # drop the trailing bias/expected-value column
    bias = contribs[:, -1]

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)  # raw, length n_features
    shap_total = mean_abs_shap.sum()
    shap_share = mean_abs_shap / shap_total if shap_total > 0 else np.zeros(n_features)

    # D5: gain-based cross-check, aligned to feature_cols by position.
    gain_dict = booster.get_score(importance_type="gain")  # {"f<i>": gain, ...}; missing = never split on
    gain_importance = np.zeros(n_features)
    for fname, val in gain_dict.items():
        idx = int(fname[1:])
        gain_importance[idx] = val
    gain_total = gain_importance.sum()
    gain_share = gain_importance / gain_total if gain_total > 0 else np.zeros(n_features)

    return {
        "split_strategy": split_strategy,
        "fold": fold,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "outer_r2": outer_r2,
        "fit_seconds": fit_seconds,
        "shap_n_requested": int(shap_n),
        "shap_n_used": int(n_shap_used),
        "shap_seed": int(shap_seed),
        "shap_seconds": shap_seconds,
        "mean_abs_shap": mean_abs_shap,
        "shap_share": shap_share,
        "shap_bias_mean": float(np.mean(bias)),
        "gain_importance": gain_importance,
        "gain_share": gain_share,
        # D8: enough to rebuild everything downstream without refitting.
        "train_idx": np.asarray(train_idx),
        "test_idx": np.asarray(test_idx),
        "shap_subsample_positions": sub_pos,  # positions WITHIN test_idx, not row ids
        "y_true_test": y_test_host,
        "y_pred_test": y_pred_test,
    }


def fit_and_attribute_all_folds(
    split_strategy, X_full, y_full, feature_cols, group_lookup,
    best_params, seed, n_outer_folds, device, shap_n, shap_seed,
):
    """
    D1-D5 across ALL n_outer_folds folds of repeat 0 for one
    split_strategy: builds the five folds via _build_repeat0_folds (no
    fold logic reimplemented), fits and attributes each one via
    _fit_and_attribute_one_fold, then adds the pooled out-of-fold R^2
    across all five folds' held-out predictions -- the SAME "pooled
    out-of-fold R^2" definition run_nested_cv() itself reports as the
    ladder's primary metric (CLAUDE.md Paper A item 1), so this
    artifact's R^2 numbers are directly comparable to the ladder table.

    Returns (fold_results, pooled_r2) -- fold_results is a list of five
    per-fold dicts (see _fit_and_attribute_one_fold), in fold order.
    """
    n_rows = X_full.shape[0]
    folds = _build_repeat0_folds(split_strategy, n_rows, group_lookup, seed=seed, n_outer_folds=n_outer_folds)

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(folds):
        fold_results.append(
            _fit_and_attribute_one_fold(
                split_strategy, fold, train_idx, test_idx, X_full, y_full, feature_cols,
                best_params, device, shap_n, shap_seed,
            )
        )

    pooled_y_true = np.concatenate([r["y_true_test"] for r in fold_results])
    pooled_y_pred = np.concatenate([r["y_pred_test"] for r in fold_results])
    pooled_r2 = float(r2_score(pooled_y_true, pooled_y_pred))

    return fold_results, pooled_r2


# ---------------------------------------------------------------------------
# D6-D7: aggregate shares to group level (mean +/- across-fold SD) and
# build the comparison table.
# ---------------------------------------------------------------------------

def _grouped_share_per_fold(feature_cols, group_of, fold_results):
    """
    For one split_strategy's list of per-fold results, return
    {group: np.array of length n_outer_folds} -- each fold's
    shap_share summed within that group.
    """
    per_group = {}
    for res in fold_results:
        fold_groups = {}
        for group, share in zip(group_of, res["shap_share"]):
            fold_groups[group] = fold_groups.get(group, 0.0) + float(share)
        for group, share in fold_groups.items():
            per_group.setdefault(group, []).append(share)
    return {group: np.array(vals) for group, vals in per_group.items()}


def _comparison_rows(shares_by_strategy):
    """
    D7: group | random mean +/- SD | chemistry mean +/- SD | delta
    (chemistry mean - random mean) | delta in units of pooled SD.

    Across-fold SD uses ddof=1 (sample SD over 5 folds), matching the
    ladder's own per_repeat_r2_std convention. pooled_sd = sqrt((sd_r^2
    + sd_c^2) / 2) -- the standard equal-n (n=5 each) two-sample pooled
    SD; see module docstring for why this is reported as a secondary
    distinguishability check, not the primary number. Sorted by |delta|
    descending.
    """
    all_groups = sorted(set(shares_by_strategy["random"]) | set(shares_by_strategy["chemistry"]))
    rows = []
    for group in all_groups:
        r_vals = shares_by_strategy["random"].get(group, np.zeros(0))
        c_vals = shares_by_strategy["chemistry"].get(group, np.zeros(0))
        r_mean, r_sd = float(r_vals.mean()), float(r_vals.std(ddof=1)) if len(r_vals) > 1 else 0.0
        c_mean, c_sd = float(c_vals.mean()), float(c_vals.std(ddof=1)) if len(c_vals) > 1 else 0.0
        delta = c_mean - r_mean
        pooled_sd = float(np.sqrt((r_sd**2 + c_sd**2) / 2))
        delta_in_pooled_sd = (delta / pooled_sd) if pooled_sd > 0 else float("inf")
        rows.append({
            "group": group,
            "random_mean_share": r_mean, "random_sd": r_sd, "random_fold_values": r_vals.tolist(),
            "chemistry_mean_share": c_mean, "chemistry_sd": c_sd, "chemistry_fold_values": c_vals.tolist(),
            "delta": delta, "pooled_sd": pooled_sd, "delta_in_pooled_sd_units": delta_in_pooled_sd,
        })
    rows.sort(key=lambda row: abs(row["delta"]), reverse=True)
    return rows


def aggregate_and_compare(feature_cols, results_by_strategy):
    """
    D6: aggregate per-feature normalized SHAP shares to (a) coarse
    source groups (magpie/cbfv/temperature) and (b) fine descriptor-
    semantic groups (see FINE_GROUP_MEMBERS), per fold, for each
    split_strategy, then reduce each group's 5 fold values to
    mean +/- across-fold SD (D6-D7).

    results_by_strategy: {"random": fold_results, "chemistry": fold_results}
    -- lists of 5 per-fold dicts each (see fit_and_attribute_all_folds).

    Returns (coarse_rows, fine_rows, unmapped).
    """
    unmapped = []
    coarse_group_of = [_coarse_group_for_column(c) for c in feature_cols]
    fine_group_of = [_fine_group_for_column(c, unmapped) for c in feature_cols]

    coarse_shares = {
        strategy: _grouped_share_per_fold(feature_cols, coarse_group_of, fold_results)
        for strategy, fold_results in results_by_strategy.items()
    }
    fine_shares = {
        strategy: _grouped_share_per_fold(feature_cols, fine_group_of, fold_results)
        for strategy, fold_results in results_by_strategy.items()
    }

    coarse_rows = _comparison_rows(coarse_shares)
    fine_rows = _comparison_rows(fine_shares)
    return coarse_rows, fine_rows, unmapped


def _rows_to_markdown(rows, title):
    lines = [
        f"### {title}", "",
        "| group | random mean share +/- SD | chemistry mean share +/- SD | delta (chem-rand) | delta / pooled SD |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        ratio_str = f"{row['delta_in_pooled_sd_units']:+.2f}" if np.isfinite(row["delta_in_pooled_sd_units"]) else "inf"
        lines.append(
            f"| {row['group']} | {row['random_mean_share']:.4f} +/- {row['random_sd']:.4f} | "
            f"{row['chemistry_mean_share']:.4f} +/- {row['chemistry_sd']:.4f} | "
            f"{row['delta']:+.4f} | {ratio_str} |"
        )
    return "\n".join(lines)


def _r2_table_markdown(results_by_strategy, pooled_r2_by_strategy):
    lines = ["## Outer R^2 per fold and pooled (D2)", "", "| split_strategy | " +
             " | ".join(f"fold {f}" for f in range(len(next(iter(results_by_strategy.values()))))) +
             " | pooled |", "|---|" + "---|" * (len(next(iter(results_by_strategy.values()))) + 1)]
    for strategy, fold_results in results_by_strategy.items():
        per_fold = " | ".join(f"{r['outer_r2']:.4f}" for r in fold_results)
        lines.append(f"| {strategy} | {per_fold} | {pooled_r2_by_strategy[strategy]:.4f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None, help="processed_data_dir override, e.g. for File A on Kaggle")
    parser.add_argument("--frozen-hyperparams", default=str(DEFAULT_FROZEN_HYPERPARAMS))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-outer-folds", type=int, default=N_OUTER_FOLDS)
    parser.add_argument("--shap-n", type=int, default=DEFAULT_SHAP_N)
    parser.add_argument("--shap-seed", type=int, default=DEFAULT_SHAP_SEED)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                         help="xgboost fit device; defaults to cuda -- the frozen zT hyperparameters were "
                              "tuned under cuda and every other CLAUDE.md number comes from cuda runs, so "
                              "cpu would introduce a gratuitous device difference. This machine has no GPU "
                              "(see CLAUDE.md Local dev environment); pass --device cpu only for a local "
                              "smoke test, run the real comparison on Kaggle with the cuda default")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / "shap_attribution" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs = {"processed_data_dir": args.data_dir} if args.data_dir else {}
    df = load_target_data(TARGET, **load_kwargs)
    feature_cols = get_feature_columns(df, feature_set=FEATURE_SET)
    assert len(feature_cols) == 397, f"expected 397 features for feature_set=full, got {len(feature_cols)}"

    X_full = df[feature_cols].to_numpy(dtype=np.float64)
    y_full = df[TARGET].to_numpy(dtype=np.float64)
    target_scale = _target_scale(TARGET)
    y_full = _transform_target(y_full, TARGET)  # no-op for zT (linear scale); kept for pipeline fidelity

    group_lookup = {
        "chemistry": df[GROUP_COL].to_numpy(),
        "composition": df["composition_id"].to_numpy(),
    }

    best_params, frozen_inner_r2 = _load_frozen_hyperparams(args.frozen_hyperparams, expected_model_type="xgboost")

    results_by_strategy = {}
    pooled_r2_by_strategy = {}
    for split_strategy in SPLIT_STRATEGIES_TO_COMPARE:
        fold_results, pooled_r2 = fit_and_attribute_all_folds(
            split_strategy, X_full, y_full, feature_cols, group_lookup,
            best_params, args.seed, args.n_outer_folds, args.device,
            args.shap_n, args.shap_seed,
        )
        results_by_strategy[split_strategy] = fold_results
        pooled_r2_by_strategy[split_strategy] = pooled_r2

    coarse_rows, fine_rows, unmapped = aggregate_and_compare(feature_cols, results_by_strategy)

    dataset_path = sorted(Path(args.data_dir or "data/processed").glob("featurized_ThermoelectricMaterials_*.csv"))[-1]
    dataset_sha256 = _sha256_file(dataset_path)

    summary = {
        "generated_at_utc": timestamp,
        "target": TARGET,
        "target_scale": target_scale,
        "feature_set": FEATURE_SET,
        "n_features": len(feature_cols),
        "seed": args.seed,
        "n_outer_folds": args.n_outer_folds,
        "device": args.device,
        "dataset": {"path": str(dataset_path), "sha256": dataset_sha256, "n_rows": int(len(df))},
        "frozen_hyperparams": {"path": str(args.frozen_hyperparams), "best_params": best_params,
                                "inner_cv_r2": frozen_inner_r2},
        "caveat": (
            "The random-split and chemistry-cluster models are trained on DIFFERENT training "
            "sets by construction, in every one of the 5 folds (the random-split model's training "
            "rows include near-duplicates of its own test rows -- that is the phenomenon under "
            "study). This compares two different fitted models per fold, not an ablation of one "
            "fixed model. Averaging over 5 folds addresses fold-to-fold sampling variation within "
            "each split_strategy; it does not eliminate this train-set-composition confound, which "
            "is inherent to comparing the two split strategies at all."
        ),
        "per_strategy": {
            strategy: {
                "pooled_r2": pooled_r2_by_strategy[strategy],
                "folds": [
                    {
                        "fold": r["fold"], "n_train": r["n_train"], "n_test": r["n_test"],
                        "outer_r2": r["outer_r2"], "fit_seconds": r["fit_seconds"],
                        "shap_n_requested": r["shap_n_requested"], "shap_n_used": r["shap_n_used"],
                        "shap_seed": r["shap_seed"], "shap_seconds": r["shap_seconds"],
                        "shap_bias_mean": r["shap_bias_mean"],
                    }
                    for r in fold_results
                ],
            }
            for strategy, fold_results in results_by_strategy.items()
        },
        "unmapped_fine_group_count": len(unmapped),
        "unmapped_fine_group_columns": [c for c, _ in unmapped],
        "coarse_group_comparison": coarse_rows,
        "fine_group_comparison": fine_rows,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_lines = [
        "# SHAP Attribution Comparison -- zT, random vs chemistry-cluster (repeat 0, all 5 folds)",
        "",
        f"Generated: {timestamp}. Dataset: `{dataset_path}` (sha256 {dataset_sha256[:12]}...). Device: {args.device}.",
        "",
        _r2_table_markdown(results_by_strategy, pooled_r2_by_strategy),
        "",
        _rows_to_markdown(coarse_rows, "Coarse group comparison (D6, D7)"),
        "",
        _rows_to_markdown(fine_rows, "Fine descriptor-semantic group comparison (D6, D7)"),
        "",
    ]
    md_lines.append(
        f"`other` fine-group column count: {len(unmapped)} "
        f"(expected 0 -- every column should resolve to a named semantic group; "
        f"see summary.json's unmapped_fine_group_columns if nonzero)."
    )
    with open(out_dir / "comparison_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # D8: per-fold, per-feature arrays, enough to rebuild a global summary
    # figure later without refitting either model.
    npz_payload = {"feature_cols": np.asarray(feature_cols, dtype=object)}
    for strategy, fold_results in results_by_strategy.items():
        for r in fold_results:
            prefix = f"{strategy}__fold{r['fold']}"
            for key in (
                "mean_abs_shap", "shap_share", "gain_importance", "gain_share",
                "train_idx", "test_idx", "shap_subsample_positions",
            ):
                npz_payload[f"{prefix}__{key}"] = r[key]
    np.savez_compressed(out_dir / "shap_arrays.npz", **npz_payload)

    print("\n".join(md_lines))
    print(f"\nWrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'comparison_table.md'}")
    print(f"Wrote {out_dir / 'shap_arrays.npz'}")


if __name__ == "__main__":
    main()
