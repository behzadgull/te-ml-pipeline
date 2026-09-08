"""
External validation against ESTM (Paper A item 6, ESTM first; teMatDb
not yet implemented).

Refit-on-full-training-then-predict: there is no saved/serialized model
anywhere in this repo (confirmed by inventory, 2026-09-07) -- "frozen"
means frozen hyperparameters (checkpoints/saved_predictions/checkpoints/
frozen_hyperparams/{S,sigma,kappa,zT}.json), not a frozen model object.
For each target, fit_frozen_external_model() builds the frozen XGBoost
hyperparameters and refits on 100% of the training table (no holdout)
-- that refit model IS "the frozen external model" item 6 validates
against.

Two INDEPENDENT dedup passes against ESTM, never merged (CLAUDE.md item
6): (a) source-DOI dedup -- drops ESTM rows sharing a normalized DOI
with any training row, tests measurement transfer; (b) composition-
cluster dedup -- drops ESTM rows sharing a chemistry_cluster_id with
any training row, tests chemistry transfer. Both dropped-row counts are
reported; the two resulting datasets are scored separately, never
intersected.

ESTM is touched by the frozen model EXACTLY ONCE, inside
run_full_validation() -- everything before that (temperature
filtering, canonicalization, featurization, the 397-column assertion,
dedup-overlap previews) is pure data-pipeline work with no model
involved, and is exactly what dry_run_inventory() runs and reports, so
correctness can be checked before the model is ever touched.

Sigma/kappa's derived-zT Duan smear factors are computed from TRAINING
out-of-fold residuals only (compute_training_smear_factors), never from
ESTM's own true values -- calibrating the correction on the same data
it then scores would be test-set leakage on the external validation.
"""

import datetime
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtransform_check import duan_smearing_correction
from src.canonicalization import DEFAULT_DOPANT_THRESHOLD_FRAC, chemistry_cluster_id, composition_id, parse_formula
from src.data_cleaning import step3_filter_temperature
from src.featurization import featurize_cbfv, featurize_magpie, get_unique_formulas
from src.nested_cv import GROUP_COL, LOG_TRANSFORM_TARGETS, MODEL_REGISTRY, get_feature_columns, load_target_data, randomized_group_kfold

TRAINING_CSV = "data/processed/featurized_ThermoelectricMaterials_2026-08-15.csv"
ESTM_PATH = Path("data/external/estm.xlsx")
FROZEN_HYPERPARAMS_DIR = Path("checkpoints/saved_predictions/checkpoints/frozen_hyperparams")
RESULTS_DIR = Path("checkpoints/external_validation")
TARGETS = ("S", "sigma", "kappa", "zT")

# ESTM's own column names -> the training-schema names the rest of this
# module expects.
ESTM_COLUMN_MAP = {
    "temperature(K)": "temperature_K",
    "seebeck_coefficient(μV/K)": "S",
    "electrical_conductivity(S/m)": "sigma",
    "thermal_conductivity(W/mK)": "kappa",
    "ZT": "zT",  # ESTM spells it "ZT" (all caps); training column is "zT" -- case-sensitive, easy to miss
}

# Confirmed by direct comparison against data_cleaning.py's
# PROPERTY_BOUNDS comments: S is microV/K, sigma is S/m, kappa is W/mK
# in BOTH the training data and ESTM's own column headers -- no unit
# conversion is needed for any of the three. Stated explicitly here
# rather than silently assumed, since a wrong assumption here would
# corrupt every downstream R^2 without raising any error.
UNIT_NOTE = (
    "S: training microV/K, ESTM microV/K -- no conversion needed. "
    "sigma: training S/m, ESTM S/m -- no conversion needed. "
    "kappa: training W/mK, ESTM W/mK -- no conversion needed."
)


def get_frozen_hyperparams(target):
    """Load target's frozen hyperparameters. Returns (best_params, model_type)."""
    path = FROZEN_HYPERPARAMS_DIR / f"{target}.json"
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    return result["best_params"], result["model_type"]


def normalize_doi(value):
    """
    Normalize a DOI/reference string for comparison: strip whitespace,
    lowercase, strip a leading 'https://doi.org/' or 'doi:' prefix.
    Returns '' for null/empty input, which never matches a real DOI.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"^https?://doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s.strip()


def load_training_data(path=TRAINING_CSV):
    """Load the featurized training table the ladder trained on."""
    return pd.read_csv(path)


def load_estm(path=ESTM_PATH):
    """Load the raw ESTM spreadsheet and rename columns to the training schema (see ESTM_COLUMN_MAP)."""
    df = pd.read_excel(path)
    return df.rename(columns=ESTM_COLUMN_MAP)


def filter_estm_temperature(estm_df):
    """
    Guard 1: apply data_cleaning.py's EXACT step3_filter_temperature
    binning rule to ESTM (300-800K window, 25K bins) -- the function is
    reused unmodified, not reimplemented. Returns (filtered_df, n_dropped).
    """
    n_before = len(estm_df)
    filtered = step3_filter_temperature(estm_df)
    n_dropped = n_before - len(filtered)
    return filtered, n_dropped


def canonicalize_estm(estm_df, dopant_threshold_frac=DEFAULT_DOPANT_THRESHOLD_FRAC):
    """
    Attach composition_id/chemistry_cluster_id to ESTM via
    parse_formula -> composition_id() -> chemistry_cluster_id() -- the
    same per-formula logic canonicalization.add_canonical_columns uses
    for training, called directly here instead of through
    add_canonical_columns since that helper requires a sample_id column
    ESTM doesn't have and external validation doesn't need.

    Returns (canonical_df_with_parse_failures_dropped, n_parse_failures).
    """
    composition_ids, cluster_ids, parse_errors = [], [], []
    for formula in estm_df["Formula"]:
        comp, error = parse_formula(formula)
        if comp is None:
            composition_ids.append(None)
            cluster_ids.append(None)
            parse_errors.append(error)
        else:
            composition_ids.append(composition_id(comp))
            cluster_ids.append(chemistry_cluster_id(comp, dopant_threshold_frac))
            parse_errors.append(None)

    out = estm_df.copy()
    out["composition_id"] = composition_ids
    out["chemistry_cluster_id"] = cluster_ids
    out["parse_error"] = parse_errors

    n_failures = int(out["parse_error"].notna().sum())
    canonical = out[out["parse_error"].isna()].drop(columns=["parse_error"]).reset_index(drop=True)
    return canonical, n_failures


def featurize_estm(canonical_df, formula_col="composition_id"):
    """
    Guard 2: MAGPIE + CBFV featurization for ESTM, mirroring
    src/featurization.py's featurize_dataset() merge logic, but calling
    featurize_cbfv with elem_prop="oliynyk" EXPLICITLY rather than
    relying on featurization.py's CBFV_PRESET module default -- so this
    path can never silently drift from training's CBFV preset even if
    that default changes later. featurize_magpie is called exactly as
    training does (no equivalent override exists there).

    Returns (featurized_df, n_unique_failed_formulas) -- the latter
    counts unique formulas that failed either featurizer, excluded from
    featurized_df by the inner merge but never silently unaccounted for.
    """
    unique_df = get_unique_formulas(canonical_df, formula_col)
    magpie_df, magpie_failures = featurize_magpie(unique_df, formula_col)
    cbfv_df, cbfv_failures = featurize_cbfv(unique_df, formula_col, elem_prop="oliynyk")

    combined = magpie_df.merge(cbfv_df, on=formula_col, how="inner")
    featurized_df = canonical_df.merge(combined, on=formula_col, how="inner")

    failed_formulas = {f["composition_id"] for f in magpie_failures} | {f["composition_id"] for f in cbfv_failures}
    return featurized_df, len(failed_formulas)


def assert_feature_columns_match(estm_featurized_df, training_df):
    """
    Guard 3: get_feature_columns() must return the identical 397 columns,
    in the identical order, for ESTM and training. Raises AssertionError
    with the exact diff otherwise -- fails loudly rather than silently
    feeding a misaligned feature matrix to the model.
    """
    training_cols = get_feature_columns(training_df)
    estm_cols = get_feature_columns(estm_featurized_df)
    if training_cols != estm_cols:
        missing = [c for c in training_cols if c not in estm_cols]
        extra = [c for c in estm_cols if c not in training_cols]
        raise AssertionError(
            f"Feature column mismatch: training has {len(training_cols)} cols, ESTM has "
            f"{len(estm_cols)}. Missing from ESTM: {missing}. Extra in ESTM: {extra}."
        )
    return training_cols


def dedup_by_doi(estm_df, training_df):
    """
    Guard 4a / dedup pass (a): drop ESTM rows whose normalized
    'reference' matches any normalized training DOI. Returns
    (surviving_df, n_dropped, n_unique_estm_dois_overlapping).
    """
    training_dois = set(training_df["DOI"].map(normalize_doi)) - {""}
    estm_norm = estm_df["reference"].map(normalize_doi)
    overlap_mask = estm_norm.isin(training_dois)
    n_unique_overlap = int(estm_norm[overlap_mask].nunique())
    surviving = estm_df[~overlap_mask].reset_index(drop=True)
    return surviving, int(overlap_mask.sum()), n_unique_overlap


def dedup_by_chemistry_cluster(estm_df, training_df):
    """
    Guard 4b / dedup pass (b): drop ESTM rows whose chemistry_cluster_id
    (computed with the same DEFAULT_DOPANT_THRESHOLD_FRAC training used)
    matches any training chemistry_cluster_id. Returns (surviving_df,
    n_dropped, n_unique_estm_clusters_overlapping).
    """
    training_clusters = set(training_df["chemistry_cluster_id"].dropna())
    overlap_mask = estm_df["chemistry_cluster_id"].isin(training_clusters)
    n_unique_overlap = int(estm_df.loc[overlap_mask, "chemistry_cluster_id"].nunique())
    surviving = estm_df[~overlap_mask].reset_index(drop=True)
    return surviving, int(overlap_mask.sum()), n_unique_overlap


def dry_run_inventory(estm_path=ESTM_PATH, training_csv=TRAINING_CSV):
    """
    Data-pipeline-only inventory: temperature-filter survival, DOI
    overlap preview, chemistry-cluster overlap preview, and the
    397-column assertion. NO model is fit or touched here -- safe to
    run and re-run freely, unlike run_full_validation(). Prints and
    returns (report_dict, estm_featurized_df, training_df) so
    run_full_validation() can reuse the featurized ESTM frame without
    recomputing it.
    """
    training_df = load_training_data(training_csv)
    estm_raw = load_estm(estm_path)
    n_estm_raw = len(estm_raw)

    estm_temp_filtered, n_dropped_temp = filter_estm_temperature(estm_raw)
    n_survive_temp = len(estm_temp_filtered)

    estm_canonical, n_parse_failures = canonicalize_estm(estm_temp_filtered)
    estm_featurized, n_featurize_failures = featurize_estm(estm_canonical)

    feature_cols = assert_feature_columns_match(estm_featurized, training_df)

    _, n_would_drop_doi, n_overlap_doi = dedup_by_doi(estm_featurized, training_df)
    _, n_would_drop_cluster, n_overlap_cluster = dedup_by_chemistry_cluster(estm_featurized, training_df)

    report = {
        "estm_raw_rows": n_estm_raw,
        "n_dropped_temperature_filter": n_dropped_temp,
        "n_survive_temperature_filter": n_survive_temp,
        "n_parse_failures": n_parse_failures,
        "n_featurize_failures_unique_formulas": n_featurize_failures,
        "n_rows_after_featurization": len(estm_featurized),
        "feature_columns_match": True,
        "n_feature_columns": len(feature_cols),
        "unique_estm_dois_overlapping_training": n_overlap_doi,
        "n_would_drop_doi_dedup": n_would_drop_doi,
        "unique_estm_clusters_overlapping_training": n_overlap_cluster,
        "n_would_drop_cluster_dedup": n_would_drop_cluster,
        "unit_handling": UNIT_NOTE,
    }

    print("=== ESTM dry-run inventory (no model touched) ===")
    for k, v in report.items():
        print(f"{k}: {v}")

    return report, estm_featurized, training_df


def compute_training_smear_factors(device="cpu", n_outer_folds=5, seed=0):
    """
    One chemistry-cluster GroupKFold pass per target (sigma, kappa),
    frozen hyperparameters (no retuning), seed=0 -- collects
    out-of-fold log10 predictions/true values and computes each
    target's Duan smearing factor from TRAINING data only, never from
    ESTM (using ESTM truths here would be test-set leakage on the
    external validation this calibrates). Prints and returns
    (smear_sigma, smear_kappa, oof_r2_dict).
    """
    smear = {}
    oof_r2 = {}
    for target in ("sigma", "kappa"):
        df = load_target_data(target)
        feature_cols = get_feature_columns(df)
        X = df[feature_cols].to_numpy(dtype=np.float64)
        y_log = np.log10(df[target].to_numpy(dtype=np.float64))
        groups = df[GROUP_COL].to_numpy()
        best_params, model_type = get_frozen_hyperparams(target)

        rng = np.random.default_rng(seed)
        oof_true, oof_pred = [], []
        for train_idx, test_idx in randomized_group_kfold(groups, n_outer_folds, rng):
            model = MODEL_REGISTRY[model_type]["build"](best_params, device)
            model.fit(X[train_idx], y_log[train_idx])
            oof_pred.append(model.predict(X[test_idx]))
            oof_true.append(y_log[test_idx])
        oof_true_arr = np.concatenate(oof_true)
        oof_pred_arr = np.concatenate(oof_pred)

        smear[target] = float(np.mean(10.0 ** (oof_true_arr - oof_pred_arr)))
        oof_r2[target] = float(r2_score(oof_true_arr, oof_pred_arr))
        print(f"{target}: OOF log10 R^2={oof_r2[target]:.4f}, smear factor={smear[target]:.6f}")

    return smear["sigma"], smear["kappa"], oof_r2


def fit_frozen_external_model(target, device="cpu"):
    """
    Build target's frozen-hyperparameter XGBoost model and fit it on
    100% of the training table (no holdout) -- "the frozen external
    model" item 6 validates against. There is no saved estimator
    anywhere in this repo (confirmed by inventory); this refit is what
    "frozen" means here -- frozen hyperparameters, not a frozen model
    object. Returns (fitted_model, feature_cols, n_training_rows_used).
    """
    df = load_target_data(target)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y_raw = df[target].to_numpy(dtype=np.float64)
    y = np.log10(y_raw) if target in LOG_TRANSFORM_TARGETS else y_raw

    best_params, model_type = get_frozen_hyperparams(target)
    model = MODEL_REGISTRY[model_type]["build"](best_params, device)
    model.fit(X, y)
    return model, feature_cols, len(df)


def score_estm_pass(estm_df, feature_cols, models, smear_sigma, smear_kappa, pass_name, save_path=None):
    """
    For one deduped ESTM dataframe: predict S/sigma/kappa directly
    (correct scale -- linear for S, log10 for sigma/kappa, matching how
    each was trained/scored throughout this project) and zT both
    directly and via the Duan-corrected derived pathway (frozen
    training smear factors, never ESTM's own truths). Returns a dict of
    per-property {r2, n}.

    save_path: if given, persist every row's chemistry_cluster_id,
    composition_id, temperature, raw property values, and every
    property's (true, pred) pair to this .npz path -- mirrors
    nested_cv.py's _save_predictions() pattern, so follow-up analysis
    (e.g. an in-distribution-only R^2 recompute) never needs to
    re-touch the model. None (default) skips saving, unchanged from
    the original behavior.
    """
    X = estm_df[feature_cols].to_numpy(dtype=np.float64)
    results = {}

    s_pred = models["S"].predict(X)
    s_true = estm_df["S"].to_numpy(dtype=np.float64)
    results["S"] = {"r2": float(r2_score(s_true, s_pred)), "n": int(len(s_true))}

    sigma_true = estm_df["sigma"].to_numpy(dtype=np.float64)
    sigma_log_pred = models["sigma"].predict(X)
    sigma_log_true = np.log10(sigma_true)
    results["sigma"] = {"r2": float(r2_score(sigma_log_true, sigma_log_pred)), "n": int(len(sigma_log_true))}

    kappa_true = estm_df["kappa"].to_numpy(dtype=np.float64)
    kappa_log_pred = models["kappa"].predict(X)
    kappa_log_true = np.log10(kappa_true)
    results["kappa"] = {"r2": float(r2_score(kappa_log_true, kappa_log_pred)), "n": int(len(kappa_log_true))}

    zt_direct_pred = models["zT"].predict(X)
    zt_true = estm_df["zT"].to_numpy(dtype=np.float64)
    results["zT_direct"] = {"r2": float(r2_score(zt_true, zt_direct_pred)), "n": int(len(zt_true))}

    T = estm_df["temperature_bin"].to_numpy(dtype=np.float64)
    d = {"S_pred": s_pred, "sigma_log_pred": sigma_log_pred, "kappa_log_pred": kappa_log_pred, "T": T}
    zt_derived_pred, _, _ = duan_smearing_correction(d, smear_sigma=smear_sigma, smear_kappa=smear_kappa)
    results["zT_derived"] = {"r2": float(r2_score(zt_true, zt_derived_pred)), "n": int(len(zt_true))}

    print(f"--- {pass_name} ---")
    for prop, res in results.items():
        print(f"  {prop}: R^2={res['r2']:.4f} (n={res['n']:,})")

    if save_path is not None:
        np.savez(
            save_path,
            chemistry_cluster_id=estm_df["chemistry_cluster_id"].to_numpy(),
            composition_id=estm_df["composition_id"].to_numpy(),
            temperature=T,
            S_true=s_true,
            S_pred=s_pred,
            sigma_true=sigma_true,
            sigma_log_true=sigma_log_true,
            sigma_log_pred=sigma_log_pred,
            kappa_true=kappa_true,
            kappa_log_true=kappa_log_true,
            kappa_log_pred=kappa_log_pred,
            zT_true=zt_true,
            zT_direct_pred=zt_direct_pred,
            zT_derived_pred=zt_derived_pred,
        )
        print(f"  saved per-row predictions to {save_path}")

    return results


def run_full_validation(device="cpu"):
    """
    THE model-touching entry point -- ESTM is touched by the frozen
    model EXACTLY ONCE, here (CLAUDE.md Paper A item 6). Must only be
    called after dry_run_inventory()'s report has been reviewed and the
    run explicitly authorized.
    """
    _, estm_featurized, training_df = dry_run_inventory()
    feature_cols = get_feature_columns(training_df)

    estm_dedup_a, n_dropped_doi, n_overlap_doi = dedup_by_doi(estm_featurized, training_df)
    estm_dedup_b, n_dropped_cluster, n_overlap_cluster = dedup_by_chemistry_cluster(estm_featurized, training_df)

    smear_sigma, smear_kappa, smear_oof_r2 = compute_training_smear_factors(device=device)

    models = {}
    training_rows_used = {}
    for target in TARGETS:
        model, _, n_rows = fit_frozen_external_model(target, device=device)
        models[target] = model
        training_rows_used[target] = n_rows

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_a = score_estm_pass(
        estm_dedup_a, feature_cols, models, smear_sigma, smear_kappa, "Pass (a) source-DOI dedup",
        save_path=RESULTS_DIR / "estm_predictions_passa.npz",
    )
    results_b = score_estm_pass(
        estm_dedup_b, feature_cols, models, smear_sigma, smear_kappa, "Pass (b) composition-cluster dedup",
        save_path=RESULTS_DIR / "estm_predictions_passb.npz",
    )

    output = {
        "provenance": {
            "frozen_hyperparams_paths": {t: str(FROZEN_HYPERPARAMS_DIR / f"{t}.json") for t in TARGETS},
            "training_csv": TRAINING_CSV,
            "estm_file": str(ESTM_PATH),
            "date": datetime.date.today().isoformat(),
        },
        "training_rows_used_for_refit": training_rows_used,
        "smear_factors": {"sigma": smear_sigma, "kappa": smear_kappa},
        "smear_training_oof_r2": smear_oof_r2,
        "unit_handling": UNIT_NOTE,
        "dedup_a_source_doi": {
            "n_dropped": n_dropped_doi,
            "n_surviving": len(estm_dedup_a),
            "unique_estm_dois_overlapping_training": n_overlap_doi,
            "results": results_a,
        },
        "dedup_b_chemistry_cluster": {
            "n_dropped": n_dropped_cluster,
            "n_surviving": len(estm_dedup_b),
            "unique_estm_clusters_overlapping_training": n_overlap_cluster,
            "results": results_b,
        },
        "zt_is_derived_not_measured_note": (
            "ESTM's ZT column is algebraically S^2*sigma*T/kappa (confirmed by spot "
            "check, matches to <0.6%), not an independent measurement -- the "
            "zT_direct/zT_derived numbers above are internal consistency checks, not "
            "independent external validation the way S/sigma/kappa are."
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "estm_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {out_path}")

    return output


if __name__ == "__main__":
    dry_run_inventory()
