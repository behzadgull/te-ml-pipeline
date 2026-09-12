"""
teMatDb external validation scoring (Paper A item 6, second external
dataset). Mirrors src/external_validation.py's ESTM protocol exactly --
reuses its generic functions unmodified (fit_frozen_external_model,
compute_training_smear_factors, get_frozen_hyperparams, score_estm_pass,
duan_smearing_correction). No src/ changes. This script lives outside
src/ deliberately, per the task's "do not change src/" constraint --
all teMatDb-specific glue (paths, column renames, the three-stratum
split) lives here; all model-touching logic is imported unchanged.
"""
import sys, io, json, datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, r"C:\Users\choha\te-ml-pipeline")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.external_validation import (
    fit_frozen_external_model, compute_training_smear_factors, get_frozen_hyperparams,
    score_estm_pass, TARGETS,
)
from src.backtransform_check import duan_smearing_correction
from src.nested_cv import get_feature_columns
from src.external_validation import load_training_data

TEMATDB_SHA = "86a9bf979e35e2b9b0a4927bc50ad031edf84ad7"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
RESULTS_DIR = Path("results") / f"{TIMESTAMP}_tematdb_external"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_BOUNDS = {
    "S": (-460.4, 562.6),
    "sigma": (957.599947648446, 1656678.2772202597),
    "kappa": (0.2833018666666667, 13.6921),
}

print("=== S1: refit all four frozen models on 100% training ===", flush=True)
training_df = load_training_data()
feature_cols = get_feature_columns(training_df)

models, training_rows_used, hyperparams_used = {}, {}, {}
for target in TARGETS:
    model, fc, n_rows = fit_frozen_external_model(target, device="cpu")
    models[target] = model
    training_rows_used[target] = n_rows
    best_params, model_type = get_frozen_hyperparams(target)
    hyperparams_used[target] = {"model_type": model_type, "best_params": best_params}
    print(f"  {target}: refit on {n_rows:,} training rows", flush=True)

print()
print("=== S1: recompute + freeze Duan smear factors (training OOF only) ===", flush=True)
smear_sigma, smear_kappa, smear_oof_r2 = compute_training_smear_factors(device="cpu")
print(f"  FROZEN smear_sigma={smear_sigma:.6f}  smear_kappa={smear_kappa:.6f}", flush=True)

# ---- load prepared scoring frame (data-prep only, done pre-flight, no model touch) ----
scoring_df = pd.read_csv(r"data/external/tematdb/_scoring_df.csv")

STRATA = {
    "a0": scoring_df["in_training"] & scoring_df["parseable"],
    "a": (~scoring_df["in_training"]) & scoring_df["parseable"],
    "b": scoring_df["in_stratum_b"],
}


def bootstrap_r2_ci(y_true, y_pred, sample_ids, n_boot=1000, seed=0):
    """95% CI on R^2, resampled at sample_id level (not row level)."""
    rng = np.random.default_rng(seed)
    unique_ids = np.unique(sample_ids)
    n_ids = len(unique_ids)
    id_to_rowidx = {sid: np.where(sample_ids == sid)[0] for sid in unique_ids}
    boot_r2 = []
    for _ in range(n_boot):
        drawn_ids = rng.choice(unique_ids, size=n_ids, replace=True)
        row_idx = np.concatenate([id_to_rowidx[sid] for sid in drawn_ids])
        yt, yp = y_true[row_idx], y_pred[row_idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_r2.append(r2_score(yt, yp))
    if not boot_r2:
        return (float("nan"), float("nan"))
    return (float(np.percentile(boot_r2, 2.5)), float(np.percentile(boot_r2, 97.5)))


def full_metrics(y_true, y_pred, sample_ids, bounds=None, values_for_ood=None):
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    ci_lo, ci_hi = bootstrap_r2_ci(y_true, y_pred, sample_ids)
    out = {"r2": r2, "rmse": rmse, "mae": mae, "n_rows": int(len(y_true)),
           "n_samples": int(len(np.unique(sample_ids))), "r2_ci95": [ci_lo, ci_hi]}
    if bounds is not None and values_for_ood is not None:
        lo, hi = bounds
        in_support = (values_for_ood >= lo) & (values_for_ood <= hi)
        n_ood = int((~in_support).sum())
        out["ood_fraction"] = float(n_ood / len(values_for_ood)) if len(values_for_ood) else None
        out["support_bounds"] = [lo, hi]
        if in_support.sum() >= 2 and len(np.unique(y_true[in_support])) >= 2:
            out["r2_in_distribution"] = float(r2_score(y_true[in_support], y_pred[in_support]))
            out["n_in_distribution"] = int(in_support.sum())
        else:
            out["r2_in_distribution"] = None
            out["n_in_distribution"] = int(in_support.sum())
    return out


all_metrics = {}
for stratum_name, mask in STRATA.items():
    df = scoring_df[mask].reset_index(drop=True)
    X = df[feature_cols].to_numpy(dtype=np.float64)
    sample_ids = df["sample_id"].to_numpy()
    n_clusters = df["chemistry_cluster_id"].nunique()
    n_samples_contrib = df["sample_id"].nunique()

    print(f"\n=== Scoring stratum ({stratum_name}): n_rows={len(df)}, n_samples={n_samples_contrib}, n_clusters={n_clusters} ===", flush=True)

    s_pred = models["S"].predict(X)
    sigma_log_pred = models["sigma"].predict(X)
    kappa_log_pred = models["kappa"].predict(X)
    zt_direct_pred = models["zT"].predict(X)

    s_true = df["S"].to_numpy(dtype=np.float64)
    sigma_true = df["sigma"].to_numpy(dtype=np.float64)
    sigma_log_true = np.log10(sigma_true)
    kappa_true = df["kappa_prop"].to_numpy(dtype=np.float64)
    kappa_log_true = np.log10(kappa_true)
    T = df["temperature_bin"].to_numpy(dtype=np.float64)

    zt_declared_true = df["zt_declared"].to_numpy(dtype=np.float64)
    zt_tep_true = df["zt_tep"].to_numpy(dtype=np.float64)

    d = {"S_pred": s_pred, "sigma_log_pred": sigma_log_pred, "kappa_log_pred": kappa_log_pred, "T": T}
    zt_derived_pred, used_sigma, used_kappa = duan_smearing_correction(d, smear_sigma=smear_sigma, smear_kappa=smear_kappa)
    assert used_sigma == smear_sigma and used_kappa == smear_kappa

    stratum_metrics = {
        "n_rows": int(len(df)), "n_samples": n_samples_contrib, "n_clusters": int(n_clusters),
        "S": full_metrics(s_true, s_pred, sample_ids, TRAINING_BOUNDS["S"], s_true),
        "sigma_log10": full_metrics(sigma_log_true, sigma_log_pred, sample_ids, None, None),
        "kappa_log10": full_metrics(kappa_log_true, kappa_log_pred, sample_ids, None, None),
        "zT_direct_vs_declared": full_metrics(zt_declared_true, zt_direct_pred, sample_ids),
        "zT_direct_vs_tep": full_metrics(zt_tep_true, zt_direct_pred, sample_ids),
        "zT_derived_vs_declared": full_metrics(zt_declared_true, zt_derived_pred, sample_ids),
        "zT_derived_vs_tep": full_metrics(zt_tep_true, zt_derived_pred, sample_ids),
    }
    # sigma/kappa OOD in their own (log10) native scale for the support check, using linear-scale bounds
    stratum_metrics["sigma_log10"]["ood_fraction"] = float(((sigma_true < TRAINING_BOUNDS["sigma"][0]) | (sigma_true > TRAINING_BOUNDS["sigma"][1])).mean())
    sigma_in_support = (sigma_true >= TRAINING_BOUNDS["sigma"][0]) & (sigma_true <= TRAINING_BOUNDS["sigma"][1])
    stratum_metrics["sigma_log10"]["support_bounds_linear"] = list(TRAINING_BOUNDS["sigma"])
    stratum_metrics["sigma_log10"]["r2_in_distribution"] = (
        float(r2_score(sigma_log_true[sigma_in_support], sigma_log_pred[sigma_in_support]))
        if sigma_in_support.sum() >= 2 and len(np.unique(sigma_log_true[sigma_in_support])) >= 2 else None
    )
    stratum_metrics["sigma_log10"]["n_in_distribution"] = int(sigma_in_support.sum())

    stratum_metrics["kappa_log10"]["ood_fraction"] = float(((kappa_true < TRAINING_BOUNDS["kappa"][0]) | (kappa_true > TRAINING_BOUNDS["kappa"][1])).mean())
    kappa_in_support = (kappa_true >= TRAINING_BOUNDS["kappa"][0]) & (kappa_true <= TRAINING_BOUNDS["kappa"][1])
    stratum_metrics["kappa_log10"]["support_bounds_linear"] = list(TRAINING_BOUNDS["kappa"])
    stratum_metrics["kappa_log10"]["r2_in_distribution"] = (
        float(r2_score(kappa_log_true[kappa_in_support], kappa_log_pred[kappa_in_support]))
        if kappa_in_support.sum() >= 2 and len(np.unique(kappa_log_true[kappa_in_support])) >= 2 else None
    )
    stratum_metrics["kappa_log10"]["n_in_distribution"] = int(kappa_in_support.sum())

    all_metrics[stratum_name] = stratum_metrics

    for prop_key in ["S", "sigma_log10", "kappa_log10", "zT_direct_vs_declared", "zT_direct_vs_tep", "zT_derived_vs_declared", "zT_derived_vs_tep"]:
        m = stratum_metrics[prop_key]
        print(f"  {prop_key:25s} R2={m['r2']:+.4f} RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} "
              f"n={m['n_rows']} CI95=[{m['r2_ci95'][0]:+.4f},{m['r2_ci95'][1]:+.4f}]", flush=True)

    # S7: save per-row predictions -- one npz per (stratum, property/target)
    npz_path = RESULTS_DIR / f"tematdb_{stratum_name}_predictions.npz"
    np.savez(
        npz_path,
        sample_id=sample_ids, chemistry_cluster_id=df["chemistry_cluster_id"].to_numpy(), temperature=T,
        S_true=s_true, S_pred=s_pred,
        sigma_true=sigma_true, sigma_log_true=sigma_log_true, sigma_log_pred=sigma_log_pred,
        kappa_true=kappa_true, kappa_log_true=kappa_log_true, kappa_log_pred=kappa_log_pred,
        zt_declared_true=zt_declared_true, zt_tep_true=zt_tep_true,
        zT_direct_pred=zt_direct_pred, zT_derived_pred=zt_derived_pred,
    )
    print(f"  saved per-row predictions to {npz_path}", flush=True)

# ---- output ----
metrics_path = RESULTS_DIR / "metrics.json"
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(all_metrics, f, indent=2)

import hashlib
def sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

run_config = {
    "tematdb_sha": TEMATDB_SHA,
    "teMatDb_samples_csv_sha256": sha256_of("data/external/tematdb/teMatDb_samples.csv"),
    "teMatDb_collocatedTEPs_csv_sha256": sha256_of("data/external/tematdb/teMatDb_collocatedTEPs.csv"),
    "frozen_hyperparameters": hyperparams_used,
    "smear_factors": {"sigma": smear_sigma, "kappa": smear_kappa},
    "smear_training_oof_r2": smear_oof_r2,
    "training_rows_used_for_refit": training_rows_used,
    "training_csv": "data/processed/featurized_ThermoelectricMaterials_2026-08-15.csv",
    "date": datetime.date.today().isoformat(),
    "preflight": {"P1": "PASS", "P2": "PASS", "P3": "PASS"},
}
with open(RESULTS_DIR / "run_config.json", "w", encoding="utf-8") as f:
    json.dump(run_config, f, indent=2)

print(f"\nSaved metrics to {metrics_path}")
print(f"Saved run_config to {RESULTS_DIR / 'run_config.json'}")
print(f"RESULTS_DIR={RESULTS_DIR}")
