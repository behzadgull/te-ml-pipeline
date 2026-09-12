import sys, json, datetime
import numpy as np
import pandas as pd
sys.path.insert(0, r'C:\Users\choha\te-ml-pipeline')
import src.external_validation as ev
from src.nested_cv import get_feature_columns
from src.backtransform_check import duan_smearing_correction
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# CONTROL RUN: identical script to tematdb_D_scoring_fileA.py, EXCEPT no
# monkeypatch is applied -- ev.load_target_data stays the ORIGINAL function,
# defaulting to File B. Isolates reimplementation effect from dataset effect
# for section D, same purpose as the ESTM control.

NEW_RESULTS_DIR = "results/20260911T134500_tematdb_control_fileB"
RNG_SEED = 0
N_BOOT = 2000

print("=== teMatDb CONTROL RUN: reimplemented script, File B (default, unmodified) ===", flush=True)
print(f"TRAINING_CSV (module default) = {ev.TRAINING_CSV}", flush=True)

scoring_df = pd.read_csv("data/external/tematdb/_scoring_df.csv")
assert len(scoring_df) == 2930
assert (scoring_df.in_training & scoring_df.parseable).sum() == 2027
assert (~scoring_df.in_training & scoring_df.parseable).sum() == 903
assert scoring_df.in_stratum_b.sum() == 549

print("\n=== S1: refit all four frozen models on 100% of File B training ===", flush=True)
models = {}
training_rows_used = {}
for target in ev.TARGETS:
    model, feature_cols_train, n_rows = ev.fit_frozen_external_model(target, device="cpu")  # no override -- File B default
    models[target] = model
    training_rows_used[target] = n_rows
    print(f"  {target}: refit on {n_rows:,} training rows", flush=True)

print("\n=== S1: recompute + freeze Duan smear factors (File B training OOF only) ===", flush=True)
smear_sigma, smear_kappa, smear_oof_r2 = ev.compute_training_smear_factors(device="cpu")  # no override -- File B default
print(f"  FROZEN smear_sigma={smear_sigma:.6f}  smear_kappa={smear_kappa:.6f}", flush=True)

feature_cols = get_feature_columns(scoring_df)
assert len(feature_cols) == 397, f"expected 397 feature columns, got {len(feature_cols)}"


def bootstrap_r2_ci(sample_ids, y_true, y_pred, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(sample_ids)
    if len(uniq) < 2:
        return (float("nan"), float("nan"))
    idx_by_sample = {s: np.where(sample_ids == s)[0] for s in uniq}
    boot_r2 = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_sample[s] for s in drawn])
        if len(np.unique(y_true[rows])) < 2:
            continue
        try:
            boot_r2.append(r2_score(y_true[rows], y_pred[rows]))
        except Exception:
            continue
    if len(boot_r2) < 10:
        return (float("nan"), float("nan"))
    return (float(np.percentile(boot_r2, 2.5)), float(np.percentile(boot_r2, 97.5)))


def score_stratum(sub, stratum_name, save_path):
    X = sub[feature_cols].to_numpy(dtype=np.float64)
    n_rows = len(sub)
    n_samples = int(sub["sample_id"].nunique())
    n_clusters = int(sub["chemistry_cluster_id"].nunique())
    print(f"\n=== Scoring stratum ({stratum_name}): n_rows={n_rows} n_samples={n_samples} n_clusters={n_clusters} ===", flush=True)

    S_true = sub["S"].to_numpy(dtype=np.float64)
    sigma_true = sub["sigma"].to_numpy(dtype=np.float64)
    kappa_true = sub["kappa_prop"].to_numpy(dtype=np.float64)
    zt_declared_true = sub["zt_declared"].to_numpy(dtype=np.float64)
    zt_tep_true = sub["zt_tep"].to_numpy(dtype=np.float64)
    T = sub["temperature_bin"].to_numpy(dtype=np.float64)
    sample_ids = sub["sample_id"].to_numpy()

    S_pred = models["S"].predict(X)
    sigma_log_pred = models["sigma"].predict(X)
    kappa_log_pred = models["kappa"].predict(X)
    zT_direct_pred = models["zT"].predict(X)

    d = {"S_pred": S_pred, "sigma_log_pred": sigma_log_pred, "kappa_log_pred": kappa_log_pred, "T": T}
    zT_derived_pred, _, _ = duan_smearing_correction(d, smear_sigma=smear_sigma, smear_kappa=smear_kappa)

    sigma_log_true = np.log10(sigma_true)
    kappa_log_true = np.log10(kappa_true)

    results = {}

    def add_metric(name, y_true, y_pred):
        r2 = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        ci = bootstrap_r2_ci(sample_ids, y_true, y_pred)
        results[name] = {"r2": r2, "rmse": rmse, "mae": mae, "n": int(len(y_true)), "r2_ci95": list(ci)}
        print(f"  {name:25s} R2={r2:+.4f} RMSE={rmse:.4f} MAE={mae:.4f} n={len(y_true)} CI95=[{ci[0]:+.4f},{ci[1]:+.4f}]", flush=True)

    add_metric("S", S_true, S_pred)
    add_metric("sigma_log10", sigma_log_true, sigma_log_pred)
    add_metric("kappa_log10", kappa_log_true, kappa_log_pred)
    add_metric("zT_direct_vs_declared", zt_declared_true, zT_direct_pred)
    add_metric("zT_direct_vs_tep", zt_tep_true, zT_direct_pred)
    add_metric("zT_derived_vs_declared", zt_declared_true, zT_derived_pred)
    add_metric("zT_derived_vs_tep", zt_tep_true, zT_derived_pred)

    np.savez(
        save_path,
        sample_id=sample_ids,
        chemistry_cluster_id=sub["chemistry_cluster_id"].to_numpy(),
        temperature=T,
        S_true=S_true, S_pred=S_pred,
        sigma_true=sigma_true, sigma_log_true=sigma_log_true, sigma_log_pred=sigma_log_pred,
        kappa_true=kappa_true, kappa_log_true=kappa_log_true, kappa_log_pred=kappa_log_pred,
        zt_declared_true=zt_declared_true, zt_tep_true=zt_tep_true,
        zT_direct_pred=zT_direct_pred, zT_derived_pred=zT_derived_pred,
    )
    print(f"  saved per-row predictions to {save_path}", flush=True)
    return {"n_rows": n_rows, "n_samples": n_samples, "n_clusters": n_clusters, "metrics": results}

a0_sub = scoring_df[scoring_df.in_training & scoring_df.parseable].reset_index(drop=True)
a_sub = scoring_df[~scoring_df.in_training & scoring_df.parseable].reset_index(drop=True)
b_sub = scoring_df[scoring_df.in_stratum_b].reset_index(drop=True)

result_a0 = score_stratum(a0_sub, "a0", f"{NEW_RESULTS_DIR}/tematdb_a0_predictions.npz")
result_a = score_stratum(a_sub, "a", f"{NEW_RESULTS_DIR}/tematdb_a_predictions.npz")
result_b = score_stratum(b_sub, "b", f"{NEW_RESULTS_DIR}/tematdb_b_predictions.npz")

metrics_out = {
    "training_csv": ev.TRAINING_CSV,
    "date": datetime.date.today().isoformat(),
    "training_rows_used_for_refit": training_rows_used,
    "smear_factors": {"sigma": smear_sigma, "kappa": smear_kappa},
    "smear_training_oof_r2": smear_oof_r2,
    "note": "CONTROL RUN: reimplemented orchestration script (same one used for File A), pointed at File B "
            "(unmodified default path). Compare against "
            "results/20260910T123047_tematdb_external/metrics.json (the original run's result).",
    "strata": {"a0": result_a0, "a": result_a, "b": result_b},
}
with open(f"{NEW_RESULTS_DIR}/metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)
print(f"\nSaved metrics to {NEW_RESULTS_DIR}/metrics.json")
print("TEMATDB_CONTROL_FILEB_DONE", flush=True)
