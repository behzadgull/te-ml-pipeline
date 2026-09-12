# teMatDb section N, step 4 (N3 metrics): label-to-label agreement (R2,
# RMSE, MAE, sample_id-level bootstrap 95% CI, N_BOOT=2000) from step 3's
# matched points, full-range and 300-800K, sigma/kappa in log10 space.
# Produced the original File B N3 R2_agree numbers CLAUDE.md's
# (now-superseded) digitization ceiling table cited.
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

SCRATCH = r"C:\Users\choha\AppData\Local\Temp\claude\C--Users-choha-te-ml-pipeline\c4b3e6d8-f82b-4c85-b9c8-4bcf710343a3\scratchpad"
RNG_SEED = 0
N_BOOT = 2000

declared_df = pd.read_csv(f"{SCRATCH}\\_step_n2_matched_points_declared.csv")
tep_df = pd.read_csv(f"{SCRATCH}\\_step_n2_matched_points_zt_tep.csv")
compids = pd.read_csv(f"{SCRATCH}\\_step_n1_tematdb_compids.csv")
doi_lookup = compids.set_index("sample_id")["DOI"].to_dict()

LOG_PROPS = {"sigma", "kappa"}


def bootstrap_r2_ci(sample_ids, y_true, y_pred, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(sample_ids)
    if len(uniq) < 2:
        return (float("nan"), float("nan"))
    boot_r2 = []
    idx_by_sample = {s: np.where(sample_ids == s)[0] for s in uniq}
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


def compute_metrics(df, prop, restrict_300_800=False):
    sub = df[df["property"] == prop].copy()
    if restrict_300_800:
        sub = sub[(sub["temperature"] >= 300) & (sub["temperature"] <= 800)]
    if sub.empty:
        return None
    y_true = sub["tematdb_value"].to_numpy(dtype=np.float64)
    y_pred = sub["train_interp_value"].to_numpy(dtype=np.float64)
    sample_ids = sub["tematdb_sample_id"].to_numpy()

    if prop.replace("_tep", "") in LOG_PROPS:
        valid = (y_true > 0) & (y_pred > 0)
        n_dropped_nonpositive = int((~valid).sum())
        y_true, y_pred, sample_ids = y_true[valid], y_pred[valid], sample_ids[valid]
        y_true, y_pred = np.log10(y_true), np.log10(y_pred)
    else:
        n_dropped_nonpositive = 0

    if len(y_true) < 2:
        return None

    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    n_points = int(len(y_true))
    n_samples = int(len(np.unique(sample_ids)))
    doi_set = {doi_lookup.get(sid) for sid in np.unique(sample_ids)}
    n_dois = int(len(doi_set))
    ci = bootstrap_r2_ci(sample_ids, y_true, y_pred)

    return {"r2": r2, "rmse": rmse, "mae": mae, "n_points": n_points, "n_samples": n_samples,
            "n_distinct_dois": n_dois, "r2_ci95": list(ci), "n_dropped_nonpositive_for_log": n_dropped_nonpositive}


results = {}
for prop in ["S", "sigma", "kappa", "zT"]:
    results[f"{prop}_full_range"] = compute_metrics(declared_df, prop, restrict_300_800=False)
    results[f"{prop}_300_800K"] = compute_metrics(declared_df, prop, restrict_300_800=True)

# zt_tep uses the separate tep_df, property label "zT_tep" inside that df
results["zT_tep_full_range"] = compute_metrics(tep_df.rename(columns={}), "zT_tep", restrict_300_800=False)
results["zT_tep_300_800K"] = compute_metrics(tep_df, "zT_tep", restrict_300_800=True)

print("=== N3: label-to-label agreement (composition-matched) ===")
for k, v in results.items():
    if v is None:
        print(f"  {k}: NO DATA")
        continue
    print(f"  {k:20s} R2={v['r2']:+.4f} RMSE={v['rmse']:.4f} MAE={v['mae']:.4f} "
          f"n={v['n_points']} n_samples={v['n_samples']} n_dois={v['n_distinct_dois']} "
          f"CI95=[{v['r2_ci95'][0]:+.4f},{v['r2_ci95'][1]:+.4f}]")

with open(f"{SCRATCH}\\_step_n3_results.json", "w") as fh:
    json.dump(results, fh, indent=2)
print("\nsaved _step_n3_results.json")
