# teMatDb section N, step 5 (N4 diagnostic): classify the OLD round-3
# temperature-proximity best-match pairs by whether their compositions
# DIFFER/SAME/UNKNOWN vs step-1's composition-identity flags, then rescore
# the OLD L1 matched points restricted to the composition-DIFFER subset --
# confirms the original temperature-proximity matching conflated composition
# mismatch with digitization disagreement. File B.
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

BASE = r"C:\Users\choha\te-ml-pipeline"
SCRATCH = r"C:\Users\choha\AppData\Local\Temp\claude\C--Users-choha-te-ml-pipeline\c4b3e6d8-f82b-4c85-b9c8-4bcf710343a3\scratchpad"
RNG_SEED = 0
N_BOOT = 2000
LOG_PROPS = {"sigma", "kappa"}

# old round-3 temperature-proximity best-match pairs (176, one per a0 sample)
old_pairs = pd.read_csv(f"{BASE}\\data\\external\\tematdb\\_step_g1_best_match.csv")
# composition-match flags for every (tematdb_sample_id, train_sample_id) DOI-shared candidate (from N1)
cand = pd.read_csv(f"{SCRATCH}\\_step_n1_candidate_pairs_matched.csv")

old_flagged = old_pairs.merge(
    cand[["tematdb_sample_id", "train_sample_id", "tematdb_composition_id", "train_composition_id", "composition_match"]],
    on=["tematdb_sample_id", "train_sample_id"], how="left",
)
print(f"old 176 proximity pairs, flag coverage: {old_flagged['composition_match'].notna().sum()} of {len(old_flagged)} found in candidate table")

# classify: DIFFER (both parsed, unequal), SAME (both parsed, equal), UNKNOWN (either side unparseable)
both_parsed = old_flagged["tematdb_composition_id"].notna() & old_flagged["train_composition_id"].notna()
differ_mask = both_parsed & (~old_flagged["composition_match"].fillna(False))
same_mask = both_parsed & old_flagged["composition_match"].fillna(False)
unknown_mask = ~both_parsed

n_differ = int(differ_mask.sum())
n_same = int(same_mask.sum())
n_unknown = int(unknown_mask.sum())
print(f"of 176 old proximity pairs: composition DIFFERS={n_differ}, composition SAME (proximity happened to also match)={n_same}, UNKNOWN (unparseable on one side)={n_unknown}")

differ_tematdb_ids = set(old_flagged.loc[differ_mask, "tematdb_sample_id"].tolist())
old_flagged.to_csv(f"{SCRATCH}\\_step_n4_old_pairs_flagged.csv", index=False)

# old L1 matched points (already computed in the prior corrections round, temperature-proximity based)
old_matched = pd.read_csv(f"{BASE}\\data\\external\\tematdb\\_step_l_matched_points.csv")
old_matched_differ = old_matched[old_matched["tematdb_sample_id"].isin(differ_tematdb_ids)].copy()
print(f"old L1 matched points restricted to composition-DIFFER pairs: {len(old_matched_differ)} of {len(old_matched)}")
old_matched_differ.to_csv(f"{SCRATCH}\\_step_n4_matched_points_differ_subset.csv", index=False)


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
    if prop in LOG_PROPS:
        valid = (y_true > 0) & (y_pred > 0)
        y_true, y_pred, sample_ids = y_true[valid], y_pred[valid], sample_ids[valid]
        y_true, y_pred = np.log10(y_true), np.log10(y_pred)
    if len(y_true) < 2:
        return None
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    ci = bootstrap_r2_ci(sample_ids, y_true, y_pred)
    return {"r2": r2, "rmse": rmse, "mae": mae, "n_points": int(len(y_true)),
            "n_samples": int(len(np.unique(sample_ids))), "r2_ci95": list(ci)}


n4_results = {}
for prop in ["S", "sigma", "kappa", "zT"]:
    n4_results[f"{prop}_full_range"] = compute_metrics(old_matched_differ, prop, restrict_300_800=False)

print("\n=== N4 DIAGNOSTIC: old temperature-proximity pairs, restricted to composition-DIFFER subset ===")
for k, v in n4_results.items():
    if v is None:
        print(f"  {k}: NO DATA")
        continue
    print(f"  {k:20s} R2={v['r2']:+.4f} RMSE={v['rmse']:.4f} MAE={v['mae']:.4f} n={v['n_points']} n_samples={v['n_samples']} CI95=[{v['r2_ci95'][0]:+.4f},{v['r2_ci95'][1]:+.4f}]")

with open(f"{SCRATCH}\\_step_n4_results.json", "w") as fh:
    json.dump({
        "n_old_pairs_composition_differ": n_differ,
        "n_old_pairs_composition_same": n_same,
        "n_old_pairs_composition_unknown": n_unknown,
        "metrics_on_differ_subset": n4_results,
    }, fh, indent=2)
print("\nsaved _step_n4_results.json")
