# Self-contained re-run of the full G1 (candidate pairs)/N1 (composition
# match)/N2 (matched points)/N3 (label agreement) pipeline against File A's
# raw curves and training data, standing in for the separate N_step1-4
# scripts (which are hardwired to File B and this session's own scratchpad
# paths). Read-only, no model touch. Feeds
# results/20260911T114356_tematdb_inventory_fileA/inventory_fileA.json.
import sys, re, json, pickle
import numpy as np
import pandas as pd
sys.path.insert(0, r'C:\Users\choha\te-ml-pipeline')
from src.canonicalization import parse_formula, composition_id
from src.data_cleaning import load_raw_curves, step1_extract_and_filter_properties, step2_integrate_and_consolidate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

BASE = r"C:\Users\choha\te-ml-pipeline"
SCRATCH = r"C:\Users\choha\AppData\Local\Temp\claude\C--Users-choha-te-ml-pipeline\c4b3e6d8-f82b-4c85-b9c8-4bcf710343a3\scratchpad"
FILE_A_RAW_DIR = f"{BASE}\\checkpoints\\saved_predictions\\te-ml-pipeline\\data\\raw"
FILE_A_CSV = f"{BASE}\\checkpoints\\saved_predictions\\te-ml-pipeline\\data\\processed\\featurized_ThermoelectricMaterials_2026-08-15.csv"
TOL_K = 5.0

def normalize_doi_inventory(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"^https?://dx\.doi\.org/", "", s)
    s = re.sub(r"^https?://doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s.strip()

print("=== Loading File A training data (DOI, sample_id, composition_id) ===", flush=True)
train_df = pd.read_csv(FILE_A_CSV, usecols=["sample_id", "DOI", "composition_id"]).drop_duplicates(subset=["sample_id"]).reset_index(drop=True)
train_df["doi_norm"] = train_df["DOI"].map(normalize_doi_inventory)
print(f"training unique sample_id rows: {len(train_df):,}", flush=True)

print("\n=== teMatDb a0 samples (DOI in training, File A) ===", flush=True)
samples = pd.read_csv(f"{BASE}\\data\\external\\tematdb\\teMatDb_samples.csv")
samples["doi_norm"] = samples["DOI"].map(normalize_doi_inventory)
training_dois = set(train_df["doi_norm"]) - {""}
a0_mask = samples["doi_norm"].isin(training_dois) & (samples["doi_norm"] != "")
a0 = samples[a0_mask].copy().reset_index(drop=True)
assert len(a0) == 176, f"expected 176 a0 samples (File A), got {len(a0)}"
print(f"a0 samples: {len(a0)}", flush=True)

comp_ids, errs = [], []
for f in a0["Composition_detailed"]:
    comp, err = parse_formula(f)
    if comp is None:
        comp_ids.append(None); errs.append(err)
    else:
        comp_ids.append(composition_id(comp)); errs.append(None)
a0["tematdb_composition_id"] = comp_ids
a0["tematdb_parse_error"] = errs
n_a0_parseable = int(a0["tematdb_composition_id"].notna().sum())
print(f"a0 parseable: {n_a0_parseable}, parse failures: {len(a0)-n_a0_parseable}   (File B: 122 parseable, 54 failed)", flush=True)

# === G1-equivalent: candidate pairs (DOI-shared), pair-median temperature distance, File A raw curves ===
print("\n=== Building DOI-shared candidate pairs (G1-equivalent, File A raw curves) ===", flush=True)
a0_dois = set(a0["doi_norm"]) - {""}
candidates = train_df[train_df["doi_norm"].isin(a0_dois)][["sample_id", "doi_norm"]].rename(columns={"sample_id": "train_sample_id"})
cand_pairs = a0[["sample_id", "doi_norm"]].rename(columns={"sample_id": "tematdb_sample_id"}).merge(candidates, on="doi_norm", how="inner")
cand_pairs = cand_pairs.drop_duplicates()
print(f"candidate pairs (DOI-shared, File A): {len(cand_pairs)}   (File B round-3 G1 universe: 1,251)", flush=True)

needed_train_ids = set(cand_pairs["train_sample_id"].unique())
print(f"unique training sample_ids needed: {len(needed_train_ids)}", flush=True)

curves_df = load_raw_curves(raw_data_dir=FILE_A_RAW_DIR)
curves_df = curves_df[curves_df["sample_id"].isin(needed_train_ids)].reset_index(drop=True)
print(f"File A raw curve rows for needed training sample_ids: {len(curves_df):,}", flush=True)
long1 = step1_extract_and_filter_properties(curves_df)
long2 = step2_integrate_and_consolidate(long1)
print(f"training long-format rows after step1+step2: {len(long2):,}", flush=True)

train_curves = {}
for (sid, prop), grp in long2.groupby(["sample_id", "property"]):
    g = grp.sort_values("temperature_K")
    train_curves[(sid, prop)] = (g["temperature_K"].to_numpy(dtype=np.float64), g["value"].to_numpy(dtype=np.float64))
print(f"training curves indexed: {len(train_curves)} (sample_id, property) groups", flush=True)

raw_tep = pd.read_csv(f"{BASE}\\data\\external\\tematdb\\teMatDb_rawTEPs.csv")
needed_tematdb_ids = set(a0["sample_id"].unique())
raw_tep = raw_tep[raw_tep["sample_id"].isin(needed_tematdb_ids)].reset_index(drop=True)
tematdb_curves = {}
for (sid, tep), grp in raw_tep.groupby(["sample_id", "tepname"]):
    g = grp.sort_values("Temperature")
    temps = g["Temperature"].to_numpy(dtype=np.float64)
    vals = g["tepvalue"].to_numpy(dtype=np.float64)
    if tep == "rho":
        vals = 1.0 / vals
    tematdb_curves[(sid, tep)] = (temps, vals)
print(f"teMatDb curves indexed: {len(tematdb_curves)} (sample_id, tepname) groups", flush=True)

PROPERTY_MAP = {"alpha": "S", "kappa": "kappa", "ZT": "zT", "rho": "sigma"}

# pooled cross-tepname pair-median distance (round-2 Step C / round-3 G1 method:
# each training point's nearest teMatDb point, restricted to their shared range first, pooled across tepnames, median per pair)
pair_medians = []
for row in cand_pairs.itertuples(index=False):
    tid, sid = row.tematdb_sample_id, row.train_sample_id
    dists = []
    for tep, prop in PROPERTY_MAP.items():
        key_te, key_tr = (tid, tep), (sid, prop)
        if key_te not in tematdb_curves or key_tr not in train_curves:
            continue
        te_t, _ = tematdb_curves[key_te]
        tr_t, _ = train_curves[key_tr]
        if len(te_t) == 0 or len(tr_t) == 0:
            continue
        lo, hi = max(te_t.min(), tr_t.min()), min(te_t.max(), tr_t.max())
        tr_in_range = tr_t[(tr_t >= lo) & (tr_t <= hi)]
        if len(tr_in_range) == 0:
            continue
        for t in tr_in_range:
            dists.append(float(np.min(np.abs(te_t - t))))
    if dists:
        pair_medians.append({"tematdb_sample_id": tid, "train_sample_id": sid, "pair_median_distance": float(np.median(dists)), "n_comparisons": len(dists)})

g1_df = pd.DataFrame(pair_medians)
print(f"\ncandidate pairs with >=1 in-range comparison: {len(g1_df)}   (File B: 1,251)", flush=True)
g1_df.to_csv(f"{SCRATCH}\\_step_g1_pair_median_fileA.csv", index=False)

# === N1: composition-identical filtering ===
print("\n=== N1: composition-identical pairing (File A) ===", flush=True)
merged = g1_df.merge(
    a0[["sample_id", "tematdb_composition_id"]].rename(columns={"sample_id": "tematdb_sample_id"}),
    on="tematdb_sample_id", how="left",
).merge(
    train_df[["sample_id", "composition_id"]].rename(columns={"sample_id": "train_sample_id", "composition_id": "train_composition_id"}),
    on="train_sample_id", how="left",
)
merged["composition_match"] = (
    merged["tematdb_composition_id"].notna() & merged["train_composition_id"].notna()
    & (merged["tematdb_composition_id"] == merged["train_composition_id"])
)
matched_pairs = merged[merged["composition_match"]]
samples_with_match = sorted(matched_pairs["tematdb_sample_id"].unique().tolist())
n_samples_with_match = len(samples_with_match)
n_total_matched_pairs = len(matched_pairs)
n_samples_no_match = len(a0) - n_samples_with_match
matches_per_sample = matched_pairs.groupby("tematdb_sample_id").size()
dist = matches_per_sample.value_counts().sort_index()

print(f"teMatDb samples with >=1 composition match: {n_samples_with_match}   (File B: 96)", flush=True)
print(f"teMatDb samples with DOI in training but NO composition match: {n_samples_no_match}   (File B: 80)", flush=True)
print(f"total matched (tematdb_sample_id, train_sample_id) pairs: {n_total_matched_pairs}   (File B: 197)", flush=True)
print("matches-per-sample distribution:", dist.to_dict(), flush=True)

n1_summary = {
    "n_a0_samples": int(len(a0)), "n_a0_parseable": n_a0_parseable,
    "n_candidate_pairs_doi_shared": len(cand_pairs), "n_candidate_pairs_ge1_comparison": len(g1_df),
    "n_candidate_pairs_composition_match": int(n_total_matched_pairs),
    "n_samples_with_ge1_match": int(n_samples_with_match),
    "n_samples_doi_in_training_no_match": int(n_samples_no_match),
    "matches_per_sample_distribution": {str(k): int(v) for k, v in dist.to_dict().items()},
}
with open(f"{SCRATCH}\\_step_n1_summary_fileA.json", "w") as fh:
    json.dump(n1_summary, fh, indent=2)

# === N2: select best pair per sample, build final curves cache (already have train_curves/tematdb_curves) ===
print("\n=== N2: best pair selection + matched points ===", flush=True)
best = matched_pairs.sort_values("pair_median_distance").drop_duplicates(subset=["tematdb_sample_id"], keep="first")
best = best[["tematdb_sample_id", "train_sample_id", "pair_median_distance", "tematdb_composition_id"]].reset_index(drop=True)
print(f"selected pairs (one per matched teMatDb sample): {len(best)}   (File B: 96)", flush=True)

def match_points(tematdb_temps, tematdb_vals, train_temps, train_vals, tol_k=TOL_K):
    if len(train_temps) < 2 or len(tematdb_temps) == 0:
        return [], len(tematdb_temps), 0
    lo, hi = train_temps.min(), train_temps.max()
    survived, n_not_bracketed, n_bracketed_but_far = [], 0, 0
    for t, v in zip(tematdb_temps, tematdb_vals):
        if t < lo or t > hi:
            n_not_bracketed += 1; continue
        nearest_dist = np.min(np.abs(train_temps - t))
        if nearest_dist > tol_k:
            n_bracketed_but_far += 1; continue
        survived.append((t, v, np.interp(t, train_temps, train_vals)))
    return survived, n_not_bracketed, n_bracketed_but_far

declared_records = []
survival_stats = {}
for prop, tepname in [("S", "alpha"), ("sigma", "rho"), ("kappa", "kappa"), ("zT", "ZT")]:
    total_considered = n_survived = n_not_bracketed = n_bracketed_but_far = 0
    for row in best.itertuples(index=False):
        tid, sid = row.tematdb_sample_id, row.train_sample_id
        key_te, key_tr = (tid, tepname), (sid, prop)
        if key_te not in tematdb_curves or key_tr not in train_curves:
            continue
        te_temps, te_vals = tematdb_curves[key_te]
        if tepname == "alpha":
            te_vals = te_vals * 1.0e6
        tr_temps, tr_vals = train_curves[key_tr]
        total_considered += len(te_temps)
        survived, nb, nf = match_points(te_temps, te_vals, tr_temps, tr_vals)
        n_not_bracketed += nb; n_bracketed_but_far += nf; n_survived += len(survived)
        for t, v, iv in survived:
            declared_records.append({"property": prop, "tematdb_sample_id": tid, "train_sample_id": sid,
                                       "temperature": t, "tematdb_value": v, "train_interp_value": iv})
    survival_stats[prop] = {"total_considered": total_considered, "survived": n_survived,
                             "dropped_not_bracketed": n_not_bracketed, "dropped_bracketed_but_over_5K": n_bracketed_but_far}
declared_df = pd.DataFrame(declared_records)
print("N2 survival (declared streams):", survival_stats, flush=True)

tep_records = []
n_zt_considered = n_tep_failed = n_tep_ok = 0
for tid in best["tematdb_sample_id"].unique():
    key_zt, key_a, key_r, key_k = (tid, "ZT"), (tid, "alpha"), (tid, "rho"), (tid, "kappa")
    if key_zt not in tematdb_curves or key_a not in tematdb_curves or key_r not in tematdb_curves or key_k not in tematdb_curves:
        continue
    zt_temps, _ = tematdb_curves[key_zt]
    a_t, a_v = tematdb_curves[key_a]; r_t, r_v = tematdb_curves[key_r]; k_t, k_v = tematdb_curves[key_k]
    n_zt_considered += len(zt_temps)
    for T in zt_temps:
        if not (a_t.min() <= T <= a_t.max() and r_t.min() <= T <= r_t.max() and k_t.min() <= T <= k_t.max()):
            n_tep_failed += 1; continue
        alpha_i = np.interp(T, a_t, a_v)
        rho_i = 1.0 / np.interp(T, r_t, r_v)
        kappa_i = np.interp(T, k_t, k_v)
        zt_tep = (alpha_i ** 2) * T / (rho_i * kappa_i)
        tep_records.append({"tematdb_sample_id": tid, "temperature": T, "zt_tep_value": zt_tep})
        n_tep_ok += 1
tep_df = pd.DataFrame(tep_records)
print(f"zt_tep recompute: considered={n_zt_considered} ok={n_tep_ok} failed={n_tep_failed}", flush=True)

tep_matched_records = []
tep_survival = {"total_considered": 0, "survived": 0, "dropped_not_bracketed": 0, "dropped_bracketed_but_over_5K": 0}
for row in best.itertuples(index=False):
    tid, sid = row.tematdb_sample_id, row.train_sample_id
    key_tr = (sid, "zT")
    if key_tr not in train_curves:
        continue
    sub = tep_df[tep_df["tematdb_sample_id"] == tid]
    if sub.empty:
        continue
    te_temps = sub["temperature"].to_numpy(dtype=np.float64)
    te_vals = sub["zt_tep_value"].to_numpy(dtype=np.float64)
    tr_temps, tr_vals = train_curves[key_tr]
    tep_survival["total_considered"] += len(te_temps)
    survived, nb, nf = match_points(te_temps, te_vals, tr_temps, tr_vals)
    tep_survival["dropped_not_bracketed"] += nb
    tep_survival["dropped_bracketed_but_over_5K"] += nf
    tep_survival["survived"] += len(survived)
    for t, v, iv in survived:
        tep_matched_records.append({"property": "zT_tep", "tematdb_sample_id": tid, "train_sample_id": sid,
                                     "temperature": t, "tematdb_value": v, "train_interp_value": iv})
tep_matched_df = pd.DataFrame(tep_matched_records)
print(f"N2 survival (zT_tep recomputed): {tep_survival}", flush=True)

declared_df.to_csv(f"{SCRATCH}\\_step_n2_matched_points_declared_fileA.csv", index=False)
tep_matched_df.to_csv(f"{SCRATCH}\\_step_n2_matched_points_zt_tep_fileA.csv", index=False)

# === N3: metrics ===
print("\n=== N3: label-to-label agreement (File A) ===", flush=True)
doi_lookup = a0.set_index("sample_id")["DOI"].to_dict()
LOG_PROPS = {"sigma", "kappa"}

def bootstrap_r2_ci(sample_ids, y_true, y_pred, n_boot=2000, seed=0):
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
        n_dropped = int((~valid).sum())
        y_true, y_pred, sample_ids = y_true[valid], y_pred[valid], sample_ids[valid]
        y_true, y_pred = np.log10(y_true), np.log10(y_pred)
    else:
        n_dropped = 0
    if len(y_true) < 2:
        return None
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    doi_set = {doi_lookup.get(sid) for sid in np.unique(sample_ids)}
    ci = bootstrap_r2_ci(sample_ids, y_true, y_pred)
    return {"r2": r2, "rmse": rmse, "mae": mae, "n_points": int(len(y_true)), "n_samples": int(len(np.unique(sample_ids))),
            "n_distinct_dois": int(len(doi_set)), "r2_ci95": list(ci), "n_dropped_nonpositive_for_log": n_dropped}

n3_results = {}
for prop in ["S", "sigma", "kappa", "zT"]:
    n3_results[f"{prop}_full_range"] = compute_metrics(declared_df, prop, restrict_300_800=False)
    n3_results[f"{prop}_300_800K"] = compute_metrics(declared_df, prop, restrict_300_800=True)
n3_results["zT_tep_full_range"] = compute_metrics(tep_matched_df, "zT_tep", restrict_300_800=False)
n3_results["zT_tep_300_800K"] = compute_metrics(tep_matched_df, "zT_tep", restrict_300_800=True)

for k, v in n3_results.items():
    if v is None:
        print(f"  {k}: NO DATA", flush=True)
        continue
    print(f"  {k:20s} R2={v['r2']:+.4f} n={v['n_points']} n_samples={v['n_samples']}", flush=True)

with open(f"{SCRATCH}\\_step_n3_results_fileA.json", "w") as fh:
    json.dump({"n1_summary": n1_summary, "n2_survival": {**survival_stats, "zT_tep": tep_survival}, "n3_results": n3_results}, fh, indent=2)
print("\nWrote _step_n3_results_fileA.json  -- C3_FILEA_DONE", flush=True)
