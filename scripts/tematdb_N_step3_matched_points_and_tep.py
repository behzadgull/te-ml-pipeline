# teMatDb section N, step 3 (N2/N3 matched-point construction): for each
# step-2 best-matched pair, linearly interpolate the training curve to each
# teMatDb raw abscissa (bracket + 5K tolerance), for S/sigma/kappa/zT
# declared streams, plus recompute zT_tep from teMatDb's own alpha/rho/kappa
# and match that against training's declared zT. File B.
import pickle
import numpy as np
import pandas as pd

SCRATCH = r"C:\Users\choha\AppData\Local\Temp\claude\C--Users-choha-te-ml-pipeline\c4b3e6d8-f82b-4c85-b9c8-4bcf710343a3\scratchpad"
TOL_K = 5.0

with open(f"{SCRATCH}\\_step_n2_curves_cache.pkl", "rb") as fh:
    cache = pickle.load(fh)
train_curves = cache["train_curves"]
tematdb_curves = cache["tematdb_curves"]
best_pairs = cache["best_pairs"]

PROPERTY_MAP = {"alpha": "S", "kappa": "kappa", "ZT": "zT", "rho": "sigma"}


def match_points(tematdb_temps, tematdb_vals, train_temps, train_vals, tol_k=TOL_K):
    """Bracket + nearest-abscissa-within-tol_k rule, identical to old L1."""
    if len(train_temps) < 2 or len(tematdb_temps) == 0:
        return [], len(tematdb_temps), 0
    lo, hi = train_temps.min(), train_temps.max()
    survived = []
    n_not_bracketed = 0
    n_bracketed_but_far = 0
    for t, v in zip(tematdb_temps, tematdb_vals):
        if t < lo or t > hi:
            n_not_bracketed += 1
            continue
        nearest_dist = np.min(np.abs(train_temps - t))
        if nearest_dist > tol_k:
            n_bracketed_but_far += 1
            continue
        interp_v = np.interp(t, train_temps, train_vals)
        survived.append((t, v, interp_v))
    return survived, n_not_bracketed, n_bracketed_but_far


# === N2 + N3: declared-property matched points (S, sigma, kappa, zT_declared) ===
declared_records = []
survival_stats = {}
for prop, tepname in [("S", "alpha"), ("sigma", "rho"), ("kappa", "kappa"), ("zT", "ZT")]:
    total_considered = 0
    n_survived = 0
    n_not_bracketed = 0
    n_bracketed_but_far = 0
    for row in best_pairs.itertuples(index=False):
        tid, sid = row.tematdb_sample_id, row.train_sample_id
        key_te = (tid, tepname)
        key_tr = (sid, prop)
        if key_te not in tematdb_curves or key_tr not in train_curves:
            continue
        te_temps, te_vals = tematdb_curves[key_te]
        if tepname == "alpha":
            te_vals = te_vals * 1.0e6  # teMatDb alpha is raw V/K; training's S is uV/K (see step1_extract_and_filter_properties's own V/K->uV/K conversion) -- must match units before comparing
        tr_temps, tr_vals = train_curves[key_tr]
        total_considered += len(te_temps)
        survived, nb, nf = match_points(te_temps, te_vals, tr_temps, tr_vals)
        n_not_bracketed += nb
        n_bracketed_but_far += nf
        n_survived += len(survived)
        for t, v, iv in survived:
            declared_records.append({"property": prop, "tematdb_sample_id": tid, "train_sample_id": sid,
                                       "temperature": t, "tematdb_value": v, "train_interp_value": iv})
    survival_stats[prop] = {"total_considered": total_considered, "survived": n_survived,
                             "dropped_not_bracketed": n_not_bracketed, "dropped_bracketed_but_over_5K": n_bracketed_but_far}

declared_df = pd.DataFrame(declared_records)
declared_df.to_csv(f"{SCRATCH}\\_step_n2_matched_points_declared.csv", index=False)
print("=== N2 survival (declared streams: S, sigma, kappa, zT_declared) ===")
for prop, stats in survival_stats.items():
    print(f"  {prop:8s} {stats}")
print(f"total matched points (declared): {len(declared_df)}")

# === zt_tep recompute stream: interpolate teMatDb's OWN alpha/rho/kappa onto its OWN ZT raw abscissa ===
tep_records = []
n_zt_points_considered = 0
n_tep_recompute_failed_bracket = 0
n_tep_recompute_ok = 0
for tid in best_pairs["tematdb_sample_id"].unique():
    key_zt = (tid, "ZT")
    key_alpha = (tid, "alpha")
    key_rho = (tid, "rho")
    key_kappa = (tid, "kappa")
    if key_zt not in tematdb_curves or key_alpha not in tematdb_curves or key_rho not in tematdb_curves or key_kappa not in tematdb_curves:
        continue
    zt_temps, _ = tematdb_curves[key_zt]
    a_t, a_v = tematdb_curves[key_alpha]
    r_t, r_v = tematdb_curves[key_rho]  # already inverted to sigma in step2 script; need RAW rho here, not inverted
    k_t, k_v = tematdb_curves[key_kappa]
    n_zt_points_considered += len(zt_temps)
    for T in zt_temps:
        # bracket check on all three component curves (self-consistency interpolation, same database)
        if not (a_t.min() <= T <= a_t.max() and r_t.min() <= T <= r_t.max() and k_t.min() <= T <= k_t.max()):
            n_tep_recompute_failed_bracket += 1
            continue
        alpha_i = np.interp(T, a_t, a_v)
        rho_i = 1.0 / np.interp(T, r_t, r_v)  # r_v was already inverted to sigma=1/rho during caching; invert back to rho
        kappa_i = np.interp(T, k_t, k_v)
        zt_tep = (alpha_i ** 2) * T / (rho_i * kappa_i)
        tep_records.append({"tematdb_sample_id": tid, "temperature": T, "zt_tep_value": zt_tep})
        n_tep_recompute_ok += 1

print(f"\n=== zt_tep recompute (teMatDb's own alpha/rho/kappa interpolated to its own ZT abscissa) ===")
print(f"ZT raw points considered: {n_zt_points_considered}, recompute succeeded (all 3 components bracket): {n_tep_recompute_ok}, failed (component curve doesn't bracket): {n_tep_recompute_failed_bracket}")

tep_df = pd.DataFrame(tep_records)
tep_df.to_csv(f"{SCRATCH}\\_step_n2_zt_tep_recomputed.csv", index=False)

# now match these recomputed values against TRAINING's declared zT curve, same bracket+5K rule
tep_matched_records = []
tep_survival = {"total_considered": 0, "survived": 0, "dropped_not_bracketed": 0, "dropped_bracketed_but_over_5K": 0}
for row in best_pairs.itertuples(index=False):
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
tep_matched_df.to_csv(f"{SCRATCH}\\_step_n2_matched_points_zt_tep.csv", index=False)
print(f"\n=== N2 survival (zT_tep recomputed stream, vs training declared zT) ===")
print(f"  zT_tep  {tep_survival}")
print(f"total matched points (zT_tep): {len(tep_matched_df)}")

import json
all_survival = dict(survival_stats)
all_survival["zT_tep"] = tep_survival
with open(f"{SCRATCH}\\_step_n2_survival.json", "w") as fh:
    json.dump(all_survival, fh, indent=2)
print("\nsaved survival stats")
