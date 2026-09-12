# teMatDb section N, step 2: pick the best (smallest pair-median-distance)
# composition-matched training sample per teMatDb sample from step 1's
# output, then build the (sample_id, property) -> (temps, values) curve
# cache from raw training curves (load_raw_curves, File B) and teMatDb's
# rawTEPs.csv, for step 3's point matching.
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\choha\te-ml-pipeline")
from src.data_cleaning import load_raw_curves, step1_extract_and_filter_properties, step2_integrate_and_consolidate

BASE = r"C:\Users\choha\te-ml-pipeline"
SCRATCH = r"C:\Users\choha\AppData\Local\Temp\claude\C--Users-choha-te-ml-pipeline\c4b3e6d8-f82b-4c85-b9c8-4bcf710343a3\scratchpad"
TOL_K = 5.0

# --- pick ONE composition-matched pair per teMatDb sample: smallest pair_median_distance among matches ---
merged = pd.read_csv(f"{SCRATCH}\\_step_n1_candidate_pairs_matched.csv")
matched = merged[merged["composition_match"] == True].copy()
best = matched.sort_values("pair_median_distance").drop_duplicates(subset=["tematdb_sample_id"], keep="first")
best = best[["tematdb_sample_id", "train_sample_id", "pair_median_distance", "tematdb_composition_id"]].reset_index(drop=True)
print(f"selected pairs (one per teMatDb sample, smallest median distance among composition-identical candidates): {len(best)}")
best.to_csv(f"{SCRATCH}\\_step_n2_selected_pairs.csv", index=False)

selected_train_ids = set(best["train_sample_id"].unique().tolist())
selected_tematdb_ids = set(best["tematdb_sample_id"].unique().tolist())
print(f"unique training sample_ids needed: {len(selected_train_ids)}; unique teMatDb sample_ids: {len(selected_tematdb_ids)}")

# --- training raw curves, restricted to the needed sample_ids (huge speedup vs full ~1M row table) ---
curves_df = load_raw_curves()
curves_df = curves_df[curves_df["sample_id"].isin(selected_train_ids)].reset_index(drop=True)
print(f"raw curve rows for needed training sample_ids (pre-filter): {len(curves_df)}")

long1 = step1_extract_and_filter_properties(curves_df)
long2 = step2_integrate_and_consolidate(long1)  # converts rho -> sigma, merges
print(f"training long-format rows after step1+step2: {len(long2)}")
print("properties present:", sorted(long2["property"].unique().tolist()))

train_curves = {}  # (sample_id, property) -> sorted (temps, values)
for (sid, prop), grp in long2.groupby(["sample_id", "property"]):
    g = grp.sort_values("temperature_K")
    train_curves[(sid, prop)] = (g["temperature_K"].to_numpy(dtype=np.float64), g["value"].to_numpy(dtype=np.float64))
print(f"training curves indexed: {len(train_curves)} (sample_id, property) groups")

# --- teMatDb raw curves per property, from rawTEPs.csv ---
raw_tep = pd.read_csv(f"{BASE}\\data\\external\\tematdb\\teMatDb_rawTEPs.csv")
raw_tep = raw_tep[raw_tep["sample_id"].isin(selected_tematdb_ids)].reset_index(drop=True)
print(f"teMatDb rawTEPs rows for selected samples: {len(raw_tep)}")

PROPERTY_MAP = {"alpha": "S", "kappa": "kappa", "ZT": "zT", "rho": "sigma"}
tematdb_curves = {}  # (sample_id, tepname) -> sorted (temps, values), tepname in {alpha,rho,kappa,ZT}
for (sid, tep), grp in raw_tep.groupby(["sample_id", "tepname"]):
    g = grp.sort_values("Temperature")
    temps = g["Temperature"].to_numpy(dtype=np.float64)
    vals = g["tepvalue"].to_numpy(dtype=np.float64)
    if tep == "rho":
        vals = 1.0 / vals  # ohm*m resistivity -> S/m conductivity, same inversion as training's step2
    tematdb_curves[(sid, tep)] = (temps, vals)
print(f"teMatDb curves indexed: {len(tematdb_curves)} (sample_id, tepname) groups")

with open(f"{SCRATCH}\\_step_n2_cache_meta.json", "w") as fh:
    json.dump({
        "n_selected_pairs": len(best),
        "n_train_curve_groups": len(train_curves),
        "n_tematdb_curve_groups": len(tematdb_curves),
    }, fh, indent=2)

import pickle
with open(f"{SCRATCH}\\_step_n2_curves_cache.pkl", "wb") as fh:
    pickle.dump({"train_curves": train_curves, "tematdb_curves": tematdb_curves, "best_pairs": best}, fh)
print("saved curves cache")
