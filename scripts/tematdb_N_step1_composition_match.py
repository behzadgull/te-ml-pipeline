# teMatDb section N, step 1: pair each stratum-a0 sample with every DOI-shared
# training sample_id and flag composition-identity matches (parse_formula ->
# composition_id, both sides). Produced reports/tematdb_inventory's section N
# N1 numbers (File B). Reads _samples_with_doi_flag.csv and round-3's
# _step_g1_pair_median.csv; writes _step_n1_*.csv/.json to scratch.
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\choha\te-ml-pipeline")
from src.canonicalization import parse_formula, composition_id

BASE = r"C:\Users\choha\te-ml-pipeline"
SCRATCH = r"C:\Users\choha\AppData\Local\Temp\claude\C--Users-choha-te-ml-pipeline\c4b3e6d8-f82b-4c85-b9c8-4bcf710343a3\scratchpad"

# --- teMatDb side: a0 samples (DOI in training), canonicalize Composition_detailed ---
samples = pd.read_csv(f"{BASE}\\data\\external\\tematdb\\_samples_with_doi_flag.csv")
a0 = samples[samples["in_training"] == True].copy().reset_index(drop=True)
assert len(a0) == 176, f"expected 176 a0 samples, got {len(a0)}"

comp_ids, errs = [], []
for f in a0["Composition_detailed"]:
    comp, err = parse_formula(f)
    if comp is None:
        comp_ids.append(None)
        errs.append(err)
    else:
        comp_ids.append(composition_id(comp))
        errs.append(None)
a0["tematdb_composition_id"] = comp_ids
a0["tematdb_parse_error"] = errs

n_a0_parseable = a0["tematdb_composition_id"].notna().sum()
print(f"a0 samples: {len(a0)}, parseable (composition_id computed): {n_a0_parseable}, parse failures: {len(a0)-n_a0_parseable}")

a0[["sample_id", "DOI", "DOI_norm", "Composition_detailed", "tematdb_composition_id", "tematdb_parse_error"]].to_csv(
    f"{SCRATCH}\\_step_n1_tematdb_compids.csv", index=False
)

# --- training side: sample_id -> composition_id lookup (from featurized training CSV, dedup) ---
train_lookup = pd.read_csv(
    f"{BASE}\\data\\processed\\featurized_ThermoelectricMaterials_2026-08-15.csv",
    usecols=["sample_id", "DOI", "composition_id", "composition"],
).drop_duplicates(subset=["sample_id"]).reset_index(drop=True)
print(f"training unique sample_id -> composition_id rows: {len(train_lookup)}")
train_lookup.to_csv(f"{SCRATCH}\\_step_n1_train_compids.csv", index=False)

# --- round-3 G1 full candidate-pair table (already DOI-paired, all training sample_ids sharing DOI with an a0 sample) ---
candidates = pd.read_csv(f"{BASE}\\data\\external\\tematdb\\_step_g1_pair_median.csv")
print(f"candidate pairs (from round3 G1, DOI-shared): {len(candidates)}, unique tematdb_sample_id: {candidates['tematdb_sample_id'].nunique()}")

merged = candidates.merge(
    a0[["sample_id", "tematdb_composition_id"]].rename(columns={"sample_id": "tematdb_sample_id"}),
    on="tematdb_sample_id", how="left",
).merge(
    train_lookup[["sample_id", "composition_id"]].rename(columns={"sample_id": "train_sample_id", "composition_id": "train_composition_id"}),
    on="train_sample_id", how="left",
)
merged["composition_match"] = (
    merged["tematdb_composition_id"].notna()
    & merged["train_composition_id"].notna()
    & (merged["tematdb_composition_id"] == merged["train_composition_id"])
)
print(f"candidate pairs with composition MATCH: {merged['composition_match'].sum()} of {len(merged)}")
merged.to_csv(f"{SCRATCH}\\_step_n1_candidate_pairs_matched.csv", index=False)

# --- N1 summary stats ---
matched_pairs = merged[merged["composition_match"]]
samples_with_match = sorted(matched_pairs["tematdb_sample_id"].unique().tolist())
n_samples_with_match = len(samples_with_match)
n_total_matched_pairs = len(matched_pairs)
n_samples_no_match = len(a0) - n_samples_with_match  # a0 total (DOI-in-training) minus those with >=1 match

matches_per_sample = matched_pairs.groupby("tematdb_sample_id").size()
dist = matches_per_sample.value_counts().sort_index()

print("\n=== N1 SUMMARY ===")
print(f"a0 samples (DOI in training): {len(a0)}")
print(f"teMatDb samples with >=1 composition match: {n_samples_with_match}")
print(f"teMatDb samples with DOI in training but NO composition match: {n_samples_no_match}")
print(f"total matched (tematdb_sample_id, train_sample_id) pairs: {n_total_matched_pairs}")
print("distribution of matches-per-sample (matches -> n_samples):")
print(dist.to_string())

import json
n1_summary = {
    "n_a0_samples": int(len(a0)),
    "n_a0_parseable": int(n_a0_parseable),
    "n_samples_with_ge1_match": int(n_samples_with_match),
    "n_samples_doi_in_training_no_match": int(n_samples_no_match),
    "n_total_matched_pairs": int(n_total_matched_pairs),
    "matches_per_sample_distribution": {str(k): int(v) for k, v in dist.to_dict().items()},
    "samples_with_match": samples_with_match,
}
with open(f"{SCRATCH}\\_step_n1_summary.json", "w") as fh:
    json.dump(n1_summary, fh, indent=2)
print("\nsaved _step_n1_summary.json")
