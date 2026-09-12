# teMatDb DOI overlap (C1) and cluster overlap (C2) recomputed against File A
# training data, compared against the original File B counts. Confirmed
# every stratum size (|a0|/|a|, parsed/failed, cluster overlap, the 36-sample
# stratum-b set) is identical between snapshots. Read-only, no model touch.
# Feeds results/20260911T114356_tematdb_inventory_fileA/inventory_fileA.json.
import sys, re, json
import numpy as np
import pandas as pd
sys.path.insert(0, r'C:\Users\choha\te-ml-pipeline')
from src.canonicalization import parse_formula, chemistry_cluster_id, DEFAULT_DOPANT_THRESHOLD_FRAC
from src.data_cleaning import step3_filter_temperature

FILE_A_CSV = "checkpoints/saved_predictions/te-ml-pipeline/data/processed/featurized_ThermoelectricMaterials_2026-08-15.csv"

def normalize_doi_inventory(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"^https?://dx\.doi\.org/", "", s)
    s = re.sub(r"^https?://doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s.strip()

# ---- load File A training data ----
training_df = pd.read_csv(FILE_A_CSV)
training_dois = set(training_df["DOI"].map(normalize_doi_inventory)) - {""}
training_clusters = set(training_df["chemistry_cluster_id"].dropna())
print(f"File A: {len(training_df):,} rows, {len(training_dois):,} unique DOIs, {len(training_clusters):,} unique chemistry_cluster_id values")

# ---- teMatDb samples ----
samples = pd.read_csv("data/external/tematdb/teMatDb_samples.csv")
samples["doi_norm"] = samples["DOI"].map(normalize_doi_inventory)

a0_mask = samples["doi_norm"].isin(training_dois) & (samples["doi_norm"] != "")
n_a0 = int(a0_mask.sum())
n_a = int((~a0_mask).sum())
print(f"\n=== C1: DOI overlap (File A) ===")
print(f"|a0| (DOI in training) = {n_a0}   |a| (DOI not in training) = {n_a}")
print(f"File B comparison: |a0|=176  |a|=96")

# ---- parse formulas (independent of training data -- should match File B exactly) ----
comp_ids, cluster_ids, errors = [], [], []
for formula in samples["Composition_detailed"]:
    comp, err = parse_formula(formula)
    if comp is None:
        comp_ids.append(None); cluster_ids.append(None); errors.append(err)
    else:
        cluster_ids.append(chemistry_cluster_id(comp, DEFAULT_DOPANT_THRESHOLD_FRAC))
        comp_ids.append(str(comp)); errors.append(None)
samples["cluster_id"] = cluster_ids
samples["parse_error"] = errors

n_parsed = int(samples["parse_error"].isna().sum())
n_failed = int(samples["parse_error"].notna().sum())
print(f"\n=== C2: Cluster overlap (File A) ===")
print(f"parsed={n_parsed}  failed={n_failed}   (File B: parsed=184 failed=88 -- should be identical, training-independent)")

parseable = samples[samples["parse_error"].isna()].copy()
unique_clusters_among_successes = set(parseable["cluster_id"].dropna())
print(f"unique clusters among successes = {len(unique_clusters_among_successes)}   (File B: 156)")

not_in_training_mask = ~parseable["cluster_id"].isin(training_clusters)
stratum_b = parseable[not_in_training_mask]
n_not_in_training = len(stratum_b)
print(f"unique clusters among successes NOT in training = {n_not_in_training}   (File B: 36)")

# ---- a0/a intersect parseable ----
a0_ids = set(samples.loc[a0_mask, "sample_id"])
a_ids = set(samples.loc[~a0_mask, "sample_id"])
parseable_ids = set(parseable["sample_id"])
a0_parseable_ids = a0_ids & parseable_ids
a_parseable_ids = a_ids & parseable_ids
print(f"\n|a0 ∩ parseable| = {len(a0_parseable_ids)}   (File B: 122)")
print(f"|a ∩ parseable| = {len(a_parseable_ids)}   (File B: 62)")

# ---- stratum change detection: which sample_ids differ between File A and File B strata ----
# Exact File-B stratum-b membership, read from corrections.json K1_cross_tab
# (the union of b_and_a0 + b_and_a_CLEAN sample_ids -- both subsets of stratum b, disjoint, summing to 36).
with open("results/20260910T123047_tematdb_external/corrections.json") as _f:
    _corr = json.load(_f)
_k1 = _corr["K_clean_chemistry_transfer_stratum"]["K1_cross_tab"]
FILE_B_STRATUM_B_SAMPLE_IDS = set(_k1["b_and_a0_DOI_in_training"]["sample_ids"]) | set(_k1["b_and_a_DOI_not_in_training_CLEAN"]["sample_ids"])
assert len(FILE_B_STRATUM_B_SAMPLE_IDS) == 36, f"expected 36, got {len(FILE_B_STRATUM_B_SAMPLE_IDS)}"
fileA_stratum_b_ids = set(stratum_b["sample_id"])
added = fileA_stratum_b_ids - FILE_B_STRATUM_B_SAMPLE_IDS
removed = FILE_B_STRATUM_B_SAMPLE_IDS - fileA_stratum_b_ids
print(f"\nStratum-b sample_id changes (File B -> File A): added={sorted(added)}  removed={sorted(removed)}")
print(f"File A stratum-b sample_ids ({len(fileA_stratum_b_ids)}): {sorted(fileA_stratum_b_ids)}")

# ---- thinned rows ----
collocated = pd.read_csv("data/external/tematdb/teMatDb_collocatedTEPs.csv")
collocated_renamed = collocated.rename(columns={"Temperature": "temperature_K"})
filtered = step3_filter_temperature(collocated_renamed)
thinned = filtered.drop_duplicates(subset=["sample_id", "temperature_bin"])

def thinned_count(sample_ids):
    return len(thinned[thinned["sample_id"].isin(sample_ids)])

n_thinned_a0_parseable = thinned_count(a0_parseable_ids)
n_thinned_a_parseable = thinned_count(a_parseable_ids)
n_thinned_b = thinned_count(fileA_stratum_b_ids)
print(f"\nThinned rows: a0∩parseable={n_thinned_a0_parseable} (File B: 2,027)   a∩parseable={n_thinned_a_parseable} (File B: 903)   b={n_thinned_b} (File B: 549)")

result = {
    "training_csv": FILE_A_CSV,
    "n_training_rows": len(training_df),
    "n_training_dois": len(training_dois),
    "n_training_clusters": len(training_clusters),
    "C1_doi_overlap": {"n_a0": n_a0, "n_a": n_a, "n_a0_parseable": len(a0_parseable_ids), "n_a_parseable": len(a_parseable_ids),
                        "thinned_a0_parseable": n_thinned_a0_parseable, "thinned_a_parseable": n_thinned_a_parseable},
    "C2_cluster_overlap": {"n_parsed": n_parsed, "n_failed": n_failed,
                            "n_unique_clusters_among_successes": len(unique_clusters_among_successes),
                            "n_not_in_training": n_not_in_training, "thinned_b": n_thinned_b,
                            "stratum_b_sample_ids": sorted(fileA_stratum_b_ids),
                            "stratum_b_changes": {"added": sorted(added), "removed": sorted(removed)}},
}
with open(r"C:\Users\choha\AppData\Local\Temp\claude\C--Users-choha-te-ml-pipeline\c4b3e6d8-f82b-4c85-b9c8-4bcf710343a3\scratchpad\tematdb_C1_C2_fileA.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nWrote scratch result json (scratchpad, not in results/ yet -- will be merged into final C output)")
