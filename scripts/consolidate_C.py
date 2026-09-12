# Merges tematdb_C1_C2_fileA.py's and tematdb_C3_fileA.py's scratch JSON
# fragments into the single final
# results/20260911T114356_tematdb_inventory_fileA/inventory_fileA.json,
# with the File-A-vs-File-B comparison fields. Pure I/O, no new computation.
import json

SCRATCH = r"C:\Users\choha\AppData\Local\Temp\claude\C--Users-choha-te-ml-pipeline\c4b3e6d8-f82b-4c85-b9c8-4bcf710343a3\scratchpad"

with open(f"{SCRATCH}\\tematdb_C1_C2_fileA.json") as f:
    c1c2 = json.load(f)
with open(f"{SCRATCH}\\_step_n1_summary_fileA.json") as f:
    n1 = json.load(f)
with open(f"{SCRATCH}\\_step_n3_results_fileA.json") as f:
    n3_full = json.load(f)

output = {
    "training_csv": "checkpoints/saved_predictions/te-ml-pipeline/data/processed/featurized_ThermoelectricMaterials_2026-08-15.csv",
    "training_csv_sha256": "e97406fa5223efa466d1e3fe4ffb2b1bc0303dc94b6adac6a824a643a9156f8d",
    "note": "Recompute of teMatDb inventory strata (C1/C2) and composition-matched digitization agreement "
            "(C3, section N) against File A, per CLAUDE.md Canonical Dataset section (commit e1ede95). "
            "Original results (reports/tematdb_inventory/, results/20260910T123047_tematdb_external/) used "
            "File B and are preserved unchanged.",
    "C1_doi_overlap": c1c2["C1_doi_overlap"],
    "C2_cluster_overlap": c1c2["C2_cluster_overlap"],
    "C3_composition_matched_digitization_agreement": {
        "N1_identity_pairing": n1,
        "N2_matched_point_survival": n3_full["n2_survival"],
        "N3_label_agreement": n3_full["n3_results"],
    },
    "strata_comparison_vs_fileB": {
        "n_a0": {"fileB": 176, "fileA": c1c2["C1_doi_overlap"]["n_a0"]},
        "n_a": {"fileB": 96, "fileA": c1c2["C1_doi_overlap"]["n_a"]},
        "n_a0_parseable": {"fileB": 122, "fileA": c1c2["C1_doi_overlap"]["n_a0_parseable"]},
        "n_a_parseable": {"fileB": 62, "fileA": c1c2["C1_doi_overlap"]["n_a_parseable"]},
        "n_parsed": {"fileB": 184, "fileA": c1c2["C2_cluster_overlap"]["n_parsed"]},
        "n_failed": {"fileB": 88, "fileA": c1c2["C2_cluster_overlap"]["n_failed"]},
        "n_unique_clusters_among_successes": {"fileB": 156, "fileA": c1c2["C2_cluster_overlap"]["n_unique_clusters_among_successes"]},
        "n_not_in_training_clusters": {"fileB": 36, "fileA": c1c2["C2_cluster_overlap"]["n_not_in_training"]},
        "stratum_b_sample_id_changes": c1c2["C2_cluster_overlap"]["stratum_b_changes"],
        "thinned_rows_a0": {"fileB": 2027, "fileA": c1c2["C1_doi_overlap"]["thinned_a0_parseable"]},
        "thinned_rows_a": {"fileB": 903, "fileA": c1c2["C1_doi_overlap"]["thinned_a_parseable"]},
        "thinned_rows_b": {"fileB": 549, "fileA": c1c2["C2_cluster_overlap"]["thinned_b"]},
        "n1_matched_sample_denominator": {"fileB": "96 of 176", "fileA": f"{n1['n_samples_with_ge1_match']} of {n1['n_a0_samples']}"},
        "r2_agree_300_800K": {
            "S": {"fileB": 0.9628, "fileA": n3_full["n3_results"]["S_300_800K"]["r2"]},
            "sigma": {"fileB": 0.9777, "fileA": n3_full["n3_results"]["sigma_300_800K"]["r2"]},
            "kappa": {"fileB": 0.9806, "fileA": n3_full["n3_results"]["kappa_300_800K"]["r2"]},
            "zT_declared": {"fileB": 0.9597, "fileA": n3_full["n3_results"]["zT_300_800K"]["r2"]},
            "zT_tep": {"fileB": 0.9607, "fileA": n3_full["n3_results"]["zT_tep_300_800K"]["r2"]},
        },
    },
}

with open("results/20260911T114356_tematdb_inventory_fileA/inventory_fileA.json", "w") as f:
    json.dump(output, f, indent=2)
print("Wrote results/20260911T114356_tematdb_inventory_fileA/inventory_fileA.json")
