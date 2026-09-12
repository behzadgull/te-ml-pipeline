# ESTM external validation re-run against File A: reimplements
# run_full_validation()'s sequence (dry_run_inventory/dedup/
# compute_training_smear_factors/fit_frozen_external_model/score_estm_pass,
# all reused unmodified from src/external_validation.py) with
# load_target_data monkeypatched to File A's directory (no src/ edit).
# Produced results/20260911T114356_estm_external_fileA/.
import sys, json, functools, datetime
sys.path.insert(0, r'C:\Users\choha\te-ml-pipeline')
import src.external_validation as ev
from src.nested_cv import get_feature_columns

FILE_A_DIR = "checkpoints/saved_predictions/te-ml-pipeline/data/processed"
FILE_A_CSV = f"{FILE_A_DIR}/featurized_ThermoelectricMaterials_2026-08-15.csv"
NEW_RESULTS_DIR = "results/20260911T114356_estm_external_fileA"

# PF2 mechanism: rebind the name inside external_validation's own module
# namespace so compute_training_smear_factors()/fit_frozen_external_model()
# (which call load_target_data(target) with no override) transparently load
# File A instead. Process-local only; no file on disk touched, no src/ edit.
ev.load_target_data = functools.partial(ev.load_target_data, processed_data_dir=FILE_A_DIR)

print(f"=== ESTM external validation, re-run against File A ({FILE_A_CSV}) ===", flush=True)

_, estm_featurized, training_df = ev.dry_run_inventory(training_csv=FILE_A_CSV)
feature_cols = get_feature_columns(training_df)

estm_dedup_a, n_dropped_doi, n_overlap_doi = ev.dedup_by_doi(estm_featurized, training_df)
estm_dedup_b, n_dropped_cluster, n_overlap_cluster = ev.dedup_by_chemistry_cluster(estm_featurized, training_df)

print("\n=== Computing training smear factors (5-fold OOF, File A) ===", flush=True)
smear_sigma, smear_kappa, smear_oof_r2 = ev.compute_training_smear_factors(device="cpu")

print("\n=== Refitting 4 frozen-hyperparameter models on 100% of File A training ===", flush=True)
models = {}
training_rows_used = {}
for target in ev.TARGETS:
    model, _, n_rows = ev.fit_frozen_external_model(target, device="cpu")
    models[target] = model
    training_rows_used[target] = n_rows
    print(f"  {target}: refit on {n_rows:,} rows", flush=True)

print("\n=== Scoring both dedup passes ===", flush=True)
results_a = ev.score_estm_pass(
    estm_dedup_a, feature_cols, models, smear_sigma, smear_kappa, "Pass (a) source-DOI dedup",
    save_path=f"{NEW_RESULTS_DIR}/estm_predictions_passa.npz",
)
results_b = ev.score_estm_pass(
    estm_dedup_b, feature_cols, models, smear_sigma, smear_kappa, "Pass (b) composition-cluster dedup",
    save_path=f"{NEW_RESULTS_DIR}/estm_predictions_passb.npz",
)

output = {
    "provenance": {
        "frozen_hyperparams_paths": {t: str(ev.FROZEN_HYPERPARAMS_DIR / f"{t}.json") for t in ev.TARGETS},
        "training_csv": FILE_A_CSV,
        "training_csv_sha256": "e97406fa5223efa466d1e3fe4ffb2b1bc0303dc94b6adac6a824a643a9156f8d",
        "estm_file": str(ev.ESTM_PATH),
        "date": datetime.date.today().isoformat(),
        "note": "Re-run against File A per CLAUDE.md's Canonical Dataset section (commit e1ede95). "
                "Original run (checkpoints/external_validation/estm_results.json) used File B and is preserved unchanged.",
    },
    "training_rows_used_for_refit": training_rows_used,
    "smear_factors": {"sigma": smear_sigma, "kappa": smear_kappa},
    "smear_training_oof_r2": smear_oof_r2,
    "unit_handling": ev.UNIT_NOTE,
    "dedup_a_source_doi": {
        "n_dropped": n_dropped_doi,
        "n_surviving": len(estm_dedup_a),
        "unique_estm_dois_overlapping_training": n_overlap_doi,
        "results": results_a,
    },
    "dedup_b_chemistry_cluster": {
        "n_dropped": n_dropped_cluster,
        "n_surviving": len(estm_dedup_b),
        "unique_estm_clusters_overlapping_training": n_overlap_cluster,
        "results": results_b,
    },
    "zt_is_derived_not_measured_note": (
        "ESTM's ZT column is algebraically S^2*sigma*T/kappa (confirmed by spot "
        "check, matches to <0.6%), not an independent measurement -- the "
        "zT_direct/zT_derived numbers above are internal consistency checks, not "
        "independent external validation the way S/sigma/kappa are."
    ),
}

out_path = f"{NEW_RESULTS_DIR}/estm_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved results to {out_path}")
print("ESTM_FILEA_DONE", flush=True)
