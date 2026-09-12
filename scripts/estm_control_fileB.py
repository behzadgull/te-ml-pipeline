import sys, json, datetime
sys.path.insert(0, r'C:\Users\choha\te-ml-pipeline')
import src.external_validation as ev
from src.nested_cv import get_feature_columns

# CONTROL RUN: identical script to estm_fileA.py, EXCEPT no monkeypatch is
# applied -- ev.load_target_data stays the ORIGINAL function, which defaults
# to PROCESSED_DATA_DIR = data/processed/ (File B). dry_run_inventory() is
# also called with its own default training_csv=TRAINING_CSV (File B).
# This isolates: does the REIMPLEMENTED orchestration sequence (calling
# dry_run_inventory/dedup/compute_training_smear_factors/fit_frozen_external_model/
# score_estm_pass directly, instead of run_full_validation()) reproduce the
# original run_full_validation()-produced results, when pointed at the SAME
# dataset (File B) the original used?

NEW_RESULTS_DIR = "results/20260911T134500_estm_control_fileB"

print(f"=== ESTM CONTROL RUN: reimplemented script, File B (default, unmodified) ===", flush=True)
print(f"TRAINING_CSV (module default) = {ev.TRAINING_CSV}", flush=True)

_, estm_featurized, training_df = ev.dry_run_inventory()  # no override -- File B default
feature_cols = get_feature_columns(training_df)

estm_dedup_a, n_dropped_doi, n_overlap_doi = ev.dedup_by_doi(estm_featurized, training_df)
estm_dedup_b, n_dropped_cluster, n_overlap_cluster = ev.dedup_by_chemistry_cluster(estm_featurized, training_df)

print("\n=== Computing training smear factors (5-fold OOF, File B) ===", flush=True)
smear_sigma, smear_kappa, smear_oof_r2 = ev.compute_training_smear_factors(device="cpu")  # no override -- File B default

print("\n=== Refitting 4 frozen-hyperparameter models on 100% of File B training ===", flush=True)
models = {}
training_rows_used = {}
for target in ev.TARGETS:
    model, _, n_rows = ev.fit_frozen_external_model(target, device="cpu")  # no override -- File B default
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
        "training_csv": ev.TRAINING_CSV,
        "estm_file": str(ev.ESTM_PATH),
        "date": datetime.date.today().isoformat(),
        "note": "CONTROL RUN: reimplemented orchestration script (same one used for File A), pointed at "
                "File B (unmodified default path) to isolate dataset effect from reimplementation effect. "
                "Compare against checkpoints/external_validation/estm_results.json (the original "
                "run_full_validation()-produced result).",
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
}

out_path = f"{NEW_RESULTS_DIR}/estm_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved results to {out_path}")
print("ESTM_CONTROL_FILEB_DONE", flush=True)
