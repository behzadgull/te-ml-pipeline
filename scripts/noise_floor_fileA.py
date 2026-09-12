# Re-runs src/noise_floor.py's compute_all() against File A's cleaned CSV
# (no src/ edit -- load_cleaned_dataset's existing processed_data_dir
# parameter is passed directly). Produced
# results/noise_floor/20260911T114356/noise_floor_inputs.json.
import sys, json, hashlib
sys.path.insert(0, r'C:\Users\choha\te-ml-pipeline')
from src.noise_floor import compute_all, report, PAPER_SCALE, CONFIRMED_CHEMISTRY_CLUSTER_R2, load_cleaned_dataset

FILE_A_DIR = "checkpoints/saved_predictions/te-ml-pipeline/data/processed"
FILE_A_CLEANED = f"{FILE_A_DIR}/cleaned_ThermoelectricMaterials_2026-08-15.csv"

df, source_path = load_cleaned_dataset(processed_data_dir=FILE_A_DIR)
print(f"Loaded {len(df):,} rows from {source_path}")

h = hashlib.sha256()
with open(source_path, 'rb') as f:
    while True:
        chunk = f.read(1 << 20)
        if not chunk:
            break
        h.update(chunk)
sha = h.hexdigest()
print(f"SHA256: {sha}")

results = compute_all(df)
rows = report(results)

output = {
    "generated_at_utc": "2026-09-11T11:43:56",
    "input_dataset": {
        "path": str(source_path),
        "sha256": sha,
        "n_rows": len(df),
        "note": "File A's cleaned CSV, per CLAUDE.md's Canonical Dataset section (commit e1ede95). "
                "Supersedes the 20260910T134042 artifact, which used File B.",
    },
    "confirmed_chemistry_cluster_r2_source": {
        "checkpoint_dir": "checkpoints/saved_predictions/checkpoints/{S,sigma,kappa,zT}_chemistry/",
        "values": CONFIRMED_CHEMISTRY_CLUSTER_R2,
        "note": "unchanged from the 20260910T134042 artifact -- these come from File A already (the confirmed ladder), not File B, so no change needed here.",
    },
    "results": {},
}

for prop in ["S", "sigma", "kappa", "zT"]:
    log_res = results[prop]["log"]
    lin_res = results[prop]["linear"]
    scale = PAPER_SCALE[prop]
    paper_r2_max = log_res["r2_max"] if scale == "log" else lin_res["r2_max"]
    confirmed = CONFIRMED_CHEMISTRY_CLUSTER_R2[prop]
    output["results"][prop] = {
        "log": {k: v for k, v in log_res.items() if k != "property"},
        "linear": {k: v for k, v in lin_res.items() if k != "property"},
        "paper_scale": scale,
        "paper_r2_max": paper_r2_max,
        "confirmed_chemistry_cluster_r2": confirmed,
        "headroom": paper_r2_max - confirmed,
    }

with open("results/noise_floor/20260911T114356/noise_floor_inputs.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nWrote results/noise_floor/20260911T114356/noise_floor_inputs.json")
