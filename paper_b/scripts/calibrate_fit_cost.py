"""
Kaggle calibration of model fit cost for the Paper B compute estimate
(methodology doc, section 8.4). Trains nothing that is kept: it only times fits.

Loads the snapfix CSV by explicit path after a SHA256 and size check (identity
read from paper_b/SHARED_DEPENDENCIES.md), keeps the rows with a non-null target
(default zT) and the Paper A feature columns (397: MAGPIE, CBFV, temperature_bin),
and times:

  * XGBoost fits on nested random subsets of n rows for n in {2k, 8k, 15k, 40k,
    150k} and (max_depth, n_estimators) in {(3, 100), (6, 350), (10, 600)}, three
    timings each, median recorded. The other hyperparameters are Paper A's frozen
    values for the target. Each timing is one fit plus one prediction on a fixed
    disjoint 5,000-row sample, as in Paper A's per-fold times. n is capped at the
    rows available less the 5,000 held out (zT has fewer than 150k rows); n_fit
    records what was used.
  * One StandardScaler + Ridge fit on CPU at n = 150k (same cap), plus 15k and 40k
    to check that ridge cost is linear in n.

Models are built with src.nested_cv.MODEL_REGISTRY, so the constructor and
arguments are Paper A's. Data placement follows Paper A's `_to_device`: cupy if it
imports, otherwise host arrays (XGBoost then bridges to the GPU on every call).
If cupy imports, both placements are timed and recorded separately.

Records the GPU (nvidia-smi), library versions, dataset SHA256, git HEAD and
tree_clean, and writes calibration_timings.csv, calibration_medians.csv,
run_config.json and a zip of the folder. Stops starting new timings after
--budget-minutes (default 25) and records that it was truncated.

Run from the repository root:
    python paper_b/scripts/calibrate_fit_cost.py --csv <snapfix csv> --out-dir /kaggle/working/calibration
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, so `src` imports when run by path
from src.nested_cv import MODEL_REGISTRY, get_feature_columns  # noqa: E402

MANIFEST_PATH = Path("paper_b/SHARED_DEPENDENCIES.md")
FROZEN_DIR = Path("checkpoints/saved_predictions/checkpoints/frozen_hyperparams")
N_GRID = (2_000, 8_000, 15_000, 40_000, 150_000)
CONFIGS = ((3, 100), (6, 350), (10, 600))  # (max_depth, n_estimators)
RIDGE_N = (15_000, 40_000, 150_000)
TIMINGS = 3
N_PREDICT = 5_000
SEED = 0
FROZEN_KEYS = ("learning_rate", "subsample", "colsample_bytree", "min_child_weight", "reg_lambda", "reg_alpha")


def sha256_file(path, chunk=1 << 24):
    """SHA256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def expected_dataset_identity():
    """(sha256, bytes) of the snapfix CSV from the manifest's data table."""
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0] == "snapfix featurized CSV":
            return cells[2], int(cells[3].replace(",", ""))
    raise ValueError(f"no snapfix data row in {MANIFEST_PATH}")


def run(cmd):
    """Stdout of a command, or None if it cannot run."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def system_info():
    """GPU, library and machine information for the run_config."""
    info = {
        "python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"]),
        "nvidia_smi_L": run(["nvidia-smi", "-L"]),
    }
    for name in ("numpy", "pandas", "sklearn", "xgboost", "optuna", "cupy", "threadpoolctl"):
        try:
            info[name] = __import__(name).__version__
        except Exception as exc:
            info[name] = f"unavailable ({type(exc).__name__})"
    try:
        from threadpoolctl import threadpool_info
        info["threadpools"] = threadpool_info()
    except Exception:
        info["threadpools"] = None
    return info


def load_data(csv, target):
    """Load the CSV after the identity check; return (X float64, y float64, feature columns, identity)."""
    expected_sha, expected_bytes = expected_dataset_identity()
    size = Path(csv).stat().st_size
    if size != expected_bytes:
        raise ValueError(f"{csv}: {size} bytes, manifest says {expected_bytes}")
    sha = sha256_file(csv)
    if sha != expected_sha:
        raise ValueError(f"{csv}: SHA256 {sha} differs from the manifest {expected_sha}")
    features = get_feature_columns(pd.read_csv(csv, nrows=0))
    df = pd.read_csv(csv, usecols=[*features, target])
    df = df[df[target].notna()]
    X = df[features].to_numpy(dtype=np.float64)
    y = df[target].to_numpy(dtype=np.float64)
    return X, y, features, {"path": str(csv), "sha256": sha, "bytes": size, "n_rows_with_target": int(len(df))}


def to_placement(arr, placement):
    """cupy array for placement 'cupy', the numpy array itself for 'host'."""
    if placement == "cupy":
        import cupy as cp
        return cp.asarray(arr)
    return arr


def sync(placement):
    """Wait for the GPU if arrays live on it."""
    if placement == "cupy":
        import cupy as cp
        cp.cuda.Stream.null.synchronize()


def time_fit(build, X_train, y_train, X_pred, placement):
    """One timing: seconds for fit and for predict."""
    model = build()
    start = time.perf_counter()
    model.fit(X_train, y_train)
    sync(placement)
    mid = time.perf_counter()
    model.predict(X_pred)
    sync(placement)
    return mid - start, time.perf_counter() - mid


def main(argv=None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", required=True, help="the snapfix featurized CSV")
    parser.add_argument("--out-dir", default=None, help="output folder (default paper_b/results/calibration/<UTC>)")
    parser.add_argument("--target", default="zT")
    parser.add_argument("--device", default="cuda", help="XGBoost device (cpu only for a local smoke test)")
    parser.add_argument("--budget-minutes", type=float, default=25.0)
    parser.add_argument("--smoke", action="store_true", help="tiny grid, for a local check of the script")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("paper_b/results/calibration") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, features, identity = load_data(args.csv, args.target)
    print(f"loaded {X.shape[0]:,} rows x {X.shape[1]} features for {args.target} in {time.perf_counter() - started:.0f} s", flush=True)
    frozen_path = FROZEN_DIR / f"{args.target}.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))["best_params"]
    base_params = {key: frozen[key] for key in FROZEN_KEYS}

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(y))
    pred_idx, pool = order[:N_PREDICT], order[N_PREDICT:]
    n_grid = (2_000,) if args.smoke else N_GRID
    configs = ((3, 100),) if args.smoke else CONFIGS
    ridge_n = (15_000,) if args.smoke else RIDGE_N

    placements = ["host"]
    if args.device == "cuda":
        try:
            import cupy  # noqa: F401
            placements.append("cupy")
        except Exception as exc:
            print(f"cupy unavailable ({type(exc).__name__}); timing host arrays only, as Paper A did", flush=True)
    xgb_build = MODEL_REGISTRY["xgboost"]["build"]
    ridge_build = MODEL_REGISTRY["ridge"]["build"]

    rows, truncated = [], False

    def over_budget():
        return (time.perf_counter() - started) / 60.0 > args.budget_minutes

    warm = to_placement(X[pool[:2000]], placements[0]), to_placement(y[pool[:2000]], placements[0])
    time_fit(lambda: xgb_build({**base_params, "n_estimators": 20, "max_depth": 3}, args.device), *warm,
             to_placement(X[pred_idx], placements[0]), placements[0])  # untimed warm-up: CUDA initialisation

    for placement in placements:
        for n in n_grid:
            n_fit = min(n, len(pool))
            train_idx = pool[:n_fit]
            Xt, yt = to_placement(X[train_idx], placement), to_placement(y[train_idx], placement)
            Xp = to_placement(X[pred_idx], placement)
            for depth, trees in configs:
                if over_budget():
                    truncated = True
                    continue
                params = {**base_params, "n_estimators": trees, "max_depth": depth}
                for repeat in range(TIMINGS):
                    fit_s, pred_s = time_fit(lambda: xgb_build(params, args.device), Xt, yt, Xp, placement)
                    rows.append({"model": "xgboost", "device": args.device, "placement": placement, "n_requested": n,
                                 "n_fit": n_fit, "max_depth": depth, "n_estimators": trees, "repeat": repeat,
                                 "fit_seconds": fit_s, "predict_seconds": pred_s})
                print(f"xgboost {placement} n={n_fit:,} depth={depth} trees={trees}: fit medians "
                      f"{np.median([r['fit_seconds'] for r in rows[-TIMINGS:]]):.2f} s "
                      f"[{(time.perf_counter() - started) / 60:.1f} min elapsed]", flush=True)
            del Xt, yt, Xp

    Xp = X[pred_idx]
    for n in ridge_n:
        if over_budget():
            truncated = True
            continue
        n_fit = min(n, len(pool))
        Xt, yt = X[pool[:n_fit]], y[pool[:n_fit]]
        for repeat in range(TIMINGS):
            fit_s, pred_s = time_fit(lambda: ridge_build({"alpha": 1.0}, "cpu"), Xt, yt, Xp, "host")
            rows.append({"model": "ridge", "device": "cpu", "placement": "host", "n_requested": n, "n_fit": n_fit,
                         "max_depth": None, "n_estimators": None, "repeat": repeat,
                         "fit_seconds": fit_s, "predict_seconds": pred_s})
        print(f"ridge n={n_fit:,}: fit median {np.median([r['fit_seconds'] for r in rows[-TIMINGS:]]):.2f} s", flush=True)

    timings = pd.DataFrame(rows)
    keys = ["model", "device", "placement", "n_requested", "n_fit", "max_depth", "n_estimators"]
    medians = timings.groupby(keys, dropna=False)[["fit_seconds", "predict_seconds"]].median().reset_index()
    timings.to_csv(out_dir / "calibration_timings.csv", index=False)
    medians.to_csv(out_dir / "calibration_medians.csv", index=False)
    git_head = run(["git", "rev-parse", "HEAD"])
    porcelain = run(["git", "status", "--porcelain"])
    config = {
        "target": args.target, "device": args.device, "placements_timed": placements, "seed": SEED,
        "n_predict_rows": N_PREDICT, "timings_per_config": TIMINGS, "n_grid": list(n_grid), "configs_depth_trees": [list(c) for c in configs],
        "ridge_n": list(ridge_n), "ridge_alpha": 1.0, "n_features": len(features), "dataset": identity,
        "frozen_hyperparams_file": str(frozen_path), "frozen_hyperparams_sha256": sha256_file(frozen_path),
        "frozen_params_used": base_params, "script_sha256": sha256_file(Path(__file__)),
        "git_head": git_head, "tree_clean": (porcelain == "") if porcelain is not None else None,
        "dirty_files": porcelain.splitlines() if porcelain else [], "budget_minutes": args.budget_minutes,
        "truncated_by_budget": truncated, "total_minutes": (time.perf_counter() - started) / 60.0,
        "smoke": args.smoke, "utc_stamp": stamp, "system": system_info(),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    archive = shutil.make_archive(str(out_dir), "zip", out_dir)
    print(f"\nWrote {out_dir} and {archive}; total {config['total_minutes']:.1f} min; truncated={truncated}", flush=True)
    pd.set_option("display.width", 200)
    print(medians.to_string(index=False))


if __name__ == "__main__":
    main()
