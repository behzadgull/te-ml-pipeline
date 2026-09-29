"""
Throughput calibration for two compute options not covered by
calibrate_fit_cost.py (methodology doc section 8, compute estimate):

  (a) k concurrent small XGBoost fits on one GPU, for specialists and the
      inner-tuning fits (both are many small, independent fits; if the GPU has
      idle capacity per fit, running several at once could shorten wall time
      without changing the seconds-of-GPU-time the compute estimate counts).
  (b) specialists run on CPU instead of GPU (a Kaggle CPU session, so no GPU
      queue at all, at the cost of a slower per-fit time).

Trains nothing that is kept; it only times fits. For a given --device (cuda or
cpu) and --concurrency k, at two representative small-fit configs (the same
(max_depth, n_estimators) bundles calibrate_fit_cost.py calibrated):
  * "tuning"     n=2,000,  max_depth=6,  n_estimators=350  (inner-tuning-fold scale)
  * "specialist" n=15,000, max_depth=10, n_estimators=600  (specialist-fold scale)
times, 3 repeats each:
  * SEQUENTIAL: k such fits run one after another, in this process
  * CONCURRENT: the same k fits run at once, one per subprocess
and reports speedup = median(sequential total) / median(concurrent total), per
config. k=1 is a sanity check (speedup must be ~1).

On device=cpu, each concurrent worker is capped to max(1, cpu_count // k)
threads (xgboost n_jobs), to avoid k workers each claiming every core; override
with --cpu-jobs-per-worker. On device=cuda, workers keep the default n_jobs
(host threads are not the shared resource there; the GPU itself is) unless
--cuda-jobs-per-worker is given.

This script does not decide the speedup; it only measures it. Loads the
snapfix CSV (SHA-checked) once, for zT's frozen hyperparameters and rows, as
calibrate_fit_cost.py does. Records the GPU/CPU, library versions, dataset
SHA256, git HEAD, tree_clean. Writes a CSV and a run_config into a bundle to
download. Checks --budget-minutes (default 12) before each repeat's sequential
half and again before its concurrent half, not only between the two named
configs (2026-09-29 fix: the first version only checked between configs, so a
single slow config -- device=cpu, concurrency=4, specialist scale -- ran for
26 minutes against a 12-minute budget with truncated_by_budget left false).

Run from the repository root:
    python paper_b/scripts/calibrate_throughput.py --csv <snapfix csv> --device cuda --concurrency 2 \
        --out-dir /kaggle/working/throughput_cuda_k2
    python paper_b/scripts/calibrate_throughput.py --csv <snapfix csv> --device cpu --concurrency 4 \
        --out-dir /kaggle/working/throughput_cpu_k4
"""

import argparse
import hashlib
import json
import multiprocessing as mp
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
import xgboost as xgb

MANIFEST_PATH = Path("paper_b/SHARED_DEPENDENCIES.md")
FROZEN_DIR = Path("checkpoints/saved_predictions/checkpoints/frozen_hyperparams")
CONFIGS = {"tuning": (2_000, 6, 350), "specialist": (15_000, 10, 600)}  # name: (n, max_depth, n_estimators)
TIMINGS = 3
N_PREDICT = 2_000
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
    for name in ("numpy", "pandas", "xgboost"):
        try:
            info[name] = __import__(name).__version__
        except Exception as exc:
            info[name] = f"unavailable ({type(exc).__name__})"
    return info


def get_features(csv):
    """The Paper A feature column names (MagpieData/CBFV_ prefixed), read from the CSV header only."""
    header = pd.read_csv(csv, nrows=0).columns
    return [c for c in header if c.startswith(("MagpieData", "CBFV_"))] + ["temperature_bin"]


def load_data(csv, target):
    """Load the CSV after the identity check; return (X float64, y float64, identity)."""
    expected_sha, expected_bytes = expected_dataset_identity()
    size = Path(csv).stat().st_size
    if size != expected_bytes:
        raise ValueError(f"{csv}: {size} bytes, manifest says {expected_bytes}")
    sha = sha256_file(csv)
    if sha != expected_sha:
        raise ValueError(f"{csv}: SHA256 {sha} differs from the manifest {expected_sha}")
    features = get_features(csv)
    df = pd.read_csv(csv, usecols=[*features, target])
    df = df[df[target].notna()]
    X = df[features].to_numpy(dtype=np.float64)
    y = df[target].to_numpy(dtype=np.float64)
    return X, y, {"path": str(csv), "sha256": sha, "bytes": size, "n_rows_with_target": int(len(df))}


def _worker(args):
    """
    One timed fit + predict, run in its own process (module-level, so it
    pickles with the default 'spawn' start method). `args` is (X_train,
    y_train, X_pred, params, device, n_jobs). Returns (fit_seconds, pid).
    """
    X_train, y_train, X_pred, params, device, n_jobs = args
    model = xgb.XGBRegressor(**params, n_jobs=n_jobs, tree_method="hist", device=device, random_state=0,
                              objective="reg:squarederror")
    start = time.perf_counter()
    model.fit(X_train, y_train)
    model.predict(X_pred)
    return time.perf_counter() - start, os.getpid()


def build_worker_args(X, y, pred_idx, pool, config_name, k, seed, params, device, n_jobs):
    """k (X_train, y_train, X_pred, params, device, n_jobs) tuples, one per worker, distinct row slices."""
    n, depth, trees = CONFIGS[config_name]
    rng = np.random.default_rng(seed)
    args = []
    for worker in range(k):
        idx = rng.choice(pool, size=min(n, len(pool)), replace=False)
        args.append((X[idx], y[idx], X[pred_idx], {**params, "max_depth": depth, "n_estimators": trees}, device, n_jobs))
    return args


def time_sequential(worker_args):
    """Total seconds to run every worker's fit one after another, in this process."""
    total = 0.0
    for args in worker_args:
        seconds, _ = _worker(args)
        total += seconds
    return total


def time_concurrent(worker_args, mp_context):
    """Wall-clock seconds for all workers' fits run at once, one process per worker (max of their individual times)."""
    with mp_context.Pool(processes=len(worker_args)) as pool:
        start = time.perf_counter()
        results = pool.map(_worker, worker_args)
        wall = time.perf_counter() - start
    fit_times = [r[0] for r in results]
    pids = {r[1] for r in results}
    if len(pids) != len(worker_args):
        raise RuntimeError(f"expected {len(worker_args)} distinct worker processes, got {len(pids)}")
    return wall, fit_times


def main(argv=None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", required=True, help="the snapfix featurized CSV")
    parser.add_argument("--out-dir", default=None, help="output folder (default paper_b/results/throughput/<UTC>)")
    parser.add_argument("--target", default="zT")
    parser.add_argument("--device", required=True, choices=("cuda", "cpu"))
    parser.add_argument("--concurrency", type=int, required=True, help="k: number of concurrent fits")
    parser.add_argument("--cpu-jobs-per-worker", type=int, default=None, help="override xgboost n_jobs per worker on device=cpu")
    parser.add_argument("--cuda-jobs-per-worker", type=int, default=-1, help="xgboost n_jobs per worker on device=cuda")
    parser.add_argument("--budget-minutes", type=float, default=12.0)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("paper_b/results/throughput") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    cpu_count = os.cpu_count() or 1
    n_jobs = (
        (args.cpu_jobs_per_worker or max(1, cpu_count // args.concurrency)) if args.device == "cpu"
        else args.cuda_jobs_per_worker
    )
    X, y, identity = load_data(args.csv, args.target)
    print(f"loaded {X.shape[0]:,} rows x {X.shape[1]} features for {args.target} in {time.perf_counter() - started:.0f} s; "
          f"device={args.device} concurrency={args.concurrency} n_jobs_per_worker={n_jobs}", flush=True)
    frozen = json.loads((FROZEN_DIR / f"{args.target}.json").read_text(encoding="utf-8"))["best_params"]
    base_params = {key: frozen[key] for key in FROZEN_KEYS}

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(y))
    pred_idx, pool = order[:N_PREDICT], order[N_PREDICT:]
    mp_context = mp.get_context("spawn")

    def over_budget():
        return (time.perf_counter() - started) / 60.0 > args.budget_minutes

    rows, truncated = [], False
    for config_name in CONFIGS:
        if over_budget():
            truncated = True
            break
        n, depth, trees = CONFIGS[config_name]
        sequential_totals, concurrent_walls, concurrent_fit_times = [], [], []
        for repeat in range(TIMINGS):
            if over_budget():
                truncated = True
                break
            worker_args = build_worker_args(X, y, pred_idx, pool, config_name, args.concurrency,
                                             SEED + repeat, base_params, args.device, n_jobs)
            sequential_totals.append(time_sequential(worker_args))
            if over_budget():  # the sequential half alone may already have used the remaining budget
                truncated = True
                break
            worker_args = build_worker_args(X, y, pred_idx, pool, config_name, args.concurrency,
                                             1000 + SEED + repeat, base_params, args.device, n_jobs)
            wall, fit_times = time_concurrent(worker_args, mp_context)
            concurrent_walls.append(wall)
            concurrent_fit_times.append(fit_times)
            rows.append({"config": config_name, "n": n, "max_depth": depth, "n_estimators": trees,
                         "concurrency": args.concurrency, "device": args.device, "n_jobs_per_worker": n_jobs,
                         "repeat": repeat, "sequential_total_seconds": sequential_totals[-1],
                         "concurrent_wall_seconds": wall, "concurrent_fit_seconds": fit_times})
        if concurrent_walls:
            seq_med, conc_med = float(np.median(sequential_totals)), float(np.median(concurrent_walls))
            speedup = seq_med / conc_med if conc_med > 0 else float("nan")
            print(f"{config_name} (n={n:,}, depth={depth}, trees={trees}): sequential median {seq_med:.2f} s, "
                  f"concurrent median {conc_med:.2f} s, speedup {speedup:.2f}x "
                  f"[{(time.perf_counter() - started) / 60:.1f} min elapsed]"
                  + (" -- truncated, fewer than 3 repeats completed" if len(concurrent_walls) < TIMINGS else ""), flush=True)
        elif truncated:
            print(f"{config_name}: skipped, over budget before any repeat completed", flush=True)

    timings = pd.DataFrame(rows)
    timings.to_csv(out_dir / "throughput_timings.csv", index=False)
    if rows:
        summary = timings.groupby(["config", "n", "max_depth", "n_estimators", "concurrency", "device", "n_jobs_per_worker"]).agg(
            sequential_median_seconds=("sequential_total_seconds", "median"),
            concurrent_median_seconds=("concurrent_wall_seconds", "median"),
        ).reset_index()
        summary["speedup"] = summary["sequential_median_seconds"] / summary["concurrent_median_seconds"]
    else:
        summary = pd.DataFrame(columns=["config", "n", "max_depth", "n_estimators", "concurrency", "device",
                                        "n_jobs_per_worker", "sequential_median_seconds", "concurrent_median_seconds", "speedup"])
    summary.to_csv(out_dir / "throughput_summary.csv", index=False)

    git_head = run(["git", "rev-parse", "HEAD"])
    porcelain = run(["git", "status", "--porcelain"])
    config = {
        "target": args.target, "device": args.device, "concurrency": args.concurrency, "n_jobs_per_worker": n_jobs,
        "cpu_count": cpu_count, "configs": CONFIGS, "seed": SEED, "n_predict_rows": N_PREDICT, "timings_per_config": TIMINGS,
        "dataset": identity, "frozen_hyperparams_file": str(FROZEN_DIR / f"{args.target}.json"),
        "frozen_hyperparams_sha256": sha256_file(FROZEN_DIR / f"{args.target}.json"),
        "frozen_params_used": base_params, "script_sha256": sha256_file(Path(__file__)),
        "git_head": git_head, "tree_clean": (porcelain == "") if porcelain is not None else None,
        "dirty_files": porcelain.splitlines() if porcelain else [], "budget_minutes": args.budget_minutes,
        "truncated_by_budget": truncated, "total_minutes": (time.perf_counter() - started) / 60.0,
        "utc_stamp": stamp, "system": system_info(),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    archive = shutil.make_archive(str(out_dir), "zip", out_dir)
    print(f"\nWrote {out_dir} and {archive}; total {config['total_minutes']:.1f} min; truncated={truncated}", flush=True)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
