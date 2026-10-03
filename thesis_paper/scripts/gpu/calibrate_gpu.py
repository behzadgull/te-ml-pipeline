"""
Calibration test for a GPU machine (written for the department's V100S): fit timings, a cost estimate for the planned analyses, and a
reproduction of the committed zT chemistry-cluster result for repeat 0.

What it does
  1. Checks the dataset by SHA256 and size (the pinned snapfix CSV) and records GPU, driver and library versions, git head and tree state.
  2. Times XGBoost fits (frozen zT hyperparameters, the Paper A constructor src.nested_cv._build_xgb_model) on n = 15,000, 60,000 and all
     training rows of one zT fold, on the GPU, and at n = 15,000 on the CPU for the speed-up; three timings each, median kept; each timing is
     one fit plus one prediction on a fixed 5,000-row sample.
  3. Converts the per-fit time into estimated GPU hours for XGBoost-based analyses (NA1 nested CV, NA3 SHAP, NA6 classifier, NA7 random-split
     direct-vs-derived), assuming cost linear in rows. Random forest and stacking (NA2) are not estimated here.
  4. Reproduces repeat 0 of the committed zT chemistry-cluster run: `python src/nested_cv.py --target zT --split-strategy chemistry --device cuda
     --n-repeats 1 --frozen-hyperparams <zT.json>` into <out-dir>/repro_zT, pools the five folds' predictions and compares R2 with the committed
     per-repeat value in reports/regen_snapfix/20260917T150000/ladder_metrics.json. The same seed and the same fold assignment are used, so a
     difference measures the GPU, not the data.

Usage (from the repository root, inside the conda environment):
    python thesis_paper/scripts/gpu/calibrate_gpu.py --csv data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv --out-dir <dir>
Options: --skip-repro (timings only), --device cpu (smoke test on a machine without a GPU: the timings then describe the CPU).
"""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src import nested_cv as ncv  # noqa: E402

DATASET_SHA256 = "d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489"
DATASET_BYTES = 974854507
FROZEN = REPO / "checkpoints" / "saved_predictions" / "checkpoints" / "frozen_hyperparams" / "zT.json"
LADDER = REPO / "reports" / "regen_snapfix" / "20260917T150000" / "ladder_metrics.json"
TARGET_ROWS = {"S": 185064, "sigma": 182755, "kappa": 121110, "zT": 129419}
REPRO_TOLERANCE = 0.005  # R2 difference regarded as "same numbers" across GPU generations; a larger one is reported, not hidden


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def timed_fit(params, device, X_tr, y_tr, X_te, repeats=3):
    """Median seconds of (fit + predict) over `repeats` runs."""
    times = []
    for _ in range(repeats):
        model = ncv._build_xgb_model(params, device)
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        model.predict(X_te)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), times


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--skip-repro", action="store_true")
    ap.add_argument("--quick", action="store_true", help="plumbing check only: tiny sizes, one timing each (the estimates are then meaningless)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    csv = Path(args.csv)
    sha, size = sha256_file(csv), csv.stat().st_size
    assert (sha, size) == (DATASET_SHA256, DATASET_BYTES), f"dataset is not the pinned snapfix CSV: {sha} {size}"
    report["dataset"] = {"path": str(csv), "sha256": sha, "bytes": size}
    report["environment"] = {
        "python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
        "xgboost": __import__("xgboost").__version__, "sklearn": __import__("sklearn").__version__, "optuna": __import__("optuna").__version__,
        "gpu": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "git_head": run(["git", "rev-parse", "HEAD"]), "git_dirty_files": (run(["git", "status", "--porcelain"]) or "").splitlines(),
    }
    print(json.dumps(report["environment"], indent=1))

    params = json.loads(FROZEN.read_text(encoding="utf-8"))["best_params"]
    df = pd.read_csv(csv)
    df = df[df["zT"].notna()].reset_index(drop=True)
    assert len(df) == TARGET_ROWS["zT"]
    cols = ncv.get_feature_columns(df)
    assert len(cols) == 397
    X = df[cols].to_numpy(dtype=np.float32)
    y = df["zT"].to_numpy(dtype=np.float32)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(df))
    test_idx, train_pool = perm[:5000], perm[5000:]
    X_te = X[test_idx]
    timings = {}
    sizes = (2000, 5000, 10000) if args.quick else (15000, 60000, len(train_pool))
    for n in sizes:
        idx = train_pool[:n]
        med, all_t = timed_fit(params, args.device, X[idx], y[idx], X_te, repeats=1 if args.quick else 3)
        timings[f"{args.device}_n{n}"] = {"median_s": med, "all_s": all_t}
        print(f"{args.device} n={n}: median {med:.1f} s")
    n_cpu = sizes[0]
    cpu_med, cpu_all = timed_fit(params, "cpu", X[train_pool[:n_cpu]], y[train_pool[:n_cpu]], X_te, repeats=1)
    timings[f"cpu_n{n_cpu}"] = {"median_s": cpu_med, "all_s": cpu_all}
    print(f"cpu n={n_cpu}: {cpu_med:.1f} s")
    report["timings"] = timings

    # cost estimate: seconds per training row from the full-size timing, applied to each target's rows
    n_ref = sizes[-1]
    per_row = timings[f"{args.device}_n{n_ref}"]["median_s"] / n_ref
    est = {}
    for t, n_rows in TARGET_ROWS.items():
        train = 0.8 * n_rows
        inner = 0.8 * train * (2 / 3)  # inner fits train on two of three inner folds of the outer training set
        est[t] = {
            "one_outer_fit_s": per_row * train,
            "nested_cv_one_repeat_h": 5 * (20 * 3 * per_row * inner + per_row * train) / 3600,  # NA1: 5 outer folds x (20 trials x 3 inner fits + 1 refit)
            "ladder_style_25_fits_h": 25 * per_row * train / 3600,  # NA3, NA6, NA7: 5 repeats x 5 folds, one fit each
        }
    report["estimates_h"] = {
        "assumption": "fit cost linear in training rows; frozen zT hyperparameters used for all targets; GPU timing as measured here",
        "per_target": est,
        "NA1_nested_cv_5_repeats_all_targets_h": 5 * sum(v["nested_cv_one_repeat_h"] for v in est.values()),
        "NA3_shap_three_targets_h": sum(est[t]["ladder_style_25_fits_h"] for t in ("S", "sigma", "kappa")),
        "NA6_classifier_h": est["S"]["ladder_style_25_fits_h"],
        "NA7_random_split_four_models_h": sum(v["ladder_style_25_fits_h"] for v in est.values()),
    }
    print(json.dumps(report["estimates_h"], indent=1))

    if not args.skip_repro:
        ck = out / "repro_zT"
        cmd = [sys.executable, str(REPO / "src" / "nested_cv.py"), "--target", "zT", "--split-strategy", "chemistry", "--device", args.device,
               "--n-repeats", "1", "--frozen-hyperparams", str(FROZEN), "--checkpoint-dir", str(ck)]
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True, cwd=REPO)
        elapsed = time.perf_counter() - t0
        ys, ps = [], []
        for f in range(ncv.N_OUTER_FOLDS):
            z = np.load(ck / f"repeat0_fold{f}_predictions.npz")
            ys.append(z["y_true"].astype(float))
            ps.append(z["y_pred"].astype(float))
        y_all, p_all = np.concatenate(ys), np.concatenate(ps)
        r2 = 1.0 - float(np.sum((y_all - p_all) ** 2) / np.sum((y_all - y_all.mean()) ** 2))
        committed = json.loads(LADDER.read_text(encoding="utf-8"))["runs"]["zT_chemistry_full"]["per_repeat_r2"][0]
        report["reproduction"] = {"command": " ".join(cmd), "seconds": elapsed, "pooled_r2_repeat0": r2, "committed_r2_repeat0": committed,
                                  "abs_difference": abs(r2 - committed), "tolerance": REPRO_TOLERANCE, "within_tolerance": abs(r2 - committed) <= REPRO_TOLERANCE}
        print(json.dumps(report["reproduction"], indent=1))
    (out / "calibration_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out / 'calibration_report.json'}")


if __name__ == "__main__":
    main()
