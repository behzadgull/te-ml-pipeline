"""
Compute estimate for the Paper B modelling harness (methodology doc, section 8).
No model is trained and no new run is made.

Step 1, cost model. Fitted on the real Kaggle calibration bundle
(paper_b/scripts/calibrate_fit_cost.py's output; see --calibration-dir), not on
Paper A's logs. Two fits per model:
  * per-bundle: for each of the three calibrated (max_depth, n_estimators)
    configs, seconds = t0 + k * n_train * n_estimators, by least squares over
    the five calibrated row counts. This is the model actually used to price
    the harness: depth and n_estimators are confounded by the calibration
    script's design (each depth was calibrated at one fixed n_estimators), so
    a fit is priced by the bundle NEAREST its own max_depth, reusing that
    bundle's t0 and its per-(row,tree) marginal cost k against the fit's own
    n_estimators.
  * global naive: seconds = t0 + k * n_train * n_estimators * max_depth, the
    single-predictor form src/nested_cv.py's search-space comment assumes
    (linear in depth). Reported only to state its fit quality; not used for
    pricing, because the per-bundle fits are markedly better (see the R^2
    printed at run time).
Ridge: seconds = t0 + k * n_train, fitted the same way on the calibration's
ridge timings.

Also compares the calibrated GPU against Paper A's own two logged speed
regimes (parsed from progress.log files, as before), at the calibration's own
(n, n_estimators, max_depth) points, to say which regime this GPU is closest to.

Step 2, harness cost (default mode; --labels-run, --super-units,
--calibration-dir). For every qualifying (target, family) pair, counts the
fits of the section 8 design and prices them with the calibrated model: tuning
once per pair (20 pooled Optuna trials x 3 inner folds, search-space-mean
parameters, section 8.1), C0, C2, C3 per fold and repeat, C1 once (its
training set does not depend on the test fold), and the specialist with
nested tuning (10 trials) inside each of its training folds. Evaluation fits
use each target's Paper A frozen (n_estimators, max_depth) as a proxy for the
tuned parameters. The super-family sensitivity analysis is priced as full
runs for the super-family units plus a C3-only rerun for every standalone
unit (an upper bound: C3 reruns only where the chosen family G differs).
Reports R in {1, 3, 5} on both GPU and (unconditionally) CPU-priced ridge.

Step 3, plans (--plans; needs --throughput-dir, a calibrate_throughput.py
bundle covering device=cuda concurrency in {2, 4} and device=cpu concurrency
1). At one fixed repeat count (--plan-repeats, default 3), compares:
  (i)   all-GPU, sequential (the calibrated model above, no concurrency).
  (ii)  GPU with the best MEASURED concurrency for the small fits (tuning_once
        and the specialist's own nested tuning and final fit); C0-C3 are never
        run concurrently here (not calibrated at that scale). "Best measured"
        is whichever of concurrency 1 (i.e. none), 2 or 4 gave the highest
        speedup in the throughput bundle; if every measured concurrency > 1
        was slower than sequential (as found on this 2xT4 box on
        2026-09-29, most likely because every worker's device='cuda' resolves
        to the same physical GPU 0, so concurrent workers contend for one
        GPU rather than using both), the best is concurrency 1 and this plan
        is identical to (i).
  (iii) specialists (their nested tuning and final fit) on CPU, everything
        else (tuning_once, C0-C3) on GPU. The CPU cost is not a fresh t0/k fit
        (the throughput bundle times only one n per bucket, not a sweep): it
        is the GPU per-bundle model's shape, rescaled by the ratio of the
        single measured CPU second at that bucket's n to what the GPU model
        predicts at that same n. This assumes the ratio holds across n within
        a bucket; it does not assume CPU cost is linear in n from scratch.
Reports GPU-hours and CPU-hours per plan, by component.

Run from the repository root:
    python paper_b/scripts/compute_estimate.py --labels-run paper_b/reports/family_labels/<UTC> \
        --super-units paper_b/reports/super_family_qualification/<UTC>/units.csv \
        --calibration-dir paper_b/reports/calibration/<UTC>
    python paper_b/scripts/compute_estimate.py --labels-run ... --super-units ... --calibration-dir ... \
        --plans i,ii,iii --plan-repeats 3 --throughput-dir paper_b/reports/calibration/<UTC>_throughput
Imports nothing from top-level src/.
"""

import argparse
import glob
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TARGETS = ("S", "sigma", "kappa", "zT")
OUT_DIR = Path("paper_b/reports/compute_estimate")
SUPER_PATH = Path("paper_b/config/super_families.yaml")
FROZEN_DIR = Path("checkpoints/saved_predictions/checkpoints/frozen_hyperparams")
LOG_GLOBS = ("kaggle_out/**/progress.log", "checkpoints/**/progress.log", "results/**/progress.log")
REPEATS = (1, 3, 5)
N_FOLDS = 5
N_INNER = 3
TUNING_TRIALS = 20  # pooled tuning, section 8.1
SPECIALIST_TRIALS = 10  # nested inside each outer training fold, section 8.1
COMPONENTS = ("tuning_once", "C0", "C1", "C2", "C3", "specialist")
GPU_CONCURRENCY_COMPONENTS = ("tuning_once", "specialist")  # small fits a GPU-concurrency plan (ii) may discount
CPU_PLAN_COMPONENTS = ("specialist",)  # what plan (iii) moves to CPU: the specialist workflow, not the once-per-pair tuning
# Search-space means (src/nested_cv.py _xgb_search_space): n_estimators 100..600 step 50, max_depth 3..10.
SPACE_N_ESTIMATORS = float(np.mean(np.arange(100, 601, 50)))
SPACE_DEPTH = float(np.mean(np.arange(3, 11)))
PLAN_NAMES = {"i": "all_gpu_sequential", "ii": "gpu_best_measured_concurrency", "iii": "specialists_on_cpu"}

HEAD = re.compile(
    r"\] target=(\w+) \(scale=\w+\): model_type=(\w+), ([\d,]+) rows, (\d+) features, split_strategy=(\w+)"
    r"(?: \([^)]*\))?, n_repeats=(\d+), hyperparameters=(\w+).*device=(\w+)"
)
FOLD = re.compile(r"^\[([\d\- :]+)\] repeat (\d+) fold (\d+): .*n_train=([\d,]+), n_test=([\d,]+)")


def sha256_file(path):
    """SHA256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_params():
    """Paper A's frozen (n_estimators, max_depth) per target."""
    out = {}
    for target in TARGETS:
        best = json.loads((FROZEN_DIR / f"{target}.json").read_text(encoding="utf-8"))["best_params"]
        out[target] = (best["n_estimators"], best["max_depth"])
    return out


def lstsq_fit(n, y):
    """Least squares t = t0 + k*n; returns (t0, k, r2, rmse)."""
    design = np.column_stack([np.ones(len(n)), n])
    (t0, k), *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ np.array([t0, k])
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    ss_res, ss_tot = float(np.sum((y - pred) ** 2)), float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(t0), float(k), r2, rmse


def load_calibration(calibration_dir):
    """
    Load a calibrate_fit_cost.py bundle and fit the cost models described in
    the module docstring's Step 1. Returns a dict: bundles (per calibrated
    max_depth: t0, k, r2, rmse, n_points, n_estimators_config), ridge (t0, k,
    r2, rmse), global_naive (t0, k, r2, rmse, over the single n*n_est*depth
    predictor, for the fit-quality comparison only), and provenance (the
    calibration run's own dataset SHA256, git HEAD, tree_clean, GPU, and this
    bundle's own file SHA256s).
    """
    calibration_dir = Path(calibration_dir)
    medians = pd.read_csv(calibration_dir / "calibration_medians.csv")
    config = json.loads((calibration_dir / "run_config.json").read_text(encoding="utf-8"))

    xgb = medians[medians.model == "xgboost"]
    bundles = {}
    for depth, group in xgb.groupby("max_depth"):
        t0, k, r2, rmse = lstsq_fit((group.n_fit * group.n_estimators).to_numpy(), group.fit_seconds.to_numpy())
        bundles[int(depth)] = {
            "t0": t0, "k": k, "r2": r2, "rmse": rmse, "n_points": int(len(group)),
            "n_estimators_config": int(group.n_estimators.iloc[0]),
        }

    ridge = medians[medians.model == "ridge"]
    ridge_t0, ridge_k, ridge_r2, ridge_rmse = lstsq_fit(ridge.n_fit.to_numpy(), ridge.fit_seconds.to_numpy())

    x = (xgb.n_fit * xgb.n_estimators * xgb.max_depth).to_numpy()
    naive_t0, naive_k, naive_r2, naive_rmse = lstsq_fit(x, xgb.fit_seconds.to_numpy())

    return {
        "bundles": bundles,
        "ridge": {"t0": ridge_t0, "k": ridge_k, "r2": ridge_r2, "rmse": ridge_rmse, "n_points": int(len(ridge))},
        "global_naive": {"t0": naive_t0, "k": naive_k, "r2": naive_r2, "rmse": naive_rmse, "n_points": int(len(xgb))},
        "provenance": {
            "calibration_dir": str(calibration_dir),
            "calibration_medians_sha256": sha256_file(calibration_dir / "calibration_medians.csv"),
            "calibration_run_config_sha256": sha256_file(calibration_dir / "run_config.json"),
            "calibration_dataset_sha256": config["dataset"]["sha256"],
            "calibration_git_head": config["git_head"],
            "calibration_tree_clean": config["tree_clean"],
            "calibration_utc_stamp": config["utc_stamp"],
            "gpu": config["system"].get("nvidia_smi_L") or config["system"].get("nvidia_smi"),
        },
    }


def select_bundle(bundles, depth):
    """The calibrated bundle whose max_depth is nearest `depth`."""
    return bundles[min(bundles, key=lambda d: abs(d - depth))]


def xgb_seconds(cal, n, n_estimators, depth):
    """Modelled seconds for one XGBoost fit (with its prediction) on n rows, per-bundle GPU model."""
    bundle = select_bundle(cal["bundles"], depth)
    return max(bundle["t0"], 0.0) + bundle["k"] * n * n_estimators


def ridge_seconds(cal, n):
    """Modelled seconds for one ridge fit on n rows."""
    return max(cal["ridge"]["t0"], 0.0) + cal["ridge"]["k"] * n


def load_throughput_bundle(throughput_dir):
    """
    Load a calibrate_throughput.py bundle (device=cuda concurrency in {2,4},
    device=cpu concurrency 1 folders expected as subfolders
    throughput_cuda_k2/, throughput_cuda_k4/, throughput_cpu_k1/, each holding
    throughput_summary.csv and run_config.json). Returns {"gpu": DataFrame of
    every cuda run's summary rows, "cpu_k1": the cpu concurrency=1 summary,
    "provenance": [{folder, dataset_sha256, git_head, tree_clean, gpu}, ...]}.
    """
    throughput_dir = Path(throughput_dir)
    gpu_rows, provenance = [], []
    cpu_k1 = None
    for folder in sorted(throughput_dir.glob("throughput_*")):
        summary_path, config_path = folder / "throughput_summary.csv", folder / "run_config.json"
        if not (summary_path.exists() and config_path.exists()):
            continue
        summary = pd.read_csv(summary_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        provenance.append({
            "folder": folder.name, "device": config["device"], "concurrency": config["concurrency"],
            "dataset_sha256": config["dataset"]["sha256"], "git_head": config["git_head"],
            "tree_clean": config["tree_clean"], "truncated_by_budget": config["truncated_by_budget"],
            "total_minutes": config["total_minutes"], "gpu": config["system"].get("nvidia_smi_L"),
        })
        if config["device"] == "cuda":
            gpu_rows.append(summary)
        elif config["device"] == "cpu" and config["concurrency"] == 1:
            cpu_k1 = summary
    if cpu_k1 is None:
        raise ValueError(f"no device=cpu concurrency=1 folder found under {throughput_dir}")
    return {"gpu": pd.concat(gpu_rows, ignore_index=True) if gpu_rows else pd.DataFrame(), "cpu_k1": cpu_k1,
            "provenance": provenance}


def best_gpu_concurrency_factor(throughput):
    """
    Per config ("tuning", "specialist"), the best speedup measured across
    every calibrated concurrency, floored at 1.0 (concurrency 1, i.e. no
    concurrency, is always an option and always has speedup 1.0 by
    definition). Returns {config: (best_speedup, best_concurrency)}.
    """
    out = {}
    for config_name, group in throughput["gpu"].groupby("config"):
        idx = group["speedup"].idxmax()
        best_speedup, best_k = float(group.loc[idx, "speedup"]), int(group.loc[idx, "concurrency"])
        if best_speedup < 1.0:
            best_speedup, best_k = 1.0, 1
        out[config_name] = (best_speedup, best_k)
    return out


def load_cpu_ratio_model(cal, throughput):
    """
    Per calibrated bucket depth (6 for "tuning", 10 for "specialist"), the
    ratio of the single measured CPU second (device=cpu, concurrency=1) to
    what the GPU per-bundle model predicts at that same (n, n_estimators,
    depth). Returns {depth: ratio}; ratio > 1 means CPU is slower.
    """
    depth_of = {"tuning": 6, "specialist": 10}
    ratios = {}
    for _, row in throughput["cpu_k1"].iterrows():
        depth = depth_of[row["config"]]
        predicted = xgb_seconds(cal, row["n"], row["n_estimators"], depth)
        ratios[depth] = float(row["sequential_median_seconds"]) / predicted
    return ratios


def cpu_seconds(cal, cpu_ratios, n, n_estimators, depth):
    """Modelled seconds for one XGBoost fit on n rows on CPU: the GPU model's shape, rescaled by the measured ratio."""
    bucket = min(cpu_ratios, key=lambda d: abs(d - depth))
    return cpu_ratios[bucket] * xgb_seconds(cal, n, n_estimators, depth)


def parse_paper_a_logs(params):
    """One row per usable Paper A progress.log: median seconds per fold and mean training size."""
    rows, seen = [], set()
    files = sorted({f for pattern in LOG_GLOBS for f in glob.glob(pattern, recursive=True)})
    for log in files:
        try:
            lines = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        head = next((HEAD.search(line) for line in lines if HEAD.search(line)), None)
        if head is None:
            continue
        target, model, n_rows, n_feat, split, _, hp, device = head.groups()
        if (model, hp, device, n_feat) != ("xgboost", "frozen", "cuda", "397"):
            continue
        folds = [FOLD.match(line) for line in lines]
        folds = [(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), int(m.group(4).replace(",", ""))) for m in folds if m]
        if len(folds) < 3:
            continue
        key = (lines[0], folds[0][0])
        if key in seen:
            continue
        seen.add(key)
        deltas = np.array([(folds[i][0] - folds[i - 1][0]).total_seconds() for i in range(1, len(folds))])
        n_train = np.array([f[1] for f in folds[1:]], dtype=float)
        keep = deltas < 5 * np.median(deltas)
        n_est, depth = params[target]
        rows.append({"log": log, "target": target, "n_train": float(n_train[keep].mean()),
                     "sec_per_fit": float(np.median(deltas[keep])), "n_estimators": n_est, "max_depth": depth})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["x"] = df["n_train"] * df["n_estimators"] * df["max_depth"]
    df["sec_per_row_tree"] = df["sec_per_fit"] / (df["n_train"] * df["n_estimators"])
    df["regime"] = "slower"
    for target, group in df.groupby("target"):
        ordered = group["sec_per_row_tree"].sort_values()
        ratios = ordered.iloc[1:].to_numpy() / ordered.iloc[:-1].to_numpy()
        cut = ordered.iloc[int(np.argmax(ratios))]
        df.loc[group.index[group["sec_per_row_tree"] <= cut], "regime"] = "faster"
    models = {}
    for regime, group in df.groupby("regime"):
        t0, k, r2, rmse = lstsq_fit(group["x"].to_numpy(), group["sec_per_fit"].to_numpy())
        models[regime] = {"t0": t0, "k": k, "r2": r2, "rmse": rmse, "n_logs": int(len(group))}
    return {"logs": df, "regimes": models}


def compare_to_paper_a(cal, paper_a):
    """
    At each calibrated XGBoost (n, n_estimators, max_depth) point, compare the
    measured seconds against what Paper A's "faster" and "slower" single-
    predictor regime models would have predicted for that same point.
    """
    rows = []
    for depth, bundle in cal["bundles"].items():
        n_est = bundle["n_estimators_config"]
        n = 124_419
        measured = bundle["t0"] + bundle["k"] * n * n_est
        row = {"max_depth": depth, "n_estimators": n_est, "n": n, "measured_seconds": measured}
        for regime, model in paper_a["regimes"].items():
            row[f"{regime}_predicted_seconds"] = model["t0"] + model["k"] * n * n_est * depth
        rows.append(row)
    return pd.DataFrame(rows)


def closest_family(target, family, own_group, candidates, rows):
    """The qualifying candidate closest to `family` in rows for the target, outside `own_group`; ties by name."""
    pool = sorted(c for c in candidates if c not in own_group and c != family)
    return min(pool, key=lambda c: (abs(rows[c] - rows[family]), c))


def build_pairs(summary, units, supers, never):
    """Family-level (target, unit) pairs, plus the super-family analysis's super-family and standalone-unit pairs."""
    totals = {t: int(summary[f"rows_{t}"].sum()) for t in TARGETS}
    rows_by = {t: dict(zip(summary["family"], summary[f"rows_{t}"])) for t in TARGETS}
    super_of = {member: name for name, group in supers.items() for member in group}
    pair_rows, super_pair_rows, standalone_rows = [], [], []
    for target in TARGETS:
        qualifying = [f for f, ok in zip(summary["family"], summary[f"qualifies_{target}"]) if ok and f not in never]
        for family in qualifying:
            own = {m for m in supers.get(super_of.get(family, ""), [])} | {family}
            g = closest_family(target, family, own, qualifying, rows_by[target])
            pair_rows.append({"target": target, "unit": family, "n_family": rows_by[target][family],
                              "g": g, "n_g": rows_by[target][g]})
        u = units[units[f"qualifies_{target}"].astype(str).str.lower() == "true"]
        unit_rows = {name: int(r) for name, r in zip(u["unit"], u[f"rows_{target}"])}
        for name in unit_rows:
            g = closest_family(target, name, {name}, list(unit_rows), unit_rows)
            record = {"target": target, "unit": name, "n_family": unit_rows[name], "g": g, "n_g": unit_rows[g]}
            (super_pair_rows if name in supers else standalone_rows).append(record)
    return totals, pd.DataFrame(pair_rows), pd.DataFrame(super_pair_rows), pd.DataFrame(standalone_rows)


def pair_costs(cal, params, n_total, n_family, n_g, repeats, cpu_ratios=None, cpu_components=frozenset(),
               tuning_speedup=1.0, specialist_speedup=1.0):
    """
    Seconds per condition, per model (xgboost, ridge), for one (target, unit)
    pair. `cpu_components` names which of tuning_once/C0/C1/C2/C3/specialist
    are priced with the CPU-ratio model (`cpu_ratios`) instead of the GPU
    per-bundle model; specialist's own nested tuning moves with it.
    `tuning_speedup`/`specialist_speedup` (>= 1.0) divide the seconds of the
    once-per-pair tuning and, within specialist, its inner-tuning fits
    (`tuning_speedup`) or its final fit (`specialist_speedup`) -- a measured
    GPU-concurrency benefit; both apply to the xgboost build only, never to
    ridge, and never to C0-C3 (not calibrated at that scale). `n_family` rows
    of the held-out unit, `n_g` rows of the structured-control family,
    `n_total` rows for the target. Returns {model: {component: seconds}}.
    """
    fold = n_family / N_FOLDS
    inner = (N_INNER - 1) / N_INNER  # each inner training set holds 2/3 of the rows
    n_est, depth = params
    n_fits = N_FOLDS * repeats
    spec_train = (N_FOLDS - 1) / N_FOLDS * n_family

    def build(gpu_price, allow_cpu, allow_concurrency):
        def price(component, n, ne, d, speedup=1.0):
            if allow_cpu and component in cpu_components and cpu_ratios is not None:
                return cpu_seconds(cal, cpu_ratios, n, ne, d)
            return gpu_price(n, ne, d) / (speedup if allow_concurrency else 1.0)

        return {
            "tuning_once": TUNING_TRIALS * N_INNER * price("tuning_once", inner * (n_total - n_family), SPACE_N_ESTIMATORS, SPACE_DEPTH, tuning_speedup),
            "C0": n_fits * price("C0", n_total - fold, n_est, depth),
            "C1": 1 * price("C1", n_total - n_family, n_est, depth),
            "C2": n_fits * price("C2", n_total - n_family, n_est, depth),
            "C3": n_fits * price("C3", n_total - fold - n_g, n_est, depth),
            "specialist": n_fits * (
                SPECIALIST_TRIALS * N_INNER * price("specialist", inner * spec_train, SPACE_N_ESTIMATORS, SPACE_DEPTH, tuning_speedup)
                + price("specialist", spec_train, n_est, depth, specialist_speedup)
            ),
        }

    # ridge is always priced on CPU regardless of plan (this project runs ridge on CPU throughout, see
    # src/nested_cv.py's MODEL_REGISTRY); only the xgboost build ever moves to the CPU-ratio model or gets
    # a concurrency discount (ridge never ran on the GPU to begin with).
    return {"xgboost": build(lambda n, ne, d: xgb_seconds(cal, n, ne, d), allow_cpu=True, allow_concurrency=True),
            "ridge": build(lambda n, ne, d: ridge_seconds(cal, n), allow_cpu=False, allow_concurrency=False)}


def c3_only(cal, params, n_total, n_family, n_g, repeats):
    """Seconds for the C3 fits alone (used for the standalone units of the super-family analysis), per model."""
    fold = n_family / N_FOLDS
    n_est, depth = params
    n = n_total - fold - n_g
    return {"xgboost": {"C3": N_FOLDS * repeats * xgb_seconds(cal, n, n_est, depth)},
            "ridge": {"C3": N_FOLDS * repeats * ridge_seconds(cal, n)}}


def hours_by_component(cal, params, totals, frame, repeats, only_c3=False, cpu_ratios=None, cpu_components=frozenset(),
                       tuning_speedup=1.0, specialist_speedup=1.0):
    """Sum, over every pair in `frame`, seconds by (model, component); return hours."""
    totals_out = {"xgboost": dict.fromkeys(COMPONENTS, 0.0), "ridge": dict.fromkeys(COMPONENTS, 0.0)}
    for _, r in frame.iterrows():
        costs = (
            c3_only(cal, params[r.target], totals[r.target], r.n_family, r.n_g, repeats)
            if only_c3
            else pair_costs(cal, params[r.target], totals[r.target], r.n_family, r.n_g, repeats, cpu_ratios,
                            cpu_components, tuning_speedup, specialist_speedup)
        )
        for model, components in costs.items():
            for component, seconds in components.items():
                totals_out[model][component] += seconds
    return {model: {component: seconds / 3600.0 for component, seconds in components.items()}
            for model, components in totals_out.items()}


def full_hours(cal, params, totals, pairs, super_pairs, standalone, repeats, cpu_ratios=None, cpu_components=frozenset(),
               tuning_speedup=1.0, specialist_speedup=1.0):
    """{model: {component: hours}} summed over family-level pairs plus the super-family analysis."""
    kwargs = {"cpu_ratios": cpu_ratios, "cpu_components": cpu_components,
              "tuning_speedup": tuning_speedup, "specialist_speedup": specialist_speedup}
    family = hours_by_component(cal, params, totals, pairs, repeats, **kwargs)
    super_a = hours_by_component(cal, params, totals, super_pairs, repeats, **kwargs)
    super_b = hours_by_component(cal, params, totals, standalone, repeats, only_c3=True)
    out = {}
    for model in ("xgboost", "ridge"):
        out[model] = {c: family[model][c] + super_a[model][c] + super_b[model][c] for c in COMPONENTS}
    return out


def git_state():
    """Return (HEAD, tree clean, dirty files)."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    dirty = [line[3:] for line in status.splitlines()]
    return head, not dirty, dirty


def run_default_mode(args, cal, params, pairs, super_pairs, standalone, totals):
    """R in {1,3,5}, no plans: the calibrated GPU model, xgboost + ridge, by component."""
    paper_a = parse_paper_a_logs(params)
    comparison = compare_to_paper_a(cal, paper_a) if paper_a else None
    rows = []
    for repeats in REPEATS:
        hours = full_hours(cal, params, totals, pairs, super_pairs, standalone, repeats)
        for model in ("xgboost", "ridge"):
            for component in COMPONENTS:
                rows.append({"repeats": repeats, "model": model, "component": component, "hours": hours[model][component]})
    long_table = pd.DataFrame(rows)
    wide = long_table.pivot_table(index=["repeats", "model"], columns="component", values="hours").reset_index()
    wide = wide[["repeats", "model", *COMPONENTS]]
    wide["total"] = wide[list(COMPONENTS)].sum(axis=1)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = OUT_DIR / stamp
    out.mkdir(parents=True)
    head, clean, dirty = git_state()
    config = {
        "mode": "default", "labels_run": str(args.labels_run), "labels_sha256": sha256_file(Path(args.labels_run) / "host_family_labels.csv"),
        "family_summary_sha256": sha256_file(Path(args.labels_run) / "family_summary.csv"),
        "super_units": args.super_units, "super_units_sha256": sha256_file(args.super_units),
        "frozen_hyperparams_sha256": {t: sha256_file(FROZEN_DIR / f"{t}.json") for t in TARGETS},
        "calibration": cal["provenance"],
        "cost_model": {"bundles": cal["bundles"], "ridge": cal["ridge"], "global_naive_fit_quality_only": cal["global_naive"]},
        "paper_a_regime_comparison": comparison.to_dict("records") if comparison is not None else None,
        "paper_a_regimes": paper_a["regimes"] if paper_a else None,
        "tuning_trials": TUNING_TRIALS, "specialist_trials": SPECIALIST_TRIALS,
        "search_space_mean_n_estimators": SPACE_N_ESTIMATORS, "search_space_mean_depth": SPACE_DEPTH,
        "repeats_grid": list(REPEATS), "no_pruning_credit": True,
        "n_family_level_pairs": int(len(pairs)), "n_super_family_units": int(len(super_pairs)),
        "n_standalone_units_super_analysis": int(len(standalone)),
        "git_head": head, "tree_clean": clean, "dirty_files": dirty, "utc_stamp": stamp,
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    pairs.to_csv(out / "family_level_pairs.csv", index=False)
    super_pairs.to_csv(out / "super_family_pairs.csv", index=False)
    standalone.to_csv(out / "super_family_analysis_standalone_units.csv", index=False)
    wide.to_csv(out / "compute_by_component.csv", index=False)
    if paper_a is not None:
        paper_a["logs"].to_csv(out / "paper_a_fit_times_for_comparison.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:,.3f}".format)
    print(f"Wrote {out}\n")
    print(f"Calibration: {cal['provenance']['calibration_dir']} (GPU: {cal['provenance']['gpu']!r})\n")
    print("Per-bundle cost model, seconds = t0 + k * n_train * n_estimators (this is what prices the harness):")
    for depth, b in sorted(cal["bundles"].items()):
        print(f"  depth={depth}, n_estimators={b['n_estimators_config']}: t0={b['t0']:.2f} s, "
              f"k={b['k']:.4e} s/(row*tree), R^2={b['r2']:.4f}, rmse={b['rmse']:.2f} s, n={b['n_points']}")
    r = cal["ridge"]
    print(f"  ridge: t0={r['t0']:.3f} s, k={r['k']:.4e} s/row, R^2={r['r2']:.4f}, rmse={r['rmse']:.3f} s, n={r['n_points']}")
    n = cal["global_naive"]
    print(f"\nGlobal naive model (t0 + k*n*n_est*depth, NOT used for pricing, fit quality only): "
          f"t0={n['t0']:.2f} s, k={n['k']:.4e}, R^2={n['r2']:.4f}, rmse={n['rmse']:.2f} s")
    if comparison is not None:
        print("\nComparison to Paper A's own logged regimes, at this calibration's own points:")
        print(comparison.to_string(index=False))
    print(f"\nQualifying (target, family) pairs: {pairs.groupby('target').size().to_dict()}, total {len(pairs)}")
    print(f"Super-family units: {len(super_pairs)}; standalone units in the super-family analysis (C3 only): {len(standalone)}")
    print(f"\nHours by component, R in {REPEATS}, {TUNING_TRIALS} pooled trials / {SPECIALIST_TRIALS} specialist trials:")
    print(wide.to_string(index=False))


def run_plans_mode(args, cal, params, pairs, super_pairs, standalone, totals):
    """One fixed repeat count, three device/concurrency plans, GPU-hours and CPU-hours by component."""
    throughput = load_throughput_bundle(args.throughput_dir)
    gpu_factors = best_gpu_concurrency_factor(throughput)
    cpu_ratios = load_cpu_ratio_model(cal, throughput)
    repeats = args.plan_repeats
    plans = [p.strip() for p in args.plans.split(",")]

    rows = []
    for plan in plans:
        if plan not in PLAN_NAMES:
            raise ValueError(f"unknown plan {plan!r}; choose from {sorted(PLAN_NAMES)}")
        cpu_components = frozenset(CPU_PLAN_COMPONENTS) if plan == "iii" else frozenset()
        tuning_speedup = gpu_factors.get("tuning", (1.0, 1))[0] if plan == "ii" else 1.0
        specialist_speedup = gpu_factors.get("specialist", (1.0, 1))[0] if plan == "ii" else 1.0
        hours = full_hours(cal, params, totals, pairs, super_pairs, standalone, repeats,
                           cpu_ratios=cpu_ratios if plan == "iii" else None, cpu_components=cpu_components,
                           tuning_speedup=tuning_speedup, specialist_speedup=specialist_speedup)
        for component in COMPONENTS:
            on_cpu = plan == "iii" and component in CPU_PLAN_COMPONENTS
            rows.append({"plan": plan, "plan_name": PLAN_NAMES[plan], "component": component,
                        "priced_on": "cpu" if on_cpu else "gpu", "gpu_xgboost_hours": 0.0 if on_cpu else hours["xgboost"][component],
                        "cpu_xgboost_hours": hours["xgboost"][component] if on_cpu else 0.0,
                        "cpu_ridge_hours": hours["ridge"][component]})
    long_table = pd.DataFrame(rows)
    combined_parts = []
    for plan in plans:
        block = long_table[long_table["plan"] == plan]
        total_row = block[["gpu_xgboost_hours", "cpu_xgboost_hours", "cpu_ridge_hours"]].sum()
        total_row["plan"], total_row["plan_name"], total_row["component"], total_row["priced_on"] = (
            plan, PLAN_NAMES[plan], "TOTAL", "",
        )
        combined_parts.append(pd.concat([block, pd.DataFrame([total_row])[list(block.columns)]], ignore_index=True))
    combined = pd.concat(combined_parts, ignore_index=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = OUT_DIR / stamp
    out.mkdir(parents=True)
    head, clean, dirty = git_state()
    config = {
        "mode": "plans", "plans": plans, "plan_repeats": repeats,
        "labels_run": str(args.labels_run), "labels_sha256": sha256_file(Path(args.labels_run) / "host_family_labels.csv"),
        "super_units": args.super_units, "super_units_sha256": sha256_file(args.super_units),
        "calibration": cal["provenance"],
        "throughput_dir": str(args.throughput_dir), "throughput_provenance": throughput["provenance"],
        "gpu_concurrency_best": {config: {"best_speedup": s, "best_concurrency": k} for config, (s, k) in gpu_factors.items()},
        "cpu_gpu_ratio_by_bucket_depth": cpu_ratios,
        "tuning_trials": TUNING_TRIALS, "specialist_trials": SPECIALIST_TRIALS,
        "n_family_level_pairs": int(len(pairs)), "n_super_family_units": int(len(super_pairs)),
        "n_standalone_units_super_analysis": int(len(standalone)),
        "git_head": head, "tree_clean": clean, "dirty_files": dirty, "utc_stamp": stamp,
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    combined.to_csv(out / "plans_comparison.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:,.3f}".format)
    print(f"Wrote {out}\n")
    print("Best measured GPU concurrency per small-fit config (floored at concurrency=1, speedup=1.0):")
    for config_name, (speedup, k) in gpu_factors.items():
        note = " (no measured concurrency beat sequential)" if k == 1 else ""
        print(f"  {config_name}: best concurrency={k}, speedup={speedup:.3f}x{note}")
    print("\nCPU/GPU ratio by bucket depth (ratio > 1 means CPU is slower than this GPU at that bucket):")
    for depth, ratio in sorted(cpu_ratios.items()):
        print(f"  depth={depth}: {ratio:.3f}x")
    print(f"\nPlans at R={repeats}, {TUNING_TRIALS} pooled trials / {SPECIALIST_TRIALS} specialist trials:")
    print(combined.to_string(index=False))


def main(argv=None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--labels-run", required=True)
    parser.add_argument("--super-units", required=True)
    parser.add_argument("--calibration-dir", required=True, help="a calibrate_fit_cost.py output bundle")
    parser.add_argument("--plans", default=None, help="comma list of i,ii,iii; enables plans mode (needs --throughput-dir)")
    parser.add_argument("--plan-repeats", type=int, default=3)
    parser.add_argument("--throughput-dir", default=None, help="a calibrate_throughput.py output bundle (plans mode)")
    args = parser.parse_args(argv)
    labels_run = Path(args.labels_run)

    params = frozen_params()
    cal = load_calibration(args.calibration_dir)
    summary = pd.read_csv(labels_run / "family_summary.csv", keep_default_na=False)
    never = set(yaml.safe_load(SUPER_PATH.read_text(encoding="utf-8"))["never_held_out"])
    supers = yaml.safe_load(SUPER_PATH.read_text(encoding="utf-8"))["super_families"]
    units = pd.read_csv(args.super_units, keep_default_na=False)
    totals, pairs, super_pairs, standalone = build_pairs(summary, units, supers, never)

    print(f"Qualifying (target, family) pairs: {pairs.groupby('target').size().to_dict()}, total {len(pairs)}")
    print(f"Super-family units: {len(super_pairs)}; standalone units in the super-family analysis (C3 only): {len(standalone)}\n")

    if args.plans:
        if not args.throughput_dir:
            parser.error("--plans requires --throughput-dir")
        run_plans_mode(args, cal, params, pairs, super_pairs, standalone, totals)
    else:
        run_default_mode(args, cal, params, pairs, super_pairs, standalone, totals)


if __name__ == "__main__":
    main()
