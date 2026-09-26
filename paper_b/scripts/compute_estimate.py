"""
Compute estimate for the Paper B modelling harness (methodology doc, section 8).
No model is trained and no new run is made.

Step 1, XGBoost fit time. Parses Paper A's recorded per-fold times from its
progress.log files (frozen-hyperparameter, full-feature, CUDA, XGBoost runs;
the time between consecutive fold lines is one fit plus its prediction and
checkpoint write). The logs fall into two speed regimes (different Kaggle
sessions; the GPU model is not recorded in them), found per target as the split
at the largest gap in seconds per (training row x tree). Within a regime,
time per fit = t0 + k * n_train * n_estimators * max_depth, by least squares.
The linear dependence on depth is an ASSUMPTION taken from the CPU measurement
recorded in src/nested_cv.py's search-space comment (57 s at depth 10 / 600
trees against 3 s at depth 3 / 100 trees); Paper A's GPU logs only contain
depths 9 and 10 and cannot test it.

Step 2, harness cost. For every qualifying (target, family) pair, counts the fits
of the section 8 design and prices them with the fitted model: tuning once per
pair (trials x 3 inner folds, search-space-mean parameters), C0, C2, C3 per
fold and repeat, C1 once (its training set does not depend on the test fold),
and the specialist with nested tuning inside each of its training folds.
Evaluation fits use each target's Paper A frozen (n_estimators, max_depth) as a
proxy for the tuned parameters. Ridge is priced as CPU time from an ASSUMED
cost per training row (Paper A recorded no Ridge timing). The super-family
sensitivity analysis is priced as full runs for the super-family units plus a
C3-only rerun for every standalone unit (an upper bound: C3 reruns only where
the chosen family G differs).

Run from the repository root:
    python paper_b/scripts/compute_estimate.py --labels-run paper_b/reports/family_labels/<UTC> \
        --super-units paper_b/reports/super_family_qualification/<UTC>/units.csv
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
TRIALS = (10, 20)
N_FOLDS = 5
N_INNER = 3
# Search-space means (src/nested_cv.py _xgb_search_space): n_estimators 100..600 step 50, max_depth 3..10.
SPACE_N_ESTIMATORS = np.mean(np.arange(100, 601, 50))
SPACE_DEPTH = np.mean(np.arange(3, 11))
# ASSUMPTION (no Ridge timing was recorded): seconds per training row for one StandardScaler + Ridge fit on
# 397 features, from about n*p^2 flops at an effective 5e10 flop/s on a 4-core Kaggle CPU (about 1.5 s at 150k rows).
RIDGE_SEC_PER_ROW = 1.0e-5

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


def parse_logs(params):
    """One row per usable progress.log: median seconds per fold and the mean training size (duplicates dropped)."""
    rows, seen = [], set()
    files = sorted({f for pattern in LOG_GLOBS for f in glob.glob(pattern, recursive=True)})
    for log in files:
        lines = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()
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
        if key in seen:  # the same run copied under kaggle_out/ and results/
            continue
        seen.add(key)
        deltas = np.array([(folds[i][0] - folds[i - 1][0]).total_seconds() for i in range(1, len(folds))])
        n_train = np.array([f[1] for f in folds[1:]], dtype=float)
        keep = deltas < 5 * np.median(deltas)  # drop resume gaps
        n_est, depth = params[target]
        rows.append({
            "log": log, "log_sha256": sha256_file(log), "target": target, "split": split,
            "n_train": float(n_train[keep].mean()), "sec_per_fit": float(np.median(deltas[keep])),
            "n_estimators": n_est, "max_depth": depth,
        })
    df = pd.DataFrame(rows)
    df["x"] = df["n_train"] * df["n_estimators"] * df["max_depth"]
    df["sec_per_row_tree"] = df["sec_per_fit"] / (df["n_train"] * df["n_estimators"])
    return df


def assign_regimes(df):
    """Label each log 'faster' or 'slower' by splitting each target at its largest gap in sec per (row x tree)."""
    df = df.copy()
    df["regime"] = "slower"
    for target, group in df.groupby("target"):
        ordered = group["sec_per_row_tree"].sort_values()
        ratios = ordered.iloc[1:].to_numpy() / ordered.iloc[:-1].to_numpy()
        cut = ordered.iloc[int(np.argmax(ratios))]
        df.loc[group.index[group["sec_per_row_tree"] <= cut], "regime"] = "faster"
    return df


def fit_models(df):
    """
    Two least-squares fits per regime, keyed (regime, overhead): "intercept"
    t = t0 + k x, and "origin" t = k x. Paper A's logs only hold fits of about
    97k to 167k rows, so the intercept is an extrapolation to the small fits
    (specialists, tuning inner folds); the two fits bracket it: "intercept" is
    the high estimate for small fits, "origin" (no fixed cost per fit) the low.
    Returns {(regime, overhead): (t0, k, n_logs, rmse)}.
    """
    models = {}
    for regime, group in df.groupby("regime"):
        y = group["sec_per_fit"].to_numpy()
        design = np.column_stack([np.ones(len(group)), group["x"]])
        (t0, k), *_ = np.linalg.lstsq(design, y, rcond=None)
        rmse = float(np.sqrt(np.mean((design @ np.array([t0, k]) - y) ** 2)))
        models[(regime, "intercept")] = (float(t0), float(k), len(group), rmse)
        k0 = float(np.sum(group["x"] * y) / np.sum(group["x"] ** 2))
        rmse0 = float(np.sqrt(np.mean((k0 * group["x"].to_numpy() - y) ** 2)))
        models[(regime, "origin")] = (0.0, k0, len(group), rmse0)
    return models


def fit_seconds(model, n, n_est, depth):
    """Modelled seconds for one XGBoost fit (with its prediction) on n training rows."""
    t0, k = model[0], model[1]
    return max(t0, 0.0) + k * n * n_est * depth


def closest_family(target, family, own_group, candidates, rows):
    """The qualifying candidate closest to `family` in rows for the target, outside `own_group`; ties by name."""
    pool = sorted(c for c in candidates if c not in own_group and c != family)
    return min(pool, key=lambda c: (abs(rows[c] - rows[family]), c))


def pair_costs(model, params, n_total, n_family, n_g, trials, repeats, ridge=False):
    """
    Seconds per condition for one (target, unit) pair. `n_family` rows of the
    held-out unit, `n_g` rows of the structured-control family, `n_total` rows for
    the target. Returns a dict of seconds by component.
    """
    fold = n_family / N_FOLDS
    inner = (N_INNER - 1) / N_INNER  # each inner training set holds 2/3 of the rows

    def price(n, n_est, depth):
        return n * RIDGE_SEC_PER_ROW if ridge else fit_seconds(model, n, n_est, depth)

    n_est, depth = params
    tune_est, tune_depth = (SPACE_N_ESTIMATORS, SPACE_DEPTH)
    n_fits = N_FOLDS * repeats
    cost = {
        "tuning_once": trials * N_INNER * price(inner * (n_total - n_family), tune_est, tune_depth),
        "C0": n_fits * price(n_total - fold, n_est, depth),
        "C1": 1 * price(n_total - n_family, n_est, depth),
        "C2": n_fits * price(n_total - n_family, n_est, depth),
        "C3": n_fits * price(n_total - fold - n_g, n_est, depth),
    }
    spec_train = (N_FOLDS - 1) / N_FOLDS * n_family
    cost["specialist"] = n_fits * (
        trials * N_INNER * price(inner * spec_train, tune_est, tune_depth) + price(spec_train, n_est, depth)
    )
    return cost


def c3_only(model, params, n_total, n_family, n_g, repeats, ridge=False):
    """Seconds for the C3 fits alone (used for the standalone units of the super-family analysis)."""
    fold = n_family / N_FOLDS
    n_est, depth = params
    n = n_total - fold - n_g
    one = n * RIDGE_SEC_PER_ROW if ridge else fit_seconds(model, n, n_est, depth)
    return N_FOLDS * repeats * one


def git_state():
    """Return (HEAD, tree clean, dirty files)."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    dirty = [line[3:] for line in status.splitlines()]
    return head, not dirty, dirty


def main(argv=None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--labels-run", required=True)
    parser.add_argument("--super-units", required=True)
    args = parser.parse_args(argv)
    labels_run = Path(args.labels_run)
    params = frozen_params()
    logs = assign_regimes(parse_logs(params))
    models = fit_models(logs)
    summary = pd.read_csv(labels_run / "family_summary.csv", keep_default_na=False)
    never = set(yaml.safe_load(SUPER_PATH.read_text(encoding="utf-8"))["never_held_out"])
    supers = yaml.safe_load(SUPER_PATH.read_text(encoding="utf-8"))["super_families"]
    super_of = {member: name for name, group in supers.items() for member in group}
    units = pd.read_csv(args.super_units, keep_default_na=False)
    totals = {t: int(summary[f"rows_{t}"].sum()) for t in TARGETS}
    rows_by = {t: dict(zip(summary["family"], summary[f"rows_{t}"])) for t in TARGETS}

    pair_rows, super_pair_rows, standalone_rows = [], [], []
    for target in TARGETS:
        qualifying = [
            f for f, ok in zip(summary["family"], summary[f"qualifies_{target}"]) if ok and f not in never
        ]
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
    pairs, super_pairs, standalone = map(pd.DataFrame, (pair_rows, super_pair_rows, standalone_rows))

    results = []
    for (regime, overhead), model in models.items():
        for repeats in REPEATS:
            for trials in TRIALS:
                def total(frame, ridge, only_c3=False):
                    seconds = 0.0
                    for _, r in frame.iterrows():
                        if only_c3:
                            seconds += c3_only(model, params[r.target], totals[r.target], r.n_family, r.n_g, repeats, ridge)
                        else:
                            seconds += sum(pair_costs(model, params[r.target], totals[r.target], r.n_family, r.n_g,
                                                      trials, repeats, ridge).values())
                    return seconds / 3600.0

                results.append({
                    "regime": regime, "small_fit_overhead": overhead, "repeats": repeats, "trials": trials,
                    "xgb_gpu_hours_family_level": total(pairs, False),
                    "xgb_gpu_hours_super_family": total(super_pairs, False) + total(standalone, False, only_c3=True),
                    "ridge_cpu_hours_family_level": total(pairs, True),
                    "ridge_cpu_hours_super_family": total(super_pairs, True) + total(standalone, True, only_c3=True),
                })
    table = pd.DataFrame(results)
    table["xgb_gpu_hours_total"] = table["xgb_gpu_hours_family_level"] + table["xgb_gpu_hours_super_family"]
    table["ridge_cpu_hours_total"] = table["ridge_cpu_hours_family_level"] + table["ridge_cpu_hours_super_family"]

    breakdown = {}  # component split, family level, slower regime, R=3, trials=20, both overhead models
    for overhead in ("intercept", "origin"):
        parts = {}
        for _, r in pairs.iterrows():
            costs = pair_costs(models[("slower", overhead)], params[r.target], totals[r.target], r.n_family, r.n_g, 20, 3)
            for name, seconds in costs.items():
                parts[name] = parts.get(name, 0.0) + seconds / 3600.0
        breakdown[overhead] = parts
    breakdown = pd.DataFrame(breakdown).rename_axis("component").reset_index()
    breakdown.columns = ["component", "gpu_hours_intercept_model", "gpu_hours_origin_model"]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = OUT_DIR / stamp
    out.mkdir(parents=True)
    head, clean, dirty = git_state()
    config = {
        "labels_run": str(labels_run), "labels_sha256": sha256_file(labels_run / "host_family_labels.csv"),
        "family_summary_sha256": sha256_file(labels_run / "family_summary.csv"),
        "super_units": args.super_units, "super_units_sha256": sha256_file(args.super_units),
        "frozen_hyperparams_sha256": {t: sha256_file(FROZEN_DIR / f"{t}.json") for t in TARGETS},
        "fit_models": {f"{r}/{o}": {"t0_seconds": m[0], "k_seconds_per_row_tree_depth": m[1], "n_logs": m[2], "rmse_seconds": m[3]}
                       for (r, o), m in models.items()},
        "n_logs_used": int(len(logs)), "logs": logs[["log", "log_sha256", "target", "split", "regime"]].to_dict("records"),
        "assumptions": {"ridge_sec_per_row": RIDGE_SEC_PER_ROW, "search_space_mean_n_estimators": float(SPACE_N_ESTIMATORS),
                        "search_space_mean_depth": float(SPACE_DEPTH), "no_pruning_credit": True,
                        "depth_linear": True, "evaluation_params": "Paper A frozen per target"},
        "git_head": head, "tree_clean": clean, "dirty_files": dirty, "utc_stamp": stamp,
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    logs.drop(columns=["log_sha256"]).to_csv(out / "paper_a_fit_times.csv", index=False)
    pairs.to_csv(out / "family_level_pairs.csv", index=False)
    super_pairs.to_csv(out / "super_family_pairs.csv", index=False)
    standalone.to_csv(out / "super_family_analysis_standalone_units.csv", index=False)
    table.to_csv(out / "compute_table.csv", index=False)
    breakdown.to_csv(out / "component_breakdown_slower_R3_T20.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:,.2f}".format)
    print(f"Wrote {out}\n")
    print(f"Logs used: {len(logs)}; fit models (t = t0 + k * n_train * n_estimators * max_depth):")
    for (regime, overhead), (t0, k, n, rmse) in models.items():
        print(f"  {regime}/{overhead}: t0 = {t0:.2f} s, k = {k:.3e} s per row-tree-depth, {n} logs, rmse {rmse:.2f} s")
    print("\nQualifying (target, family) pairs (family level, other_oxide and unassignable excluded):",
          pairs.groupby("target").size().to_dict(), "total", len(pairs))
    print("Super-family units:", len(super_pairs), "; standalone units in the super-family analysis:", len(standalone))
    print("\n", table.to_string(index=False))
    print("\nComponent split, family level, slower regime, R=3, 20 trials (XGBoost GPU hours, both overhead models):")
    print(breakdown.to_string(index=False))


if __name__ == "__main__":
    main()
