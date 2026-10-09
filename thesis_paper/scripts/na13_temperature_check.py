"""
NA13, exploratory follow-up (docs/decisions.md, "Exploratory follow-up to NA13, 2026-10-10"; post hoc, after the results were seen, descriptive only, it does not change the pre-registered paired verdict).

The NA13 selection never kept `temperature_bin` (0 of 100 folds). For every one of the 100 committed NA13 units this script refits the target's frozen XGBoost on that unit's selected features PLUS `temperature_bin`,
on the same outer training rows, scored on the same outer test rows, on `cuda` (the machine and device of the committed units), and reports its R2 next to the unit's committed `r2_selected` and `r2_all397`.

Inputs: the committed NA13 bundle folders (results/na13_<T>/<stamp>, found by glob, exactly one per target; manifests verified, the pin, device cuda, 25 units each: the same checks as na13_analysis.py), the snapfix CSV
(size and SHA256 checked), the frozen XGBoost hyperparameters. The folds are rebuilt from the CSV with the same seed and fold code and checked against each unit's saved y_true. A unit whose selected list already contains
`temperature_bin` stops the script. Control: for the first unit of each target the selected-only fit is repeated and compared with the committed `r2_selected` (recorded, not used to drop anything).

Refuses (unless --smoke) when: the git tree is not clean, the device is not cuda, a tiny cuda fit fails, the out-dir lies inside the clone (the outputs must not dirty the tree), the CSV is not the snapfix CSV, or a bundle fails its checks.
--smoke: CPU only, the first unit (repeat 0, fold 0) of each target, a dirty tree allowed, never a result.

Output (--out-dir, outside the clone): units/<target>_repeat<r>_fold<f>.json and .npz (resumable: existing units are skipped), summary.json (per target: pooled per-repeat R2 of the committed selected fit, of
selected + temperature_bin and of the committed all-397 fit; per-fold differences with mean, SD, minimum, maximum; no interval, no test), README.md (generated), manifest.json, run_config.json.

Usage (repository root, on spcai3):
    python thesis_paper/scripts/na13_temperature_check.py --out-dir ~/runs/na13_temperature --dataset ~/data/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "kaggle"))
sys.path.insert(0, str(REPO))
import run_record as rr  # noqa: E402
import harness as H  # noqa: E402
import na13_analysis as A  # noqa: E402
from src import nested_cv as ncv  # noqa: E402

TEMP = "temperature_bin"
FROZEN = REPO / "checkpoints" / "saved_predictions" / "checkpoints" / "frozen_hyperparams"
TARGETS = A.TARGETS
N_REPEATS, N_FOLDS, SEED = A.N_REPEATS, A.N_FOLDS, A.SEED


def refuse(msg):
    """Stop with a refusal."""
    raise SystemExit(f"NA13 temperature check REFUSED: {msg}")


def find_bundle(target):
    """The one committed bundle folder of a target."""
    base = REPO / "thesis_paper" / "results" / f"na13_{target}"
    found = sorted(d for d in base.glob("*") if d.is_dir() and (d / "status.json").exists())
    if len(found) != 1:
        refuse(f"expected exactly one committed bundle folder under {base}, found {len(found)}")
    return found[0]


def spread(v):
    """mean, SD (ddof 1), min, max."""
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "min": float(v.min()), "max": float(v.max())}


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="outside the clone")
    ap.add_argument("--dataset", default=str(REPO / "data" / "processed" / H.DATASET_NAME))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--smoke", action="store_true", help="CPU, repeat 0 fold 0 of each target, dirty tree allowed; never a result")
    ap.add_argument("--max-units", type=int, default=None, help="testing only: stop after this many units")
    args = ap.parse_args()
    out = Path(args.out_dir).expanduser().resolve()
    if REPO == out or REPO in out.parents:
        refuse(f"the out-dir {out} lies inside the clone; the outputs must not dirty the tree")
    device = "cpu" if args.smoke else args.device
    if not args.smoke and device != "cuda":
        refuse("the device is not cuda (a CPU run is only allowed with --smoke, and is never a result)")
    ds = Path(args.dataset).expanduser()
    prov = rr.provenance({"dataset": str(ds)}, __file__)
    if prov["inputs"]["dataset"]["sha256"] != A.DATASET_SHA256 or prov["inputs"]["dataset"]["bytes"] != H.DATASET_BYTES:
        refuse("the CSV is not the snapfix CSV (SHA256 or size differs)")
    if not args.smoke and prov["tree_clean"] is not True:
        refuse(f"the git tree is not clean: {prov['dirty_files'][:5]}")
    if device == "cuda":
        try:
            import xgboost as xgb
            xgb.XGBRegressor(n_estimators=5, device="cuda", tree_method="hist").fit(np.random.rand(500, 8), np.random.rand(500))
        except Exception as e:  # noqa: BLE001
            refuse(f"a tiny xgboost device='cuda' fit failed: {type(e).__name__}: {e}")
    tmp = Path(tempfile.mkdtemp())
    bundles, idents = {}, {}
    for t in TARGETS:
        d = find_bundle(t)
        p, ident = A.open_bundle(str(d), tmp)
        tt, units, cfg = A.validate_bundle(p, str(d))
        if tt != t:
            refuse(f"{d} is a bundle of {tt}, not {t}")
        bundles[t] = (p, units)
        idents[f"bundle:{t}"] = ident
    (out / "units").mkdir(parents=True, exist_ok=True)
    df_all = pd.read_csv(ds)
    done_this_run, rows_by_target = 0, {t: [] for t in TARGETS}
    t_start = time.perf_counter()
    for target in TARGETS:
        p, units = bundles[target]
        df = df_all[df_all[target].notna()].reset_index(drop=True)
        cols = ncv.get_feature_columns(df)
        if TEMP not in cols:
            refuse(f"{TEMP} is not among the model features")
        ti = cols.index(TEMP)
        col_ix = {c: i for i, c in enumerate(cols)}
        X = df[cols].to_numpy(dtype=np.float64)
        y = ncv._transform_target(df[target].to_numpy(dtype=np.float64), target)
        groups = df[ncv.GROUP_COL].to_numpy()
        hp = json.loads((FROZEN / f"{target}.json").read_text(encoding="utf-8"))["best_params"]
        rng_master = np.random.default_rng(SEED)
        for r in range(N_REPEATS):
            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            for f, (tr, te) in enumerate(ncv.outer_splits("chemistry", len(df), {"chemistry": groups}, N_FOLDS, rng)):
                if args.smoke and not (r == 0 and f == 0):
                    continue
                uid = f"{target}_repeat{r}_fold{f}"
                meta = units[uid]
                a = np.load(p / "units" / f"{uid}.npz")
                if len(a["y_true"]) != len(te) or not np.allclose(a["y_true"], y[te], rtol=0, atol=1e-6):
                    refuse(f"{uid}: the rebuilt fold differs from the unit's (rows or y_true)")
                if TEMP in meta["selected"]:
                    refuse(f"{uid}: the committed selected list already contains {TEMP}")
                jp, np_ = out / "units" / f"{uid}.json", out / "units" / f"{uid}.npz"
                if jp.exists() and np_.exists():
                    rec = json.loads(jp.read_text(encoding="utf-8"))
                else:
                    if args.max_units is not None and done_this_run >= args.max_units:
                        continue
                    idx = [col_ix[c] for c in meta["selected"]] + [ti]
                    t0 = time.perf_counter()
                    model = ncv._build_xgb_model(hp, device)
                    model.fit(X[tr][:, idx], y[tr])
                    pred = np.asarray(model.predict(X[te][:, idx]), dtype=float)
                    sec = time.perf_counter() - t0
                    r2_new = H.r2(y[te], pred)
                    rec = {"target": target, "repeat": r, "fold": f, "device": device, "n_features_fitted": len(idx), "r2_selected_plus_temperature": r2_new,
                           "r2_selected_committed": meta["r2_selected"], "r2_all397_committed": meta["r2_all397"],
                           "delta_vs_selected": r2_new - meta["r2_selected"], "delta_vs_all397": r2_new - meta["r2_all397"], "seconds_fit": sec}
                    if r == 0 and f == 0:  # control: the selected-only fit on this machine and device against the committed value
                        m2 = ncv._build_xgb_model(hp, device)
                        sel_ix = [col_ix[c] for c in meta["selected"]]
                        m2.fit(X[tr][:, sel_ix], y[tr])
                        c_r2 = H.r2(y[te], np.asarray(m2.predict(X[te][:, sel_ix]), dtype=float))
                        rec.update({"control_selected_only_refit_r2": c_r2, "control_equals_committed_r2_selected": bool(c_r2 == meta["r2_selected"]),
                                    "control_abs_difference": abs(c_r2 - meta["r2_selected"])})
                    np.savez_compressed(np_, y_true=y[te], y_pred=pred)
                    jp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
                    done_this_run += 1
                    print(f"{uid}: selected + temperature {rec['r2_selected_plus_temperature']:.4f} | committed selected {rec['r2_selected_committed']:.4f}, all 397 {rec['r2_all397_committed']:.4f} [{device}] ({sec:.1f}s)"
                          + (f" | control refit equals committed: {rec['control_equals_committed_r2_selected']}" if "control_equals_committed_r2_selected" in rec else ""), flush=True)
                rows_by_target[target].append((r, f, rec, a))
    n_units = sum(len(v) for v in rows_by_target.values())
    complete = n_units == 100 and not args.smoke
    summary = {"exploratory": True, "device": device, "units": n_units, "complete": complete, "targets": {}}
    if complete:
        for target in TARGETS:
            rs = sorted(rows_by_target[target], key=lambda x: (x[0], x[1]))
            pr_sel, pr_new, pr_all = [], [], []
            for r in range(N_REPEATS):
                grp = [x for x in rs if x[0] == r]
                yt = np.concatenate([x[3]["y_true"] for x in grp])
                pr_sel.append(H.r2(yt, np.concatenate([x[3]["y_pred"] for x in grp])))
                pr_all.append(H.r2(yt, np.concatenate([x[3]["y_pred_all397"] for x in grp])))
                pr_new.append(H.r2(yt, np.concatenate([np.load(out / "units" / f"{target}_repeat{r}_fold{x[1]}.npz")["y_pred"] for x in grp])))
            dsel = [x[2]["delta_vs_selected"] for x in rs]
            dall = [x[2]["delta_vs_all397"] for x in rs]
            ctrl = [x[2] for x in rs if "control_equals_committed_r2_selected" in x[2]]
            summary["targets"][target] = {
                "pooled_per_repeat_r2": {"committed_selected": pr_sel, "selected_plus_temperature": pr_new, "committed_all397": pr_all,
                                         "committed_selected_mean": float(np.mean(pr_sel)), "selected_plus_temperature_mean": float(np.mean(pr_new)), "committed_all397_mean": float(np.mean(pr_all))},
                "per_fold_delta_selected_plus_temperature_minus_committed_selected": spread(dsel),
                "per_fold_delta_selected_plus_temperature_minus_all397": spread(dall),
                "n_folds_selected_plus_temperature_above_selected": int(sum(d > 0 for d in dsel)), "n_folds_selected_plus_temperature_above_all397": int(sum(d > 0 for d in dall)),
                "control_selected_only_refit_equals_committed": [{"repeat": c["repeat"], "fold": c["fold"], "equal": c["control_equals_committed_r2_selected"], "abs_difference": c["control_abs_difference"]} for c in ctrl]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    lines = ["# NA13 exploratory follow-up: the selected features plus temperature_bin", "",
             "EXPLORATORY and descriptive (docs/decisions.md, 2026-10-10, post hoc, after the NA13 results were seen). No threshold, test or interval is attached; it does not change the pre-registered paired verdict.", "",
             f"{n_units} units; device {device}; complete {complete}; smoke {bool(args.smoke)}.", ""]
    if complete:
        lines += ["| Target | R2 committed selected | R2 selected + temperature_bin | R2 committed all 397 | per-fold difference to selected, mean (min to max) | per-fold difference to all 397, mean (min to max) |", "|---|---|---|---|---|---|"]
        for t in TARGETS:
            o = summary["targets"][t]
            a_, b_ = o["per_fold_delta_selected_plus_temperature_minus_committed_selected"], o["per_fold_delta_selected_plus_temperature_minus_all397"]
            lines.append(f"| {t} | {o['pooled_per_repeat_r2']['committed_selected_mean']:.4f} | {o['pooled_per_repeat_r2']['selected_plus_temperature_mean']:.4f} | {o['pooled_per_repeat_r2']['committed_all397_mean']:.4f} | "
                         f"{a_['mean']:+.4f} ({a_['min']:+.4f} to {a_['max']:+.4f}) | {b_['mean']:+.4f} ({b_['min']:+.4f} to {b_['max']:+.4f}) |")
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = {x.relative_to(out).as_posix(): H.sha256_file(x) for x in sorted(out.rglob("*")) if x.is_file() and x.name not in ("manifest.json", "run_config.json")}
    (out / "manifest.json").write_text(json.dumps({"files": files}, indent=1), encoding="utf-8")
    prov["inputs"].update({k: {"sha256": v} for k, v in idents.items()})
    rr.write_json(out / "run_config.json", {**prov, "device": device, "smoke": bool(args.smoke), "units": n_units, "complete": complete,
                                            "accepted_as_result": bool(complete and prov["tree_clean"] and device == "cuda" and not args.smoke), "seconds_this_session": time.perf_counter() - t_start})
    print(f"wrote {out} ({n_units} units; accepted_as_result {bool(complete and prov['tree_clean'] and device == 'cuda' and not args.smoke)})")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
