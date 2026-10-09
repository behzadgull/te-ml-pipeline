"""
NA13 analysis: the paired comparison of the thesis's feature selection with all 397 features, as pre-registered in docs/decisions.md (2026-10-07 entry, pre-registration 2026-10-09, items 1, 2 and 5).
Written and committed before any NA13 result exists.

Input: the four NA13 bundles, one per target (`na13_S`, `na13_sigma`, `na13_kappa`, `na13_zT`: directories or .tar.gz written by scripts/kaggle/na13_feature_selection.py), 25 units each.
The script REFUSES to run unless every one of these holds, for every bundle: status complete and accepted_as_result; every session at the pinned code commit (PIN), tree_clean true, allow_dirty false, not a smoke run;
the pinned dataset SHA256; device cuda (parameters and every unit); `paired_all397`, `lasso_max_iter` 20000 and the thesis's k in the parameters; manifest SHA256 values all verified; exactly the 25 units
(target, repeat 0 to 4, fold 0 to 4) once each; the four bundles cover the four targets once each, 100 units in all.

Per target it reports (all computed here from the units, not copied from the bundles' results.json):
  - the pooled per-repeat R2 of the selected-feature fit and of the all-397 fit (same fold, machine, device), their paired difference per repeat (selected minus all 397) with mean and SD over the 5 repeats,
    and the per-fold difference (25 folds) with its spread;
  - THE CLAIM: the paired cluster-bootstrap interval of the pooled difference, exactly as scripts/model_comparison.py does it for Table 7: folds and chemistry-cluster ids rebuilt from the snapfix CSV (same seed and fold code),
    per-cluster sums per repeat, whole clusters resampled with replacement and carried in every repeat, 2000 draws, seed 0 (one generator, targets in the order S, sigma, kappa, zT), the statistic the mean over the 5 repeats
    of the pooled R2 difference, interval = 2.5th and 97.5th percentiles; the verdict text says only whether the interval contains zero;
  - alongside, not used for the claim: the committed 397-feature rung (cuda, Kaggle) per repeat and per fold, this machine's all-397 fit against it, the cross-device committed-minus-selected difference;
  - the counts after the Pearson filter, after the Lasso and selected, each with mean, SD, minimum, maximum over the 25 folds; the chosen-alpha position and the share of units at the grid minimum (per target and over the 100
    units); the convergence-warning totals (all, inner paths, folds with a warning, folds whose final refit did not converge); the selection frequency of every feature and the features selected in every fold.

Output (a new UTC-stamped folder, or --out-dir): analysis.json, per_fold.csv (100 rows), selection_frequency.csv, README.md (generated), run_config.json (inputs' SHA256, code commit and tree state, this script's SHA256).

Usage (repository root):
    python thesis_paper/scripts/na13_analysis.py --bundles <na13_S>,<na13_sigma>,<na13_kappa>,<na13_zT>
"""

import argparse
import csv
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "kaggle"))
sys.path.insert(0, str(REPO))
import run_record as rr  # noqa: E402
import harness as H  # noqa: E402
from src import nested_cv as ncv  # noqa: E402

PIN = "aad78f175952cc3d3e2bfd6d14eef94eec98b0f7"
DATASET = "data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv"
DATASET_SHA256 = "d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489"
LADDER = "results/ladder_regen_snapfix/20260917T150000"
LADDER_METRICS = "reports/regen_snapfix/20260917T150000/ladder_metrics.json"
TARGETS = ("S", "sigma", "kappa", "zT")
K_THESIS = {"S": 25, "sigma": 44, "kappa": 39, "zT": 32}
N_REPEATS, N_FOLDS, SEED = 5, 5, 0
LASSO_MAX_ITER = 20000
SPREAD_KEYS = ("n_after_pearson", "n_after_lasso", "n_selected")


def refuse(msg):
    """Stop with a refusal."""
    raise SystemExit(f"NA13 analysis REFUSED: {msg}")


def open_bundle(path, tmp):
    """(directory, SHA256 of the file that identifies the bundle) of a bundle given as a directory or a .tar.gz; manifest verified."""
    p = Path(path)
    ident = None
    if p.is_file():
        ident = H.sha256_file(p)
        with tarfile.open(p) as tf:
            tf.extractall(tmp / p.name.replace(".", "_"))
        sub = tmp / p.name.replace(".", "_")
        inner = [d for d in sub.iterdir() if d.is_dir()]
        p = inner[0] if len(inner) == 1 and not (sub / "status.json").exists() else sub
    if not (p / "status.json").exists():
        refuse(f"{path}: no status.json")
    man = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
    for rel, sha in man["files"].items():
        if not (p / rel).exists() or H.sha256_file(p / rel) != sha:
            refuse(f"{path}: {rel} does not match its manifest SHA256")
    extra = {f.relative_to(p).as_posix() for f in p.rglob("*") if f.is_file()} - set(man["files"]) - {"manifest.json"}
    if extra:
        refuse(f"{path}: files outside the manifest: {sorted(extra)[:3]}")
    return p, ident or H.sha256_file(p / "manifest.json")


def validate_bundle(p, path):
    """The pre-registered refusal conditions, for one bundle; returns (target, units meta, config of the last session)."""
    st = json.loads((p / "status.json").read_text(encoding="utf-8"))
    if not (st.get("complete") is True and st.get("accepted_as_result") is True and st.get("units_done") == st.get("units_total") == N_REPEATS * N_FOLDS):
        refuse(f"{path}: status is not complete, accepted, {N_REPEATS * N_FOLDS} of {N_REPEATS * N_FOLDS} units: {st}")
    cfgs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((p / "run_configs").glob("session_*.json"))]
    if not cfgs:
        refuse(f"{path}: no run_configs")
    for c in cfgs:
        if c["git_head"] != PIN:
            refuse(f"{path}: a session ran at {c['git_head']}, not at the pinned {PIN}")
        if c["tree_clean"] is not True or c["allow_dirty"] is not False or c["smoke"] is not False:
            refuse(f"{path}: a session has tree_clean {c['tree_clean']}, allow_dirty {c['allow_dirty']}, smoke {c['smoke']}")
        if c["dataset_sha256"] != DATASET_SHA256:
            refuse(f"{path}: a session used dataset {c['dataset_sha256']}")
        if c.get("device") != "cuda" or c["params"].get("device") != "cuda":
            refuse(f"{path}: device is {c.get('device')} / {c['params'].get('device')}, not cuda")
    c = cfgs[-1]
    prm = c["params"]
    if len(prm["targets"]) != 1 or prm["targets"][0] not in TARGETS:
        refuse(f"{path}: the bundle is not a single-target bundle of {TARGETS}: {prm['targets']}")
    t = prm["targets"][0]
    if not (prm["n_repeats"] == N_REPEATS and prm["n_folds"] == N_FOLDS and prm["seed"] == SEED and prm.get("paired_all397") is True and prm["lasso_max_iter"] == LASSO_MAX_ITER and prm["k"] == K_THESIS):
        refuse(f"{path}: design parameters differ from the pre-registration: { {k: prm.get(k) for k in ('n_repeats', 'n_folds', 'seed', 'paired_all397', 'lasso_max_iter', 'k')} }")
    units = {}
    for u in sorted((p / "units").glob("*.json")):
        meta = json.loads(u.read_text(encoding="utf-8"))["meta"]
        units[u.stem] = meta
    expect = {f"{t}_repeat{r}_fold{f}" for r in range(N_REPEATS) for f in range(N_FOLDS)}
    if set(units) != expect or len(units) != len(expect):
        refuse(f"{path}: the units are not exactly the {len(expect)} of {t} (repeats 0 to 4 x folds 0 to 4) once each: missing {sorted(expect - set(units))[:3]}, unexpected {sorted(set(units) - expect)[:3]}")
    for uid, m in units.items():
        if (m["target"], m["repeat"], m["fold"]) != (t, int(uid.split("repeat")[1].split("_")[0]), int(uid.split("fold")[1])) or m.get("device") != "cuda":
            refuse(f"{path}: unit {uid} has target/repeat/fold {m['target']}/{m['repeat']}/{m['fold']} and device {m.get('device')}")
        for k in ("r2_selected", "r2_all397", "delta_r2_selected_minus_all397"):
            if k not in m:
                refuse(f"{path}: unit {uid} lacks {k}")
    return t, units, c


def spread(v):
    """mean, SD (ddof 1), min, max."""
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "min": float(v.min()), "max": float(v.max())}


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", required=True, help="the four na13_<target> bundles (directories or .tar.gz), comma-separated")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", default=None, help="default thesis_paper/results/na13_analysis/<UTC>")
    args = ap.parse_args()
    if args.n_boot != 2000 and args.out_dir is None:
        refuse("--n-boot differs from the pre-registered 2000; use --out-dir for a test run")
    tmp = Path(tempfile.mkdtemp())
    found = {}
    idents = {}
    for path in [x for x in args.bundles.split(",") if x]:
        p, ident = open_bundle(path, tmp)
        t, units, cfg = validate_bundle(p, path)
        if t in found:
            refuse(f"two bundles for target {t}")
        found[t] = (p, units, cfg)
        idents[f"bundle:{t}"] = ident
    if set(found) != set(TARGETS):
        refuse(f"the bundles cover {sorted(found)}, not the four targets {list(TARGETS)}")
    if sum(len(v[1]) for v in found.values()) != 100:
        refuse("the four bundles do not cover 100 units once")

    inputs = {"dataset": DATASET, "ladder_metrics": LADDER_METRICS}
    prov = rr.provenance(inputs, __file__)
    if prov["inputs"]["dataset"]["sha256"] != DATASET_SHA256:
        refuse("the CSV is not the snapfix CSV")
    ladder = json.loads((REPO / LADDER_METRICS).read_text(encoding="utf-8"))["runs"]
    rng_boot = np.random.default_rng(SEED)
    tq = stats.t.ppf(0.975, N_REPEATS - 1)
    pct = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    out = {"analysis": "na13", "pin": PIN, "device": "cuda", "n_boot": args.n_boot, "seed": SEED, "targets": {}}
    rows, freq_rows = [], []
    for target in TARGETS:
        p, units, cfg = found[target]
        df = pd.read_csv(REPO / DATASET, usecols=[ncv.GROUP_COL, target])
        df = df[df[target].notna()].reset_index(drop=True)
        y = ncv._transform_target(df[target].to_numpy(dtype=np.float64), target)
        codes, uniq = pd.factorize(df[ncv.GROUP_COL].to_numpy())
        groups = df[ncv.GROUP_COL].to_numpy()
        nc = len(uniq)
        rng_master = np.random.default_rng(SEED)
        yy, cl, ps, pa = [], [], [], []
        for r in range(N_REPEATS):
            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            order, p_sel, p_all = [], [], []
            for f, (_, te) in enumerate(ncv.outer_splits("chemistry", len(df), {"chemistry": groups}, N_FOLDS, rng)):
                a = np.load(p / "units" / f"{target}_repeat{r}_fold{f}.npz")
                meta = units[f"{target}_repeat{r}_fold{f}"]
                if len(a["y_true"]) != len(te) or meta["n_test"] != len(te) or not np.allclose(a["y_true"], y[te], rtol=0, atol=1e-6):
                    refuse(f"{target} repeat {r} fold {f}: the rebuilt fold differs from the unit's (rows or y_true)")
                order.append(te)
                p_sel.append(np.asarray(a["y_pred"], dtype=float))
                p_all.append(np.asarray(a["y_pred_all397"], dtype=float))
                if abs(H.r2(a["y_true"], a["y_pred"]) - meta["r2_selected"]) > 1e-9 or abs(H.r2(a["y_true"], a["y_pred_all397"]) - meta["r2_all397"]) > 1e-9:
                    refuse(f"{target} repeat {r} fold {f}: the unit's recorded R2 does not match its saved predictions")
            order = np.concatenate(order)
            yy.append(y[order]); cl.append(codes[order]); ps.append(np.concatenate(p_sel)); pa.append(np.concatenate(p_all))
        S = {k: np.zeros((N_REPEATS, nc)) for k in ("n", "sy", "syy", "sse_sel", "sse_all")}
        for r in range(N_REPEATS):
            S["n"][r] = np.bincount(cl[r], minlength=nc)
            S["sy"][r] = np.bincount(cl[r], weights=yy[r], minlength=nc)
            S["syy"][r] = np.bincount(cl[r], weights=yy[r] ** 2, minlength=nc)
            S["sse_sel"][r] = np.bincount(cl[r], weights=(yy[r] - ps[r]) ** 2, minlength=nc)
            S["sse_all"][r] = np.bincount(cl[r], weights=(yy[r] - pa[r]) ** 2, minlength=nc)

        def r2s(w):
            n, sy, syy = S["n"] @ w, S["sy"] @ w, S["syy"] @ w
            sst = syy - sy ** 2 / n
            return 1 - (S["sse_sel"] @ w) / sst, 1 - (S["sse_all"] @ w) / sst  # per repeat

        full_sel, full_all = r2s(np.ones(nc))
        b_sel, b_all = [], []
        for _ in range(args.n_boot):
            rs, ra = r2s(np.bincount(rng_boot.integers(0, nc, nc), minlength=nc).astype(float))
            b_sel.append(float(rs.mean())); b_all.append(float(ra.mean()))
        b_sel, b_all = np.array(b_sel), np.array(b_all)
        d_boot = b_sel - b_all
        d_rep = full_sel - full_all
        lo, hi = pct(d_boot)
        metas = [units[f"{target}_repeat{r}_fold{f}"] for r in range(N_REPEATS) for f in range(N_FOLDS)]
        committed = ladder[f"{target}_chemistry_full"]
        c_fold = np.array([json.loads((REPO / LADDER / f"{target}_chemistry_full" / f"repeat{r}_fold{f}.json").read_text(encoding="utf-8"))["outer_r2"] for r in range(N_REPEATS) for f in range(N_FOLDS)])
        all_fold = np.array([m["r2_all397"] for m in metas])
        d_fold = np.array([m["delta_r2_selected_minus_all397"] for m in metas])
        counts = {}
        for m in metas:
            for name in m["selected"]:
                counts[name] = counts.get(name, 0) + 1
        grid = [len(m["lasso_alphas"]) - 1 for m in metas]
        at_min = [m["lasso_alpha_index"] == g for m, g in zip(metas, grid)]
        verdict = ("the interval contains zero: a null result, no difference between the selected features and all 397 is established" if lo <= 0 <= hi
                   else ("the interval excludes zero, selected features LOWER than all 397" if hi < 0 else "the interval excludes zero, selected features HIGHER than all 397"))
        out["targets"][target] = {
            "n_rows": int(len(df)), "n_clusters": int(nc), "bundle_identity_sha256": idents[f"bundle:{target}"], "device": "cuda",
            "pooled_per_repeat_r2": {"selected": full_sel.tolist(), "all397": full_all.tolist(), "selected_mean": float(full_sel.mean()), "selected_sd": float(full_sel.std(ddof=1)),
                                     "all397_mean": float(full_all.mean()), "all397_sd": float(full_all.std(ddof=1))},
            "paired_difference_selected_minus_all397": {
                "per_repeat": d_rep.tolist(), "mean": float(d_rep.mean()), "sd": float(d_rep.std(ddof=1)),
                "ci95_cluster_bootstrap": [lo, hi], "share_of_bootstrap_draws_selected_above_all397": float((d_boot > 0).mean()),
                "ci95_t_across_repeats_secondary": [float(d_rep.mean() - tq * d_rep.std(ddof=1) / np.sqrt(N_REPEATS)), float(d_rep.mean() + tq * d_rep.std(ddof=1) / np.sqrt(N_REPEATS))],
                "ci95_cluster_bootstrap_of_selected_mean": pct(b_sel), "ci95_cluster_bootstrap_of_all397_mean": pct(b_all), "verdict": verdict},
            "per_fold_delta_r2": {**spread(d_fold), "n_folds": len(d_fold), "n_folds_selected_above_all397": int((d_fold > 0).sum())},
            "committed_rung_alongside_not_used_for_the_claim": {
                "per_repeat_r2": committed["per_repeat_r2"], "mean": committed["per_repeat_r2_mean"], "sd": committed["per_repeat_r2_std"],
                "all397_this_machine_minus_committed_per_fold": {**spread(all_fold - c_fold), "max_abs": float(np.max(np.abs(all_fold - c_fold)))},
                "committed_minus_selected_cross_device_pooled_mean": float(committed["per_repeat_r2_mean"] - full_sel.mean())},
            "counts_over_the_25_folds": {k: spread([m[k] for m in metas]) for k in SPREAD_KEYS} | {"k_thesis": K_THESIS[target]},
            "lasso_alpha": spread([m["lasso_alpha"] for m in metas]),
            "alpha_grid_position": {**spread([m["lasso_alpha_index"] for m in metas]), "grid_size": grid[0] + 1, "units_at_grid_minimum": int(sum(at_min)), "share_at_grid_minimum": float(np.mean(at_min))},
            "convergence": {"warnings_total": int(sum(m["n_convergence_warnings"] for m in metas)), "warnings_inner_paths_total": int(sum(m["n_convergence_warnings_inner_paths"] for m in metas)),
                            "folds_with_warnings": int(sum(m["n_convergence_warnings"] > 0 for m in metas)), "folds_final_refit_not_converged": int(sum(not m["final_refit_converged"] for m in metas))},
            "selected_in_every_fold": sorted(n for n, c in counts.items() if c == len(metas)),
            "n_distinct_features_ever_selected": len(counts)}
        for (r, f), m, cf in zip([(r, f) for r in range(N_REPEATS) for f in range(N_FOLDS)], metas, c_fold):
            rows.append({"target": target, "repeat": r, "fold": f, "n_train": m["n_train"], "n_test": m["n_test"], "n_after_pearson": m["n_after_pearson"], "n_after_lasso": m["n_after_lasso"], "n_selected": m["n_selected"],
                         "lasso_alpha_index": m["lasso_alpha_index"], "lasso_alpha": m["lasso_alpha"], "warnings": m["n_convergence_warnings"], "warnings_inner_paths": m["n_convergence_warnings_inner_paths"],
                         "final_refit_converged": m["final_refit_converged"], "r2_selected": m["r2_selected"], "r2_all397": m["r2_all397"], "delta_r2_selected_minus_all397": m["delta_r2_selected_minus_all397"],
                         "committed_rung_fold_r2": cf, "seconds_selected_fit": m["seconds_selected_fit"], "seconds_all397_fit": m["seconds_all397_fit"], "seconds_unit": m["seconds"]})
        for name, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            freq_rows.append({"target": target, "feature": name, "folds_selected": c, "share": c / len(metas)})
        print(target, "selected", round(float(full_sel.mean()), 4), "all397", round(float(full_all.mean()), 4), "paired diff", round(float(d_rep.mean()), 5), "CI", [round(lo, 5), round(hi, 5)], flush=True)

    n_all = sum(t["alpha_grid_position"]["units_at_grid_minimum"] for t in out["targets"].values())
    out["overall_100_units"] = {"units": 100, "units_with_chosen_alpha_at_grid_minimum": n_all, "share": n_all / 100,
                                "warnings_total": sum(t["convergence"]["warnings_total"] for t in out["targets"].values()),
                                "folds_with_warnings": sum(t["convergence"]["folds_with_warnings"] for t in out["targets"].values()),
                                "folds_final_refit_not_converged": sum(t["convergence"]["folds_final_refit_not_converged"] for t in out["targets"].values())}
    d = Path(args.out_dir) if args.out_dir else rr.new_run_dir("na13_analysis")
    d.mkdir(parents=True, exist_ok=True)
    rr.write_json(d / "analysis.json", out)
    for name, rs in (("per_fold.csv", rows), ("selection_frequency.csv", freq_rows)):
        with open(d / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rs[0]))
            w.writeheader(); w.writerows(rs)
    prov["inputs"].update({k: {"sha256": v} for k, v in idents.items()})
    rr.write_json(d / "run_config.json", {**prov, "bundles": args.bundles, "n_boot": args.n_boot})
    lines = ["# NA13 analysis: selected features against all 397 features, paired, on one machine and device (cuda)", "",
             f"Generated by `scripts/na13_analysis.py` from the four bundles (code pin `{PIN[:7]}`, 100 units); design: `docs/decisions.md`, 2026-10-09 pre-registration. "
             f"Code state of this analysis: commit {prov['git_head'][:7]}, tree_clean {prov['tree_clean']}.", "",
             "| Target | R2 selected | R2 all 397 | Paired difference (selected minus all 397), mean over 5 repeats | Cluster-bootstrap 95% interval | Verdict |", "|---|---|---|---|---|---|"]
    for t in TARGETS:
        o = out["targets"][t]
        pdiff = o["paired_difference_selected_minus_all397"]
        lines.append(f"| {t} | {o['pooled_per_repeat_r2']['selected_mean']:.4f} | {o['pooled_per_repeat_r2']['all397_mean']:.4f} | {pdiff['mean']:+.4f} | [{pdiff['ci95_cluster_bootstrap'][0]:+.4f}, {pdiff['ci95_cluster_bootstrap'][1]:+.4f}] | {pdiff['verdict']} |")
    ov = out["overall_100_units"]
    lines += ["", f"- Chosen alpha at the grid minimum in {ov['units_with_chosen_alpha_at_grid_minimum']} of 100 units; convergence warnings in total {ov['warnings_total']}, folds with a warning {ov['folds_with_warnings']}, folds whose final refit did not converge {ov['folds_final_refit_not_converged']}.",
              "- The committed rung values, the per-fold differences, the counts with spread and the selection frequencies are in `analysis.json`, `per_fold.csv` and `selection_frequency.csv`; they are not used for the claim."]
    (d / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", d)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
