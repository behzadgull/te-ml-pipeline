"""
Final models on ALL rows of each target (S, sigma, kappa, zT; frozen hyperparameters of each target) and their predictions for
  - the JARVIS dft_3d entries with a Seebeck coefficient (NA10 data, featurised; 600 K), scored for S only, and
  - the Materials Project perovskite candidates (NA11 data, featurised), predicted on the thesis temperature grid 300 to 800 K in 100 K steps.
Nothing here is cross-validated: these are the models that the thesis screening uses, fitted once on the whole snapfix data, exactly as the ESTM
external validation refits on 100% of the training data. The carrier-type classifier of na6_classifier.py, when given, decides the sign of S
where it disagrees with the regressor.

Inputs (all verified by SHA256 before use; the expected hashes are arguments so that nothing is read unverified):
  --jarvis-csv / --jarvis-sha256     data/external/jarvis/jarvis_dft3d_seebeck_featurized.csv from na10_jarvis_prep.py
  --mp-csv / --mp-sha256             data/external/mp/mp_perovskite_candidates_featurized_<stamp>.csv from na11_mp_filter_featurize.py (optional, later)
  --classifier-dir / --classifier-sha256   the na6 output directory (its final_classifier.json is hashed)
sigma and kappa predictions are back-transformed from log10 with the smearing factors computed from training out-of-fold residuals before ESTM was
touched (results/external_snapfix/20260917T160553/estm_results.json); the log10 predictions are kept too.

Units: fit_<target> (model saved under models/), pred_jarvis, pred_mp. JARVIS scoring (S): R2 against the p-type and the n-type value, a sign-matched
R2 (the p value where the predicted sign is positive, else the n value), sign agreement, and the same on the ABX3-stoichiometry and cluster-unseen
subsets. JARVIS Seebeck values are constant-relaxation-time DFT values at one doping, so this is a cross-domain test, not a like-for-like score.

Smoke mode: ~3,000 training rows, 20 trees, first 200 rows of each external file.

Usage:
    python thesis_paper/scripts/kaggle/na_final_models.py --out-dir /kaggle/working/final --expect-commit <sha> --device cuda \
        --jarvis-csv <path> --jarvis-sha256 <sha>
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
import harness as H  # noqa: E402
from src import nested_cv as ncv  # noqa: E402

ROOT = H.repo_root()
FROZEN_DIR = ROOT / "checkpoints" / "saved_predictions" / "checkpoints" / "frozen_hyperparams"
ESTM_RESULTS = ROOT / "results" / "external_snapfix" / "20260917T160553" / "estm_results.json"
TARGETS = ("S", "sigma", "kappa", "zT")
TARGET_ROWS = {"S": 185064, "sigma": 182755, "kappa": 121110, "zT": 129419}
T_GRID = (300, 400, 500, 600, 700, 800)


def verified(path, sha):
    p = Path(path)
    if H.sha256_file(p) != sha:
        raise SystemExit(f"{p} does not match the expected SHA256 {sha}")
    return p


def predict_all(models, X, smear, clf=None):
    """Predictions of the four models for a feature matrix; derived zT; sign overrides when a classifier is given."""
    out = {"S_reg": models["S"].predict(X).astype(float), "sigma_log10": models["sigma"].predict(X).astype(float),
           "kappa_log10": models["kappa"].predict(X).astype(float), "zT_direct": models["zT"].predict(X).astype(float)}
    S = out["S_reg"].copy()
    if clf is not None:
        p_type = clf.predict_proba(X)[:, 1] > 0.5
        override = (np.sign(S) != 0) & ((S > 0) != p_type)
        S = np.where(p_type, np.abs(S), -np.abs(S))
        out["sign_overridden"] = override
    out["S"] = S
    out["sigma"] = (10.0 ** out["sigma_log10"]) * smear["sigma"]
    out["kappa"] = (10.0 ** out["kappa_log10"]) * smear["kappa"]
    return out


def main():
    ap = argparse.ArgumentParser()
    H.add_common_args(ap)
    ap.add_argument("--jarvis-csv"); ap.add_argument("--jarvis-sha256")
    ap.add_argument("--mp-csv"); ap.add_argument("--mp-sha256")
    ap.add_argument("--classifier-dir"); ap.add_argument("--classifier-sha256")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.smoke:
        args.device = "cpu"
    for a, b in (("jarvis_csv", "jarvis_sha256"), ("mp_csv", "mp_sha256"), ("classifier_dir", "classifier_sha256")):
        if bool(getattr(args, a)) != bool(getattr(args, b)):
            raise SystemExit(f"--{a.replace('_', '-')} and --{b.replace('_', '-')} go together")
    hp = {t: json.loads((FROZEN_DIR / f"{t}.json").read_text(encoding="utf-8"))["best_params"] for t in TARGETS}
    if args.smoke:
        hp = {t: {**p, "n_estimators": 20, "max_depth": min(p["max_depth"], 4)} for t, p in hp.items()}
    smear = json.loads(ESTM_RESULTS.read_text(encoding="utf-8"))["smear_factors"]
    jarvis = verified(args.jarvis_csv, args.jarvis_sha256) if args.jarvis_csv else None
    mp = verified(args.mp_csv, args.mp_sha256) if args.mp_csv else None
    clf_file = verified(Path(args.classifier_dir) / "final_classifier.json", args.classifier_sha256) if args.classifier_dir else None
    sess = H.Session("na_final_models", args, {"seed": args.seed, "smoke": args.smoke, "hyperparams": hp, "smear": smear,
                                               "jarvis_sha256": args.jarvis_sha256, "mp_sha256": args.mp_sha256, "classifier_sha256": args.classifier_sha256})
    df_all = H.load_frame(sess)
    cols = ncv.get_feature_columns(df_all)
    (sess.out / "models").mkdir(exist_ok=True)
    unit_ids = [f"fit_{t}" for t in TARGETS] + (["pred_jarvis"] if jarvis else []) + (["pred_mp"] if mp else [])
    models = {}
    for t in TARGETS:
        uid = f"fit_{t}"
        path = sess.out / "models" / f"{t}.json"
        model = ncv._build_xgb_model(hp[t], args.device)
        if sess.done(uid):
            model.load_model(str(path))
            models[t] = model
            continue
        if sess.out_of_time():
            print("time budget reached before", uid, flush=True)
            break
        df = df_all[df_all[t].notna()].reset_index(drop=True)
        if args.smoke:
            df = H.smoke_subset(df, ncv.GROUP_COL, 3000)
        elif len(df) != TARGET_ROWS[t]:
            raise SystemExit(f"{t}: {len(df)} rows, expected {TARGET_ROWS[t]}")
        t0 = time.perf_counter()
        X, y = df[cols].to_numpy(dtype=np.float64), ncv._transform_target(df[t].to_numpy(dtype=np.float64), t)
        model.fit(ncv._to_device(X, args.device), ncv._to_device(y, args.device))
        model.save_model(str(path))
        models[t] = model
        sess.save(uid, {"target": t, "n_rows": int(len(df)), "scale": ncv._target_scale(t), "seconds": time.perf_counter() - t0, "model_file": f"models/{t}.json"})
        print(f"{uid}: fitted on {len(df):,} rows ({time.perf_counter() - t0:.0f}s)", flush=True)
    clf = None
    if clf_file:
        clf = xgb.XGBClassifier()
        clf.load_model(str(clf_file))
    results = {}
    if len(models) == len(TARGETS):
        if jarvis and not sess.done("pred_jarvis"):
            ext = pd.read_csv(jarvis)
            if args.smoke:
                ext = ext.head(200)
            miss = [c for c in cols if c not in ext.columns]
            assert not miss, f"JARVIS file lacks feature columns {miss[:3]}"
            pr = predict_all(models, ext[cols].to_numpy(dtype=np.float64), smear, clf)
            keep = ["jid", "formula", "composition_id", "chemistry_cluster_id", "n_seebeck", "p_seebeck", "composition_seen", "cluster_seen", "abx3_stoichiometry"]
            out = ext[[c for c in keep if c in ext.columns]].copy()
            for k, v in pr.items():
                out[k] = v
            out.to_csv(sess.out / "predictions_jarvis.csv", index=False, lineterminator="\n")
            p, n = ext["p_seebeck"].to_numpy(float), ext["n_seebeck"].to_numpy(float)
            sign_matched = np.where(pr["S"] > 0, p, n)
            subsets = {"all": np.ones(len(ext), bool)}
            for name in ("abx3_stoichiometry", "cluster_seen", "composition_seen"):
                if name in ext.columns:
                    subsets[name] = ext[name].to_numpy(bool)
                    subsets["not_" + name] = ~ext[name].to_numpy(bool)
            res = {}
            for name, m in subsets.items():
                if m.sum() < 3:
                    continue
                res[name] = {"n": int(m.sum()), "r2_vs_p": H.r2(p[m], pr["S"][m]), "r2_vs_n": H.r2(n[m], pr["S"][m]), "r2_sign_matched": H.r2(sign_matched[m], pr["S"][m]),
                             "sign_agreement_with_p_value": float(np.mean((pr["S"][m] > 0) == (p[m] > 0))), "mae_sign_matched": float(np.mean(np.abs(sign_matched[m] - pr["S"][m])))}
            results["jarvis_S"] = res
            sess.save("pred_jarvis", {"n": int(len(ext)), "scoring": res, "file": "predictions_jarvis.csv", "classifier_used": bool(clf)})
        if mp and not sess.done("pred_mp"):
            ext = pd.read_csv(mp)
            if args.smoke:
                ext = ext.head(50)
            rows = []
            for T in T_GRID:
                e = ext.copy()
                e["temperature_bin"] = T
                pr = predict_all(models, e[cols].to_numpy(dtype=np.float64), smear, clf)
                part = e[[c for c in ("material_id", "formula", "ehull", "gap", "spg_number", "spg_symbol", "is_stable") if c in e.columns]].copy()
                part["T_K"] = T
                derived = ((pr["S"] / 1e6) ** 2) * pr["sigma"] * T / pr["kappa"]
                for k, v in {**pr, "zT_derived": derived}.items():
                    part[k] = v
                rows.append(part)
            out = pd.concat(rows, ignore_index=True)
            out.to_csv(sess.out / "predictions_mp.csv", index=False, lineterminator="\n")
            sess.save("pred_mp", {"n_candidates": int(len(ext)), "temperatures": list(T_GRID), "file": "predictions_mp.csv", "classifier_used": bool(clf)})
    if all(sess.done(u) for u in unit_ids):
        results["units"] = unit_ids
        for u in ("pred_jarvis", "pred_mp"):
            if sess.done(u):
                results.setdefault(u, sess.load(u)[0])
    sess.finish(len(unit_ids), results or None)


if __name__ == "__main__":
    main()
