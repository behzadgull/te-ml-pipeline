"""
External validation of the thesis paper's final models: ESTM (two passes, in-support) and teMatDb (three strata), scored with the SAVED boosters.

Why this script exists (docs/decisions.md, 2026-10-10): the earlier ESTM and teMatDb results (results/external_snapfix/20260917T160553) came from an
in-process refit of the frozen hyperparameters whose producing script is not in the repository and whose folder records no commit. The thesis paper's
final models (thesis_paper/results/final_a, SHA256 of every model in its manifest) are used here instead, so that the external numbers and the
screening predictions come from the same four fitted models.

What it does, in order:
  1. Loads the four boosters from final_a.tar.gz (checked against the SHA256 of the committed manifest; the tarball's own SHA256 is checked against the README value).
  2. Smear factors for the back-transform of sigma and kappa: the Duan factor of the committed out-of-fold predictions of the chemistry-cluster rung
     (results/ladder_regen_snapfix/20260917T150000, 5 repeats x 5 folds, every file's SHA256 recorded), the mean over the five repeats. Training residuals only.
  3. ESTM: the unchanged data path of src/external_validation.py (temperature window, canonicalisation, featurisation, the 397-column assertion), the two
     dedup passes against the snapfix training CSV, scoring with the saved models, and the in-support rule of scripts/insupport_share.py (the joint
     training range of S, sigma and kappa; the row counts 2,709 and 1,196 are asserted because they depend on the data only).
  4. teMatDb: the cached featurised scoring frame data/external/tematdb/_scoring_df.csv (SHA256 recorded), strata a0 (DOI in training), a (DOI not in training)
     and b (chemistry cluster absent from training; the sample ids are those of results/external_snapfix/20260917T160553/tematdb_inventory_snapfix.json, rebuilt
     with the fixed cluster definition), sample-level bootstrap intervals (2000 draws, seed 0).
Nothing is refitted and no hyperparameter is touched. Output folder (outside the clone with --allow-dirty, which is for tests and is never a result):
estm_results.json, insupport_share.json, tematdb_scoring.json, the per-row prediction .npz files, run_config.json, README.md.

Usage (repository root, clean tree):
    python thesis_paper/scripts/external_rescore.py --models-tar C:/Users/<you>/Downloads/final_a.tar.gz
"""

import argparse
import datetime
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import run_record as rr  # noqa: E402
import src.external_validation as ev  # noqa: E402
from src.backtransform_check import duan_smearing_correction  # noqa: E402
from src.nested_cv import get_feature_columns  # noqa: E402

TRAIN = "data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv"
TRAIN_SHA256 = "d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489"
ESTM_XLSX = "data/external/estm.xlsx"
NA9_RUN = "thesis_paper/results/na9/20261003T132121"  # records the SHA256 of the ESTM file that the committed ESTM counts were made from
TEMATDB = "data/external/tematdb/_scoring_df.csv"
INVENTORY = "results/external_snapfix/20260917T160553/tematdb_inventory_snapfix.json"
FINAL_A = "thesis_paper/results/final_a/20261004T102855"
FINAL_A_TAR_SHA256 = "91719958320c4a3c9b387ae670d15e7295a4624c8fe2ab6d42728b374eb63874"
LADDER = "results/ladder_regen_snapfix/20260917T150000"
LADDER_METRICS = "reports/regen_snapfix/20260917T150000/ladder_metrics.json"
TARGETS = ("S", "sigma", "kappa", "zT")
N_BOOT, SEED = 2000, 0
# per-property min and max of the snapfix training rows (same constants as scripts/insupport_share.py), asserted against the training frame below
TRAINING_BOUNDS = {"S": (-461.0258, 562.5), "sigma": (958.7831, 1656678.2772), "kappa": (0.2830364, 13.7753)}
EXPECTED_IN_SUPPORT = {"a": 2709, "b": 1196}
EXPECTED_TEMATDB = {"a0": (2027, 122), "a": (903, 61), "b": (412, 27)}


def load_models(tar_path, workdir):
    """Extract the four boosters, check them against the committed manifest, return {target: XGBRegressor} and their SHA256."""
    import xgboost as xgb

    manifest = json.loads((REPO / FINAL_A / "manifest.json").read_text(encoding="utf-8"))["files"]
    with tarfile.open(tar_path) as t:
        names = {Path(m.name).as_posix(): m for m in t.getmembers() if m.isfile()}
        models, shas = {}, {}
        for target in TARGETS:
            hit = [n for n in names if n.endswith(f"models/{target}.json")]
            assert len(hit) == 1, f"models/{target}.json: {len(hit)} matches in the tarball"
            dest = Path(workdir) / f"{target}.json"
            dest.write_bytes(t.extractfile(names[hit[0]]).read())
            sha = rr.sha256_file(dest)
            want = next((v["sha256"] if isinstance(v, dict) else v) for k, v in manifest.items() if k.endswith(f"models/{target}.json"))
            assert sha == want, f"{target}: model SHA256 differs from the committed manifest"
            m = xgb.XGBRegressor()
            m.load_model(str(dest))
            assert m.get_booster().num_features() == 397, f"{target}: {m.get_booster().num_features()} features"
            models[target], shas[target] = m, sha
    return models, shas


def smear_from_rung():
    """Duan smearing factors of sigma and kappa from the committed out-of-fold predictions of the chemistry-cluster rung (log10 space), and the file SHA256s."""
    out, oof_r2, files = {}, {}, {}
    for t in ("sigma", "kappa"):
        per, r2s = [], []
        for r in range(5):
            yt, yp = [], []
            for f in range(5):
                p = REPO / LADDER / f"{t}_chemistry_full" / f"repeat{r}_fold{f}_predictions.npz"
                files[str(p.relative_to(REPO)).replace("\\", "/")] = rr.sha256_file(p)
                z = np.load(p)
                yt.append(z["y_true"])
                yp.append(z["y_pred"])
            yt, yp = np.concatenate(yt), np.concatenate(yp)
            per.append(float(np.mean(10.0 ** (yt - yp))))
            r2s.append(float(r2_score(yt, yp)))
        out[t] = {"factor": float(np.mean(per)), "per_repeat": per, "min": min(per), "max": max(per)}
        oof_r2[t] = float(np.mean(r2s))
    return out, oof_r2, files


def in_support_mask(z, bounds):
    """Rows inside the training per-property range of S, sigma and kappa simultaneously."""
    mask = np.ones(len(z["S_true"]), dtype=bool)
    for prop, (lo, hi) in bounds.items():
        col = z[{"S": "S_true", "sigma": "sigma_true", "kappa": "kappa_true"}[prop]]
        mask &= (col >= lo) & (col <= hi)
    return mask


def r2(y, yhat):
    """Coefficient of determination."""
    return float(1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def insupport_report(out_dir, bounds, ladder_metrics):
    """The in-support results of both ESTM passes, in the schema of reports/insupport_share/*/insupport_share.json."""
    cols = {"S": ("S_true", "S_pred"), "sigma": ("sigma_log_true", "sigma_log_pred"), "kappa": ("kappa_log_true", "kappa_log_pred"), "zT": ("zT_true", "zT_direct_pred")}
    res = {}
    for letter, name in (("a", "DOI-disjoint"), ("b", "cluster-disjoint")):
        z = np.load(out_dir / f"estm_predictions_pass{letter}.npz", allow_pickle=True)
        ins = in_support_mask(z, bounds)
        assert int(ins.sum()) == EXPECTED_IN_SUPPORT[letter], (letter, int(ins.sum()))
        props = {}
        for prop, (tc, pc) in cols.items():
            y, yh = np.asarray(z[tc], float), np.asarray(z[pc], float)
            internal = ladder_metrics["runs"][f"{prop}_chemistry_full"]["per_repeat_r2_mean"]
            full, sup = r2(y, yh), r2(y[ins], yh[ins])
            e2 = (y - yh) ** 2
            props[prop] = {"r2_full": full, "r2_in_support": sup, "r2_internal_chemistry": internal, "drop_from_internal": internal - full,
                           "recovered_share": (sup - full) / (internal - full), "ood_sse_share": float(e2[~ins].sum() / e2.sum())}
        res[letter] = {"stratum": name, "n": int(len(ins)), "n_in_support": int(ins.sum()), "ood_row_fraction": float((~ins).mean()), "properties": props}
    return res


def bootstrap_r2_ci(sample_ids, y_true, y_pred, n_boot=N_BOOT, seed=SEED):
    """95% interval of R2 by resampling whole samples (sample_id) with replacement: 2.5th and 97.5th percentile."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(sample_ids)
    if len(uniq) < 2:
        return [float("nan"), float("nan")]
    idx = {s: np.where(sample_ids == s)[0] for s in uniq}
    out = []
    for _ in range(n_boot):
        rows = np.concatenate([idx[s] for s in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(np.unique(y_true[rows])) < 2:
            continue
        out.append(r2_score(y_true[rows], y_pred[rows]))
    if len(out) < 10:
        return [float("nan"), float("nan")]
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def score_tematdb_stratum(sub, name, models, feature_cols, smear, out_dir):
    """Predict one teMatDb stratum with the saved models and score it against the declared and the recomputed zT."""
    X = sub[feature_cols].to_numpy(dtype=np.float64)
    sid = sub["sample_id"].to_numpy()
    S_true, sigma_true, kappa_true = (sub[c].to_numpy(dtype=np.float64) for c in ("S", "sigma", "kappa_prop"))
    zt_decl, zt_tep = sub["zt_declared"].to_numpy(dtype=np.float64), sub["zt_tep"].to_numpy(dtype=np.float64)
    T = sub["temperature_bin"].to_numpy(dtype=np.float64)
    S_pred, sig_pred, kap_pred, zt_dir = (models[t].predict(X) for t in ("S", "sigma", "kappa", "zT"))
    zt_der, _, _ = duan_smearing_correction({"S_pred": S_pred, "sigma_log_pred": sig_pred, "kappa_log_pred": kap_pred, "T": T}, smear_sigma=smear["sigma"], smear_kappa=smear["kappa"])
    sig_log_true, kap_log_true = np.log10(sigma_true), np.log10(kappa_true)
    metrics = {}
    for key, yt, yp in (("S", S_true, S_pred), ("sigma_log10", sig_log_true, sig_pred), ("kappa_log10", kap_log_true, kap_pred),
                        ("zT_direct_vs_declared", zt_decl, zt_dir), ("zT_direct_vs_tep", zt_tep, zt_dir),
                        ("zT_derived_vs_declared", zt_decl, zt_der), ("zT_derived_vs_tep", zt_tep, zt_der)):
        metrics[key] = {"r2": float(r2_score(yt, yp)), "rmse": float(np.sqrt(mean_squared_error(yt, yp))), "mae": float(mean_absolute_error(yt, yp)),
                        "n": int(len(yt)), "r2_ci95": bootstrap_r2_ci(sid, yt, yp)}
        print(f"  {name} {key:26s} R2 {metrics[key]['r2']:+.4f} CI95 [{metrics[key]['r2_ci95'][0]:+.4f}, {metrics[key]['r2_ci95'][1]:+.4f}] n={len(yt)}", flush=True)
    np.savez(out_dir / f"tematdb_{name}_predictions.npz", sample_id=sid, temperature=T, S_true=S_true, S_pred=S_pred, sigma_log_true=sig_log_true, sigma_log_pred=sig_pred,
             kappa_log_true=kap_log_true, kappa_log_pred=kap_pred, zt_declared_true=zt_decl, zt_tep_true=zt_tep, zT_direct_pred=zt_dir, zT_derived_pred=zt_der)
    return {"n_rows": int(len(sub)), "n_samples": int(len(np.unique(sid))), "metrics": metrics}


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-tar", required=True, help="final_a.tar.gz (the bundle whose README records its SHA256)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--allow-dirty", action="store_true", help="test run: the output folder must lie outside the clone and accepted_as_result is false")
    args = ap.parse_args()
    inputs = {"training_csv": TRAIN, "estm_xlsx": ESTM_XLSX, "tematdb_scoring_frame": TEMATDB, "tematdb_inventory": INVENTORY,
              "final_a_manifest": f"{FINAL_A}/manifest.json", "ladder_metrics": LADDER_METRICS, "na9_run_config": f"{NA9_RUN}/run_config.json"}
    prov = rr.provenance(inputs, __file__)
    assert prov["inputs"]["training_csv"]["sha256"] == TRAIN_SHA256, "not the snapfix CSV"
    na9 = json.loads((REPO / NA9_RUN / "run_config.json").read_text(encoding="utf-8"))
    assert prov["inputs"]["estm_xlsx"]["sha256"] == na9["inputs"]["estm_xlsx"]["sha256"], "the ESTM file differs from the one NA9 read"
    if not args.allow_dirty and not prov["tree_clean"]:
        raise SystemExit(f"working tree is not clean: {prov['dirty_files'][:5]}")
    stamp = rr.utc_stamp()
    out_dir = Path(args.out_dir) if args.out_dir else REPO / "thesis_paper" / "results" / "external_rescore" / stamp
    if args.allow_dirty:
        assert REPO.resolve() not in out_dir.resolve().parents, "a --allow-dirty run must write outside the clone"
    tar_sha = rr.sha256_file(args.models_tar)
    assert tar_sha == FINAL_A_TAR_SHA256, f"final_a.tar.gz SHA256 {tar_sha} differs from the README value"
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        models, model_sha = load_models(args.models_tar, td)
    smear, smear_oof_r2, rung_files = smear_from_rung()
    print("smear factors from the committed rung:", {t: round(v["factor"], 4) for t, v in smear.items()}, flush=True)
    smear_f = {t: v["factor"] for t, v in smear.items()}

    # ---- ESTM
    report, estm_feat, train_df = ev.dry_run_inventory(estm_path=REPO / ESTM_XLSX, training_csv=REPO / TRAIN)
    feature_cols = get_feature_columns(train_df)
    bounds_now = {p: (float(train_df[p].min()), float(train_df[p].max())) for p in ("S", "sigma", "kappa")}
    for p, (lo, hi) in TRAINING_BOUNDS.items():
        assert abs(bounds_now[p][0] - lo) < 1e-3 * max(1, abs(lo)) and abs(bounds_now[p][1] - hi) < 1e-3 * max(1, abs(hi)), (p, bounds_now[p], (lo, hi))
    dedup_a, n_drop_a, n_overlap_doi = ev.dedup_by_doi(estm_feat, train_df)
    dedup_b, n_drop_b, n_overlap_cl = ev.dedup_by_chemistry_cluster(estm_feat, train_df)
    res_a = ev.score_estm_pass(dedup_a, feature_cols, models, smear_f["sigma"], smear_f["kappa"], "Pass (a) source-DOI dedup", save_path=out_dir / "estm_predictions_passa.npz")
    res_b = ev.score_estm_pass(dedup_b, feature_cols, models, smear_f["sigma"], smear_f["kappa"], "Pass (b) chemistry-cluster dedup", save_path=out_dir / "estm_predictions_passb.npz")
    del train_df
    estm_out = {
        "provenance": {"training_csv": TRAIN, "training_csv_sha256": prov["inputs"]["training_csv"]["sha256"], "estm_file": ESTM_XLSX,
                       "estm_file_sha256": prov["inputs"]["estm_xlsx"]["sha256"], "models": "the four saved boosters of thesis_paper/results/final_a (no refit)",
                       "model_sha256": model_sha, "date": datetime.date.today().isoformat()},
        "smear_factors": smear_f, "smear_source": f"mean over the 5 repeats of the Duan factor of the out-of-fold predictions of {LADDER}/{{sigma,kappa}}_chemistry_full",
        "smear_per_repeat": {t: smear[t]["per_repeat"] for t in smear}, "smear_training_oof_r2": smear_oof_r2,
        "inventory": {k: v for k, v in report.items() if k != "unit_handling"},
        "dedup_a_source_doi": {"n_dropped": n_drop_a, "n_surviving": len(dedup_a), "unique_estm_dois_overlapping_training": n_overlap_doi, "results": res_a},
        "dedup_b_chemistry_cluster": {"n_dropped": n_drop_b, "n_surviving": len(dedup_b), "unique_estm_clusters_overlapping_training": n_overlap_cl, "results": res_b},
    }
    (out_dir / "estm_results.json").write_text(json.dumps(estm_out, indent=2), encoding="utf-8")
    ladder_metrics = json.loads((REPO / LADDER_METRICS).read_text(encoding="utf-8"))
    ins = insupport_report(out_dir, TRAINING_BOUNDS, ladder_metrics)
    (out_dir / "insupport_share.json").write_text(json.dumps(ins, indent=2), encoding="utf-8")

    # ---- teMatDb
    sc = pd.read_csv(REPO / TEMATDB)
    assert len(sc) == 2930 and get_feature_columns(sc) == feature_cols, "the cached teMatDb frame has a different feature layout"
    inv = json.loads((REPO / INVENTORY).read_text(encoding="utf-8"))
    b_ids = inv["C2_cluster_overlap"]["stratum_b_sample_ids"]
    sc["in_stratum_b"] = sc["sample_id"].isin(b_ids)  # the fixed cluster definition (the cached flag is the pre-fix one: 549 rows)
    strata = {"a0": sc[sc["in_training"] & sc["parseable"]], "a": sc[~sc["in_training"] & sc["parseable"]], "b": sc[sc["in_stratum_b"]]}
    for k, (nr, ns) in EXPECTED_TEMATDB.items():
        assert (len(strata[k]), strata[k]["sample_id"].nunique()) == (nr, ns), (k, len(strata[k]), strata[k]["sample_id"].nunique())
    tm = {k: score_tematdb_stratum(v.reset_index(drop=True), k, models, feature_cols, smear_f, out_dir) for k, v in strata.items()}
    tm_out = {"training_csv": TRAIN, "training_csv_sha256": prov["inputs"]["training_csv"]["sha256"], "scoring_frame": TEMATDB,
              "scoring_frame_sha256": prov["inputs"]["tematdb_scoring_frame"]["sha256"], "stratum_b_sample_ids_from": INVENTORY, "model_sha256": model_sha,
              "smear_factors": smear_f, "n_boot": N_BOOT, "seed": SEED, "bootstrap_unit": "sample_id",
              "strata_definition": {"a0": "sample DOI is in the training data (parseable composition)", "a": "sample DOI is not in the training data (parseable)",
                                    "b": "chemistry cluster (fixed definition) absent from the training data"},
              "strata": tm}
    (out_dir / "tematdb_scoring.json").write_text(json.dumps(tm_out, indent=2), encoding="utf-8")

    cfg = {**prov, "script_args": {"models_tar_sha256": tar_sha}, "model_sha256": model_sha, "rung_prediction_sha256": rung_files, "device_for_prediction": "cpu",
           "xgboost": __import__("xgboost").__version__, "numpy": np.__version__, "pandas": pd.__version__, "allow_dirty": bool(args.allow_dirty),
           "accepted_as_result": bool(prov["tree_clean"] and not args.allow_dirty)}
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# External validation of the thesis paper's final models (ESTM and teMatDb)\n\n"
        "Written by `scripts/external_rescore.py`; see its docstring and `docs/decisions.md` (2026-10-10). The saved boosters of `results/final_a` score ESTM "
        "(two passes never pooled, in-support rule) and teMatDb (strata a0, a, b; stratum b is small and is described qualitatively). Smear factors for the "
        "back-transform are the Duan factors of the committed out-of-fold predictions of the chemistry-cluster rung. Nothing is refitted.\n", encoding="utf-8")
    print("wrote", out_dir, flush=True)


if __name__ == "__main__":
    main()
