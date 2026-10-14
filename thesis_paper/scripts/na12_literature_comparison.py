"""
NA12: literature validation. Compare out-of-fold predictions of the committed chemistry-cluster CV with thermoelectric values read from the literature, for the compounds fixed in
docs/decisions.md (2026-10-05, pre-registration B). It does not use the final models.

Input: docs/na12_literature_values.csv, one row per literature value, with columns
  lit_id, compound_label, composition, property, T_K, value, unit, doi, location, value_kind, digitisation_method, notes
  composition   the formula exactly as the paper gives it, dopants included (parsed with pymatgen as the training compositions were)
  property      S, sigma, kappa, zT, or resistivity (converted to sigma)
  unit          S: uV/K, mV/K, V/K; sigma: S/m, S/cm; resistivity: ohm m, ohm cm, mohm cm; kappa: W/mK; zT: 1
  location      page and figure or table the value was read from
  value_kind    table, text or digitised; a digitised value needs a digitisation_method (tool and axis calibration)
Values come from the PDFs only. Rows with T_K outside [300, 800) K are kept in the output but are not compared (the cleaning pipeline bins 300 K up to 800 K exclusive, in 25 K bins).

Predictions. The out-of-fold predictions of results/ladder_regen_snapfix/<stamp>/{S,sigma,kappa,zT}_chemistry_full (the row's chemistry cluster is held out; 5 repeats x 5 folds) are
mapped to the rows of the snapfix CSV by rebuilding the folds (same seeds and fold code, every unit checked against the rebuild). Two levels, labelled:
  exact          the training rows with the same composition_id at the same 25 K temperature bin as the literature value (mean over those rows, then over the 5 repeats)
  cluster-level  the training rows of the same chemistry cluster at that bin (mean over the rows, then over the repeats)
If a level has no row it is reported as missing; nothing is refitted. sigma and kappa are predicted in log10; they are back-transformed without smearing.
Aggregation (docs/decisions.md, amendment 2, 2026-10-14): values of one paper, one composition (composition_id) and one property in the same 25 K bin are averaged into one value before
the comparison; n and the identifiers are kept (n_aggregated, aggregated_lit_ids). Values outside [300, 800) K are not aggregated and not compared.
Reporting: per paper; any overall summary weights papers equally (fraction inside the band and median absolute error within each paper, then the median over papers); a paper whose DOI is
among the DOIs of the training data is a training-source paper, reported separately and never pooled with the independent ones.
Per value it reports whether the DOI is among the DOIs of the training data, whether the composition and the cluster are in the training data, and whether training has a row with the same
DOI and composition (the literature value itself is then probably a training label, although its cluster is held out in the prediction).
Error: (prediction - literature) / |literature|, and for sigma and kappa the log10 ratio, next to the measurement uncertainty band of the Alleno round-robin (RELATIVE_UNCERTAINTY of
src/noise_floor.py: S 0.06, sigma 0.08, kappa 0.11, zT 0.19, read from the file); "inside the band" means |relative error| <= that value. No other threshold is used.

Usage (from the repository root):
    python thesis_paper/scripts/na12_literature_comparison.py                (compare; needs a filled values file)
    python thesis_paper/scripts/na12_literature_comparison.py --check-only   (validate the values file only)
    python thesis_paper/scripts/na12_literature_comparison.py --self-test    (aggregation and summary on synthetic data; needs no dataset and no prediction)
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import run_record as rr  # noqa: E402
from src import canonicalization as canon  # noqa: E402
from src import nested_cv as ncv  # noqa: E402

VALUES = "thesis_paper/docs/na12_literature_values.csv"
DATASET = "data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv"
DATASET_SHA256 = "d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489"
LADDER = "results/ladder_regen_snapfix/20260917T150000"
TARGETS = ("S", "sigma", "kappa", "zT")
LOG_TARGETS = ("sigma", "kappa")
N_REPEATS, N_FOLDS, SEED = 5, 5, 0
T_MIN, T_MAX, BIN = 300.0, 800.0, 25.0
COLUMNS = ["lit_id", "compound_label", "composition", "property", "T_K", "value", "unit", "doi", "location", "value_kind", "digitisation_method", "notes"]
UNITS = {"S": {"uV/K": 1.0, "mV/K": 1e3, "V/K": 1e6}, "sigma": {"S/m": 1.0, "S/cm": 100.0}, "kappa": {"W/mK": 1.0}, "zT": {"1": 1.0},
         "resistivity": {"ohm m": 1.0, "ohm cm": 1e-2, "mohm cm": 1e-5}}  # resistivity factors give ohm m; sigma = 1 / rho


def read_uncertainties():
    """RELATIVE_UNCERTAINTY of src/noise_floor.py, read from the file (not imported: the module pulls in the whole noise-floor machinery)."""
    tree = ast.parse((REPO / "src" / "noise_floor.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "RELATIVE_UNCERTAINTY":
            return ast.literal_eval(node.value)
    raise SystemExit("RELATIVE_UNCERTAINTY not found in src/noise_floor.py")


def norm_doi(x):
    """Lower-case DOI without a URL or doi: prefix."""
    s = str(x).strip().lower()
    for pre in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        if s.startswith(pre):
            s = s[len(pre):]
    return s


def load_values(path):
    """Read and validate the literature values; returns a DataFrame with the SI value of every row (property S, sigma, kappa or zT)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    problems = []
    if df["lit_id"].duplicated().any():
        problems.append("duplicate lit_id")
    rows = []
    for _, r in df.iterrows():
        where = f"{r['lit_id']}"
        if r["property"] not in UNITS:
            problems.append(f"{where}: property {r['property']!r}")
            continue
        if r["unit"] not in UNITS[r["property"]]:
            problems.append(f"{where}: unit {r['unit']!r} not allowed for {r['property']}")
            continue
        try:
            val, temp = float(r["value"]), float(r["T_K"])
        except ValueError:
            problems.append(f"{where}: value or T_K is not a number")
            continue
        if not r["doi"].strip() or not r["location"].strip() or not r["composition"].strip():
            problems.append(f"{where}: doi, location and composition are required")
        if r["value_kind"] not in ("table", "text", "digitised"):
            problems.append(f"{where}: value_kind must be table, text or digitised")
        if r["value_kind"] == "digitised" and not r["digitisation_method"].strip():
            problems.append(f"{where}: a digitised value needs a digitisation_method")
        if canon.parse_formula(r["composition"])[0] is None:
            problems.append(f"{where}: composition {r['composition']!r} is not parseable")
        conv = val * UNITS[r["property"]][r["unit"]]
        if r["property"] == "resistivity":
            if conv <= 0:
                problems.append(f"{where}: resistivity must be positive")
                continue
            prop, si = "sigma", 1.0 / conv
        else:
            prop, si = r["property"], conv
        rows.append({**r.to_dict(), "T_K": temp, "value": val, "target": prop, "value_target_units": si})
    if problems:
        raise SystemExit("values file problems:\n  " + "\n  ".join(problems))
    return pd.DataFrame(rows)


def aggregate_bins(vals):
    """Amendment 2 (a): one value per (DOI, composition_id, property, 25 K bin) for the rows inside [T_MIN, T_MAX): the arithmetic mean of value_target_units, with n_aggregated and the
    identifiers. Rows outside the window are passed through (n_aggregated = 1) and are never compared."""
    out, groups = [], {}
    for _, r in vals.iterrows():
        d = r.to_dict()
        if not (T_MIN <= d["T_K"] < T_MAX):
            d.update(n_aggregated=1, aggregated_lit_ids=d["lit_id"])
            out.append(d)
            continue
        comp, _ = canon.parse_formula(d["composition"])
        key = (norm_doi(d["doi"]), canon.composition_id(comp), d["target"], T_MIN + BIN * np.floor((d["T_K"] - T_MIN) / BIN))
        groups.setdefault(key, []).append(d)
    for key, members in groups.items():
        first = dict(members[0])
        first["value_target_units"] = float(np.mean([m["value_target_units"] for m in members]))
        first["T_K"] = float(np.mean([m["T_K"] for m in members]))
        first["n_aggregated"] = len(members)
        first["aggregated_lit_ids"] = ";".join(m["lit_id"] for m in members)
        if len(members) > 1:
            first["lit_id"] = f"{members[0]['lit_id']}+{len(members) - 1}"
            first["value"] = first["value_target_units"]
            first["unit"] = {"S": "uV/K", "sigma": "S/m", "kappa": "W/mK", "zT": "1"}[first["target"]]
            first["value_kind"] = "+".join(sorted({m["value_kind"] for m in members}))
        out.append(first)
    cols = list(vals.columns) + ["n_aggregated", "aggregated_lit_ids"]
    return pd.DataFrame(out)[cols].reset_index(drop=True)


def _paper_block(g):
    """Per property and level: compared values, inside the band, fraction, median absolute relative error (one paper)."""
    block = {}
    for t in TARGETS:
        for level in ("exact", "cluster"):
            s = g[(g["target"] == t) & (g[f"{level}_n_rows"] > 0)]
            n = int(len(s))
            inside = int(s[f"{level}_inside_band"].fillna(False).astype(bool).sum())
            block[f"{t}_{level}"] = {"n_compared": n, "n_inside_band": inside, "fraction_inside_band": (inside / n) if n else None,
                                     "median_abs_rel_error": float(s[f"{level}_rel_error"].abs().median()) if n else None}
    return block


def summarise(res):
    """Amendment 2 (b) and (c). res has one row per aggregated value with columns paper, doi_in_training, target, in_window and, per level, n_rows, inside_band and rel_error.
    Returns the results per paper, the overall summary of the independent papers (papers weigh equally) and the training-source papers separately."""
    by_paper = {}
    for paper, g in res.groupby("paper"):
        by_paper[paper] = {"training_source": bool(g["doi_in_training"].all()), "n_values": int(len(g)), "n_in_window": int(g["in_window"].sum()),
                           "by_property_and_level": _paper_block(g)}
    indep = {k: v for k, v in by_paper.items() if not v["training_source"]}
    train = {k: v for k, v in by_paper.items() if v["training_source"]}
    overall = {"n_papers": len(indep), "papers": sorted(indep), "by_property_and_level": {}}
    keys = [f"{t}_{level}" for t in TARGETS for level in ("exact", "cluster")]
    for k in keys:
        fr = [v["by_property_and_level"][k]["fraction_inside_band"] for v in indep.values() if v["by_property_and_level"][k]["n_compared"]]
        er = [v["by_property_and_level"][k]["median_abs_rel_error"] for v in indep.values() if v["by_property_and_level"][k]["n_compared"]]
        overall["by_property_and_level"][k] = {"n_papers_with_comparisons": len(fr), "mean_over_papers_of_fraction_inside_band": float(np.mean(fr)) if fr else None,
                                               "median_over_papers_of_median_abs_rel_error": float(np.median(er)) if er else None}
    return {"by_paper": by_paper, "overall_independent_papers": overall, "training_source_papers": train}


def self_test():
    """Synthetic checks of the aggregation and the summary; no dataset, no prediction."""
    rows = []

    def row(i, doi, comp, target, T, v):
        rows.append({"lit_id": i, "doi": doi, "composition": comp, "target": target, "T_K": T, "value": v, "unit": "uV/K", "value_target_units": v, "value_kind": "digitised"})
    row("a1", "10.1/A", "CsSnI3", "S", 310.0, 100.0)    # same bin [300, 325) as a2
    row("a2", "10.1/A", "CsSnI3", "S", 320.0, 140.0)
    row("a3", "10.1/A", "CsSnI3", "S", 325.0, 90.0)     # next bin
    row("a4", "10.1/A", "CsSnI3", "sigma", 310.0, 5.0)  # another property
    row("a5", "10.1/A", "LaCoO3", "S", 310.0, 7.0)      # another composition
    row("b1", "10.1/B", "CsSnI3", "S", 310.0, 50.0)     # another paper
    row("a6", "10.1/A", "CsSnI3", "S", 298.15, 11.0)    # below the window: not aggregated
    row("a7", "10.1/A", "CsSnI3", "S", 301.0, 12.0)     # in the first bin with a1 and a2
    row("a8", "10.1/A", "CsSnI3", "S", 800.0, 13.0)     # at 800 K: outside the window
    df = pd.DataFrame(rows)
    agg = aggregate_bins(df)
    first = agg[(agg["doi"] == "10.1/A") & (agg["composition"] == "CsSnI3") & (agg["target"] == "S") & (agg["T_K"] > 300) & (agg["T_K"] < 325)]
    assert len(first) == 1, "a1, a2 and a7 must give one value"
    assert first.iloc[0]["n_aggregated"] == 3 and abs(first.iloc[0]["value_target_units"] - (100 + 140 + 12) / 3) < 1e-9
    assert set(first.iloc[0]["aggregated_lit_ids"].split(";")) == {"a1", "a2", "a7"}
    assert len(agg) == len(df) - 2, f"{len(df)} rows must become {len(df) - 2}, not {len(agg)}"
    assert (agg[agg["T_K"] < 300]["n_aggregated"] == 1).all() and (agg[agg["T_K"] >= 800]["n_aggregated"] == 1).all()
    assert agg[agg["lit_id"] == "b1"].iloc[0]["n_aggregated"] == 1 and agg[agg["lit_id"] == "a5"].iloc[0]["n_aggregated"] == 1
    # summary: paper P has 10 values (all inside), paper Q has 1 value (outside), paper F is a training source (all inside)
    recs = []
    for i in range(10):
        recs.append({"paper": "P", "doi_in_training": False, "target": "S", "in_window": True, "exact_n_rows": 1, "exact_inside_band": True, "exact_rel_error": 0.01,
                     "cluster_n_rows": 0, "cluster_inside_band": None, "cluster_rel_error": np.nan})
    recs.append({"paper": "Q", "doi_in_training": False, "target": "S", "in_window": True, "exact_n_rows": 1, "exact_inside_band": False, "exact_rel_error": 0.5,
                 "cluster_n_rows": 0, "cluster_inside_band": None, "cluster_rel_error": np.nan})
    for i in range(5):
        recs.append({"paper": "F", "doi_in_training": True, "target": "S", "in_window": True, "exact_n_rows": 1, "exact_inside_band": True, "exact_rel_error": 0.001,
                     "cluster_n_rows": 0, "cluster_inside_band": None, "cluster_rel_error": np.nan})
    sm = summarise(pd.DataFrame(recs))
    o = sm["overall_independent_papers"]["by_property_and_level"]["S_exact"]
    assert sm["overall_independent_papers"]["n_papers"] == 2 and sm["overall_independent_papers"]["papers"] == ["P", "Q"]
    assert abs(o["mean_over_papers_of_fraction_inside_band"] - 0.5) < 1e-12, "papers must weigh equally (0.5), not rows (10/11)"
    assert "F" not in sm["overall_independent_papers"]["papers"] and "F" in sm["training_source_papers"], "training-source papers are reported separately"
    assert sm["by_paper"]["F"]["training_source"] and not sm["by_paper"]["P"]["training_source"]
    print("self-test passed: aggregation (9 rows -> 7, mean and n recorded, outside-window rows untouched) and summary (papers weigh equally, training source separate)")


def oof_predictions(df, target):
    """(rows of the training CSV with a value of `target`, out-of-fold predictions shape (n_repeats, n) in the model's scale); every saved unit is checked against the rebuilt fold."""
    sub = df[df[target].notna()].reset_index(drop=True)
    y = sub[target].to_numpy(float)
    y = np.log10(y) if target in LOG_TARGETS else y
    groups = sub[ncv.GROUP_COL].to_numpy()
    pred = np.full((N_REPEATS, len(sub)), np.nan)
    rng_master = np.random.default_rng(SEED)
    d = REPO / LADDER / f"{target}_chemistry_full"
    for r in range(N_REPEATS):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        for f, (_, te) in enumerate(ncv.outer_splits("chemistry", len(sub), {"chemistry": groups}, N_FOLDS, rng)):
            u = np.load(d / f"repeat{r}_fold{f}_predictions.npz")
            assert len(u["y_pred"]) == len(te) and np.allclose(u["y_true"], y[te]), f"{target} repeat {r} fold {f}: the rebuilt fold differs from the saved one"
            pred[r, te] = u["y_pred"]
    assert not np.isnan(pred).any()
    return sub, pred


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", default=str(REPO / VALUES))
    ap.add_argument("--dataset", default=str(REPO / DATASET))
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    vals = load_values(args.values)
    if len(vals) == 0:
        print("the values file has no rows yet: fill it from the PDFs (docs/decisions.md, pre-registration B) and rerun")
        return 3
    print(f"{len(vals)} literature values, {vals['doi'].map(norm_doi).nunique()} DOIs, {vals['compound_label'].nunique()} compounds: valid")
    n_raw = len(vals)
    vals = aggregate_bins(vals)
    print(f"after the aggregation per paper, composition, property and 25 K bin (amendment 2): {len(vals)} values")
    if args.check_only:
        return 0
    prov = rr.provenance({"dataset": str(Path(args.dataset).resolve()), "values": str(Path(args.values).resolve()), "noise_floor": "src/noise_floor.py"}, __file__)
    assert prov["inputs"]["dataset"]["sha256"] == DATASET_SHA256, "not the snapfix CSV"
    eps = read_uncertainties()
    df = pd.read_csv(args.dataset, usecols=["DOI", "composition_id", ncv.GROUP_COL, "temperature_bin", *TARGETS])
    train_dois = set(df["DOI"].dropna().map(norm_doi))
    train_comps, train_clusters = set(df["composition_id"].dropna()), set(df[ncv.GROUP_COL].dropna())
    doi_comp = set(zip(df["DOI"].dropna().map(norm_doi), df.loc[df["DOI"].notna(), "composition_id"]))
    oof = {t: oof_predictions(df, t) for t in TARGETS}
    out = []
    for _, r in vals.iterrows():
        comp, _ = canon.parse_formula(r["composition"])
        cid, clid = canon.composition_id(comp), canon.chemistry_cluster_id(comp)
        doi = norm_doi(r["doi"])
        in_window = T_MIN <= r["T_K"] < T_MAX
        tbin = T_MIN + BIN * np.floor((r["T_K"] - T_MIN) / BIN) if in_window else np.nan
        row = {**r.to_dict(), "composition_id": cid, "chemistry_cluster_id": clid, "temperature_bin": tbin, "in_window": in_window, "doi_in_training": doi in train_dois,
               "composition_in_training": cid in train_comps, "cluster_in_training": clid in train_clusters, "doi_and_composition_in_training": (doi, cid) in doi_comp}
        t = r["target"]
        sub, pred = oof[t]
        for level, mask in (("exact", (sub["composition_id"] == cid)), ("cluster", (sub[ncv.GROUP_COL] == clid))):
            row[f"{level}_n_rows"] = 0
            row[f"{level}_pred"] = row[f"{level}_pred_sd_repeats"] = row[f"{level}_rel_error"] = row[f"{level}_log10_ratio"] = np.nan
            row[f"{level}_inside_band"] = None
            if not in_window:
                continue
            m = (mask & (sub["temperature_bin"] == tbin)).to_numpy()
            if not m.any():
                continue
            per_repeat = pred[:, m].mean(axis=1)  # model scale, one value per repeat
            lin = 10.0 ** per_repeat if t in LOG_TARGETS else per_repeat
            lit = r["value_target_units"]
            row[f"{level}_n_rows"] = int(m.sum())
            row[f"{level}_pred"] = float(np.mean(lin))
            row[f"{level}_pred_sd_repeats"] = float(np.std(lin, ddof=1))
            if lit != 0:
                row[f"{level}_rel_error"] = float((np.mean(lin) - lit) / abs(lit))
                row[f"{level}_inside_band"] = bool(abs(row[f"{level}_rel_error"]) <= eps[t])
                if t in LOG_TARGETS and lit > 0:
                    row[f"{level}_log10_ratio"] = float(np.mean(per_repeat) - np.log10(lit))
        row["relative_uncertainty_band"] = eps[t]
        row["paper"] = doi
        out.append(row)
    res = pd.DataFrame(out)
    summary = {"n_values_read": int(n_raw), "n_values_after_aggregation": int(len(res)), "n_in_window": int(res["in_window"].sum()), "relative_uncertainty": eps,
               "n_doi_in_training": int(res["doi_in_training"].sum()), "n_composition_in_training": int(res["composition_in_training"].sum()),
               "n_cluster_in_training": int(res["cluster_in_training"].sum()), "n_doi_and_composition_in_training": int(res["doi_and_composition_in_training"].sum()),
               **summarise(res)}
    d = Path(args.out_dir) if args.out_dir else REPO / "thesis_paper" / "results" / "na12" / rr.utc_stamp()
    d.mkdir(parents=True, exist_ok=True)
    res.to_csv(d / "comparison.csv", index=False, lineterminator="\n")
    rr.write_json(d / "summary.json", summary)
    ladder_hashes = {str(p.relative_to(REPO)): rr.sha256_file(p) for t in TARGETS for p in sorted((REPO / LADDER / f"{t}_chemistry_full").glob("repeat*_fold*_predictions.npz"))}
    rr.write_json(d / "run_config.json", {**prov, "ladder_dir": LADDER, "ladder_prediction_files_sha256": ladder_hashes, "seed": SEED})
    print("wrote", d)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
