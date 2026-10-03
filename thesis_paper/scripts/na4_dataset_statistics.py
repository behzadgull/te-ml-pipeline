"""
NA4: dataset statistics of the Paper A data, from the featurized snapfix CSV (the rows the models are trained and scored on).

Reports, per target: coverage, mean, median, SD, range, histogram counts (for Figure 6); rows with zT above 3; the share of p-type rows
among S rows; rows per temperature bin (overall and per target); and the share of perovskite-family rows from the committed Paper B
family labels (composition-only labels, so the share is a lower bound for any perovskite definition that needs structure).

Usage (from the repository root):  python thesis_paper/scripts/na4_dataset_statistics.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import paper_a_values as pav  # noqa: E402
import run_record as rr  # noqa: E402

CSV = "data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv"
CSV_SHA = "d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489"  # recorded in the Paper A runs on this file
FAMILIES = "paper_b/reports/family_labels/20260926T181709/family_summary.csv"
T4 = pav.TARGETS
BINS = {"S": 60, "sigma": 60, "kappa": 60, "zT": 60}


def main():
    """Entry point."""
    prov = rr.provenance({"featurized_csv": CSV, "family_summary": FAMILIES}, __file__)
    assert prov["inputs"]["featurized_csv"]["sha256"] == CSV_SHA, "the featurized CSV is not the one the Paper A runs used"
    df = pd.read_csv(REPO / CSV, usecols=["temperature_bin", "S", "sigma", "kappa", "zT"])
    n_total = len(df)
    assert n_total == pav._json(pav.GROUPING)["n_rows_total"]
    res = {"n_rows": n_total, "per_target": {}}
    for t in T4:
        x = df[t].dropna()
        assert len(x) == pav._json(pav.LADDER)["runs"][f"{t}_chemistry_full"]["n_rows_header"]
        d = {"n": int(len(x)), "coverage": float(len(x) / n_total), "mean": float(x.mean()), "median": float(x.median()),
             "std": float(x.std()), "min": float(x.min()), "max": float(x.max())}
        if t in ("sigma", "kappa"):
            lx = np.log10(x)
            d.update({"log10_mean": float(lx.mean()), "log10_median": float(lx.median()), "log10_std": float(lx.std())})
            hist_x = lx
        else:
            hist_x = x
        counts, edges = np.histogram(hist_x, bins=BINS[t])
        d["histogram"] = {"space": "log10" if t in ("sigma", "kappa") else "linear", "edges": edges.tolist(), "counts": counts.tolist()}
        res["per_target"][t] = d
    zt = df["zT"].dropna()
    res["zT_above_3"] = {"n": int((zt > 3.0).sum()), "share": float((zt > 3.0).mean())}
    s = df["S"].dropna()
    res["S_p_type"] = {"n_positive": int((s > 0).sum()), "share_positive": float((s > 0).mean()), "n_negative": int((s < 0).sum()), "n_zero": int((s == 0).sum())}
    res["rows_per_temperature_bin"] = {
        "all_rows": {str(int(k)): int(v) for k, v in df["temperature_bin"].value_counts().sort_index().items()},
        **{t: {str(int(k)): int(v) for k, v in df.loc[df[t].notna(), "temperature_bin"].value_counts().sort_index().items()} for t in T4},
    }
    for t in ("all_rows", *T4):
        c = res["rows_per_temperature_bin"][t]
        res["rows_per_temperature_bin"][f"{t}_summary"] = {"densest_bin": max(c, key=c.get), "densest_n": max(c.values()),
                                                           "sparsest_bin": min(c, key=c.get), "sparsest_n": min(c.values()),
                                                           "n_800": c.get("800"), "n_600": c.get("600")}
    fam = pd.read_csv(REPO / FAMILIES).set_index("family")
    res["family_rows"] = {}
    for f in ("perovskite_titanate", "manganite", "cobaltite", "other_oxide", "unassignable"):
        res["family_rows"][f] = {t: {"rows": int(fam.loc[f, f"rows_{t}"]), "share": float(fam.loc[f, f"rows_{t}"] / res["per_target"][t]["n"])} for t in T4}
    out = rr.new_run_dir("na4")
    rr.write_json(out / "run_config.json", prov)
    rr.write_json(out / "results.json", res)
    print(f"wrote {out}")
    for t in T4:
        d = res["per_target"][t]
        print(t, d["n"], f"{d['coverage']:.3f}", f"mean {d['mean']:.4g} median {d['median']:.4g} std {d['std']:.4g} range {d['min']:.4g}..{d['max']:.4g}")
    print("zT>3", res["zT_above_3"], "S p", res["S_p_type"])
    for t in ("all_rows", *T4):
        print(t, res["rows_per_temperature_bin"][f"{t}_summary"])
    print(res["family_rows"]["perovskite_titanate"])


if __name__ == "__main__":
    main()
