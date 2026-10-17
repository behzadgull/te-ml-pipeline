"""
NA12: literature validation. Compare out-of-fold predictions of the committed chemistry-cluster CV with thermoelectric values read from the literature, for the compounds fixed in
docs/decisions.md (2026-10-05, pre-registration B). It does not use the final models.

Input: docs/na12_literature_values.csv, one row per literature value, with columns
  lit_id, compound_label, composition, property, T_K, value, unit, doi, location, value_kind, digitisation_method, notes
  composition   the formula exactly as the paper gives it, dopants included (parsed with pymatgen as the training compositions were); for the Ohta et al. crystals, whose doping is below
                5 at% and given only as a carrier concentration, the host formula SrTiO3 (amendment 3)
  compound_label the sample as the paper states it (the key of the per-sample aggregation)
  property      S, sigma, kappa, zT, or resistivity (converted to sigma)
  unit          S: uV/K, mV/K, V/K; sigma: S/m, S/cm; resistivity: ohm m, ohm cm, mohm cm; kappa: W/mK; zT: 1
  location      page and figure or table the value was read from
  value_kind    table, text or digitised; a digitised value needs a digitisation_method (tool and axis calibration)
  notes         tokens read by this script: [cluster-level] (amendment 3: compared at the cluster level only) and [edge-sensitivity: also bin B] (amendments 4 to 6: the temperature lies
                within 2 u of a 25 K bin edge; B is the neighbouring bin, its lower edge in K)
Values come from the PDFs only. Rows with T_K outside [300, 800) K (room-temperature values below 300 K, values at or above 800 K) are kept in the output, marked "recorded, not compared",
and are never compared (the cleaning pipeline bins 300 K up to 800 K exclusive, in 25 K bins).

Predictions. The out-of-fold predictions of results/ladder_regen_snapfix/<stamp>/{S,sigma,kappa,zT}_chemistry_full (the row's chemistry cluster is held out; 5 repeats x 5 folds) are
mapped to the rows of the snapfix CSV by rebuilding the folds (same seeds and fold code, every unit checked against the rebuild). Two levels, labelled:
  exact          the training rows with the same composition_id at the same 25 K temperature bin as the literature value (mean over those rows, then over the 5 repeats); not computed for a
                 [cluster-level] value (the composition of an Ohta et al. sample is not its host formula)
  cluster-level  the training rows of the same chemistry cluster at that bin (mean over the rows, then over the repeats)
If a level has no row it is reported as missing; nothing is refitted. sigma and kappa are predicted in log10; they are back-transformed without smearing.
Aggregation (docs/decisions.md, amendments 2 and 3): values of one paper, one composition (composition_id), one sample (compound_label, as the paper states it, not a derived formula) and one
property in the same 25 K bin are averaged into one value before the comparison; n and the identifiers are kept (n_aggregated, aggregated_lit_ids). Values outside [300, 800) K are not
aggregated and not compared. The members of a group must agree on [cluster-level] and on the edge-sensitivity bin, else the run stops.
Edge sensitivity (amendments 4 to 6): a value marked [edge-sensitivity: also bin B] is compared in its floor bin (the headline) and also in bin B; both results are in comparison.csv
(columns *_alt_*) and summary.json reports the overall summary of the independent papers for the headline run and for the run in which every edge value is evaluated in its neighbouring bin,
and the edge values whose result changes. A neighbouring bin outside [300, 800) K, or with no training row, is reported as missing.
Reporting: per paper; any overall summary weights papers equally (fraction inside the band and median absolute error within each paper, then the median over papers); a paper whose DOI is
among the DOIs of the training data is a training-source paper, reported separately and never pooled with the independent ones (in the edge-sensitivity summaries too).
Per value it reports whether the DOI is among the DOIs of the training data, whether the composition and the cluster are in the training data (the composition flags do not apply to
[cluster-level] values), and whether training has a row with the same DOI and composition (the literature value itself is then probably a training label, although its cluster is held out).
Error: (prediction - literature) / |literature|, and for sigma and kappa the log10 ratio, next to the measurement uncertainty band of the Alleno round-robin (RELATIVE_UNCERTAINTY of
src/noise_floor.py: S 0.06, sigma 0.08, kappa 0.11, zT 0.19, read from the file); "inside the band" means |relative error| <= that value. No other threshold is used.

Usage (from the repository root):
    python thesis_paper/scripts/na12_literature_comparison.py                (compare; needs a filled values file)
    python thesis_paper/scripts/na12_literature_comparison.py --check-only   (validate the values file and count the tokens; no dataset, no prediction)
    python thesis_paper/scripts/na12_literature_comparison.py --self-test    (tokens, aggregation, levels, edge sensitivity and summaries on synthetic data; needs no dataset and no prediction)
"""

import argparse
import ast
import json
import re
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
ROOM_TEMPERATURE_K = 298.15  # 25 C: below the window, recorded and not compared (amendment 3)
COLUMNS = ["lit_id", "compound_label", "composition", "property", "T_K", "value", "unit", "doi", "location", "value_kind", "digitisation_method", "notes"]
UNITS = {"S": {"uV/K": 1.0, "mV/K": 1e3, "V/K": 1e6}, "sigma": {"S/m": 1.0, "S/cm": 100.0}, "kappa": {"W/mK": 1.0}, "zT": {"1": 1.0},
         "resistivity": {"ohm m": 1.0, "ohm cm": 1e-2, "mohm cm": 1e-5}}  # resistivity factors give ohm m; sigma = 1 / rho
CLUSTER_TOKEN = "[cluster-level]"
EDGE_RE = re.compile(r"\[edge-sensitivity: also bin ([0-9]+(?:\.[0-9]+)?)\]")
FIELDS = ("n_rows", "pred", "pred_sd_repeats", "rel_error", "log10_ratio", "inside_band")  # per level, in comparison.csv as <level>_<field> and <level>_alt_<field>


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


def in_window(T):
    """True if the temperature is inside [300, 800) K, the only range that is compared."""
    return bool(T_MIN <= T < T_MAX)


def floor_bin(T):
    """Lower edge (K) of the 25 K bin of the pipeline that contains T (floor bins, [300, 325) is 300)."""
    return float(T_MIN + BIN * np.floor((T - T_MIN) / BIN))


def parse_tokens(notes, T):
    """The tokens of a notes field: (cluster_only, alt_bin (nan if none), problem or None). The edge-sensitivity bin must be the floor bin of T plus or minus 25 K."""
    cluster_only = CLUSTER_TOKEN in notes
    found = EDGE_RE.findall(notes)
    if len(found) > 1:
        return cluster_only, np.nan, "more than one edge-sensitivity token"
    if not found:
        return cluster_only, np.nan, None
    alt = float(found[0])
    if in_window(T) and abs(alt - floor_bin(T)) != BIN:
        return cluster_only, alt, f"edge-sensitivity bin {alt:g} is not adjacent to the floor bin {floor_bin(T):g} of {T:g} K"
    return cluster_only, alt, None


def not_compared_reason(T):
    """Why a value is recorded and not compared (None if it is compared)."""
    if T < T_MIN:
        return "recorded, not compared (below 300 K, e.g. room temperature)"
    if T >= T_MAX:
        return "recorded, not compared (at or above 800 K)"
    return None


def load_values(path):
    """Read and validate the literature values; returns a DataFrame with the SI value of every row (property S, sigma, kappa or zT) and the parsed notes tokens."""
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
        cluster_only, alt, prob = parse_tokens(r["notes"], temp)
        if prob:
            problems.append(f"{where}: {prob}")
        conv = val * UNITS[r["property"]][r["unit"]]
        if r["property"] == "resistivity":
            if conv <= 0:
                problems.append(f"{where}: resistivity must be positive")
                continue
            prop, si = "sigma", 1.0 / conv
        else:
            prop, si = r["property"], conv
        rows.append({**r.to_dict(), "T_K": temp, "value": val, "target": prop, "value_target_units": si, "cluster_only": cluster_only, "alt_bin": alt})
    if problems:
        raise SystemExit("values file problems:\n  " + "\n  ".join(problems))
    return pd.DataFrame(rows)


def aggregate_bins(vals):
    """Amendments 2 and 3: one value per (DOI, composition_id, sample (compound_label), property, 25 K bin) for the rows inside [T_MIN, T_MAX): the arithmetic mean of value_target_units,
    with n_aggregated and the identifiers. The members of a group must agree on [cluster-level] and on the edge-sensitivity bin. Rows outside the window are passed through
    (n_aggregated = 1) and are never compared."""
    vals = vals.copy()
    for col, default in (("compound_label", ""), ("cluster_only", False), ("alt_bin", np.nan)):
        if col not in vals.columns:
            vals[col] = default
    out, groups = [], {}
    for _, r in vals.iterrows():
        d = r.to_dict()
        if not in_window(d["T_K"]):
            d.update(n_aggregated=1, aggregated_lit_ids=d["lit_id"])
            out.append(d)
            continue
        comp, _ = canon.parse_formula(d["composition"])
        key = (norm_doi(d["doi"]), canon.composition_id(comp), d["compound_label"], d["target"], floor_bin(d["T_K"]))
        groups.setdefault(key, []).append(d)
    for key, members in groups.items():
        first = dict(members[0])
        if len({bool(m["cluster_only"]) for m in members}) > 1:
            raise SystemExit(f"rows of one aggregation group disagree on [cluster-level]: {[m['lit_id'] for m in members]}")
        if len({None if pd.isna(m["alt_bin"]) else float(m["alt_bin"]) for m in members}) > 1:
            raise SystemExit(f"rows of one aggregation group disagree on the edge-sensitivity bin: {[m['lit_id'] for m in members]}")
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


def level_result(level_mask, target, tbin, sub, pred, lit, eps_t):
    """One level at one bin: the training rows of the level at the bin, mean over the rows then over the repeats; missing (n_rows 0) if the bin is outside the window or has no row."""
    out = {"n_rows": 0, "pred": np.nan, "pred_sd_repeats": np.nan, "rel_error": np.nan, "log10_ratio": np.nan, "inside_band": None}
    if not (isinstance(tbin, float) and not np.isnan(tbin) and in_window(tbin)):
        return out
    m = (level_mask & (sub["temperature_bin"] == tbin)).to_numpy()
    if not m.any():
        return out
    per_repeat = pred[:, m].mean(axis=1)  # model scale, one value per repeat
    lin = 10.0 ** per_repeat if target in LOG_TARGETS else per_repeat
    out["n_rows"] = int(m.sum())
    out["pred"] = float(np.mean(lin))
    out["pred_sd_repeats"] = float(np.std(lin, ddof=1))
    if lit != 0:
        out["rel_error"] = float((np.mean(lin) - lit) / abs(lit))
        out["inside_band"] = bool(abs(out["rel_error"]) <= eps_t)
        if target in LOG_TARGETS and lit > 0:
            out["log10_ratio"] = float(np.mean(per_repeat) - np.log10(lit))
    return out


def compare_value(r, cid, clid, sub, pred, eps):
    """The comparison columns of one aggregated value: <level>_<field> at its floor bin and <level>_alt_<field> at the edge-sensitivity bin (amendments 4 to 6); the exact level is not
    computed for a [cluster-level] value (amendment 3); a value outside the window gets no result and a reason."""
    t, lit = r["target"], r["value_target_units"]
    win = in_window(r["T_K"])
    tbin = floor_bin(r["T_K"]) if win else np.nan
    alt = r["alt_bin"] if win else np.nan
    cluster_only = bool(r["cluster_only"])
    row = {"in_window": win, "temperature_bin": tbin, "status": not_compared_reason(r["T_K"]) or "compared", "alt_bin": alt, "cluster_only": cluster_only,
           "exact_level": "not applicable ([cluster-level] value)" if cluster_only else "computed"}
    masks = {"exact": sub["composition_id"] == cid, "cluster": sub[ncv.GROUP_COL] == clid}
    for level, mask in masks.items():
        if level == "exact" and cluster_only:
            res = {"n_rows": 0, "pred": np.nan, "pred_sd_repeats": np.nan, "rel_error": np.nan, "log10_ratio": np.nan, "inside_band": None}
            res_alt = dict(res)
        else:
            res = level_result(mask, t, tbin, sub, pred, lit, eps[t])
            res_alt = level_result(mask, t, alt, sub, pred, lit, eps[t]) if not pd.isna(alt) else {"n_rows": 0, "pred": np.nan, "pred_sd_repeats": np.nan, "rel_error": np.nan,
                                                                                                    "log10_ratio": np.nan, "inside_band": None}
        for f in FIELDS:
            row[f"{level}_{f}"] = res[f]
            row[f"{level}_alt_{f}"] = res_alt[f]
    row["alt_status"] = ("" if pd.isna(alt) else ("alt bin outside the window" if not in_window(alt) else ("computed" if (row["cluster_alt_n_rows"] or row["exact_alt_n_rows"]) else "no training row in the alt bin")))
    return row


def swap_in_alt(res):
    """The run in which every edge value is evaluated in its neighbouring bin: the *_alt_* columns replace the headline columns of the rows that have an edge-sensitivity bin."""
    alt = res.copy()
    edge = alt["alt_bin"].notna() & alt["in_window"].astype(bool)
    for level in ("exact", "cluster"):
        for f in FIELDS:
            alt[f"{level}_{f}"] = alt[f"{level}_{f}"].astype(object)
            alt.loc[edge, f"{level}_{f}"] = alt.loc[edge, f"{level}_alt_{f}"]
    return alt


def _paper_block(g):
    """Per property and level: compared values, inside the band, fraction, median absolute relative error (one paper)."""
    block = {}
    for t in TARGETS:
        for level in ("exact", "cluster"):
            s = g[(g["target"] == t) & (g[f"{level}_n_rows"].astype(float) > 0)]
            n = int(len(s))
            inside = int((s[f"{level}_inside_band"] == True).sum())  # noqa: E712
            block[f"{t}_{level}"] = {"n_compared": n, "n_inside_band": inside, "fraction_inside_band": (inside / n) if n else None,
                                     "median_abs_rel_error": float(s[f"{level}_rel_error"].astype(float).abs().median()) if n else None}
    return block


def summarise(res):
    """Amendments 2 and 3 (b), (c). res has one row per aggregated value with columns paper, doi_in_training, target, in_window, cluster_only and, per level, n_rows, inside_band and rel_error.
    Returns the results per paper, the overall summary of the independent papers (papers weigh equally) and the training-source papers separately."""
    by_paper = {}
    for paper, g in res.groupby("paper"):
        by_paper[paper] = {"training_source": bool(g["doi_in_training"].all()), "n_values": int(len(g)), "n_in_window": int(g["in_window"].sum()),
                           "n_cluster_level_only": int(g["cluster_only"].astype(bool).sum()), "by_property_and_level": _paper_block(g)}
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


def edge_changes(res, alt):
    """The edge values whose result differs between the headline run and the neighbouring-bin run: per level, the in-band status and the relative error in both."""
    out = []
    for i in res.index[res["alt_bin"].notna() & res["in_window"].astype(bool)]:
        for level in ("exact", "cluster"):
            a, b = res.loc[i, f"{level}_n_rows"], alt.loc[i, f"{level}_n_rows"]
            if not a and not b:
                continue
            ra, rb = res.loc[i, f"{level}_rel_error"], alt.loc[i, f"{level}_rel_error"]
            out.append({"lit_id": res.loc[i, "lit_id"], "level": level, "bin": float(res.loc[i, "temperature_bin"]), "alt_bin": float(res.loc[i, "alt_bin"]),
                        "headline_inside_band": res.loc[i, f"{level}_inside_band"], "alt_inside_band": alt.loc[i, f"{level}_inside_band"],
                        "headline_rel_error": None if pd.isna(ra) else float(ra), "alt_rel_error": None if pd.isna(rb) else float(rb),
                        "status_changes": bool(res.loc[i, f"{level}_inside_band"] != alt.loc[i, f"{level}_inside_band"] or bool(a) != bool(b))})
    return out


def _syn_rec(paper, in_train, target, inside, rel, cluster_only=False, alt=np.nan):
    """A synthetic result record for the summaries."""
    rec = {"paper": paper, "doi_in_training": in_train, "target": target, "in_window": True, "cluster_only": cluster_only, "alt_bin": alt, "lit_id": f"{paper}-{target}-{rel}", "temperature_bin": 350.0}
    for level in ("exact", "cluster"):
        for f in FIELDS:
            rec[f"{level}_{f}"] = 0 if f == "n_rows" else (None if f == "inside_band" else np.nan)
    rec.update({"cluster_n_rows": 1, "cluster_inside_band": inside, "cluster_rel_error": rel})
    if not cluster_only:
        rec.update({"exact_n_rows": 1, "exact_inside_band": inside, "exact_rel_error": rel})
    return rec


def self_test():
    """Synthetic checks of the tokens, the aggregation, the levels, the edge sensitivity, the not-compared rows and the summaries; no dataset, no prediction."""
    eps = {"S": 0.06, "sigma": 0.08, "kappa": 0.11, "zT": 0.19}
    # ---- tokens
    assert parse_tokens("[cluster-level] [edge-sensitivity: also bin 325] x", 350.5)[:2] == (True, 325.0) and parse_tokens("", 350.5)[2] is None
    assert parse_tokens("[cluster-level] note", 400.0) == (True, parse_tokens("[cluster-level]", 400.0)[1], None) and np.isnan(parse_tokens("[cluster-level]", 400.0)[1])
    assert parse_tokens("[edge-sensitivity: also bin 400]", 351.0)[2] is not None, "a non-adjacent bin must be a problem"
    assert parse_tokens("[edge-sensitivity: also bin 325] [edge-sensitivity: also bin 375]", 350.5)[2] is not None, "two edge tokens must be a problem"
    assert parse_tokens("[edge-sensitivity: also bin 700]", 700.6)[2] is not None, "the floor bin itself is not a neighbouring bin"
    assert parse_tokens("[edge-sensitivity: also bin 675]", 700.6)[2] is None and parse_tokens("[edge-sensitivity: also bin 725]", 751.0)[2] is None
    # ---- room temperature and 800 K are recorded, not compared
    for T in (295.8, ROOM_TEMPERATURE_K, 299.99, 800.0, 802.1, 852.8):
        assert not in_window(T) and not_compared_reason(T) is not None
    for T in (300.0, 351.0, 799.9):
        assert in_window(T) and not_compared_reason(T) is None
    assert floor_bin(300.0) == 300.0 and floor_bin(324.9) == 300.0 and floor_bin(325.0) == 325.0 and floor_bin(799.9) == 775.0
    # ---- aggregation: per paper, composition, sample and property
    rows = []

    def row(i, doi, comp, label, target, T, v, cluster_only=False, alt=np.nan):
        rows.append({"lit_id": i, "doi": doi, "composition": comp, "compound_label": label, "target": target, "T_K": T, "value": v, "unit": "uV/K", "value_target_units": v,
                     "value_kind": "digitised", "cluster_only": cluster_only, "alt_bin": alt})
    row("a1", "10.1/A", "CsSnI3", "film", "S", 310.0, 100.0)    # same bin [300, 325) as a2
    row("a2", "10.1/A", "CsSnI3", "film", "S", 320.0, 140.0)
    row("a3", "10.1/A", "CsSnI3", "film", "S", 325.0, 90.0)     # next bin
    row("a4", "10.1/A", "CsSnI3", "film", "sigma", 310.0, 5.0)  # another property
    row("a5", "10.1/A", "LaCoO3", "film", "S", 310.0, 7.0)      # another composition
    row("b1", "10.1/B", "CsSnI3", "film", "S", 310.0, 50.0)     # another paper
    row("a6", "10.1/A", "CsSnI3", "film", "S", ROOM_TEMPERATURE_K, 11.0)    # below the window: not aggregated
    row("a7", "10.1/A", "CsSnI3", "film", "S", 301.0, 12.0)     # in the first bin with a1 and a2
    row("a8", "10.1/A", "CsSnI3", "film", "S", 800.0, 13.0)     # at 800 K: outside the window
    row("o1", "10.1/O", "SrTiO3", "La-doped SrTiO3, n = 8.4e19 cm^-3 (sample (a))", "S", 351.0, -422.0, True)    # Ohta samples: one host formula, different stated samples
    row("o2", "10.1/O", "SrTiO3", "La-doped SrTiO3, n = 8.4e20 cm^-3 (sample (b))", "S", 351.0, -154.0, True)
    row("o3", "10.1/O", "SrTiO3", "La-doped SrTiO3, n = 8.4e19 cm^-3 (sample (a))", "S", 360.0, -430.0, True)  # same sample and bin as o1
    df = pd.DataFrame(rows)
    agg = aggregate_bins(df)
    first = agg[(agg["doi"] == "10.1/A") & (agg["composition"] == "CsSnI3") & (agg["target"] == "S") & (agg["T_K"] > 300) & (agg["T_K"] < 325)]
    assert len(first) == 1, "a1, a2 and a7 must give one value"
    assert first.iloc[0]["n_aggregated"] == 3 and abs(first.iloc[0]["value_target_units"] - (100 + 140 + 12) / 3) < 1e-9
    assert set(first.iloc[0]["aggregated_lit_ids"].split(";")) == {"a1", "a2", "a7"}
    assert len(agg) == len(df) - 3, f"{len(df)} rows must become {len(df) - 3}, not {len(agg)}"
    assert (agg[agg["T_K"] < 300]["n_aggregated"] == 1).all() and (agg[agg["T_K"] >= 800]["n_aggregated"] == 1).all()
    assert agg[agg["lit_id"] == "b1"].iloc[0]["n_aggregated"] == 1 and agg[agg["lit_id"] == "a5"].iloc[0]["n_aggregated"] == 1
    oh = agg[agg["doi"] == "10.1/O"]
    assert len(oh) == 2, "two Ohta samples with the same host formula and bin must stay two values (per-sample key), o1 and o3 (same sample) merge"
    sa = oh[oh["compound_label"].str.contains("sample \\(a\\)")].iloc[0]
    assert sa["n_aggregated"] == 2 and abs(sa["value_target_units"] - (-426.0)) < 1e-9 and set(sa["aggregated_lit_ids"].split(";")) == {"o1", "o3"}
    assert oh[oh["compound_label"].str.contains("sample \\(b\\)")].iloc[0]["n_aggregated"] == 1
    assert bool(oh["cluster_only"].all()), "[cluster-level] must survive the aggregation"
    for bad, msg in ((dict(cluster_only=False), "cluster-level"), (dict(alt=325.0), "edge-sensitivity")):
        rr2 = [dict(x) for x in rows if x["doi"] == "10.1/O" and "(a)" in x["compound_label"]]
        rr2[1]["cluster_only"] = bad.get("cluster_only", rr2[1]["cluster_only"])
        rr2[1]["alt_bin"] = bad.get("alt", rr2[1]["alt_bin"])
        try:
            aggregate_bins(pd.DataFrame(rr2))
        except SystemExit as e:
            assert msg in str(e)
        else:
            raise AssertionError(f"members that disagree on {msg} must stop the run")
    # ---- levels, cluster-level only, edge sensitivity, on a synthetic training set
    sub = pd.DataFrame({"composition_id": ["C1", "C1", "C2", "C2", "C3"], ncv.GROUP_COL: ["G1", "G1", "G1", "G1", "G2"],
                        "temperature_bin": [350.0, 350.0, 350.0, 325.0, 350.0]})
    pred = np.array([[10.0 * (i + 1) + r for i in range(5)] for r in range(N_REPEATS)])  # row i, repeat r: 10 (i + 1) + r
    base = {"doi": "10.1/X", "composition": "x", "compound_label": "x", "target": "S", "value_target_units": 100.0, "cluster_only": False, "alt_bin": np.nan, "T_K": 355.0}
    r1 = compare_value(base, "C1", "G1", sub, pred, eps)
    assert r1["exact_n_rows"] == 2 and abs(r1["exact_pred"] - (np.mean([10, 20]) + 2.0)) < 1e-9, "exact: rows 0 and 1 at bin 350, mean over rows, then over repeats (+2)"
    assert r1["cluster_n_rows"] == 3 and abs(r1["cluster_pred"] - (np.mean([10, 20, 30]) + 2.0)) < 1e-9 and r1["status"] == "compared" and r1["alt_status"] == ""
    r2 = compare_value({**base, "cluster_only": True}, "C1", "G1", sub, pred, eps)
    assert r2["exact_n_rows"] == 0 and r2["exact_inside_band"] is None and "not applicable" in r2["exact_level"], "a [cluster-level] value has no exact level"
    assert r2["cluster_n_rows"] == 3 and r2["cluster_pred"] == r1["cluster_pred"], "its cluster level is unchanged"
    r3 = compare_value({**base, "T_K": 349.0, "alt_bin": 350.0}, "C2", "G1", sub, pred, eps)   # floor bin 325, alt bin 350: both reported
    assert r3["temperature_bin"] == 325.0 and r3["exact_n_rows"] == 1 and abs(r3["exact_pred"] - (40.0 + 2.0)) < 1e-9
    assert r3["exact_alt_n_rows"] == 1 and abs(r3["exact_alt_pred"] - (30.0 + 2.0)) < 1e-9, "the alt bin 350 holds only row 2 of composition C2"
    assert r3["cluster_n_rows"] == 1 and r3["cluster_alt_n_rows"] == 3 and r3["alt_status"] == "computed"
    r4 = compare_value({**base, "T_K": 301.0, "alt_bin": 275.0}, "C1", "G1", sub, pred, eps)   # alt bin below the window
    assert r4["alt_status"] == "alt bin outside the window" and r4["cluster_alt_n_rows"] == 0 and r4["exact_alt_n_rows"] == 0
    r5 = compare_value({**base, "T_K": 750.0, "alt_bin": 775.0}, "C1", "G1", sub, pred, eps)   # alt bin in the window but no training row
    assert r5["alt_status"] == "no training row in the alt bin" and r5["cluster_n_rows"] == 0
    for T in (ROOM_TEMPERATURE_K, 295.8, 800.0, 852.8):
        rn = compare_value({**base, "T_K": T, "alt_bin": 325.0}, "C1", "G1", sub, pred, eps)
        assert not rn["in_window"] and rn["status"].startswith("recorded, not compared") and rn["exact_n_rows"] == 0 and rn["cluster_n_rows"] == 0 and rn["cluster_alt_n_rows"] == 0, \
            "values outside the window get no result, alt bin included"
    lit_sd = compare_value({**base, "value_target_units": 41.0}, "C2", "G1", sub, pred, eps)["exact_rel_error"]
    assert abs(lit_sd - ((np.mean([30.0]) + 2.0 - 41.0) / 41.0)) < 1e-9, "relative error (prediction - literature) / |literature|"
    # ---- summaries: papers weigh equally, training source separate, edge-sensitivity run swaps the neighbouring bin in
    recs = []
    for i in range(10):
        recs.append(_syn_rec("P", False, "S", True, 0.01 + i * 1e-4))
    recs.append(_syn_rec("Q", False, "S", False, 0.5))
    for i in range(5):
        recs.append(_syn_rec("F", True, "S", True, 0.001 + i * 1e-4))
    res = pd.DataFrame(recs)
    sm = summarise(res)
    o = sm["overall_independent_papers"]["by_property_and_level"]["S_exact"]
    assert sm["overall_independent_papers"]["n_papers"] == 2 and sm["overall_independent_papers"]["papers"] == ["P", "Q"]
    assert abs(o["mean_over_papers_of_fraction_inside_band"] - 0.5) < 1e-12, "papers must weigh equally (0.5), not rows (10/11)"
    assert "F" not in sm["overall_independent_papers"]["papers"] and "F" in sm["training_source_papers"], "training-source papers are reported separately"
    assert sm["by_paper"]["F"]["training_source"] and not sm["by_paper"]["P"]["training_source"]
    # a cluster-level-only paper O (an independent paper) with two edge values, one of which falls outside the band in the neighbouring bin; a training-source edge value in F2
    recs = [_syn_rec("O", False, "S", True, 0.01, cluster_only=True, alt=325.0), _syn_rec("O", False, "S", True, 0.02, cluster_only=True), _syn_rec("O", False, "S", True, 0.03, cluster_only=True, alt=375.0),
            _syn_rec("F2", True, "S", True, 0.001, cluster_only=True, alt=325.0)]
    for rec, a in zip(recs, (dict(n_rows=1, inside_band=False, rel_error=0.4), None, dict(n_rows=0, inside_band=None, rel_error=np.nan), dict(n_rows=1, inside_band=False, rel_error=0.9))):
        for level in ("exact", "cluster"):
            for f in FIELDS:
                rec[f"{level}_alt_{f}"] = a.get(f) if (a and level == "cluster" and f in a) else (0 if f == "n_rows" else (np.nan if f != "inside_band" else None))
    res2 = pd.DataFrame(recs)
    alt = swap_in_alt(res2)
    head, altsum = summarise(res2), summarise(alt)
    ho = head["overall_independent_papers"]["by_property_and_level"]["S_cluster"]
    ao = altsum["overall_independent_papers"]["by_property_and_level"]["S_cluster"]
    assert ho["mean_over_papers_of_fraction_inside_band"] == 1.0, "headline: all three O values inside the band"
    assert abs(ao["mean_over_papers_of_fraction_inside_band"] - 0.5) < 1e-12, "neighbouring-bin run: first value outside, second unchanged, third missing (1 of 2 compared inside)"
    assert head["by_paper"]["O"]["n_cluster_level_only"] == 3 and head["by_paper"]["O"]["by_property_and_level"]["S_exact"]["n_compared"] == 0, "cluster-level values never count at the exact level"
    assert "F2" in altsum["training_source_papers"] and "F2" not in altsum["overall_independent_papers"]["papers"], "training source stays separate in the neighbouring-bin run too"
    ch = edge_changes(res2, alt)
    assert sorted(c["lit_id"] for c in ch if c["status_changes"]) == sorted(res2.loc[i, "lit_id"] for i in (0, 2, 3)), "the three edge values whose result changes are listed (changed band, missing in the neighbouring bin)"
    assert all(c["level"] == "cluster" for c in ch), "no exact-level change is reported for [cluster-level] values"
    print("self-test passed: tokens (valid, non-adjacent bin, two tokens), room-temperature and 800 K rows not compared, per-sample aggregation (Ohta samples with one host formula stay separate,"
          " same sample merges, disagreeing members stop the run), levels (exact, cluster, [cluster-level] has no exact level), edge sensitivity (both bins, alt bin outside the window or"
          " empty reported), summaries (papers weigh equally, training source separate, neighbouring-bin run)")


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
    print(f"{len(vals)} literature values, {vals['doi'].map(norm_doi).nunique()} DOIs, {vals['compound_label'].nunique()} samples/compounds: valid")
    n_raw = len(vals)
    wmask = vals["T_K"].map(in_window)
    print(f"  inside [300, 800) K: {int(wmask.sum())}; recorded, not compared: {int((vals['T_K'] < T_MIN).sum())} below 300 K, {int((vals['T_K'] >= T_MAX).sum())} at or above 800 K")
    print(f"  [cluster-level] rows: {int(vals['cluster_only'].sum())}; edge-sensitivity rows: {int(vals['alt_bin'].notna().sum())} ({int((vals['alt_bin'].notna() & wmask).sum())} inside the window)")
    vals = aggregate_bins(vals)
    print(f"after the aggregation per paper, composition, sample, property and 25 K bin (amendments 2 and 3): {len(vals)} values "
          f"({int(vals['T_K'].map(in_window).sum())} to compare, {int(vals['cluster_only'].sum())} cluster-level only, {int((vals['alt_bin'].notna() & vals['T_K'].map(in_window)).sum())} with an edge-sensitivity bin)")
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
        sub, pred = oof[r["target"]]
        row = {**r.to_dict(), "composition_id": cid, "chemistry_cluster_id": clid, "doi_in_training": doi in train_dois, "cluster_in_training": clid in train_clusters,
               "composition_in_training": (cid in train_comps) and not bool(r["cluster_only"]), "doi_and_composition_in_training": ((doi, cid) in doi_comp) and not bool(r["cluster_only"])}
        row.update(compare_value(r, cid, clid, sub, pred, eps))
        row["relative_uncertainty_band"] = eps[r["target"]]
        row["paper"] = doi
        out.append(row)
    res = pd.DataFrame(out)
    alt = swap_in_alt(res)
    changes = edge_changes(res, alt)
    summary = {"n_values_read": int(n_raw), "n_values_after_aggregation": int(len(res)), "n_in_window": int(res["in_window"].sum()), "n_cluster_level_only": int(res["cluster_only"].sum()),
               "relative_uncertainty": eps, "n_doi_in_training": int(res["doi_in_training"].sum()), "n_composition_in_training": int(res["composition_in_training"].sum()),
               "n_cluster_in_training": int(res["cluster_in_training"].sum()), "n_doi_and_composition_in_training": int(res["doi_and_composition_in_training"].sum()),
               **summarise(res),
               "edge_sensitivity": {"n_edge_values_in_window": int((res["alt_bin"].notna() & res["in_window"].astype(bool)).sum()),
                                    "headline_overall_independent_papers": summarise(res)["overall_independent_papers"], "neighbouring_bin_overall_independent_papers": summarise(alt)["overall_independent_papers"],
                                    "training_source_headline": summarise(res)["training_source_papers"], "training_source_neighbouring_bin": summarise(alt)["training_source_papers"],
                                    "edge_values": changes}}
    d = Path(args.out_dir) if args.out_dir else REPO / "thesis_paper" / "results" / "na12" / rr.utc_stamp()
    d.mkdir(parents=True, exist_ok=True)
    res.to_csv(d / "comparison.csv", index=False, lineterminator="\n")
    rr.write_json(d / "summary.json", summary)
    ladder_hashes = {str(p.relative_to(REPO)): rr.sha256_file(p) for t in TARGETS for p in sorted((REPO / LADDER / f"{t}_chemistry_full").glob("repeat*_fold*_predictions.npz"))}
    rr.write_json(d / "run_config.json", {**prov, "ladder_dir": LADDER, "ladder_prediction_files_sha256": ladder_hashes, "seed": SEED})
    print("wrote", d)
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
