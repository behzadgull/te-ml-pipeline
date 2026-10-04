"""
NA10: how the final models' Seebeck predictions compare with the JARVIS dft_3d values (BoltzTraP, constant relaxation time, 600 K, one fixed doping).
A cross-domain description, not a validation: JARVIS values are computed, not measured, at one temperature and one doping, for ideal crystals, and the models
were trained on measured values of real, doped samples.

How the two signs are defined (stated because the first scoring in na_final_models.py chose them in a way that cannot be tested)
  - Predicted sign: the sign of the predicted S (the regressor's S in this run; when a predictions file carries the carrier-type classifier's decision, the column
    `S` already has the classifier's sign and `sign_overridden` marks the entries where it overruled the regressor).
  - JARVIS sign: JARVIS gives two values per entry, one for n-type and one for p-type doping. For a semiconductor they have opposite signs (n negative, p
    positive): the sign is set by the doping JARVIS imposed, not by the material, so it cannot be compared with a predicted carrier type. Only for entries whose two values
    have the SAME sign (metals and semimetals, where the carrier type is a property of the band structure) does JARVIS define a sign: sign(p) = sign(n). Sign agreement is
    therefore computed on that same-sign subset only; the opposite-sign entries are counted and excluded from it. (The earlier `sign_agreement_with_p_value` of
    na_final_models.py compared the predicted sign with the p value, which is positive for almost every semiconductor, so it measured the share of predicted p-type
    entries, not agreement.)
  - Magnitude reference: mean(|n|, |p|) for every entry (the two values of a metal are nearly equal; for a semiconductor the mean is a doping-independent scale). The rank
    correlation is Spearman between |predicted S| and that reference; for the same-sign subset the signed Spearman between predicted S and the common JARVIS value and the
    R2 are reported too.
Strata: all entries; same-sign and opposite-sign entries; ABX3 stoichiometry; chemistry cluster seen / unseen in the training data; composition seen / unseen;
entries with a near-zero reference (< 20 uV/K, where the sign is not defined in practice) are dropped from the sign analysis only (threshold stated in the output).
Metal / semiconductor split: the JARVIS optb88vdw gap (read from the NA10 entries by jid) separates metals and semimetals (gap < GAP_METAL = 0.05 eV) from semiconductors
(gap >= 0.05 eV); the strata "metal_or_semimetal", "semiconductor" and their seen / unseen parts are analysed like the others. The sign test covers mostly metals: the same-sign
entries are almost all metallic (a semiconductor's n and p values have opposite signs by construction), so the count of same-sign entries per gap class is reported.
Intervals: chemistry-cluster bootstrap (clusters resampled with replacement, n_boot draws, seed 0).
Counts per step are read from the NA10 preparation (results/na10/<stamp>/summary.json) and extended with the splits made here.

Usage (from the repository root):
    python thesis_paper/scripts/na10_jarvis_analysis.py --predictions thesis_paper/results/final_a/<stamp>/predictions_jarvis.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_record as rr  # noqa: E402

NA10_RUN = "thesis_paper/results/na10/20261003T133155"
MIN_REF = 20.0  # uV/K: below this the sign of the reference is not meaningful
GAP_METAL = 0.05  # eV, JARVIS optb88vdw gap: below it an entry is treated as a metal or semimetal, from it upwards as a semiconductor (a stated choice, not tuned)


def boot_ci(stat, groups, idx, n_boot, rng):
    """Cluster-bootstrap percentile interval of stat(row_indices) over the rows `idx`, resampling the clusters of `groups[idx]`."""
    codes, _ = pd.factorize(groups[idx])
    nc = codes.max() + 1
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(nc + 1))
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, nc, nc)
        rows = np.concatenate([idx[order[starts[c]:starts[c + 1]]] for c in pick])
        vals.append(stat(rows))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    pred_path = Path(args.predictions)
    prov = rr.provenance({"predictions": str(pred_path.resolve().relative_to(REPO)), "na10_summary": f"{NA10_RUN}/summary.json",
                          "na10_entries": f"{NA10_RUN}/jarvis_entries.csv"}, __file__)
    d = pd.read_csv(pred_path)
    prep = json.loads((REPO / NA10_RUN / "summary.json").read_text(encoding="utf-8"))
    assert len(d) == prep["featurised_entries"], "the predictions must cover every featurised JARVIS entry"
    ent = pd.read_csv(REPO / NA10_RUN / "jarvis_entries.csv", usecols=["jid", "optb88vdw_bandgap"])
    assert d["jid"].tolist() == ent["jid"].tolist(), "the predictions and the NA10 entries must list the same JARVIS ids in the same order"
    gap = ent["optb88vdw_bandgap"].to_numpy(float)
    metal = gap < GAP_METAL
    seen = d["cluster_seen"].to_numpy(bool)
    n, p = d["n_seebeck"].to_numpy(float), d["p_seebeck"].to_numpy(float)
    S = d["S"].to_numpy(float)
    ref_abs = (np.abs(n) + np.abs(p)) / 2
    same_sign = (np.sign(n) == np.sign(p)) & (np.sign(n) != 0)
    opposite = (np.sign(n) != np.sign(p)) & (np.sign(n) != 0) & (np.sign(p) != 0)
    common = np.where(same_sign, p, np.nan)
    groups = d["chemistry_cluster_id"].to_numpy()
    strata = {"all": np.ones(len(d), bool), "same_sign_entries": same_sign, "opposite_sign_entries": opposite,
              "abx3_stoichiometry": d["abx3_stoichiometry"].to_numpy(bool), "cluster_seen": d["cluster_seen"].to_numpy(bool),
              "cluster_unseen": ~d["cluster_seen"].to_numpy(bool), "composition_seen": d["composition_seen"].to_numpy(bool),
              "composition_unseen": ~d["composition_seen"].to_numpy(bool),
              "metal_or_semimetal": metal, "semiconductor": ~metal, "metal_cluster_seen": metal & seen, "metal_cluster_unseen": metal & ~seen,
              "semiconductor_cluster_seen": ~metal & seen, "semiconductor_cluster_unseen": ~metal & ~seen}
    rng = np.random.default_rng(0)
    out_strata = {}
    for name, m in strata.items():
        idx = np.where(m)[0]
        if len(idx) < 30:
            out_strata[name] = {"n": int(len(idx)), "note": "fewer than 30 entries, not analysed"}
            continue
        rho = lambda rows: float(stats.spearmanr(np.abs(S[rows]), ref_abs[rows])[0])  # noqa: E731
        row = {"n": int(len(idx)), "n_clusters": int(pd.Series(groups[idx]).nunique()),
               "spearman_abs_pred_vs_mean_abs_jarvis": rho(idx), "spearman_ci95_cluster_bootstrap": boot_ci(rho, groups, idx, args.n_boot, rng),
               "median_abs_pred": float(np.median(np.abs(S[idx]))), "median_abs_jarvis_mean": float(np.median(ref_abs[idx])),
               "share_predicted_p_type": float((S[idx] > 0).mean())}
        sg = idx[same_sign[idx] & (ref_abs[idx] >= MIN_REF)]
        if len(sg) >= 30:
            agree = lambda rows: float((np.sign(S[rows]) == np.sign(common[rows])).mean())  # noqa: E731
            sres = {"n": int(len(sg)), "sign_agreement": agree(sg), "sign_agreement_ci95_cluster_bootstrap": boot_ci(agree, groups, sg, args.n_boot, rng),
                    "share_jarvis_positive": float((common[sg] > 0).mean()), "share_predicted_positive": float((S[sg] > 0).mean())}
            sres["signed_spearman"] = float(stats.spearmanr(S[sg], common[sg])[0])
            sres["r2_signed"] = float(1 - ((common[sg] - S[sg]) ** 2).sum() / ((common[sg] - common[sg].mean()) ** 2).sum())
            row["sign_analysis_same_sign_entries_with_reference_ge_20"] = sres
        out_strata[name] = row
    counts = {"jarvis_entries_total": prep["entries_total"], "with_seebeck_n_and_p": prep["with_both"], "featurised": prep["featurised_entries"],
              "predicted": int(len(d)), "same_sign_entries": int(same_sign.sum()), "opposite_sign_entries": int(opposite.sum()),
              "metal_or_semimetal_gap_lt_0.05": int(metal.sum()), "semiconductor_gap_ge_0.05": int((~metal).sum()),
              "same_sign_in_metals": int((same_sign & metal).sum()), "same_sign_in_semiconductors": int((same_sign & ~metal).sum()),
              "opposite_sign_in_metals": int((opposite & metal).sum()), "opposite_sign_in_semiconductors": int((opposite & ~metal).sum()),
              "same_sign_ref_ge_20_in_metals": int((same_sign & metal & (ref_abs >= MIN_REF)).sum()),
              "same_sign_ref_ge_20_in_semiconductors": int((same_sign & ~metal & (ref_abs >= MIN_REF)).sum()),
              "neither": int(len(d) - same_sign.sum() - opposite.sum()), "same_sign_with_reference_ge_20_uV_per_K": int((same_sign & (ref_abs >= MIN_REF)).sum()),
              "composition_seen": prep["composition_seen"], "cluster_seen": prep["cluster_seen"], "abx3_stoichiometry": prep["abx3_stoichiometry"],
              "abx3_stoichiometry_cluster_unseen": prep["abx3_stoichiometry_cluster_unseen"]}
    out = {"counts_per_step": counts, "strata": out_strata, "min_reference_uV_per_K": MIN_REF, "gap_metal_threshold_eV": GAP_METAL, "n_boot": args.n_boot,
           "predicted_sign_source": "classifier" if "sign_overridden" in d.columns else "regressor",
           "definitions": "see the module docstring: sign is analysed only where JARVIS gives one (n and p values of the same sign); magnitude reference is mean(|n|,|p|)"}
    o = REPO / "thesis_paper" / "results" / "na10_analysis" / rr.utc_stamp()
    o.mkdir(parents=True, exist_ok=True)
    rr.write_json(o / "analysis.json", out)
    rr.write_json(o / "run_config.json", prov)
    print("wrote", o)
    print(json.dumps({"counts": counts, "strata": out_strata}, indent=1)[:6000])


if __name__ == "__main__":
    main()
