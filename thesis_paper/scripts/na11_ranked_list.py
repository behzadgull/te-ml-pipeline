"""
NA11: the ranked lists of the Materials Project candidates from the final-model predictions, under the pre-registered rules of docs/decisions.md (2026-10-05, committed before the G3 output
was unpacked).

Per candidate (Materials Project entry): at each of the 6 temperatures (300 to 800 K, 100 K steps) the direct zT prediction; the candidate's maximum over the grid, the temperature of the maximum
(ties: the lower temperature), and the other predictions at that temperature: S with the classifier's sign (`S`), the regressor's S (`S_reg`), sigma, kappa. The override fraction is the share of
(candidate, temperature) predictions whose sign the classifier changed, and the share of candidates with at least one changed prediction.

Rule A. Shortlist = compounds of the main list (E_hull <= 0.05 eV/atom, gap <= 0.6 eV, main toxic-element list) that are in the candidate set of every E_hull tightening of the sensitivity run
(<= 0.025 and <= 0) and whose chemistry cluster is in the training data; recomputed here from the committed tables and asserted to equal the four ids fixed in the pre-registration
(mp-3163 BaSnO3, mp-614013 CsSnI3, mp-1288145 LaCoO3, mp-5163 LaRhO3). Ordered by the maximum predicted zT, highest first. Every candidate carries the literature label of
docs/candidate_literature_labels.csv; the full lists are ranked within the groups seen / unseen chemistry cluster, with the stability label (E_hull = 0 or metastable). No predicted value is
treated as a finding: the lists order hypotheses.

Output: thesis_paper/results/na11_ranked/<UTC>/{ranked_all_409.csv, ranked_main_30.csv, shortlist.csv, summary.json, run_config.json}.
Usage (from the repository root):  python thesis_paper/scripts/na11_ranked_list.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_record as rr  # noqa: E402

PRED = "thesis_paper/results/final_b/20261004T201433/predictions_mp.csv"
LISTS = "thesis_paper/results/na11_candidate_novelty/20261004T185101/candidate_lists.csv"
MEMBERSHIP = "thesis_paper/results/na11_sensitivity/20261004T135205/membership.csv"
LABELS = "thesis_paper/docs/candidate_literature_labels.csv"
PRE_REGISTERED_SHORTLIST = ["mp-3163", "mp-614013", "mp-1288145", "mp-5163"]
GRID = (300, 400, 500, 600, 700, 800)
MAIN_TOXIC = "Tl,Hg,Cd,As,Be excluded (main)"


def main():
    """Entry point."""
    prov = rr.provenance({"predictions_mp": PRED, "candidate_lists": LISTS, "membership": MEMBERSHIP, "labels": LABELS}, __file__)
    p = pd.read_csv(REPO / PRED)
    assert len(p) == 409 * len(GRID) and sorted(p["T_K"].unique()) == list(GRID)
    c = pd.read_csv(REPO / LISTS)
    lab = pd.read_csv(REPO / LABELS)[["material_id", "previously_studied_as_thermoelectric", "doi"]].rename(columns={"doi": "literature_doi"})
    rows = []
    for mid, g in p.groupby("material_id", sort=False):
        g = g.sort_values("T_K")
        i = int(np.argmax(g["zT_direct"].to_numpy()))  # first maximum: the lower temperature on ties
        r = g.iloc[i]
        rows.append({"material_id": mid, "formula": r["formula"], "ehull": r["ehull"], "gap": r["gap"], "is_stable": bool(r["is_stable"]), "spg_symbol": r["spg_symbol"],
                     "zT_max": float(r["zT_direct"]), "T_at_zT_max": int(r["T_K"]), "S_at_max": float(r["S"]), "S_reg_at_max": float(r["S_reg"]), "sigma_at_max": float(r["sigma"]),
                     "kappa_at_max": float(r["kappa"]), "sign_overridden_at_max": bool(r["sign_overridden"]), "n_sign_overridden_of_6": int(g["sign_overridden"].sum()),
                     "p_type_at_max": bool(r["S"] > 0)})
    d = pd.DataFrame(rows).merge(c[["material_id", "in_main", "in_E_hull_0.10", "in_gap_cap_0.9", "in_E_hull_0.10_cap_0.9", "cluster_seen_any"]], on="material_id", how="left")
    d = d.merge(lab, on="material_id", how="left")
    assert len(d) == 409 and d["in_main"].notna().all()
    d["stability"] = np.where(d["ehull"] <= 1e-9, "on the hull", "metastable")
    d["group"] = np.where(d["cluster_seen_any"], "cluster seen in training", "cluster unseen")
    # rule A
    m = pd.read_csv(REPO / MEMBERSHIP)
    sets = {e: set(m[(m["toxic"] == MAIN_TOXIC) & (m["e_hull_max"] == e) & (m["gap_cap"] == "0.6 (main)")]["material_id"]) for e in ("0.05 (main)", "0.025", "0.0 (stable only)")}
    main_ids = sets["0.05 (main)"]
    assert main_ids == set(d[d["in_main"]]["material_id"])
    survive = main_ids & sets["0.025"] & sets["0.0 (stable only)"]
    shortlist_ids = sorted(survive & set(d[d["cluster_seen_any"]]["material_id"]))
    assert shortlist_ids == sorted(PRE_REGISTERED_SHORTLIST), f"the shortlist differs from the pre-registered one: {shortlist_ids}"
    d["shortlist"] = d["material_id"].isin(shortlist_ids)
    # ranks: within group by zT_max (all 409 and the main 30 separately)
    d = d.sort_values(["group", "zT_max"], ascending=[False, False]).reset_index(drop=True)
    d["rank_in_group_all_409"] = d.groupby("group")["zT_max"].rank(ascending=False, method="first").astype(int)
    mn = d[d["in_main"]].copy()
    mn["rank_in_group_main_30"] = mn.groupby("group")["zT_max"].rank(ascending=False, method="first").astype(int)
    sl = d[d["shortlist"]].sort_values("zT_max", ascending=False).copy()
    sl["shortlist_order"] = range(1, len(sl) + 1)
    out = REPO / "thesis_paper" / "results" / "na11_ranked" / rr.utc_stamp()
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "ranked_all_409.csv", index=False, lineterminator="\n")
    mn.sort_values(["group", "rank_in_group_main_30"], ascending=[False, True]).to_csv(out / "ranked_main_30.csv", index=False, lineterminator="\n")
    sl.to_csv(out / "shortlist.csv", index=False, lineterminator="\n")
    ov = lambda mask: {"entries": int(mask.sum()), "predictions": int(mask.sum() * len(GRID)),  # noqa: E731
                       "predictions_overridden": int(p[p["material_id"].isin(d[mask]["material_id"])]["sign_overridden"].sum()),
                       "share_predictions_overridden": float(p[p["material_id"].isin(d[mask]["material_id"])]["sign_overridden"].mean()),
                       "entries_with_any_override": int((d[mask]["n_sign_overridden_of_6"] > 0).sum()),
                       "entries_overridden_at_max": int(d[mask]["sign_overridden_at_max"].sum())}
    summary = {"n_entries": 409, "grid_K": list(GRID), "shortlist_ids": shortlist_ids, "shortlist_order": sl["formula"].tolist(),
               "shortlist_zT_max": {r["formula"]: [float(r["zT_max"]), int(r["T_at_zT_max"])] for _, r in sl.iterrows()},
               "override_all_409": ov(np.ones(len(d), bool)), "override_main_30": ov(d["in_main"].to_numpy()), "override_shortlist": ov(d["shortlist"].to_numpy()),
               "main_30": {"seen": int((mn["group"] == "cluster seen in training").sum()), "unseen": int((mn["group"] == "cluster unseen").sum()),
                           "zT_max_range": [float(mn["zT_max"].min()), float(mn["zT_max"].max())], "on_the_hull": int((mn["stability"] == "on the hull").sum()),
                           "p_type_at_max": int(mn["p_type_at_max"].sum()), "T_at_max_counts": {int(k): int(v) for k, v in mn["T_at_zT_max"].value_counts().sort_index().items()}},
               "all_409": {"T_at_max_counts": {int(k): int(v) for k, v in d["T_at_zT_max"].value_counts().sort_index().items()}, "zT_max_range": [float(d["zT_max"].min()), float(d["zT_max"].max())]}}
    rr.write_json(out / "summary.json", summary)
    rr.write_json(out / "run_config.json", prov)
    print("wrote", out)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
