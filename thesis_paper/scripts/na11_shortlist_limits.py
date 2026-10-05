"""
NA11: physics check of the pre-registered shortlist (docs/decisions.md, 2026-10-05; the rule and its order are NOT changed). Added after the predictions were seen, at the maintainer's request:

  1. Thermal limit. docs/shortlist_thermal_limits.csv holds, per shortlist compound, the melting or decomposition temperature (or the statement that none was found at or below 800 K) with its source.
     Predictions are reported only for grid temperatures below an unconditional limit; a conditional limit (LaCoO3 in a reducing atmosphere) is flagged, not applied.
  2. Secondary view. The ranking by the predicted zT at SECONDARY_T (600 K), because the primary rule's maxima sit at the 800 K edge of the grid (406 of 409 compounds), a boundary value.

Output: thesis_paper/results/na11_shortlist_limits/<UTC>/{shortlist_limits.csv, summary.json, run_config.json}.
Usage (from the repository root):  python thesis_paper/scripts/na11_shortlist_limits.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_record as rr  # noqa: E402

PRED = "thesis_paper/results/final_b/20261004T201433/predictions_mp.csv"
LIMITS = "thesis_paper/docs/shortlist_thermal_limits.csv"
RANKED = "thesis_paper/results/na11_ranked/20261005T053426/shortlist.csv"
SECONDARY_T = 600  # K, the secondary view; a stated choice (the middle of the grid)
GRID = (300, 400, 500, 600, 700, 800)


def main():
    """Entry point."""
    prov = rr.provenance({"predictions_mp": PRED, "limits": LIMITS, "shortlist": RANKED}, __file__)
    p = pd.read_csv(REPO / PRED)
    lim = pd.read_csv(REPO / LIMITS)
    sl = pd.read_csv(REPO / RANKED)
    assert set(lim["material_id"]) == set(sl["material_id"]) and len(lim) == 4
    rows = []
    for _, r in sl.iterrows():
        l = lim[lim["material_id"] == r["material_id"]].iloc[0]
        g = p[p["material_id"] == r["material_id"]].sort_values("T_K")
        assert list(g["T_K"]) == list(GRID)
        ul = float(l["unconditional_limit_K"]) if pd.notna(l["unconditional_limit_K"]) else None
        cl = float(l["conditional_limit_K"]) if pd.notna(l["conditional_limit_K"]) else None
        ok = g["T_K"] < ul if ul is not None else pd.Series(True, index=g.index)
        gk = g[ok]
        i = gk["zT_direct"].to_numpy().argmax()
        rows.append({"material_id": r["material_id"], "formula": r["formula"], "unconditional_limit_K": ul, "conditional_limit_K": cl, "limit_kind": l["limit_kind"],
                     "grid_max_reported_K": int(gk["T_K"].max()), "n_grid_points_reported": int(len(gk)), "zT_max_primary_all_grid": float(r["zT_max"]), "T_at_zT_max_primary": int(r["T_at_zT_max"]),
                     "zT_max_within_limit": float(gk["zT_direct"].iloc[i]), "T_at_zT_max_within_limit": int(gk["T_K"].iloc[i]),
                     "zT_at_secondary_T": float(g[g["T_K"] == SECONDARY_T]["zT_direct"].iloc[0]),
                     "primary_value_inside_limit": bool(ul is None or r["T_at_zT_max"] < ul)})
    d = pd.DataFrame(rows)
    d["rank_primary"] = d["zT_max_primary_all_grid"].rank(ascending=False, method="first").astype(int)
    d["rank_within_limit"] = d["zT_max_within_limit"].rank(ascending=False, method="first").astype(int)
    d["rank_at_secondary_T"] = d["zT_at_secondary_T"].rank(ascending=False, method="first").astype(int)
    d = d.sort_values("rank_at_secondary_T")
    out = REPO / "thesis_paper" / "results" / "na11_shortlist_limits" / rr.utc_stamp()
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "shortlist_limits.csv", index=False, lineterminator="\n")
    summary = {"secondary_T_K": SECONDARY_T, "order_primary": d.sort_values("rank_primary")["formula"].tolist(), "order_within_limit": d.sort_values("rank_within_limit")["formula"].tolist(),
               "order_at_secondary_T": d["formula"].tolist(), "limited_compounds": d[d["unconditional_limit_K"].notna() & (d["unconditional_limit_K"] <= 800)]["formula"].tolist()}
    rr.write_json(out / "summary.json", summary)
    rr.write_json(out / "run_config.json", prov)
    print("wrote", out)
    print(d.to_string())
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
