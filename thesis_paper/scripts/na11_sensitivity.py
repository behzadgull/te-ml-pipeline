"""
NA11 screening sensitivity (no new Materials Project query): how the ranked candidate list changes when the screening criteria are varied one at a time
and together. Reads the committed table of every lead- and radioactive-free ABX3 compound of the query (abx3_candidates.csv of the NA11 filter run), which
holds E_hull (<= 0.05 eV/atom, the query limit), the PBE gap (up to 4.5 eV), the structural verdict and the element list.

Axes
  toxic list   main: Tl, Hg, Cd, As, Be excluded;  relaxed: only Hg and Cd excluded (the RoHS-restricted elements), so Tl, As and Be are allowed;
  E_hull       main: <= 0.05 eV/atom;  tighter: <= 0.025 and <= 0 (stable only).  A looser limit (0.10) cannot be evaluated here: the stored query stops at
               0.05, so it would need a new query (not run, by instruction);
  gap cap      main: <= 0.6 eV;  relaxed: <= 0.9 eV.
Everything else is as in the main chain: perovskite-type by the connectivity test, anti-perovskites excluded, gap window 0.1 to 3.0 eV.

The ranking by predicted zT is not part of this step (no model has predicted these compounds yet); what is reported is which compounds are in the candidate set
under which variant, and which compounds of the main list survive every tightening. Re-rank the variants with the predicted zT once the final-model predictions
exist (thesis_paper/scripts/kaggle/na_final_models.py, --mp-csv).

Output: thesis_paper/results/na11_sensitivity/<UTC>/{counts.json, membership.csv, run_config.json}.
Usage (from the repository root):  python thesis_paper/scripts/na11_sensitivity.py
"""

import itertools
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_record as rr  # noqa: E402

REPO = HERE.parents[1]
FILTER_RUN = "thesis_paper/results/na11/20261004T100423"
TOXIC_MAIN = {"Tl", "Hg", "Cd", "As", "Be"}
TOXIC_RELAXED = {"Hg", "Cd"}
GAP_WINDOW = (0.1, 3.0)
E_HULL = {"0.05 (main)": 0.05, "0.025": 0.025, "0.0 (stable only)": 0.0}
GAP_CAP = {"0.6 (main)": 0.6, "0.9": 0.9}
TOXIC = {"Tl,Hg,Cd,As,Be excluded (main)": TOXIC_MAIN, "only Hg,Cd excluded": TOXIC_RELAXED}


def main():
    """Entry point."""
    prov = rr.provenance({"abx3_candidates": f"{FILTER_RUN}/abx3_candidates.csv", "filter_run_config": f"{FILTER_RUN}/run_config.json"}, __file__)
    df = pd.read_csv(REPO / FILTER_RUN / "abx3_candidates.csv")
    base = df[df["perovskite_type"] & ~df["anti_perovskite"] & df["gap"].between(*GAP_WINDOW)].copy()
    base["els"] = base["elements"].str.split()
    variants = {}
    for (tn, tset), (en, e), (gn, g) in itertools.product(TOXIC.items(), E_HULL.items(), GAP_CAP.items()):
        sel = base[(base["ehull"] <= e + 1e-9) & (base["gap"] <= g) & ~base["els"].apply(lambda xs: bool(set(xs) & tset))]
        variants[(tn, en, gn)] = set(sel["material_id"])
    main_key = ("Tl,Hg,Cd,As,Be excluded (main)", "0.05 (main)", "0.6 (main)")
    main = variants[main_key]
    assert len(main) == 30, len(main)  # the committed ranked list
    ranked = pd.read_csv(REPO / FILTER_RUN / "ranked_list_f6.csv")
    assert set(ranked["material_id"]) == main, "the main variant must reproduce ranked_list_f6.csv"
    info = base.set_index("material_id")
    rows = []
    for k, ids in variants.items():
        for mid in sorted(ids):
            rows.append({"toxic": k[0], "e_hull_max": k[1], "gap_cap": k[2], "material_id": mid, "formula": info.loc[mid, "formula"], "gap": info.loc[mid, "gap"],
                         "ehull": info.loc[mid, "ehull"], "in_main_list": mid in main})
    out = REPO / "thesis_paper" / "results" / "na11_sensitivity" / rr.utc_stamp()
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "membership.csv", index=False, lineterminator="\n")
    tight = [v for k, v in variants.items() if k[0] == main_key[0] and k[2] == main_key[2]]  # E_hull tightenings at the main toxic list and cap
    survive = set.intersection(*tight)
    counts = {"variants": {f"{k[0]} | E_hull <= {k[1]} | gap <= {k[2]}": len(v) for k, v in variants.items()},
              "main_list_size": len(main),
              "main_list_surviving_every_e_hull_tightening": sorted(info.loc[sorted(survive), "formula"]),
              "n_surviving_every_e_hull_tightening": len(survive),
              "main_list_stable_only_E_hull_0": len(variants[(main_key[0], "0.0 (stable only)", main_key[2])]),
              "added_by_allowing_Tl_As_Be": sorted(info.loc[sorted(variants[("only Hg,Cd excluded", "0.05 (main)", "0.6 (main)")] - main), "formula"]),
              "added_by_gap_cap_0.9": sorted(info.loc[sorted(variants[(main_key[0], "0.05 (main)", "0.9")] - main), "formula"]),
              "added_by_both": sorted(info.loc[sorted(variants[("only Hg,Cd excluded", "0.05 (main)", "0.9")] - main), "formula"]),
              "lost_at_E_hull_0.025": sorted(info.loc[sorted(main - variants[(main_key[0], "0.025", "0.6 (main)")]), "formula"]),
              "lost_at_E_hull_0": sorted(info.loc[sorted(main - variants[(main_key[0], "0.0 (stable only)", "0.6 (main)")]), "formula"]),
              "e_hull_0.10_not_evaluated": "the stored query stops at 0.05 eV/atom"}
    rr.write_json(out / "counts.json", counts)
    rr.write_json(out / "run_config.json", prov)
    print("wrote", out)
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
