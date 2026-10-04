"""
NA11: how many of the Materials Project candidates fall in chemistry clusters (and compositions) that the training data already contain? The same definition of "seen" as the NA10
JARVIS analysis (na10_jarvis_prep.py): the entry's chemistry_cluster_id / composition_id occurs among the rows of the snapfix training CSV; here also per target, i.e. among the rows
that have a measured value of that property (S, sigma, kappa, zT), because each final model is trained on its own target's rows.

The candidate lists are those of the committed sensitivity run (results/na11_sensitivity/<stamp>/membership.csv), all with the main toxic-element list and the gap window 0.1 to 3.0 eV:
  main                  E_hull <= 0.05 eV/atom, gap <= 0.6 eV    (the 30 compounds of the ranked list)
  E_hull_0.10           E_hull <= 0.10, gap <= 0.6
  gap_cap_0.9           E_hull <= 0.05, gap <= 0.9
  E_hull_0.10_cap_0.9   both relaxed
  all_perovskite_type   the 409 featurised perovskite-type compounds at E_hull <= 0.10 within the gap window (before the 0.6 eV cap)
Each list is a superset of the main list; the compounds ADDED by a relaxation (list minus main) are counted separately, because they are the ones that are not in the main list.
Also written: the membership of single compounds (candidate_lists.csv: E_hull, gap, which lists, seen flags), including BaZrSe3, which is in no list at E_hull <= 0.05.

Usage (from the repository root):  python thesis_paper/scripts/na11_candidate_novelty.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_record as rr  # noqa: E402

MP_CSV = "data/external/mp/mp_perovskite_candidates_featurized_20261004T135055.csv"
MP_SHA256 = "6386c09781667cc86a52d23dee1ee6f86ea899c39cea4d22f4ca10e143aec95c"
TRAIN = "data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv"
TRAIN_SHA256 = "d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489"
SENS_RUN = "thesis_paper/results/na11_sensitivity/20261004T135205"
TARGETS = ("S", "sigma", "kappa", "zT")
MAIN_TOXIC = "Tl,Hg,Cd,As,Be excluded (main)"
LISTS = {"main": ("0.05 (main)", "0.6 (main)"), "E_hull_0.10": ("0.10", "0.6 (main)"), "gap_cap_0.9": ("0.05 (main)", "0.9"), "E_hull_0.10_cap_0.9": ("0.10", "0.9")}


def main():
    """Entry point."""
    prov = rr.provenance({"mp_candidates": MP_CSV, "training_csv": TRAIN, "membership": f"{SENS_RUN}/membership.csv"}, __file__)
    assert prov["inputs"]["mp_candidates"]["sha256"] == MP_SHA256, "not the 409-row MP candidate file"
    assert prov["inputs"]["training_csv"]["sha256"] == TRAIN_SHA256, "not the snapfix CSV"
    mp = pd.read_csv(REPO / MP_CSV, usecols=["material_id", "formula", "ehull", "gap", "is_stable", "spg_symbol", "composition_id", "chemistry_cluster_id"])
    assert len(mp) == 409 and mp["material_id"].is_unique
    tr = pd.read_csv(REPO / TRAIN, usecols=["composition_id", "chemistry_cluster_id", *TARGETS])
    seen_cluster = {"any": set(tr["chemistry_cluster_id"].dropna())}
    seen_comp = {"any": set(tr["composition_id"].dropna())}
    for t in TARGETS:
        sub = tr[tr[t].notna()]
        seen_cluster[t], seen_comp[t] = set(sub["chemistry_cluster_id"].dropna()), set(sub["composition_id"].dropna())
    for k in ("any", *TARGETS):
        mp[f"cluster_seen_{k}"] = mp["chemistry_cluster_id"].isin(seen_cluster[k])
        mp[f"composition_seen_{k}"] = mp["composition_id"].isin(seen_comp[k])
    mem = pd.read_csv(REPO / SENS_RUN / "membership.csv")
    ids = {name: set(mem[(mem["toxic"] == MAIN_TOXIC) & (mem["e_hull_max"] == e) & (mem["gap_cap"] == g)]["material_id"]) for name, (e, g) in LISTS.items()}
    assert len(ids["main"]) == 30 and ids["main"] <= ids["E_hull_0.10"] and ids["main"] <= ids["gap_cap_0.9"] and ids["E_hull_0.10"] | ids["gap_cap_0.9"] <= ids["E_hull_0.10_cap_0.9"]
    ids["all_perovskite_type"] = set(mp["material_id"])
    for name, s in ids.items():
        assert s <= set(mp["material_id"]), f"{name}: compounds missing from the featurised candidate file"
        mp[f"in_{name}"] = mp["material_id"].isin(s)

    def summarise(sel):
        out = {"n": int(len(sel)), "unique_clusters": int(sel["chemistry_cluster_id"].nunique()), "unique_compositions": int(sel["composition_id"].nunique())}
        for k in ("any", *TARGETS):
            out[f"cluster_seen_{k}"] = int(sel[f"cluster_seen_{k}"].sum())
            out[f"cluster_unseen_{k}"] = int((~sel[f"cluster_seen_{k}"]).sum())
        out["composition_seen_any"] = int(sel["composition_seen_any"].sum())
        return out

    counts = {"definition_of_seen": "the entry's chemistry_cluster_id (composition_id) occurs among the training rows (any target) or among the rows that have a measured value of that target",
              "lists": {}, "added_by_relaxation_relative_to_main": {}}
    for name, s in ids.items():
        counts["lists"][name] = summarise(mp[mp["material_id"].isin(s)])
        if name != "main":
            counts["added_by_relaxation_relative_to_main"][name] = summarise(mp[mp["material_id"].isin(s - ids["main"])])
    counts["seen_compounds_in_main"] = sorted(mp[mp["in_main"] & mp["cluster_seen_any"]]["formula"])
    counts["seen_compounds_in_E_hull_0.10_list"] = sorted(mp[mp["in_E_hull_0.10"] & mp["cluster_seen_any"]]["formula"])
    bz = mp[mp["formula"] == "BaZrSe3"]
    counts["BaZrSe3"] = [{"material_id": r["material_id"], "ehull_eV_per_atom": float(r["ehull"]), "gap_eV_PBE": float(r["gap"]), "is_stable": bool(r["is_stable"]), "spg_symbol": r["spg_symbol"],
                          "lists": [n for n in ids if r[f"in_{n}"]], "cluster_seen_any": bool(r["cluster_seen_any"])} for _, r in bz.iterrows()]
    out = REPO / "thesis_paper" / "results" / "na11_candidate_novelty" / rr.utc_stamp()
    out.mkdir(parents=True, exist_ok=True)
    cols = ["material_id", "formula", "ehull", "gap", "is_stable", "spg_symbol", *[f"in_{n}" for n in ids], *[f"cluster_seen_{k}" for k in ("any", *TARGETS)], "composition_seen_any"]
    mp[cols].sort_values(["ehull", "formula"]).to_csv(out / "candidate_lists.csv", index=False, lineterminator="\n")
    rr.write_json(out / "counts.json", counts)
    rr.write_json(out / "run_config.json", prov)
    print("wrote", out)
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
