"""
NA11 step 2: filter the stored Materials Project query, test perovskite connectivity, featurise. No predictions.

Filter chain (the count after every filter is recorded, in this order)
  f0  energy above hull <= 0.05 eV/atom (query A, all materials)
  f1  band gap 0.1 to 3.0 eV
  f2  lead-free and free of radioactive/unstable elements (Tc, Pm, Po, At, Rn, Fr, Ra, Ac, Th, Pa, U, Np, Pu and the heavier actinides)
  f3  ABX3 stoichiometry (three elements, 1:1:3 after reduction; query B structures)
        reported alongside: how many f3 compounds the old space-group test (Pm-3m, Pnma, R-3c, I4/mcm, Imma, P4/mbm, Cmcm) would have
        kept, and how many are anti-perovskites (the element present three times is the most electropositive)
  f4  perovskite-type by the structural connectivity test (perovskite_test.py), anti-perovskites excluded
  f5  band gap <= 0.6 eV
  f6  free of Tl, Hg, Cd, As and Be (the ranked list)
The f4 compounds are featurised (MAGPIE and CBFV with the Oliynyk set, exactly as the training data and ESTM), so that the later filters
can be changed without refeaturising; temperature is added at prediction time. Nothing is predicted.

Usage (from the repository root, project environment):
    python thesis_paper/scripts/na11_mp_filter_featurize.py --query data/external/mp/mp_query_<stamp>.json
    python thesis_paper/scripts/na11_mp_filter_featurize.py --standin-jarvis      # code-path test only, on JARVIS ABX3 entries; NOT Materials Project
"""

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import perovskite_test as pt  # noqa: E402
import run_record as rr  # noqa: E402
from pymatgen.core import Composition, Structure  # noqa: E402
from src import external_validation as ev  # noqa: E402
from src.nested_cv import get_feature_columns  # noqa: E402

TRAIN = "data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv"
RADIOACTIVE = {"Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"}
LEAD = {"Pb"}
TOXIC = {"Tl", "Hg", "Cd", "As", "Be"}
PEROVSKITE_SPACE_GROUPS = {221: "Pm-3m", 62: "Pnma", 167: "R-3c", 140: "I4/mcm", 74: "Imma", 127: "P4/mbm", 63: "Cmcm"}
GAP_WINDOW = (0.1, 3.0)
GAP_MAX_FINAL = 0.6
E_HULL_MAX = 0.05


def standin_from_jarvis(path):
    """A query file with the same schema built from JARVIS ABX3 entries, to exercise the code only (the result is not Materials Project data)."""
    from jarvis.core.atoms import Atoms

    with zipfile.ZipFile(REPO / "data/external/jarvis/jdft_3d-9-24-2025.json.zip") as z:
        data = json.loads(z.read("jdft_3d-9-24-2025.json"))
    a, b = [], []
    for d in data:
        try:
            ehull, gap = float(d["ehull"]), float(d["optb88vdw_bandgap"])
        except (TypeError, ValueError):
            continue
        if ehull > E_HULL_MAX:
            continue
        s = Atoms.from_dict(d["atoms"]).pymatgen_converter()
        els = sorted(e.symbol for e in s.composition.elements)
        a.append({"material_id": d["jid"], "formula": d["formula"], "elements": els, "ehull": ehull, "gap": gap})
        if len(els) == 3 and GAP_WINDOW[0] <= gap <= GAP_WINDOW[1] and pt.abx3_elements(s) is not None:
            b.append({"material_id": d["jid"], "formula": d["formula"], "ehull": ehull, "gap": gap, "is_stable": ehull == 0.0,
                      "spg_number": int(d["spg_number"]), "spg_symbol": str(d["spg_symbol"]), "structure": s.as_dict()})
    Path(path).write_text(json.dumps({"meta": {"STANDIN_NOT_MP": True, "n_A": len(a), "n_B": len(b)}, "A": a, "B": b}), encoding="utf-8")


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="data/external/mp/mp_query_<stamp>.json written by na11_mp_query.py")
    ap.add_argument("--standin-jarvis", action="store_true")
    args = ap.parse_args()
    if bool(args.query) == bool(args.standin_jarvis):
        ap.error("give exactly one of --query and --standin-jarvis")
    standin = args.standin_jarvis
    tmp = Path(tempfile.mkdtemp())
    qpath = tmp / "standin_query.json" if standin else REPO / args.query
    if standin:
        import os

        os.environ["PYTHONUTF8"] = "1"
        standin_from_jarvis(qpath)
    q = json.loads(qpath.read_text(encoding="utf-8"))
    A, B = q["A"], q["B"]

    counts = {"f0_ehull_le_0.05": len(A)}
    f1 = [m for m in A if GAP_WINDOW[0] <= m["gap"] <= GAP_WINDOW[1]]
    counts["f1_gap_0.1_to_3.0"] = len(f1)
    f2 = [m for m in f1 if not (set(m["elements"]) & (RADIOACTIVE | LEAD))]
    counts["f2_lead_and_radioactive_free"] = len(f2)
    counts["f2_removed_lead"] = sum(1 for m in f1 if set(m["elements"]) & LEAD)
    counts["f2_removed_radioactive"] = sum(1 for m in f1 if set(m["elements"]) & RADIOACTIVE)
    ids_f1_3 = {m["material_id"] for m in f1 if len(m["elements"]) == 3}
    ids_b = {m["material_id"] for m in B}
    counts["B_matches_A_three_element_set"] = ids_b == ids_f1_3
    ok_ids = {m["material_id"] for m in f2}
    rows = []
    for m in B:
        if m["material_id"] not in ok_ids:
            continue
        s = Structure.from_dict(m["structure"])
        el = pt.abx3_elements(s)
        if el is None:
            continue
        v = pt.classify(s)
        els = {e.symbol for e in s.composition.elements}
        rows.append({**{k: m[k] for k in ("material_id", "formula", "ehull", "gap", "is_stable", "spg_number", "spg_symbol")},
                     "x_site": v["x"], "anti_perovskite": v["anti_perovskite"], "perovskite_type": v["perovskite"], "b_site": v["b"], "dimensionality": v["dimensionality"],
                     "reason": v["reason"], "space_group_rule": m["spg_number"] in PEROVSKITE_SPACE_GROUPS,
                     "toxic_element": bool(els & TOXIC), "elements": " ".join(sorted(els))})
    df = pd.DataFrame(rows)
    counts["f3_abx3"] = len(df)
    counts["f3_anti_perovskite"] = int(df["anti_perovskite"].sum())
    counts["f3_space_group_rule"] = int(df["space_group_rule"].sum())
    f4 = df[df["perovskite_type"]]
    counts["f4_connectivity_perovskite_type"] = len(f4)
    counts["f4_and_space_group_rule"] = int(f4["space_group_rule"].sum())
    counts["f4_not_in_space_group_rule"] = int((~f4["space_group_rule"]).sum())
    counts["space_group_rule_but_not_connectivity"] = int((df["space_group_rule"] & ~df["perovskite_type"] & ~df["anti_perovskite"]).sum())
    counts["f4_by_space_group"] = {str(k): int(v) for k, v in f4["spg_number"].value_counts().items()}
    f5 = f4[f4["gap"] <= GAP_MAX_FINAL]
    counts["f5_gap_le_0.6"] = len(f5)
    f6 = f5[~f5["toxic_element"]]
    counts["f6_toxic_free_ranked_list"] = len(f6)

    # featurise the f4 compounds exactly like ESTM / JARVIS
    feat_in = f4.assign(Formula=f4["formula"], temperature_bin=600).reset_index(drop=True)
    canonical, n_parse = ev.canonicalize_estm(feat_in)
    feat, n_fail = ev.featurize_estm(canonical)
    train_head = pd.read_csv(REPO / TRAIN, nrows=5)
    assert get_feature_columns(feat) == get_feature_columns(train_head), "feature columns differ from the training columns"
    counts.update({"featurised": len(feat), "parse_failures": n_parse, "featurisation_failed_unique_formulas": n_fail, "n_feature_columns": len(get_feature_columns(feat))})

    out_base = tmp / "out" if standin else REPO / "thesis_paper" / "results" / "na11"
    out_dir = out_base / rr.utc_stamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    store = (tmp / "store") if standin else (REPO / "data" / "external" / "mp")
    store.mkdir(parents=True, exist_ok=True)
    feat_path = store / ("standin_featurized.csv" if standin else f"mp_perovskite_candidates_featurized_{out_dir.name}.csv")
    feat.to_csv(feat_path, index=False)
    df.to_csv(out_dir / "abx3_candidates.csv", index=False, lineterminator="\n")
    inputs = {"training_csv": TRAIN, "perovskite_test": "thesis_paper/scripts/perovskite_test.py"}
    prov = rr.provenance(inputs, __file__)
    prov["query"] = {"path": str(qpath) if standin else args.query, "sha256": rr.sha256_file(qpath), "meta": q["meta"]}
    prov["featurised_matrix"] = {"path": str(feat_path), "sha256": rr.sha256_file(feat_path), "bytes": feat_path.stat().st_size}
    prov["standin_not_mp"] = standin
    prov["perovskite_test_parameters"] = {"BOND_FACTOR": pt.BOND_FACTOR, "MIN_OPPOSITE_ANGLE": pt.MIN_OPPOSITE_ANGLE}
    rr.write_json(out_dir / "run_config.json", prov)
    rr.write_json(out_dir / "counts.json", counts)
    print(("STAND-IN (JARVIS, not Materials Project) -> " if standin else "wrote ") + str(out_dir))
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
