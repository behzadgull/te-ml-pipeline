"""
Builds docs/candidate_literature_labels.csv: for every candidate of the largest reported list (E_hull <= 0.10 and gap cap 0.9, 73 Materials Project entries, 60 distinct formulas; it contains the
main 30, the E_hull 0.10 list of 44 and the gap-cap 0.9 list of 49) the label "previously studied as a thermoelectric: yes / no evidence found", with DOI, kind, confidence and a one-line basis.

Literature only: it reads the committed candidate lists (E_hull, gap, list membership, the seen-in-training flag; no prediction) and the table LABELS below, which was compiled from
Crossref searches (scripts/candidate_literature_search.py, output in reports/candidate_literature/<stamp>) and web searches. Criterion for "yes": the title or abstract of the cited paper states a
thermoelectric (Seebeck coefficient, power factor, zT or thermoelectric performance) measurement or calculation of that compound, undoped or doped. A label is per formula: every Materials
Project entry (polymorph) of the formula gets it. "no evidence found" means that none of the searches listed in `search_used` returned such a paper; it is not a statement that none exists.
Confidence: high = the compound is named in the title or the abstract; medium = the abstract covers a series that contains the compound, or implies it (stated in the basis).

Usage (from the repository root):  python thesis_paper/scripts/make_candidate_labels.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_record as rr  # noqa: E402

LISTS = "thesis_paper/results/na11_candidate_novelty/20261004T185101/candidate_lists.csv"
SEARCH_RUN = "thesis_paper/reports/candidate_literature/20261005T045225"
OUT = REPO / "thesis_paper" / "docs" / "candidate_literature_labels.csv"
SUMMARY = REPO / "thesis_paper" / "docs" / "candidate_literature_labels_summary.json"

# formula: (kind, doi, confidence, basis)
YES = {
    "BaHfS3": ("calculation", "10.1016/j.cocom.2025.e01202", "high", "Title: 'Evaluation of chalcogenide perovskites BaHfS3 for charge transport in energy conversion applications: Solar cells, photocatalytic water splitting and thermoelectric' (El Hidaoui 2026)"),
    "BaSnO3": ("experiment", "10.1016/j.mseb.2009.10.002", "high", "Title: 'High-temperature thermoelectric properties of La-doped BaSnO3 ceramics' (Yasukawa 2010)"),
    "BaZrSe3": ("calculation", "10.1021/acs.jpclett.3c02940", "medium", "Abstract (arXiv:2310.13851, J. Phys. Chem. Lett. 14:11465): ZT of BaZrS3 raised from 0.58 to 0.91 at 500 K 'by replacing S atom with Se and Ti-alloying'; the Se compound is BaZrSe3, which the v1 abstract does not name (search summaries of the published abstract give 'n-type BaZrSe3 ... ZT 0.42 at 300 K', not re-read)"),
    "CaMnO3": ("experiment", "10.1109/ict.2006.331291", "high", "Title: 'Effect of the Yb substitutions on the thermoelectric properties of CaMnO3' (Flahaut 2006); many other titles, e.g. 10.1063/1.3505756 (Okuda 2010)"),
    "CsEuCl3": ("calculation", "10.1016/j.rinp.2024.107980", "high", "Title: 'An ab initio study to investigate the physical properties of CsEuX3 (X=Cl, Br, and I) ...' (Al-Reyahi 2024); the abstract (search summary, Crossref has none) reports a thermoelectric figure of merit for CsEuCl3"),
    "CsGeBr3": ("calculation", "10.1007/s42247-023-00553-5", "high", "Title: 'Comprehensive study of CsGeBr3 perovskite: optical, electronic, and thermoelectric properties' (Srinivas 2023)"),
    "CsSnBr3": ("experiment+calculation", "10.1016/j.cplett.2020.137637", "high", "Title: 'Thermoelectric properties of all-inorganic perovskite CsSnBr3: A combined experimental and theoretical study' (Zhang 2020)"),
    "CsSnI3": ("experiment", "10.1016/j.orgel.2019.105488", "high", "Title: 'Interface engineering using Y2O3 scaffold to enhance the thermoelectric performance of CsSnI3 thin film' (Baranwal 2020); also 10.1021/acsaem.2c01936 (Sebastia-Luna 2022, CsSnI3 films with tunable thermoelectric properties)"),
    "LaCoO3": ("experiment", "10.1080/14786435.2016.1263404", "high", "Title: 'Understanding the thermoelectric properties of LaCoO3 compound' (Singh 2016); many others, e.g. 10.1063/1.4980675"),
    "LaNiO3": ("experiment", "10.1007/s13391-013-0034-0", "high", "Title: 'Thermoelectric properties of a doped LaNiO3 perovskite system prepared using a spark-plasma sintering process' (Tak 2013); arXiv:2108.09969 'Understanding the Seebeck coefficient of LaNiO3 compound in the temperature range 300-620 K'"),
    "LaRhO3": ("experiment", "10.1007/s11664-009-0666-x", "high", "Title: 'Thermoelectric Properties of B-Site Substituted LaRhO3' (Shibasaki 2009)"),
    "NdCoO3": ("experiment", "10.1016/j.ceramint.2020.04.113", "high", "Title: 'Thermoelectric properties of the SmCoO3 and NdCoO3 cobalt oxides' (Dudnikov 2020)"),
    "RbGeI3": ("calculation", "10.1016/j.mtcomm.2021.102650", "high", "Title: 'Ab initio DFT determination of structural, mechanical, optoelectronic, thermoelectric and thermodynamic properties of RbGeI3 inorganic perovskite ...' (Deepthi Jayan 2021)"),
    "RbSnBr3": ("calculation", "10.1016/j.cocom.2022.e00761", "high", "Title: 'First principles calculations of the inorganic halide perovskite RbSnBr3: Optical and thermoelectric properties of its three phases' (Bouchikhi 2022)"),
    "YCoO3": ("experiment", "10.1088/1674-1056/24/4/047202", "medium", "Abstract of 'Electrical transport properties of YCo1-xMnxO3 (0 <= x <= 0.2) prepared by sol-gel process' (Liu 2015, Chin. Phys. B 24:047202): Seebeck coefficient and resistivity measured from 200 K to 780 K for the series, which includes x = 0 (YCoO3)"),
}
# web searches (standard mode, 2026-10-05) made in addition to the four Crossref queries per formula
WEB = {
    "PrRhO3 NdRhO3 SmRhO3 rare-earth rhodate perovskite RRhO3 thermoelectric Seebeck coefficient": ("DyRhO3", "NdRhO3", "PrRhO3", "SmRhO3", "TbRhO3"),
    "AlSiP3 OR AgAuO3 OR InRhO3 OR BiRhO3 OR CeLuO3 thermoelectric properties": ("AlSiP3", "AgAuO3", "InRhO3", "BiRhO3", "CeLuO3"),
    "ErNiO3 RNiO3 rare-earth nickelate thermopower Seebeck coefficient thermoelectric": ("ErNiO3",),
    "CaVO3 thermoelectric Seebeck coefficient power factor": ("CaVO3",),
    "SrBiO3 KBiO3 thermoelectric properties": ("SrBiO3", "KBiO3"),
    "NaCuF3 KCrF3 LiAgF3 NaVF3 fluoroperovskite thermoelectric properties DFT": ("NaCuF3", "KCrF3", "LiAgF3", "NaVF3"),
    "CsInI3 KInI3 RbInI3 KInBr3 halide perovskite thermoelectric properties first-principles": ("CsInI3", "KInI3", "RbInI3", "KInBr3"),
    "EuZrO3 EuHfO3 EuZrS3 EuHfS3 perovskite thermoelectric properties": ("EuZrO3", "EuHfO3", "EuZrS3", "EuHfS3"),
    "LaWN3 perovskite nitride thermoelectric": ("LaWN3",),
    "CsEuCl3 RbEuCl3 europium halide perovskite thermoelectric properties": ("RbEuCl3",),
    "ErCrO3 rare-earth chromite thermoelectric Seebeck coefficient": ("ErCrO3",),
    "MnSnO3 OR MnVO3 OR NiBiO3 OR ZnBiO3 OR CaBiO3 perovskite thermoelectric Seebeck": ("MnSnO3", "MnVO3", "NiBiO3", "ZnBiO3", "CaBiO3"),
    "LuMnO3 OR ScCoO3 OR MgMnO3 OR LuVO3 OR CaCrO3 thermoelectric properties": ("LuMnO3", "ScCoO3", "MgMnO3", "LuVO3", "CaCrO3"),
}
NOTES = {
    "SrBiO3": "Sr1-xKxBiO3 (x = 0.45 to 0.6) thermopower exists (K-doped superconductor), not SrBiO3 itself; not counted",
    "KBiO3": "no paper found for the compound",
    "LaWN3": "lattice thermal conductivity calculations exist (phonon transport, not thermoelectric by title or abstract); not counted",
    "ErCrO3": "RCo0.5Cr0.5O3 (R = Dy, Ho, Er) and doped chromites are reported as thermoelectrics, not ErCrO3 itself; not counted",
    "NdRhO3": "NdRhO3 thermodynamic and structural papers only",
}


def main():
    """Entry point."""
    prov = rr.provenance({"candidate_lists": LISTS, "search_flagged": f"{SEARCH_RUN}/flagged.csv"}, __file__)
    c = pd.read_csv(REPO / LISTS)
    u = c[c["in_E_hull_0.10_cap_0.9"]].copy().sort_values(["formula", "material_id"]).reset_index(drop=True)
    assert len(u) == 73 and u["formula"].nunique() == 60
    flagged = set(pd.read_csv(REPO / SEARCH_RUN / "flagged.csv")["formula"])
    assert flagged <= set(YES), f"the Crossref search flagged formulas that are not labelled: {sorted(flagged - set(YES))}"
    assert set(YES) <= set(u["formula"])
    web_by_formula = {}
    for q, fs in WEB.items():
        for f in fs:
            assert f in set(u["formula"]), f
            web_by_formula.setdefault(f, []).append(q)
    rows = []
    for _, r in u.iterrows():
        f = r["formula"]
        base = {"material_id": r["material_id"], "formula": f, "in_main_30": bool(r["in_main"]), "in_E_hull_0.10_44": bool(r["in_E_hull_0.10"]), "in_gap_cap_0.9_49": bool(r["in_gap_cap_0.9"]),
                "in_both_relaxed_73": True, "ehull_eV_per_atom": round(float(r["ehull"]), 4), "cluster_seen_in_training": bool(r["cluster_seen_any"])}
        if f in YES:
            kind, doi, conf, basis = YES[f]
            base.update({"previously_studied_as_thermoelectric": "yes", "kind": kind, "doi": doi, "confidence": conf, "basis": basis, "search_used": "Crossref (4 queries) " + SEARCH_RUN})
        else:
            q = "; ".join(f"web search: {x}" for x in web_by_formula.get(f, []))
            base.update({"previously_studied_as_thermoelectric": "no evidence found", "kind": "", "doi": "", "confidence": "",
                         "basis": NOTES.get(f, "no paper whose title or abstract reports a thermoelectric measurement or calculation of the compound was returned"),
                         "search_used": f"Crossref (4 queries: '<formula> thermoelectric', '<formula> Seebeck coefficient electrical conductivity', '<formula>', '<formula> thermopower power factor figure of merit first-principles') {SEARCH_RUN}" + (("; " + q) if q else "")})
        rows.append(base)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, lineterminator="\n")
    summary = {"search_run": SEARCH_RUN, "n_entries": int(len(out)), "n_formulas": int(out["formula"].nunique()), "lists": {}}
    for name, col in (("main_30", "in_main_30"), ("E_hull_0.10_44", "in_E_hull_0.10_44"), ("gap_cap_0.9_49", "in_gap_cap_0.9_49"), ("both_relaxed_73", "in_both_relaxed_73")):
        s = out[out[col]]
        yes = s[s["previously_studied_as_thermoelectric"] == "yes"]
        summary["lists"][name] = {"entries": int(len(s)), "yes": int(len(yes)), "no_evidence": int(len(s) - len(yes)), "formulas": int(s["formula"].nunique()),
                                  "formulas_yes": int(yes["formula"].nunique()), "yes_seen_cluster": int(yes["cluster_seen_in_training"].sum()),
                                  "no_evidence_seen_cluster": int(s[s["previously_studied_as_thermoelectric"] != "yes"]["cluster_seen_in_training"].sum())}
    sl = out[out["formula"].isin(["BaSnO3", "CsSnI3", "LaCoO3", "LaRhO3"])]
    summary["shortlist_rule_A_labels"] = {f: g["previously_studied_as_thermoelectric"].iloc[0] for f, g in sl.groupby("formula")}
    rr.write_json(SUMMARY, {**summary, "provenance": prov})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
