"""
Literature search for the MP screening candidates: has the compound been studied as a thermoelectric (measurement or calculation)? Literature only: it reads the committed candidate lists
(results/na11_candidate_novelty/<stamp>/candidate_lists.csv, which holds E_hull, gap and list membership but no prediction) and queries Crossref; it opens no prediction file.

For every distinct formula of the compounds in the largest reported list (E_hull <= 0.10 and gap cap 0.9), two Crossref bibliographic queries are made,
    "<formula> thermoelectric"   and   "<formula> Seebeck coefficient electrical conductivity",
and the top RESULTS_PER_QUERY hits of each are stored with DOI, title (markup stripped), year, journal and the abstract when Crossref has one. A hit is flagged for review when the normalised
title contains the normalised formula (case-insensitive, subscript digits and markup removed) together with a thermoelectric term, or when the abstract does. The flags only direct the manual
review (docs/candidate_literature_labels.csv records the basis of every label, which is a title or abstract statement); nothing is labelled automatically.

Output: thesis_paper/reports/candidate_literature/<UTC>/{crossref_hits.json, flagged.csv, run_config.json}.
Usage (from the repository root):  python thesis_paper/scripts/candidate_literature_search.py
"""

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_record as rr  # noqa: E402

LISTS = "thesis_paper/results/na11_candidate_novelty/20261004T185101/candidate_lists.csv"
RESULTS_PER_QUERY = 10
QUERIES = ("{f} thermoelectric", "{f} Seebeck coefficient electrical conductivity")
TE_TERMS = ("thermoelectric", "seebeck", "power factor", "figure of merit", "thermopower", "zt")
SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def norm(text):
    """Lower-case text without markup, spaces and subscript characters."""
    t = re.sub(r"<[^>]+>", "", text or "").translate(SUB)
    return re.sub(r"\s+", "", t).lower()


def crossref(query):
    """Top hits of a Crossref bibliographic query."""
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode({"query.bibliographic": query, "rows": RESULTS_PER_QUERY,
                                                                      "select": "DOI,title,author,issued,container-title,abstract"})
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "te-ml-pipeline literature check (research use)"}), timeout=60) as r:
        items = json.load(r)["message"]["items"]
    out = []
    for i in items:
        out.append({"doi": i["DOI"], "title": re.sub(r"<[^>]+>", "", (i.get("title") or [""])[0]).strip(), "year": (i.get("issued", {}).get("date-parts") or [[None]])[0][0],
                    "journal": (i.get("container-title") or [""])[0], "first_author": ((i.get("author") or [{}])[0]).get("family"),
                    "abstract": re.sub(r"<[^>]+>", "", i.get("abstract") or "").strip()})
    return out


def main():
    """Entry point."""
    prov = rr.provenance({"candidate_lists": LISTS}, __file__)
    c = pd.read_csv(REPO / LISTS)
    formulas = sorted(c[c["in_E_hull_0.10_cap_0.9"]]["formula"].unique())
    hits, flagged = {}, []
    for f in formulas:
        nf = norm(f)
        hits[f] = {}
        for q in QUERIES:
            query = q.format(f=f)
            for attempt in range(3):
                try:
                    res = crossref(query)
                    break
                except Exception:  # network hiccup: retry
                    time.sleep(3)
            else:
                res = []
            hits[f][query] = res
            for h in res:
                t, a = norm(h["title"]), norm(h["abstract"])
                te_t, te_a = any(k in t for k in TE_TERMS), any(k in a for k in TE_TERMS)
                if (nf in t and te_t) or (nf in a and te_a) or (nf in t and te_a):
                    flagged.append({"formula": f, "query": query, "doi": h["doi"], "year": h["year"], "first_author": h["first_author"], "journal": h["journal"], "title": h["title"],
                                    "formula_in_title": nf in t, "te_in_title": te_t, "formula_in_abstract": nf in a, "te_in_abstract": te_a})
            time.sleep(0.3)
    d = REPO / "thesis_paper" / "reports" / "candidate_literature" / rr.utc_stamp()
    d.mkdir(parents=True, exist_ok=True)
    rr.write_json(d / "crossref_hits.json", hits)
    pd.DataFrame(flagged).drop_duplicates(["formula", "doi"]).to_csv(d / "flagged.csv", index=False, lineterminator="\n")
    rr.write_json(d / "run_config.json", {**prov, "queries": list(QUERIES), "results_per_query": RESULTS_PER_QUERY, "n_formulas": len(formulas)})
    print("wrote", d, "formulas", len(formulas), "flagged rows", len(flagged))


if __name__ == "__main__":
    main()
