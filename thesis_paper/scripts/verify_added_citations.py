"""
Verify the references that were added to paper/refs.bib after the citation report (the 12 keys in ADDED_KEYS, numbered 63 to 74 in the report), against the registration agency's record of the DOI: Crossref, or DataCite for the arXiv (10.48550) and Zenodo
(10.5281) DOIs. For each entry the title, the first author, the year and, for articles, the journal, volume and pages of the bib entry are compared with
the record. One row per entry is appended to reports/citation_verification.csv (same columns; "ref" is the position in refs.bib; `doi_source` is
"refs.bib, added after the report"); the comparison details (including the journal) are written to reports/citation_verification_added.json.

Usage (repository root):  python thesis_paper/scripts/verify_added_citations.py
"""

import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import verify_citations as vc  # noqa: E402

BIB = REPO / "thesis_paper" / "paper" / "refs.bib"
CSV = REPO / "thesis_paper" / "reports" / "citation_verification.csv"
DETAIL = REPO / "thesis_paper" / "reports" / "citation_verification_added.json"
# The 12 entries added to refs.bib after the citation report, in the order they had in refs.bib when they were verified (positions 63 to 74; `ref` in the
# report). Two uncited entries (hira2018temperature, kanas2022tuning, positions 59 and 60) were removed from refs.bib on 2026-10-10, so the positions in the
# file are now two lower; the keys are what identify an entry.
ADDED_KEYS = ["katsura2025starrydata", "ho2026physicsinspired", "probst2019hyperparameters", "gao2022defect", "lin2021inorganic", "monacelli2023first",
              "lee2017transparent", "jacob1995phase", "radovic2008thermal", "alleno2015invited", "ryu2025tematdb", "ryu2025tematdbv116"]
FIRST_ADDED = 63
COLUMNS = ["ref", "reference_text", "doi", "doi_source", "url", "metadata_verdict", "title_similarity", "first_author_in_ref", "year_in_ref", "volume_in_ref",
           "page_in_ref", "crossref_title", "crossref_year", "crossref_volume", "crossref_page", "abstract_available"]


def strip_latex(s):
    s = re.sub(r"\\textit\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
    s = s.replace("{\\'e}", "e").replace("{\\'o}", "o").replace("{\\\"o}", "o").replace("{\\\"u}", "u")
    return re.sub(r"[{}]", "", s).strip()


def field(body, name):
    m = re.search(rf"\b{name}\s*=\s*\{{(.*?)\}},?\s*\n", body + "\n", re.S)  # the last field of an entry has no newline after it
    return strip_latex(m.group(1)) if m else ""


def parse_bib():
    out = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}\n", BIB.read_text(encoding="utf-8"), re.S):
        out.append({"type": m.group(1), "key": m.group(2), "title": field(m.group(3), "title"), "author": field(m.group(3), "author"), "year": field(m.group(3), "year"),
                    "journal": field(m.group(3), "journal"), "volume": field(m.group(3), "volume"), "pages": field(m.group(3), "pages"), "doi": field(m.group(3), "doi").lower(),
                    "publisher": field(m.group(3), "publisher"), "number": field(m.group(3), "number")})
    return out


def datacite_work(doi):
    d = vc.get(f"https://api.datacite.org/dois/{doi}")
    if not d:
        return None
    a = d["data"]["attributes"]
    year = a.get("publicationYear")
    return {"title": [a["titles"][0]["title"]], "author": [{"family": c.get("familyName") or c.get("name", "").split(",")[0]} for c in a.get("creators", [])],
            "issued": {"date-parts": [[year]]}, "container-title": [a.get("publisher") or ""], "volume": "", "page": "", "_agency": "DataCite",
            "_type": a.get("types", {}).get("resourceTypeGeneral", "")}


def main():
    entries = parse_bib()
    report = list(csv.DictReader(open(CSV, encoding="utf-8-sig", newline="")))
    already = {r["reference_text"] for r in report}
    # every entry from FIRST_ADDED on is checked against the record of its own DOI, also where the same DOI is in the earlier report
    by_key = {e["key"]: e for e in entries}
    missing = [k for k in ADDED_KEYS if k not in by_key or not by_key[k]["doi"]]
    assert not missing, f"entries without a DOI or not in refs.bib: {missing}"
    todo = [(FIRST_ADDED + n, by_key[k]) for n, k in enumerate(ADDED_KEYS)]
    print(f"{len(entries)} entries, {len(report)} rows in the report, {len(todo)} to verify")
    rows, details, mismatches = [], [], []
    for pos, e in todo:
        first_family = e["author"].split(" and ")[0].split(",")[0].strip()
        ref_text = f"{e['author'].replace(' and ', ', ')}, {e['title']}, " + (", ".join(x for x in (e["journal"] or e["publisher"], (e["volume"] and f"vol. {e['volume']}"), e["pages"]) if x)) + f" ({e['year']}). https://doi.org/{e['doi']}"
        agency = "DataCite" if e["doi"].startswith(("10.48550/", "10.5281/")) else "Crossref"
        work = datacite_work(e["doi"]) if agency == "DataCite" else vc.crossref_work(e["doi"])
        if not work:
            rows.append({"ref": pos, "reference_text": ref_text, "doi": e["doi"], "doi_source": "refs.bib, added after the report", "url": "", "metadata_verdict": "not found"})
            mismatches.append((pos, e["key"], "the DOI is not registered at " + agency))
            continue
        # Crossref titles carry markup ("CsSnI<sub>3</sub>") and line breaks; the comparison is on the text
        work["title"] = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", t)).strip() for t in work.get("title") or [""]]
        cmp_ = vc.compare(ref_text, work)
        bw = set(vc.words(e["title"]))
        omitted = [w for w in vc.words(work["title"][0]) if w not in bw]
        notes = [f"the record's title has words the bib title lacks: {' '.join(omitted)}"] if len(omitted) / max(1, len(vc.words(work["title"][0]))) > 0.15 else []
        rec_pages = work.get("page", "") or work.get("article-number", "")
        if e["type"] == "article" and rec_pages and not e["pages"]:
            notes.append(f"the bib entry has no pages; the record gives {rec_pages}")
        container = " ".join(work.get("container-title") or [""])
        jw = vc.words(container)
        jb = vc.words(e["journal"] or e["publisher"])
        journal_sim = round(sum(1 for w in jw if w in set(jb)) / max(1, len(jw)), 2) if (e["journal"] or e["publisher"]) else None
        year_ok = str(cmp_["crossref_year"]) == e["year"]
        rec_family = (work.get("author") or [{}])[0].get("family", "")
        # DataCite creators may be unstructured ("Byungki Ryu"); the surname is then one of the name's words
        author_ok = first_family.lower() in {w.lower() for w in re.split(r"[\s,]+", rec_family)}
        ok = cmp_["title_similarity"] >= 0.6 and author_ok and year_ok and (journal_sim is None or journal_sim >= 0.6 or agency == "DataCite")
        if e["type"] == "article" and e["volume"]:
            ok = ok and str(work.get("volume", "")) == e["volume"]
        problems = []
        if cmp_["title_similarity"] < 0.6:
            problems.append(f"title similarity {cmp_['title_similarity']}")
        if not author_ok:
            problems.append(f"first author {first_family!r} against {(work.get('author') or [{}])[0].get('family', '')!r}")
        if not year_ok:
            problems.append(f"year {e['year']} against {cmp_['crossref_year']}")
        if journal_sim is not None and journal_sim < 0.6 and agency != "DataCite":
            problems.append(f"journal {e['journal']!r} against {container!r}")
        if e["type"] == "article" and e["volume"] and str(work.get("volume", "")) != e["volume"]:
            problems.append(f"volume {e['volume']} against {work.get('volume', '')}")
        if problems:
            mismatches.append((pos, e["key"], "; ".join(problems)))
        rows.append({"ref": pos, "reference_text": ref_text, "doi": e["doi"], "doi_source": "refs.bib, added after the report", "url": f"https://doi.org/{e['doi']}",
                     "metadata_verdict": "match" if ok else "partial", "title_similarity": cmp_["title_similarity"], "first_author_in_ref": author_ok, "year_in_ref": year_ok,
                     "volume_in_ref": cmp_["volume_in_ref"], "page_in_ref": cmp_["page_in_ref"], "crossref_title": cmp_["crossref_title"], "crossref_year": cmp_["crossref_year"],
                     "crossref_volume": cmp_["crossref_volume"], "crossref_page": cmp_["crossref_page"], "abstract_available": ""})
        details.append({"position": pos, "key": e["key"], "doi": e["doi"], "agency": agency, "bib": {k: e[k] for k in ("title", "author", "year", "journal", "volume", "pages")},
                        "record": {"title": cmp_["crossref_title"], "first_author": (work.get("author") or [{}])[0].get("family", ""), "year": cmp_["crossref_year"],
                                   "container": container, "volume": work.get("volume", ""), "page": work.get("page", "") or work.get("article-number", "")},
                        "journal_similarity": journal_sim, "problems": problems, "notes": notes})
    new = [r for r in rows if r["reference_text"] not in already]
    with open(CSV, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        for r in new:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
    DETAIL.write_text(json.dumps(details, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"appended {len(new)} rows to {CSV.name}; details in {DETAIL.name}")
    for d in details:
        print(f"{d['position']:3d} {d['key']:28s} {d['agency']:8s} {'OK ' if not d['problems'] else 'CHECK'} | {d['record']['first_author']} {d['record']['year']} | {d['record']['title'][:70]} | {d['record']['container'][:40]} | journal sim {d['journal_similarity']}")
    print("mismatches:", mismatches or "none")
    for d in details:
        for n in d["notes"]:
            print(f"note {d['position']} {d['key']}: {n}")


if __name__ == "__main__":
    main()
