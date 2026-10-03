"""
Verify the thesis paper's 62 references and its LITERATURE claims against external metadata.

For every numbered reference in paper.md:
  1. take the DOI printed in the reference, or find one by a Crossref bibliographic search of the reference text;
  2. fetch the Crossref record and compare it with the reference text: first-author surname, year, volume, first page, and the
     similarity of the titles (share of the reference-title words found in the Crossref title);
  3. fetch the abstract from OpenAlex (when it has one) for the claim check below.
For every LITERATURE row of reports/claim_inventory.csv that cites reference numbers, check which numbers of the claim occur in the cited
abstracts. A number missing from the abstract does not mean the claim is wrong: it means the full text has to be read (the verdict says so).

Raw API responses, including abstracts, are cached in data/external/citations/cache.json (gitignored; abstracts are publisher text and are
not committed). The committed outputs hold only identifiers, match verdicts and the numbers found.

Outputs: thesis_paper/reports/citation_verification.csv (per reference), claim_citation_check.csv (per LITERATURE claim).

Usage (from the repository root):  python thesis_paper/scripts/verify_citations.py [--refresh]
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_record as rr  # noqa: E402

PAPER = REPO / "thesis_paper" / "paper" / "paper.md"
INVENTORY = REPO / "thesis_paper" / "reports" / "claim_inventory.csv"
CACHE = REPO / "data" / "external" / "citations" / "cache.json"
OUT_REFS = REPO / "thesis_paper" / "reports" / "citation_verification.csv"
OUT_CLAIMS = REPO / "thesis_paper" / "reports" / "claim_citation_check.csv"
HEADERS = {"User-Agent": "thesis-paper-citation-check/1.0"}
# DOIs fixed by hand after the automatic search failed or matched the wrong record (each was checked against the Crossref/DataCite record)
OVERRIDE_DOI = {
    2: "10.1201/9781420049718.ch34", 8: "10.1002/adfm.202411152", 17: "10.1201/9781420038903", 23: "10.1109/ict.2006.331291",
    35: "10.48550/arXiv.2509.00299", 42: "10.1007/978-0-387-84858-7", 44: "10.1016/j.commatsci.2019.109203", 46: "10.48550/arXiv.1705.07874",
    59: "10.1021/acsaem.2c02012", 60: "10.1039/c8ra01691g",
}
# References without a DOI (NeurIPS and JMLR papers): checked on the publisher's page instead
URL_REFS = {
    53: "https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html",
    55: "https://www.jmlr.org/papers/v12/pedregosa11a.html",
    56: "https://www.jmlr.org/papers/v11/cawley10a.html",
    61: "https://www.jmlr.org/papers/v9/shafer08a.html",
}
STOP = set("the of and a an in on for to with from by at as is are its their via using based new high".split())


def get(url, params=None, retries=4):
    """GET JSON with a few retries; None on 404."""
    for k in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=40)
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 * (k + 1))
    return None


def parse_references():
    """[(number, reference text)] from the plain-text reference list in paper.md."""
    text = PAPER.read_text(encoding="utf-8").split("# References", 1)[1]
    refs = []
    for m in re.finditer(r"(?m)^\\\[(\d+)\\\] (.+)$", text):
        t = re.sub(r"<!--.*?-->", "", m.group(2)).strip()
        refs.append((int(m.group(1)), t))
    assert [n for n, _ in refs] == list(range(1, len(refs) + 1)), "reference numbers are not 1..N"
    return refs


def words(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2]


def split_ref(text):
    """(authors, title, rest) of 'A. Author, B. Author, Title, Journal vol (year) pages.' as best as a heuristic can."""
    t = re.sub(r"https?://\S+", "", text).strip()
    m = re.search(r"(?:[A-Z]\.(?:-?[A-Z]\.)*\s*[A-Z][^,]*,\s*)+(?:et al\.,\s*)?", t)
    authors = m.group(0) if m else ""
    return authors, t[len(authors):] if m and t.startswith(authors) else t


def doi_of(text):
    m = re.search(r"10\.\d{4,9}/[^\s<>]+", text)
    return m.group(0).rstrip(".,;)") if m else None


def crossref_search(text):
    """Best Crossref candidate for a free-text reference."""
    _, rest = split_ref(text)
    data = get("https://api.crossref.org/works", {"query.bibliographic": re.sub(r"https?://\S+", "", text)[:300], "rows": 3})
    return (data or {}).get("message", {}).get("items", [])


def crossref_work(doi):
    d = get(f"https://api.crossref.org/works/{doi}")
    return d["message"] if d else None


def openalex_abstract(doi):
    d = get(f"https://api.openalex.org/works/doi:{doi}")
    if not d:
        return None, None
    inv = d.get("abstract_inverted_index")
    if not inv:
        return None, d.get("id")
    pos = {}
    for w, idxs in inv.items():
        for i in idxs:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos)), d.get("id")


def compare(ref_text, work):
    """Match diagnostics between the reference text and a Crossref work."""
    title = " ".join(work.get("title", [""]))
    rt = set(words(ref_text))
    tw = words(title)
    sim = sum(1 for w in tw if w in rt) / max(1, len(tw))
    au = work.get("author", [])
    first = au[0].get("family", "") if au else ""
    year = None
    for k in ("issued", "published-print", "published-online"):
        try:
            year = work[k]["date-parts"][0][0]
            break
        except (KeyError, IndexError, TypeError):
            continue
    page = (work.get("page") or "").split("-")[0]
    art = str(work.get("article-number") or "")
    return {
        "crossref_title": title, "crossref_first_author": first, "crossref_year": year, "crossref_volume": work.get("volume", ""),
        "crossref_page": work.get("page", "") or art, "title_similarity": round(sim, 2),
        "first_author_in_ref": bool(first) and first.lower() in ref_text.lower(),
        "year_in_ref": year is not None and str(year) in ref_text,
        "volume_in_ref": bool(work.get("volume")) and re.search(rf"\b{re.escape(str(work['volume']))}\b", ref_text) is not None,
        "page_in_ref": bool(page or art) and (re.search(rf"\b{re.escape(page)}\b", ref_text) is not None if page else art in ref_text),
    }


NUM = re.compile(r"(?<![\w.])[-−+≈~∼<>]?\d[\d,]*(?:\.\d+)?%?")


def numbers(s):
    """Numeric tokens of a claim, normalised (thousands separators and signs dropped)."""
    out = []
    for m in NUM.findall(s):
        t = re.sub(r"[-−+≈~∼<>,]", "", m)
        if t and t not in ("1", "2", "3", "4", "5") or "." in t or "%" in t:
            out.append(t)
    return list(dict.fromkeys(out))


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() and not args.refresh else {}
    refs = parse_references()
    rows = []
    for n, text in refs:
        key = str(n)
        if n in URL_REFS and cache.get(key, {}).get("via") != "publisher page":
            page = requests.get(URL_REFS[n], headers=HEADERS, timeout=40)
            title_words = words(split_ref(text)[1])[:6]
            body = re.sub(r"<[^>]+>", " ", page.text).lower()
            hit = sum(1 for w in title_words if w in body) / max(1, len(title_words))
            cache[key] = {"doi": None, "via": "publisher page", "work": None, "abstract": None, "openalex": None, "url": URL_REFS[n], "page_status": page.status_code, "title_words_on_page": round(hit, 2)}
        if n in OVERRIDE_DOI and cache.get(key, {}).get("doi") != OVERRIDE_DOI[n]:
            doi = OVERRIDE_DOI[n]
            work = crossref_work(doi)
            if work is None:  # DataCite DOIs (arXiv) are not in Crossref
                d = get(f"https://api.datacite.org/dois/{doi}")
                if d:
                    a = d["data"]["attributes"]
                    work = {"title": [a["titles"][0]["title"]], "author": [{"family": (a["creators"][0].get("familyName") or "")}], "issued": {"date-parts": [[a.get("publicationYear")]]}, "DOI": doi}
            abstract, oa = openalex_abstract(doi)
            cache[key] = {"doi": doi, "via": "checked by hand", "work": work, "abstract": abstract, "openalex": oa}
            time.sleep(0.3)
        if key not in cache:
            doi = doi_of(text)
            via = "printed" if doi else None
            work = crossref_work(doi) if doi else None
            if work is None:
                best = None
                for item in crossref_search(text):
                    c = compare(text, item)
                    if best is None or c["title_similarity"] > best[1]["title_similarity"]:
                        best = (item, c)
                if best and best[1]["title_similarity"] >= 0.6:
                    work, doi, via = best[0], best[0]["DOI"], "found by search"
            abstract, oa = openalex_abstract(doi) if doi else (None, None)
            cache[key] = {"doi": doi, "via": via, "work": work, "abstract": abstract, "openalex": oa}
            time.sleep(0.3)
        c = cache[key]
        cmp_ = compare(text, c["work"]) if c["work"] else {}
        ok = bool(c["work"]) and cmp_["title_similarity"] >= 0.6 and cmp_["first_author_in_ref"]
        if c["via"] == "publisher page":
            ok = c["page_status"] == 200 and c["title_words_on_page"] >= 0.8
        rows.append({"ref": n, "reference_text": text, "doi": c["doi"] or "", "doi_source": c["via"] or "", "url": c.get("url", ""), "metadata_verdict": "match" if ok else ("partial" if c["work"] else "not found"),
                     **{k: cmp_.get(k, "") for k in ("title_similarity", "first_author_in_ref", "year_in_ref", "volume_in_ref", "page_in_ref", "crossref_title", "crossref_year", "crossref_volume", "crossref_page")},
                     "abstract_available": bool(c["abstract"])})
    CACHE.write_text(json.dumps(cache), encoding="utf-8")
    with open(OUT_REFS, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} references: " + ", ".join(f"{v} {sum(1 for r in rows if r['metadata_verdict'] == v)}" for v in ("match", "partial", "not found")))

    # claims
    inv = list(csv.DictReader(open(INVENTORY, encoding="utf-8-sig")))
    claims = []
    for r in inv:
        if r["status"] != "LITERATURE" or r["kind"] != "CLAIM":
            continue
        cited = sorted({int(x) for blob in re.findall(r"\\?\[([\d,\s–-]+)\\?\]", r["text"]) for x in re.split(r"[,\s]+", blob) if x.isdigit()})
        # table rows cite through the author names in the row text; the sentences above cite by number
        nums = numbers(r["current_value"] or r["text"])
        found, missing, support = [], [], []
        for n in cited:
            ab = (cache.get(str(n)) or {}).get("abstract")
            if not ab:
                support.append(f"[{n}] no abstract")
                continue
            hit = [x for x in nums if x in ab.replace(",", "")]
            support.append(f"[{n}] {len(hit)}/{len(nums)} numbers in abstract")
            found += hit
        missing = [x for x in nums if x not in found]
        verdict = "no citation number in the text" if not cited else ("all numbers in an abstract" if nums and not missing else ("no numbers to check" if not nums else "full text needed"))
        claims.append({"id": r["id"], "location": r["location"], "claim_value": r["current_value"], "cited_refs": " ".join(map(str, cited)), "numbers_checked": " ".join(nums),
                       "numbers_found_in_abstract": " ".join(dict.fromkeys(found)), "numbers_not_found": " ".join(missing), "abstract_check": "; ".join(support), "verdict": verdict})
    with open(OUT_CLAIMS, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(claims[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(claims)
    from collections import Counter

    print(f"{len(claims)} LITERATURE claims:", dict(Counter(c["verdict"] for c in claims)))


if __name__ == "__main__":
    main()
