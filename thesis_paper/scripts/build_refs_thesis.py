"""
Build thesis_paper/paper/refs.bib from the thesis paper's 62 references and convert the numbered in-text citations in paper.md to
pandoc citation keys.

Each reference's BibTeX entry comes, in this order, from
  1. paper/refs.bib (Paper A), when the DOI is already there: the entry and key are reused as they were verified there;
  2. a manual entry for the references that have no DOI (NeurIPS and JMLR papers, taken from the reference text and checked on the
     publisher page by verify_citations.py);
  3. the Crossref record fetched by verify_citations.py (cached in data/external/citations/cache.json), turned into BibTeX here.
Keys are surname + year + first content word of the title, as in Paper A.

The script rewrites paper.md once: the plain-text reference list is removed (pandoc builds it from refs.bib) and every numbered citation
such as [12] or [2, 3] becomes [@key] or [@key1; @key2]. It refuses to run if the numbered list is already gone, and it checks that every
number maps to a key and that every key is cited. Reference [30] (Starrydata2) cites both the 2019 paper the thesis lists and the 2025
database paper that Paper A cites.

NOTE: single use. It needs the numbered reference list, which --apply removes from paper.md; refs.bib was then corrected by hand for
reference 17 (Rowe, first edition 2005, whose DOI Crossref dates by the 2018 reissue).

NOTE: single use. It needs the numbered reference list, which --apply removes from paper.md. After the run, refs.bib was corrected by
hand for reference 17 (Rowe, first edition 2005; Crossref dates the DOI by its 2018 reissue).

Usage (from the repository root):  python thesis_paper/scripts/build_refs_thesis.py [--apply]
Without --apply it only writes refs.bib and reports what the text conversion would do.
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import verify_citations as vc  # noqa: E402

PAPER = REPO / "thesis_paper" / "paper" / "paper.md"
OUT = REPO / "thesis_paper" / "paper" / "refs.bib"
PAPER_A_BIB = REPO / "paper" / "refs.bib"
CACHE = vc.CACHE
STOP = vc.STOP | {"data", "study", "review"}

MANUAL = {
    46: ("lundberg2017unified", """@inproceedings{lundberg2017unified,
  title = {A Unified Approach to Interpreting Model Predictions},
  author = {Lundberg, Scott M. and Lee, Su-In},
  booktitle = {Advances in Neural Information Processing Systems 30 ({NIPS} 2017)},
  pages = {4765--4774},
  year = {2017},
  eprint = {1705.07874},
  archiveprefix = {arXiv},
  doi = {10.48550/arXiv.1705.07874},
  url = {https://arxiv.org/abs/1705.07874}
}"""),
    53: ("ke2017lightgbm", """@inproceedings{ke2017lightgbm,
  title = {{LightGBM}: A Highly Efficient Gradient Boosting Decision Tree},
  author = {Ke, Guolin and Meng, Qi and Finley, Thomas and Wang, Taifeng and Chen, Wei and Ma, Weidong and Ye, Qiwei and Liu, Tie-Yan},
  booktitle = {Advances in Neural Information Processing Systems 30 ({NIPS} 2017)},
  pages = {3146--3154},
  year = {2017},
  url = {https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html}
}"""),
    56: ("cawley2010overfitting", """@article{cawley2010overfitting,
  title = {On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation},
  author = {Cawley, Gavin C. and Talbot, Nicola L. C.},
  journal = {Journal of Machine Learning Research},
  volume = {11},
  pages = {2079--2107},
  year = {2010},
  url = {https://www.jmlr.org/papers/v11/cawley10a.html}
}"""),
    61: ("shafer2008tutorial", """@article{shafer2008tutorial,
  title = {A Tutorial on Conformal Prediction},
  author = {Shafer, Glenn and Vovk, Vladimir},
  journal = {Journal of Machine Learning Research},
  volume = {9},
  pages = {371--421},
  year = {2008},
  eprint = {0706.3188},
  archiveprefix = {arXiv},
  url = {https://www.jmlr.org/papers/v9/shafer08a.html}
}"""),
}
REUSE_BY_NUMBER = {35: "ma2025reexamining", 55: "pedregosa2011scikit"}  # preprint and JMLR entries already verified in Paper A


def ascii_key(s):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower())


def clean_title(t):
    t = re.sub(r"<[^>]+>", "", t)
    return re.sub(r"\s+", " ", t).strip()


def bib_entries(path):
    """{key: (doi lowercase or '', entry text)} of a .bib file."""
    out = {}
    for part in re.split(r"(?m)^(?=@\w+\{)", path.read_text(encoding="utf-8")):
        m = re.match(r"@\w+\{([^,\s]+),", part)
        if m:
            d = re.search(r"doi\s*=\s*[{\"]([^}\"]+)", part)
            out[m.group(1)] = ((d.group(1).lower() if d else ""), part.rstrip() + "\n")
    return out


def crossref_to_bib(n, work, used):
    """BibTeX text and key from a Crossref work."""
    typ = {"journal-article": "article", "proceedings-article": "inproceedings", "book": "book", "book-chapter": "incollection", "posted-content": "misc"}.get(work.get("type"), "misc")
    au = work.get("author") or work.get("editor") or []
    first = au[0].get("family", "x") if au else "x"
    try:
        year = work["issued"]["date-parts"][0][0]
    except (KeyError, IndexError, TypeError):
        year = ""
    title = clean_title((work.get("title") or [""])[0])
    word = next((w for w in re.findall(r"[A-Za-z0-9]+", title.lower()) if w not in STOP and len(w) > 2), "work")
    key = ascii_key(first) + str(year) + ascii_key(word)
    while key in used:
        key += "x"
    authors = " and ".join(f"{a.get('family', '')}, {a.get('given', '')}".strip(", ") for a in au)
    fields = [("title", title), ("author", authors)]
    container = clean_title((work.get("container-title") or [""])[0])
    if container:
        fields.append(("journal" if typ == "article" else "booktitle", container))
    for k in ("volume", "issue", "publisher"):
        if work.get(k):
            fields.append((k if k != "issue" else "number", str(work[k])))
    if work.get("page"):
        fields.append(("pages", work["page"].replace("-", "--")))
    elif work.get("article-number"):
        fields.append(("pages", str(work["article-number"])))
    fields += [("year", str(year)), ("doi", work.get("DOI", "")), ("url", f"https://doi.org/{work.get('DOI', '')}")]
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields if v)
    return key, f"@{typ}{{{key},\n{body}\n}}\n"


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    refs = vc.parse_references()
    paper_a = bib_entries(PAPER_A_BIB)
    by_doi = {d: (k, e) for k, (d, e) in paper_a.items() if d}
    keys, entries, used = {}, {}, set()
    for n, _text in refs:
        c = cache[str(n)]
        if n in REUSE_BY_NUMBER:
            k = REUSE_BY_NUMBER[n]
            keys[n], entries[k] = k, paper_a[k][1]
        elif n in MANUAL:
            k, e = MANUAL[n]
            keys[n], entries[k] = k, e
        elif c["doi"] and c["doi"].lower() in by_doi:
            k, e = by_doi[c["doi"].lower()]
            keys[n], entries[k] = k, e
        else:
            k, e = crossref_to_bib(n, c["work"], used | set(entries))
            if n == 17:  # Crossref dates the DOI by its 2018 reissue; the thesis cites the 2005 first edition (Macro to Nano)
                e = e.replace("year = {2018}", "year = {2005}").replace("title = {Thermoelectrics Handbook}", "title = {Thermoelectrics Handbook: Macro to Nano}")
            keys[n], entries[k] = k, e
        used.add(keys[n])
    extra = ["katsura2019datadriven"]  # [30]: the 2019 paper listed in the thesis plus the 2025 database paper (Paper A)
    keys_30 = [keys[30], "katsura2025starrydata"]
    for k in keys_30:
        entries[k] = paper_a[k][1] if k in paper_a else entries[k]
    for k in ("ho2026physicsinspired",):  # cited in the introduction and Section 2.7 as the existing thermoelectric split comparison
        entries[k] = paper_a[k][1]
    assert len(set(keys.values())) == len(keys), "two references share a key"
    header = ("% Bibliography of the thesis paper, built by thesis_paper/scripts/build_refs_thesis.py from the 62 references of the manuscript.\n"
              "% Entries whose DOI is in Paper A's refs.bib are copied from it; the rest come from the Crossref record of the DOI (checked by\n"
              "% verify_citations.py, see reports/citation_verification.csv); four references without a DOI are entered by hand.\n\n")
    OUT.write_bytes((header + "\n".join(entries.values())).encode("utf-8"))
    print(f"wrote {OUT}: {len(entries)} entries for {len(refs)} references")

    # numbered citations -> keys
    text = PAPER.read_bytes().decode("utf-8").replace("\r\n", "\n")
    head, sep, tail = text.partition("# References")
    assert sep and re.search(r"(?m)^\\\[1\\\] ", tail), "the numbered reference list is not in paper.md (already converted?)"
    cited = set()

    def conv(m):
        nums = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
        ks = []
        for x in nums:
            assert x in keys, x
            ks += keys_30 if x == 30 else [keys[x]]
            cited.add(x)
        return "[" + "; ".join("@" + k for k in dict.fromkeys(ks)) + "]"

    # pandoc writes \[12\]; comments (RED notes) keep their plain text and are not converted
    parts = re.split(r"(<!--.*?-->)", head, flags=re.S)
    parts = [p if p.startswith("<!--") else re.sub(r"\\\[(\d+(?:\s*,\s*\d+)*)\\\]", conv, p) for p in parts]
    todo = "".join(c + "\n" for c in re.findall(r"<!-- TODO.*?-->", tail, flags=re.S))
    new = "".join(parts) + "# References\n\n" + todo
    uncited = sorted(set(keys) - cited)
    left = re.findall(r"\\\[\d+(?:\s*,\s*\d+)*\\\]", "".join(p for p in parts if not p.startswith("<!--")))
    print(f"citations converted: {len(cited)} of {len(keys)} references cited; never cited: {uncited}; numbered citations left: {left}")
    if args.apply:
        assert not left
        PAPER.write_bytes(new.encode("utf-8"))
        print("paper.md rewritten")


if __name__ == "__main__":
    main()
