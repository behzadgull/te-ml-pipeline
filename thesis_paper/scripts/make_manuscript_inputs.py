"""
The three small files by which the value and figure scripts depend on the manuscript sources, written once from those sources and committed, so that thesis_values.py and make_figures_thesis.py
run without paper.md, docs/decisions.md or source/pdfs/README.md:

  inputs/manuscript_sections.json   section number -> heading of paper.md (levels 2 and 3), for the section numbers drawn into Figure 4
  inputs/na12_amendments.json       the numbers of the NA12 amendments to pre-registration B in docs/decisions.md (1 to N without gaps)
  inputs/na12_papers.json           the papers of the NA12 table of source/pdfs/README.md (file name and DOI)

Usage (repository root):
    python thesis_paper/scripts/make_manuscript_inputs.py           # rewrite the three files
    python thesis_paper/scripts/make_manuscript_inputs.py --check   # exit 1 if a committed file differs from what the sources give (needs the sources)
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # thesis_paper/
INPUTS = ROOT / "inputs"
PAPER = ROOT / "paper" / "paper.md"
DECISIONS = ROOT / "docs" / "decisions.md"
PDF_README = ROOT / "source" / "pdfs" / "README.md"


def read(path):
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def sections():
    """Section number -> heading, for the headings of levels 2 and 3 that start with a number."""
    heads = {m.group(1): m.group(2).strip() for m in re.finditer(r"^#{2,3} (\d(?:\.\d+)*) (.+)$", read(PAPER), flags=re.M)}
    assert heads, "no numbered headings found in paper.md"
    return heads


def na12_amendments():
    """The amendment numbers of pre-registration B (numbered 1 to N without gaps; the pre-registration itself is not counted)."""
    nums = sorted(int(m.group(1)) for m in re.finditer(r"^## .*NA12, amendment (\d+) to pre-registration B", read(DECISIONS), re.M))
    assert nums == list(range(1, len(nums) + 1)), nums
    return {"numbers": nums}


def na12_papers():
    """The papers of the NA12 table: the rows of the block after the heading 'NA12 literature validation'."""
    block = read(PDF_README).split("## NA12 literature validation", 1)[1]
    rows = [[c.strip() for c in ln.split("|")[1:3]] for ln in block.splitlines() if re.match(r"\| `[a-z]+\d{4}\.pdf` \|", ln)]
    files = [r[0].strip("`") for r in rows]
    assert len(set(files)) == len(files), files
    return {"papers": [{"file": f, "doi": r[1]} for f, r in zip(files, rows)]}


def outputs():
    return {"manuscript_sections.json": sections(), "na12_amendments.json": na12_amendments(), "na12_papers.json": na12_papers()}


def render(obj):
    return json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=False) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if a committed file differs from what the sources give")
    args = ap.parse_args()
    bad = []
    for name, obj in outputs().items():
        text = render(obj)
        f = INPUTS / name
        if args.check:
            if not f.exists() or f.read_bytes().decode("utf-8") != text:
                bad.append(name)
        else:
            INPUTS.mkdir(exist_ok=True)
            f.write_bytes(text.encode("utf-8"))
            print("wrote", f.relative_to(ROOT.parent))
    if args.check:
        print("manuscript inputs are current" if not bad else f"out of date: {bad}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
