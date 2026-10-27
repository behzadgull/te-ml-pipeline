"""
Round-trip check of the restructuring of paper.md (2026-10-27): compares the manuscript before (a git revision) with the manuscript after (paper.md, supplement.md and highlights.md together).

  A  Number markers: every <!--v:key-->, <!--c:id--> and <!--d:id--> (key and value) of the old text, as a multiset, against the new text. Removed and added markers are listed.
  B  Sentences: after normalising labels ("Table [[tbl:x]]" and "Supplementary Table [[stbl:x]]" both become "Table [[x]]"), section and equation numbers, every sentence of the old
     running text that is not in the new text (main plus Supplement) is listed, and every new sentence that is not in the old text.
  C  Tables: the generated blocks (BEGIN/END TABLE) of old and new, compared after the same normalisation.
  D  Plain numbers outside markers (digits in running text that carry no marker).

Usage (repository root):  python thesis_paper/scripts/check_restructure.py [<old git revision>]     (default 999494b)
"""

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO / "thesis_paper" / "paper"
GIT = r"C:\Program Files\Git\cmd\git.exe"
MARK = re.compile(r"<!--([vcd]):([\w.\-]+)-->(.*?)<!--/\1-->")


def old_text(rev):
    exe = GIT if Path(GIT).exists() else "git"
    return subprocess.run([exe, "show", f"{rev}:thesis_paper/paper/paper.md"], cwd=REPO, capture_output=True, check=True).stdout.decode("utf-8").replace("\r\n", "\n")


def normalise(text):
    """Labels, section numbers, equation numbers and markdown noise removed so that moved text compares equal."""
    t = re.sub(r"<!--\s*(?:BEGIN|END) TABLE [^>]*-->", "", text)
    t = re.sub(r"<!--(?!/?[vcd][:\-]).*?-->", "", t, flags=re.S)
    t = re.sub(r"\[\[(?:tbl|fig|stbl|sfig|snote):(\w+)\]\]", r"[[\1]]", t)
    t = re.sub(r"\b(?:Supplementary )?(Table|Figure|Note)\b", lambda m: m.group(1), t)
    t = re.sub(r"\[\[(?:stbl:)?amend\]\]", "S1", t)
    t = re.sub(r"\[\[(?:stbl:)?hyper\]\]", "S2", t)
    t = re.sub(r"\bFig\.", "Figure", t)
    t = re.sub(r"\bSections? \d[\d.]*(?: and \d[\d.]*)?", "Section §", t)
    t = re.sub(r"\((?:S?\d+)\)\s*$", "", t, flags=re.M)
    t = re.sub(r"\((?:\d+)\)(?=\s|$)", "", t) if False else t
    return t


def sentences(text):
    out = []
    for para in re.split(r"\n\s*\n", normalise(text)):
        para = para.strip()
        if not para or para.startswith("|") or para.startswith("#") or para.startswith("!["):
            continue
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z*\[<(\\$])", para.replace("\n", " ")):
            s = s.strip(" -")
            if len(s.split()) >= 4:
                out.append(re.sub(r"\s+", " ", s))
    return out


def table_blocks(text):
    return {m.group(1): normalise(m.group(2)) for m in re.finditer(r"<!-- BEGIN TABLE (\w+) -->\n(.*?)<!-- END TABLE \1 -->", text, re.S)}


def main():
    rev = sys.argv[1] if len(sys.argv) > 1 else "999494b"
    old = old_text(rev)
    new = "\n\n".join((PAPER_DIR / f).read_bytes().decode("utf-8").replace("\r\n", "\n") for f in ("paper.md", "supplement.md", "highlights.md") if (PAPER_DIR / f).exists())
    # A
    mo, mn = Counter(m.groups()[:2] + (m.group(3),) for m in MARK.finditer(old)), Counter(m.groups()[:2] + (m.group(3),) for m in MARK.finditer(new))
    print("A. number markers: old", sum(mo.values()), "new", sum(mn.values()))
    gone, extra = mo - mn, mn - mo
    print("   in the old text and not in the new:", sum(gone.values()))
    for k, n in sorted(gone.items()):
        print("     -", k, "x", n)
    print("   in the new text and not in the old (the highlights and the sentences written for the new structure):", sum(extra.values()))
    for k, n in sorted(extra.items()):
        print("     +", k, "x", n)
    # B
    so, sn = sentences(old), sentences(new)
    set_n, set_o = set(sn), set(so)
    miss = [s for s in so if s not in set_n]
    add = [s for s in sn if s not in set_o]
    print(f"\nB. sentences: old {len(so)}, new {len(sn)}; old sentences not found in the new text: {len(miss)}")
    for s in miss:
        print("   OLD ONLY:", s[:330])
    print(f"   new sentences not in the old text: {len(add)}")
    for s in add:
        print("   NEW ONLY:", s[:330])
    # C
    to, tn = table_blocks(old), table_blocks(new)
    diff = [k for k in sorted(set(to) | set(tn)) if to.get(k) != tn.get(k)]
    print("\nC. generated tables: old", len(to), "new", len(tn), "blocks that differ after normalisation:", diff)
    # D
    def plain_numbers(t):
        t = MARK.sub("", normalise(t))
        t = re.sub(r"\[\[[^\]]*\]\]", "", t)
        t = re.sub(r"^\|.*$", "", t, flags=re.M)
        return Counter(re.findall(r"(?<![\w.\-])\d+(?:[.,]\d+)*(?![\w.\-])", t))
    po, pn = plain_numbers(old), plain_numbers(new)
    print("\nD. plain numbers (outside markers and tables): only in old:", dict(po - pn), "| only in new:", dict(pn - po))
    return 0


if __name__ == "__main__":
    sys.exit(main())
