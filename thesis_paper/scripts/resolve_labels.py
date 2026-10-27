"""
Automatic numbering of the tables, figures and supplementary items of the manuscript (thesis_paper/paper/paper.md) and its Supplement.

A table, figure or supplementary item is never numbered by hand: it carries a label, and the number is assigned here, in the order of the captions, when the manuscript is built.

  Main text   Caption: **Table [[tbl:ranked]]:** ...   **Figure [[fig:ladder]]:** ...        Reference: Table [[tbl:ranked]], Tables [[tbl:ranked]] and [[tbl:limits]], Fig. [[fig:ladder]]a
  Supplement  Caption: **Supplementary Table [[stbl:x]]:** ...   **Supplementary Figure [[sfig:x]]:** ...   a heading "## Supplementary Note [[snote:x]]: Title"
              Reference: Supplementary Table [[stbl:x]]   (resolved to "S1", "S2", ... separately for tables, figures and notes)

The Supplement is the concatenation of SUPPLEMENT_FILES in that order (generated tables first, then supplement.md). Numbers run 1, 2, 3 ... in the order in which the captions appear, so
inserting, moving or deleting an item renumbers every reference. Checked on every build (and by --check): every reference has a caption; every label has exactly one caption; every caption
is cited at least once outside itself and outside the alt text of its image; and, in the main text, the first mention of a table or figure is not earlier than the first mention of the one
numbered before it. References to the figures and tables of OTHER papers ("Shibasaki et al., Fig. 5") are plain text and are not labels.

Usage (repository root):
    python thesis_paper/scripts/resolve_labels.py --check                       # exit 1 on any problem
    python thesis_paper/scripts/resolve_labels.py --order                       # print the numbers assigned
    python thesis_paper/scripts/resolve_labels.py --write <main.md> <supp.md>   # the main text and the Supplement with every label replaced by its number
"""

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER_DIR = HERE.parents[1] / "thesis_paper" / "paper"
PAPER = PAPER_DIR / "paper.md"
SUPPLEMENT_FILES = [PAPER_DIR / "supplement_amendments.md", PAPER_DIR / "supplement_hyperparameters.md", PAPER_DIR / "supplement.md"]

TOKEN = re.compile(r"\[\[(tbl|fig|stbl|sfig|snote):(\w+)\]\]")
DEF_MAIN = re.compile(r"\*\*(Table|Figure) \[\[(tbl|fig):(\w+)\]\]:\*\*")
DEF_SUPP = re.compile(r"(?:\*\*Supplementary (Table|Figure) |^## Supplementary (Note) )\[\[(stbl|sfig|snote):(\w+)\]\][.:](?:\*\*)?", re.M)
WORD = {"tbl": "Table", "fig": "Figure", "stbl": "Table", "sfig": "Figure", "snote": "Note"}
IMAGE_ALT = re.compile(r"!\[(?:Supplementary )?Figure $")
KINDS = ("tbl", "fig", "stbl", "sfig", "snote")


def load_supplement():
    """The Supplement as one text (the files of SUPPLEMENT_FILES that exist, in order)."""
    parts = [f.read_bytes().decode("utf-8").replace("\r\n", "\n") for f in SUPPLEMENT_FILES if f.exists()]
    return "\n\n".join(parts)


def scan(main, supp):
    """Return (numbers, problems): numbers {kind: {label: n}} by order of caption, and the list of problems found."""
    problems = []
    numbers = {k: {} for k in KINDS}
    defs = {}  # (kind, label) -> (where, start, end)
    for where, text in (("main", main), ("supp", supp)):
        if where == "main":
            found = [(m.group(2), m.group(3), m.group(1), m.span()) for m in DEF_MAIN.finditer(text)]
        else:
            found = [(m.group(3), m.group(4), m.group(1) or m.group(2), m.span()) for m in DEF_SUPP.finditer(text)]
        for kind, label, word, span in found:
            if WORD[kind] != word:
                problems.append(f"caption word {word!r} does not match label kind {kind}:{label}")
            if label in numbers[kind]:
                problems.append(f"{kind}:{label} has more than one caption")
            else:
                numbers[kind][label] = len(numbers[kind]) + 1
                defs[(kind, label)] = (where, span[0], span[1])
    first, cited = {}, {}
    for where, text in (("main", main), ("supp", supp)):
        for m in TOKEN.finditer(text):
            kind, label = m.groups()
            if label not in numbers[kind]:
                problems.append(f"reference to {kind}:{label} has no caption")
                continue
            if where == "main":
                first.setdefault((kind, label), m.start())
            w, d0, d1 = defs[(kind, label)]
            in_def = w == where and d0 <= m.start() < d1
            in_alt = bool(IMAGE_ALT.search(text[max(0, m.start() - 30):m.start()]))
            if not in_def and not in_alt:
                cited[(kind, label)] = cited.get((kind, label), 0) + 1
    for kind in KINDS:
        for label in numbers[kind]:
            if (kind, label) not in cited:
                problems.append(f"{kind}:{label} has a caption but is never cited in the text")
    for kind in ("tbl", "fig"):
        order = sorted(numbers[kind], key=lambda lb: numbers[kind][lb])
        for a, b in zip(order, order[1:]):
            if (kind, a) in first and (kind, b) in first and first[(kind, b)] < first[(kind, a)]:
                problems.append(f"{WORD[kind]} {numbers[kind][b]} ({b}) is mentioned before {WORD[kind]} {numbers[kind][a]} ({a})")
    return numbers, problems


def resolve(text, numbers):
    """The text with every label replaced by its number ('S<n>' for the supplementary kinds)."""
    def sub(m):
        kind, label = m.groups()
        n = numbers[kind][label]
        return str(n) if kind in ("tbl", "fig") else f"S{n}"

    return TOKEN.sub(sub, text)


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=str(PAPER))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--order", action="store_true")
    ap.add_argument("--write", nargs=2, metavar=("MAIN_OUT", "SUPP_OUT"), default=None)
    args = ap.parse_args()
    main_text = Path(args.paper).read_bytes().decode("utf-8").replace("\r\n", "\n")
    supp_text = load_supplement()
    numbers, problems = scan(main_text, supp_text)
    n_refs = len(TOKEN.findall(main_text)) + len(TOKEN.findall(supp_text))
    if args.order:
        for kind in KINDS:
            for lb, n in numbers[kind].items():
                pre = "S" if kind in ("stbl", "sfig", "snote") else ""
                print(f"{'Supplementary ' if pre else ''}{WORD[kind]} {pre}{n}: {lb}")
    if problems:
        for p in problems:
            print("PROBLEM:", p, file=sys.stderr)
        return 1
    if args.write:
        out_main, out_supp = resolve(main_text, numbers), resolve(supp_text, numbers)
        assert not TOKEN.search(out_main) and not TOKEN.search(out_supp)
        Path(args.write[0]).write_bytes(out_main.encode("utf-8"))
        Path(args.write[1]).write_bytes(out_supp.encode("utf-8"))
        print(f"wrote {args.write[0]} and {args.write[1]} ({len(numbers['tbl'])} tables, {len(numbers['fig'])} figures, {len(numbers['stbl'])} supplementary tables, "
              f"{len(numbers['sfig'])} supplementary figures, {len(numbers['snote'])} supplementary notes, {n_refs} label tokens resolved)")
    elif args.check:
        print(f"labels resolve: {len(numbers['tbl'])} tables, {len(numbers['fig'])} figures, {len(numbers['stbl'])} supplementary tables, {len(numbers['sfig'])} supplementary figures, "
              f"{len(numbers['snote'])} supplementary notes, {n_refs} label tokens, no problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
