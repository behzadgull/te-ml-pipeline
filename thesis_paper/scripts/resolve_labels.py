"""
Automatic numbering of the tables and figures of thesis_paper/paper/paper.md.

In paper.md (and in the generators that write its table blocks) a table or figure is never numbered by hand: it carries a label, and the number is assigned here, in the order
of the captions, when the manuscript is built.

  Caption of a table:   **Table [[tbl:ranked]]:** ...        Caption of a figure:   **Figure [[fig:ladder]]:** ...     (the image alt text is ![Figure [[fig:ladder]]](...))
  Reference:            Table [[tbl:ranked]], Tables [[tbl:ranked]] and [[tbl:limits]], Fig. [[fig:ladder]], Fig. [[fig:ladder]]a (a panel letter stays a suffix)

Numbers run 1, 2, 3 ... in the order in which the captions appear in the file, separately for tables and for figures, so inserting, moving or deleting a table renumbers every
reference. Checked on every build (and by --check): every reference has a caption; every label has exactly one caption; every caption is cited at least once outside itself; and the
first mention of every table and figure is not earlier than the first mention of the one numbered before it (a figure or table numbered 3 must not be mentioned before one numbered 4).
References to the figures and tables of OTHER papers ("Shibasaki et al., Fig. 5") are plain text and are not labels.

Usage (repository root):
    python thesis_paper/scripts/resolve_labels.py --check                     # exit 1 on any problem
    python thesis_paper/scripts/resolve_labels.py --order                     # print the numbers assigned
    python thesis_paper/scripts/resolve_labels.py --write <output.md>         # paper.md with every label replaced by its number
"""

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1] / "thesis_paper" / "paper" / "paper.md"

TOKEN = re.compile(r"\[\[(tbl|fig):(\w+)\]\]")
DEF = re.compile(r"\*\*(Table|Figure) \[\[(tbl|fig):(\w+)\]\]:\*\*")
KIND = {"tbl": "Table", "fig": "Figure"}


def scan(text):
    """Return (numbers, problems): numbers {'tbl': {label: n}, 'fig': {...}} by order of caption, and the list of problems found."""
    problems = []
    numbers = {"tbl": {}, "fig": {}}
    for m in DEF.finditer(text):
        word, kind, label = m.groups()
        if KIND[kind] != word:
            problems.append(f"caption word {word!r} does not match label kind {kind}:{label}")
        if label in numbers[kind]:
            problems.append(f"{kind}:{label} has more than one caption")
        else:
            numbers[kind][label] = len(numbers[kind]) + 1
    first, cited = {}, {}
    defs = {(m.group(2), m.group(3)): m.span() for m in DEF.finditer(text)}
    for m in TOKEN.finditer(text):
        kind, label = m.groups()
        if label not in numbers[kind]:
            problems.append(f"reference to {kind}:{label} has no caption")
            continue
        first.setdefault((kind, label), m.start())
        d0, d1 = defs.get((kind, label), (-1, -1))
        if not (d0 <= m.start() < d1):
            cited[(kind, label)] = cited.get((kind, label), 0) + 1
    for kind in numbers:
        for label in numbers[kind]:
            if (kind, label) not in cited:
                problems.append(f"{kind}:{label} has a caption but is never cited in the text")
    for kind in numbers:
        order = sorted(numbers[kind], key=lambda lb: numbers[kind][lb])
        for a, b in zip(order, order[1:]):
            if (kind, a) in first and (kind, b) in first and first[(kind, b)] < first[(kind, a)]:
                problems.append(f"{KIND[kind]} {numbers[kind][b]} ({b}) is mentioned before {KIND[kind]} {numbers[kind][a]} ({a})")
    return numbers, problems


def resolve(text, numbering=None):
    """The text with every label replaced by its number; `numbering` ({'tbl': {label: str}, 'fig': {...}}) overrides the numbers (used to verify a conversion)."""
    numbers, problems = scan(text)
    assert not problems or numbering is not None, problems
    use = numbering or {k: {lb: str(n) for lb, n in v.items()} for k, v in numbers.items()}
    return TOKEN.sub(lambda m: use[m.group(1)][m.group(2)], text), numbers, problems


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=str(PAPER))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--order", action="store_true")
    ap.add_argument("--write", default=None)
    args = ap.parse_args()
    text = Path(args.paper).read_bytes().decode("utf-8").replace("\r\n", "\n")
    numbers, problems = scan(text)
    n_refs = len(TOKEN.findall(text))
    if args.order:
        for kind in ("tbl", "fig"):
            for lb, n in numbers[kind].items():
                print(f"{KIND[kind]} {n}: {lb}")
    if problems:
        for p in problems:
            print("PROBLEM:", p, file=sys.stderr)
        return 1
    if args.write:
        out, _, _ = resolve(text)
        assert not TOKEN.search(out)
        Path(args.write).write_bytes(out.encode("utf-8"))
        print(f"wrote {args.write} ({len(numbers['tbl'])} tables, {len(numbers['fig'])} figures, {n_refs} label tokens resolved)")
    elif args.check:
        print(f"labels resolve: {len(numbers['tbl'])} tables, {len(numbers['fig'])} figures, {n_refs} label tokens, no problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
