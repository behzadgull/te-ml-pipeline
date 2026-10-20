"""
Fill the number markers of thesis_paper/docs/reframe_draft.md from the committed artifacts (the value dictionary of paper.md, no blocks) and export it
to thesis_paper/docs/reframe_draft.docx with pandoc, for the supervisor. The markers are kept in the markdown (pandoc drops them from the docx), so
no number of the draft is typed.

Usage (repository root; set PANDOC if pandoc is not on PATH):
    python thesis_paper/scripts/export_reframe_draft.py            # fill the markers and write the docx
    python thesis_paper/scripts/export_reframe_draft.py --check    # exit 1 if the markers are out of date
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import make_thesis_values as m  # noqa: E402
import thesis_values as tv  # noqa: E402

DRAFT = REPO / "thesis_paper" / "docs" / "reframe_draft.md"
DOCX = REPO / "thesis_paper" / "docs" / "reframe_draft.docx"


def fill(text):
    """The text with every inline and design marker recomputed; unknown keys raise."""
    v = tv.values()
    v.update(m.extra_values(v))
    reg = m.an.load_registry()

    def sub(mt):
        if mt.group(1) not in v:
            raise KeyError(f"unknown inline value key {mt.group(1)!r}")
        return f"<!--v:{mt.group(1)}-->{v[mt.group(1)]}<!--/v-->"

    def sub_design(mt):
        row = reg.get(mt.group(1))
        if row is None or row["class"] != "DESIGN":
            raise KeyError(f"design marker {mt.group(1)!r} has no DESIGN row")
        return f"<!--d:{mt.group(1)}-->{tv.config_value(row['source_or_config'])}<!--/d-->"

    return m.DESIGN_MARK.sub(sub_design, m.INLINE.sub(sub, text))


def main():
    """Fill and export."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    old = DRAFT.read_bytes().decode("utf-8").replace("\r\n", "\n")
    new = fill(old)
    if args.check:
        print("reframe_draft.md markers are current" if new == old else "reframe_draft.md is out of date")
        return 0 if new == old else 1
    DRAFT.write_bytes(new.encode("utf-8"))
    pandoc = os.environ.get("PANDOC", "pandoc")
    subprocess.run([pandoc, str(DRAFT), "--resource-path", str(DRAFT.parent), "-o", str(DOCX)], check=True, cwd=REPO)
    print("wrote", DRAFT, "and", DOCX)
    return 0


if __name__ == "__main__":
    sys.exit(main())
