"""
NA12: Supplementary Table S1, the pre-registration and its amendments, generated from the headings and opening sentences of docs/decisions.md (nothing is typed here). Writes thesis_paper/paper/supplement_amendments.md.

For each entry: the number, the date, the subject (the parenthesis of the heading) and the timing as the entry states it ("made before ...", "made after ...", from the heading or, for the first two amendments, the first
sentence of the entry that begins "Made before"). The full rules are in docs/decisions.md; this table lists them and says when each was made relative to the data.

Usage (repository root):  python thesis_paper/scripts/make_supplement_amendments.py
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_record as rr  # noqa: E402

DECISIONS = REPO / "thesis_paper" / "docs" / "decisions.md"
OUT = REPO / "thesis_paper" / "paper" / "supplement_amendments.md"
HEAD = re.compile(r"^## (\d{4}-\d{2}-\d{2}): NA12, amendment (\d+) to pre-registration B \((.*?)\)(.*)$")


def main():
    text = DECISIONS.read_text(encoding="utf-8").replace("\r\n", "\n")
    lines = text.split("\n")
    rows = []
    # the pre-registration itself: the entry of 2026-10-05 that holds item B
    for i, ln in enumerate(lines):
        if ln.startswith("### B. NA12 literature validation: design"):
            date = next(lines[j][3:13] for j in range(i, -1, -1) if lines[j].startswith("## 20"))
            tail = re.search(r"\((.*)\)", ln)
            rows.append(("pre-registration", date, "Design: compounds, predictions compared (out-of-fold, exact and cluster level), per-value facts, error measure and band, values from the PDFs only", tail.group(1) if tail else "", ""))
    nums = []
    for i, ln in enumerate(lines):
        m = HEAD.match(ln)
        if not m:
            continue
        date, no, subject, rest = m.group(1), int(m.group(2)), m.group(3), m.group(4).strip()
        timing = rest.lstrip(";").strip() if rest.startswith("; made") else ""
        body = []
        for ln2 in lines[i + 1:]:
            if ln2.startswith("## "):
                break
            body.append(ln2)
        disclosure = "yes" if any("**Disclosure" in b for b in body) else ""
        if not timing:
            mm = re.search(r"(Made before[^.]*\.)", " ".join(body))
            timing = (mm.group(1).rstrip(".") if mm else rest.lstrip(";").strip())
        rows.append((str(no), date, subject, timing, disclosure))
        nums.append(no)
    assert nums == list(range(1, len(nums) + 1)), nums
    prov = rr.provenance({"decisions": "thesis_paper/docs/decisions.md"}, __file__)
    out = ["**Supplementary Table [[stbl:amend]]:** The pre-registration of the literature comparison (NA12) and its amendments, with the date of each and when it was made relative to the data, as the entries of `docs/decisions.md` state. "
           "Every entry was written before the comparison was run and before any out-of-fold prediction for these compounds was looked at; entries marked as made after a test run or after the digitising of a figure were made after the "
           "digitised values of that figure had been seen. A disclosure states what came into view that was not meant to be read (values of a source while it was inventoried, or the reason an amendment was prompted). The full rules are in `docs/decisions.md`.", "",
           "| **No.** | **Date** | **Subject** | **Timing, as stated in the entry** | **Entry contains a disclosure** |", "|---|---|---|---|---|"]
    out += [f"| {no} | {d} | {subj} | {t} | {dis} |" for no, d, subj, t, dis in rows]
    out += ["", f"Generated from `docs/decisions.md` (SHA256 `{prov['inputs']['decisions']['sha256']}`) at commit `{prov['git_head']}`."]
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote", OUT, f"({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
