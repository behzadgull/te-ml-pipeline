"""
Thesis-carryover check (standing rule of 2026-10-04: no number from the thesis is trusted).

Extracts every number from the original supervisor manuscript (thesis_paper/source/Perovskite_Thermoelectric_Manuscript.docx, converted to markdown by
pandoc, which keeps the sub- and superscripts and the colour-coded notes as text) with the same tokenizer and exclusions as audit_numbers.py, then lists
every number of the current paper.md whose value also occurs in the thesis, with the paper context, the audit class of the paper occurrence
(GENERATED / CITED / DESIGN / UNCLASSIFIED) and up to three thesis contexts. Values are compared as absolute normalised numbers ("0.60" = "0.6",
"-0.005" = "0.005", "184,167" = "184167"), so a match is a candidate carry-over, not proof: a small integer or a round value can coincide by chance. The
output separates the matches that matter (value with a decimal point, a thousands separator or at least three digits) from small integers
(`trivial` true), which are listed too.

Output: thesis_paper/reports/thesis_carryover/<UTC>/{carryover.csv, thesis_numbers.csv, summary.json, run_config.json}.

Usage (from the repository root; pandoc on PATH or in $PANDOC):
    python thesis_paper/scripts/thesis_carryover_check.py
"""

import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_numbers as an  # noqa: E402
import run_record as rr  # noqa: E402

REPO = HERE.parents[1]
DOCX = REPO / "thesis_paper" / "source" / "Perovskite_Thermoelectric_Manuscript.docx"
OUT_ROOT = REPO / "thesis_paper" / "reports" / "thesis_carryover"


def thesis_markdown():
    """The original manuscript as pandoc markdown."""
    exe = os.environ.get("PANDOC") or "pandoc"
    return subprocess.run([exe, str(DOCX), "-f", "docx", "-t", "markdown", "--wrap=none"], capture_output=True, check=True).stdout.decode("utf-8")


def numbers_of(text):
    """[(token, value, context)] of the numbers that survive the audit exclusions."""
    clean, _, _ = an.prepare(text)
    excl = an.exclusion_spans(clean)
    out = []
    for s, e, tok in an.tokens(clean):
        if any(a <= s < b for (a, b, _) in excl):
            continue
        out.append((tok, an.normalise(tok), an.context(clean, s, e, 70)))
    return out


def trivial(value):
    """Small integers coincide by chance; everything with a decimal point, a thousands separator or 3+ digits does not."""
    return re.fullmatch(r"\d{1,2}", value) is not None


def main():
    """Entry point."""
    md = thesis_markdown()
    thesis = numbers_of(md)
    by_value = defaultdict(list)
    for tok, val, ctx in thesis:
        by_value[val].append((tok, ctx))
    paper_text = (REPO / "thesis_paper" / "paper" / "paper.md").read_bytes().decode("utf-8")
    rows = an.audit(paper_text, an.load_registry())
    out = []
    for r in rows:
        if r["class"].startswith("EXCLUDED"):
            continue
        hits = by_value.get(r["value"], [])
        if not hits:
            continue
        out.append({"paper_line": r["line"], "paper_token": r["token"], "value": r["value"], "paper_class": r["class"], "paper_id": r["id"], "trivial": trivial(r["value"]),
                    "thesis_occurrences": len(hits), "paper_context": r["context"], "thesis_context_1": hits[0][1],
                    "thesis_context_2": hits[1][1] if len(hits) > 1 else "", "thesis_context_3": hits[2][1] if len(hits) > 2 else "", "paper_section": r["section"]})
    stamp = rr.utc_stamp()
    d = OUT_ROOT / stamp
    d.mkdir(parents=True, exist_ok=True)
    fields = list(out[0].keys())
    with open(d / "carryover.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    cnt = Counter(v for _, v, _ in thesis)
    with open(d / "thesis_numbers.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["value", "occurrences_in_thesis", "first_context"])
        for v, n in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
            w.writerow([v, n, by_value[v][0][1]])
    nontriv = [o for o in out if not o["trivial"]]
    summary = {"thesis_numbers_total": len(thesis), "thesis_distinct_values": len(cnt),
               "paper_numbers_matching_a_thesis_value": len(out), "of_which_small_integers": len(out) - len(nontriv),
               "nontrivial_matches": len(nontriv), "nontrivial_by_paper_class": dict(Counter(o["paper_class"] for o in nontriv)),
               "nontrivial_distinct_values": len({o["value"] for o in nontriv}),
               "nontrivial_unclassified_distinct_values": len({o["value"] for o in nontriv if o["paper_class"] == "UNCLASSIFIED"})}
    prov = rr.provenance({"thesis_docx": str(DOCX.relative_to(REPO)), "paper": "thesis_paper/paper/paper.md"}, __file__)
    rr.write_json(d / "summary.json", summary)
    rr.write_json(d / "run_config.json", prov)
    print("wrote", d)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
