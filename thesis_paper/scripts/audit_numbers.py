"""
Number audit of thesis_paper/paper/paper.md (standing rule of 2026-10-04: no number from the thesis is trusted).

Every number in the manuscript (text, tables, captions) must be exactly one of
  GENERATED  a marker <!--v:key-->value<!--/v--> filled from a committed artifact by make_thesis_values.py, or a value in a generated table block
             (BEGIN TABLE / END TABLE) that the audit can trace to a value of the generator's dictionary (see below);
  CITED      a marker <!--c:id-->value<!--/c--> whose id has a row in thesis_paper/docs/number_registry.csv with class CITED, verified = yes and the
             source and location (page, table or equation) the value was read from;
  DESIGN     a marker <!--d:id-->value<!--/d--> whose id has a row with class DESIGN, the config key that holds the constant, and the justification.
Anything else is UNCLASSIFIED and fails the build (--strict). Excluded, and counted separately in the summary so that the exclusions can be
reviewed: section numbers in headings and front matter (author block), table/figure/section/step/equation cross-references and equation numbers, list
enumerators such as (1), citation keys and bracketed reference numbers, link targets and image attributes, digits inside subscripts and superscripts that
are not powers of ten (chemical formulas, unit exponents, R^2^), and the digits inside LaTeX math (exponents and coefficients of the equations; reported
as MATH). Reference-list entries live in refs.bib, not in paper.md.

Generated table blocks: every block is rendered again with a dictionary whose values are tagged, so that the characters that come from a value of the
generator's dictionary can be identified; the untagged rendering must equal the block in paper.md. A number in a block is GENERATED only if it lies in a
tagged span. Typed constants in a generator (captions, labels such as "Random 80/20", literature rows) and values that a generator reads straight from an
artifact without going through its dictionary (for example the frozen hyperparameters in Table 4) are UNCLASSIFIED; the second kind is a false alarm to be
fixed by routing the value through the dictionary.

Usage (from the repository root):
    python thesis_paper/scripts/audit_numbers.py                 # write thesis_paper/reports/number_audit/<UTC>/ and print the summary
    python thesis_paper/scripts/audit_numbers.py --strict        # exit 1 if any number is unclassified or a marker has no valid registry row
    python thesis_paper/scripts/audit_numbers.py --no-write
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_record as rr  # noqa: E402

REPO = HERE.parents[1]
PAPER = REPO / "thesis_paper" / "paper" / "paper.md"
REGISTRY = REPO / "thesis_paper" / "docs" / "number_registry.csv"
OUT_ROOT = REPO / "thesis_paper" / "reports" / "number_audit"

MARKER = re.compile(r"<!--([vcd]):([\w.\-]+)-->(.*?)<!--/\1-->")
KEEP_COMMENT = re.compile(r"<!--\s*(?:/?[vcd](?::|-->)|BEGIN TABLE|END TABLE)")
COMMENT = re.compile(r"<!--.*?-->", re.S)
BLOCK = re.compile(r"<!-- BEGIN TABLE (\d+[a-z]?) -->\n(.*?)<!-- END TABLE \1 -->", re.S)
UNIT_SUFFIX = {"K", "eV", "meV", "nm", "GPa", "MPa", "Å", "h", "s", "mV", "V", "W"}
NL = "\n"

EXCLUSIONS = [  # (category, regex, group holding the excluded text or 0)
    ("LINK", re.compile(r"\]\([^)\n]*\)"), 0),
    ("LINK", re.compile(r"https?://\S+"), 0),
    ("LINK", re.compile(r"\{[#.a-z][^}\n]*\}"), 0),
    ("CITATION", re.compile(r"\[@[^\]\n]*\]"), 0),
    ("CITATION", re.compile(r"\[\^[^\]\n]*\]"), 0),
    ("REFNUM", re.compile(r"\\\[\d+(?:\s*[,–\-]+\s*\d+)*\\\]"), 0),
    ("SECTION", re.compile(r"^#{1,6}\s+((?:[A-Z]\s+)|(?:\d+(?:\.\d+)*\s+))", re.M), 1),
    ("XREF", re.compile(r"\b(?:Tables?|Figures?|Figs?\.|Sections?|Secs?\.|Eqs?\.|Equations?|Appendix|Steps?)\s*\(?\d+(?:\.\d+)*[a-z]?\)?"
                        r"(?:\s*(?:,|and|–|-+)\s*\(?\d+(?:\.\d+)*[a-z]?\)?)*"), 0),
    ("EQNUM", re.compile(r"(?<=\$)\s*\(\d+\)\s*$", re.M), 0),
    ("ENUM", re.compile(r"(?<![\w)])\(\d{1,2}\)(?=\s*[*A-Za-z])"), 0),
    ("MATH", re.compile(r"\$\$.*?\$\$", re.S), 0),
    ("MATH", re.compile(r"\$[^$\n]+\$"), 0),
]
NOTATION = [re.compile(r"~[^~\s]+~"), re.compile(r"\^[^^\s]+\^")]  # sub- and superscripts, matched on the text with the math masked out
TOKEN = re.compile(r"(?<![\w.])10\^[−-]?\d+\^|(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?")


def load_registry():
    """The registry as {id: row}; an absent file is an empty registry."""
    if not REGISTRY.exists():
        return {}
    with open(REGISTRY, encoding="utf-8", newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def registry_problem(kind, mid, reg):
    """Why a c: or d: marker is not acceptable, or None."""
    r = reg.get(mid)
    if r is None:
        return "no registry row"
    if kind == "c":
        if r["class"] != "CITED" or r["verified"].strip().lower() != "yes" or not r["source_or_config"].strip() or not r["location_or_justification"].strip():
            return "CITED needs class CITED, verified yes, a source and a page/table/equation location"
    else:
        if r["class"] != "DESIGN" or not r["source_or_config"].strip() or not r["location_or_justification"].strip():
            return "DESIGN needs class DESIGN, a config key and a justification"
    return None


def normalise(tok):
    """Absolute value of a token as a canonical string ('0.60' and '0.6' agree; thousands separators, the sign and a % are dropped)."""
    t = tok.replace(",", "").replace("%", "").replace("−", "-").strip()
    if t.startswith("10^"):
        return "10^" + t[3:].strip("^").replace("−", "-")
    try:
        d = abs(Decimal(t))
    except InvalidOperation:
        return t
    return format(d.normalize(), "f")


def blank(m):
    """Replacement that keeps newlines and offsets."""
    return re.sub(r"[^\n]", " ", m.group(0))


def prepare(text):
    """(clean text with the same offsets, marker spans, block regions). Generic HTML comments (the kept red notes) are blanked; marker tags too."""
    text = text.replace("\r\n", NL)
    text = COMMENT.sub(lambda m: m.group(0) if KEEP_COMMENT.match(m.group(0)) else blank(m), text)
    markers = [(m.start(3), m.end(3), m.group(1), m.group(2)) for m in MARKER.finditer(text)]
    for m in list(MARKER.finditer(text)):
        text = text[:m.start(0)] + " " * (m.start(3) - m.start(0)) + text[m.start(3):m.end(3)] + " " * (m.end(0) - m.end(3)) + text[m.end(0):]
    regions = [(m.start(2), m.end(2), m.group(1)) for m in BLOCK.finditer(text)]
    text = COMMENT.sub(blank, text)
    return text, markers, regions


def exclusion_spans(text):
    """[(start, end, category)] over the cleaned text."""
    out = []
    for cat, rx, g in EXCLUSIONS:
        for m in rx.finditer(text):
            if m.start(g) < m.end(g):
                out.append((m.start(g), m.end(g), cat))
    first = re.search(r"^\*\*Abstract\*\*", text, re.M)
    if first:
        out.append((0, first.start(), "FRONT"))
    masked = text
    for a, b, c in out:
        if c == "MATH":
            masked = masked[:a] + re.sub(r"[^\n]", " ", masked[a:b]) + masked[b:]
    for rx in NOTATION:
        for m in rx.finditer(masked):
            out.append((m.start(), m.end(), "NOTATION"))
    return out


def tokens(text):
    """[(start, end, token text)] for every number in `text` (powers of ten, thousands separators, decimals, a leading minus, a trailing %)."""
    out = []
    for m in TOKEN.finditer(text):
        s, e = m.start(), m.end()
        if text[s:e].startswith("10^"):
            out.append((s, e, text[s:e]))
            continue
        if s >= 2 and text[s - 1] == "-" and text[s - 2].isalpha():  # identifiers such as mp-149
            continue
        if text[e:e + 1].isalpha():
            run = re.match(r"[A-Za-zÅ]+", text[e:]).group(0)
            if run not in UNIT_SUFFIX:
                continue  # 3D, 2D, d10 and the like
        if s >= 1 and text[s - 1] == "−":
            s -= 1
        elif s >= 2 and text[s - 1] == "-" and text[s - 2] in " (":
            s -= 1
        if text[e:e + 1] == "%":
            e += 1
        out.append((s, e, text[s:e]))
    return out


def context(text, s, e, w=55):
    """A one-line context around a span."""
    return re.sub(r"\s+", " ", text[max(0, s - w):s] + "[" + text[s:e] + "]" + text[e:e + w]).strip()


def section_of(text, pos):
    """Heading (number and title) in force at `pos`."""
    last = ""
    for m in re.finditer(r"^#{1,6}\s+(.*)$", text[:pos], re.M):
        last = m.group(1).strip()
    return last


_TRACES = {}
S1, S2, S3 = chr(1), chr(2), chr(3)


class Tag(str):
    """A value of the generator's dictionary that remembers its key: formatting it into an f-string leaves sentinels around the text."""

    def __new__(cls, val, key):
        o = super().__new__(cls, val)
        o.key = key
        return o

    def __format__(self, spec):
        return S1 + self.key + S2 + str.__format__(str(self), spec) + S3


class TrackV(dict):
    """The generator's value dictionary, handing out Tag objects."""

    def __getitem__(self, k):
        return Tag(dict.__getitem__(self, k), k)

    def get(self, k, default=None):
        return Tag(dict.__getitem__(self, k), k) if k in self else default


def block_traces(paper_text):
    """{block number: [(start, end, key)]}: which characters of each generated table block come from a value of the generator's dictionary, found by
    rendering every block with tagged values and checking that the untagged rendering equals the block in paper.md."""
    if _TRACES:
        return _TRACES
    import make_thesis_values as mtv
    import thesis_values as tv

    v = tv.values()
    v.update(mtv.extra_values(v))
    tracked = TrackV(v)
    sentinel = re.compile(S1 + "([^" + S2 + "]*)" + S2 + "(.*?)" + S3, re.S)
    for name, fn in mtv.BLOCKS.items():
        n = name.split()[1]
        raw = fn(tracked) + NL
        plain, out, i = "", [], 0
        for m in sentinel.finditer(raw):
            plain += raw[i:m.start()]
            st = len(plain)
            plain += m.group(2)
            out.append((st, len(plain), m.group(1)))
            i = m.end()
        plain += raw[i:]
        want = re.search(rf"<!-- BEGIN TABLE {n} -->\n(.*?)<!-- END TABLE {n} -->", paper_text.replace("\r\n", NL), re.S)
        if want is None or want.group(1) != plain:
            raise SystemExit(f"block {n}: paper.md is not the current rendering of its generator; run make_thesis_values.py first")
        _TRACES[n] = out
    return _TRACES


def audit(paper_text, reg):
    """Classify every number; returns a list of dict rows."""
    text, markers, regions = prepare(paper_text)
    excl = exclusion_spans(text)
    line_starts = [0] + [m.end() for m in re.finditer(NL, text)]

    def line_no(pos):
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    rows = []
    for s, e, tok in tokens(text):
        cat = next((c for (a, b, c) in excl if a <= s < b), None)
        row = {"line": line_no(s), "token": tok, "value": normalise(tok), "section": "", "context": context(text, s, e), "class": "", "id": "", "problem": ""}
        if cat:
            row["class"] = "EXCLUDED:" + cat
            rows.append(row)
            continue
        row["section"] = section_of(text, s)
        mk = next(((k, i) for (a, b, k, i) in markers if a <= s < b), None)
        if mk:
            kind, mid = mk
            if kind == "v":
                row["class"], row["id"] = "GENERATED", mid
            else:
                prob = registry_problem(kind, mid, reg)
                row["class"], row["id"] = ("CITED" if kind == "c" else "DESIGN"), mid
                if prob:
                    row["class"], row["problem"] = "INVALID_MARKER", prob
            rows.append(row)
            continue
        reg_hit = next((r for r in regions if r[0] <= s < r[1]), None)
        if reg_hit:
            traced = block_traces(paper_text).get(reg_hit[2]) or []
            hit = next((k for (a, b, k) in traced if a <= s - reg_hit[0] < b), None)
            if hit:
                row["class"], row["id"] = "GENERATED", f"table {reg_hit[2]}: {hit}"
            else:
                line = text[text.rfind(NL, 0, s) + 1:text.find(NL, s)]
                if re.search(r"\\\[\d+", line):
                    why = "literature row typed in make_thesis_values.py"
                elif not line.lstrip().startswith("|") or "**" in line:
                    why = "caption or label typed in the generator"
                else:
                    why = "cell not traced to a value of the generator's dictionary (typed, or read from an artifact outside it)"
                row["class"], row["problem"] = "UNCLASSIFIED", f"table {reg_hit[2]}: {why}"
            rows.append(row)
            continue
        row["class"] = "UNCLASSIFIED"
        rows.append(row)
    return rows


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=str(PAPER))
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    reg = load_registry()
    prov = rr.provenance({"paper": str(Path(args.paper).resolve().relative_to(REPO))}, __file__) if not args.no_write else None
    rows = audit(Path(args.paper).read_bytes().decode("utf-8"), reg)
    c = Counter(r["class"] for r in rows)
    bad = [r for r in rows if r["class"] in ("UNCLASSIFIED", "INVALID_MARKER")]
    by_sec = Counter(r["section"].split(" ")[0] if r["section"] else "(front)" for r in bad)
    summary = {"numbers_total": len(rows), "by_class": dict(sorted(c.items())), "unclassified_or_invalid": len(bad), "unclassified_by_section": dict(sorted(by_sec.items()))}
    if not args.no_write:
        out = OUT_ROOT / rr.utc_stamp()
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "audit.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["line", "token", "value", "class", "id", "problem", "section", "context"], lineterminator=NL)
            w.writeheader()
            w.writerows(rows)
        rr.write_json(out / "summary.json", summary)
        rr.write_json(out / "run_config.json", {**prov, "registry_rows": len(reg)})
        print("wrote", out)
    print(json.dumps(summary, indent=1))
    if args.strict and bad:
        print(f"FAIL: {len(bad)} numbers are unclassified or carry an invalid marker (see audit.csv)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
