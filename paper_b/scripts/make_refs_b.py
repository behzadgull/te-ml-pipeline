"""
Build paper_b/paper/refs.bib from the citation keys used in paper_b/paper/paper.md.

Each cited key is copied verbatim from the Paper A bibliography (paper/refs.bib), so that the entries
keep the verification already done there. The key `companionA` is a placeholder for Paper A itself
until it has a DOI or preprint. A cited key that is in neither place stops the script.

Usage (from the repository root):
    python paper_b/scripts/make_refs_b.py           # write paper_b/paper/refs.bib
    python paper_b/scripts/make_refs_b.py --check   # exit 1 if refs.bib is out of date
"""

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper_b" / "paper" / "paper.md"
OUT = REPO / "paper_b" / "paper" / "refs.bib"
SOURCE = REPO / "paper" / "refs.bib"

PLACEHOLDERS = {
    "companionA": """@unpublished{companionA,
  author = {Gull, Muhammad Behzad},
  title = {Validation Inflation, Descriptor Saturation, and the Label-Noise Ceiling in Thermoelectric Property Prediction},
  note = {Companion paper (Paper A); manuscript in preparation. Placeholder entry: replace with the DOI or preprint when available},
  year = {2026}
}
""",
}


def cited_keys(text):
    """Citation keys in pandoc `[@a; @b]` form, in order of first appearance, ignoring HTML comments."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    keys = []
    for m in re.finditer(r"\[(@[^\]]+)\]", text):
        for k in re.findall(r"@([A-Za-z0-9_:.-]+)", m.group(1)):
            k = k.rstrip(".")
            if k not in keys:
                keys.append(k)
    return keys


def entries(bib_text):
    """Map key -> full entry text for a .bib file."""
    parts = re.split(r"(?m)^(?=@\w+\{)", bib_text)
    out = {}
    for part in parts:
        m = re.match(r"@\w+\{([^,\s]+),", part)
        if m:
            out[m.group(1)] = part.rstrip() + "\n"
    return out


def build():
    """Return the text of refs.bib."""
    keys = cited_keys(PAPER.read_text(encoding="utf-8"))
    source = entries(SOURCE.read_text(encoding="utf-8"))
    chunks = []
    for k in keys:
        if k in PLACEHOLDERS:
            chunks.append(PLACEHOLDERS[k].rstrip() + "\n")
        elif k in source:
            chunks.append(source[k])
        else:
            raise KeyError(f"cited key {k!r} is not in paper/refs.bib and has no placeholder")
    return "\n".join(chunks), keys


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text, keys = build()
    if args.check:
        cur = OUT.read_bytes().decode("utf-8").replace("\r\n", "\n") if OUT.exists() else ""
        if cur != text:
            print("refs.bib is out of date: run python paper_b/scripts/make_refs_b.py")
            return 1
        print(f"refs.bib is current ({len(keys)} entries)")
        return 0
    OUT.write_bytes(text.encode("utf-8"))
    print(f"wrote {OUT} ({len(keys)} entries): {', '.join(keys)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
