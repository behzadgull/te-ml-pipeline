"""
Check (or write) thesis_paper/SHARED_DEPENDENCIES.md: the committed Paper A files the thesis paper reads, each pinned by SHA256,
and the rule that nothing under thesis_paper/ imports from the top-level src/ package.

Text files are hashed with CRLF converted to LF, so a Windows checkout with core.autocrlf hashes the same as the committed blob;
binary files (the Git LFS .npz) are hashed as they are, and need `git lfs pull` in a fresh clone.

Usage (from the repository root):
    python thesis_paper/scripts/check_shared_dependencies.py            # verify; exit 1 on a missing file, a changed hash or a src import
    python thesis_paper/scripts/check_shared_dependencies.py --write    # regenerate the manifest (after a deliberate change)
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_a_values as pav  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "thesis_paper" / "SHARED_DEPENDENCIES.md"
SOURCE_DOCX = "thesis_paper/source/Perovskite_Thermoelectric_Manuscript.docx"
TEXT_SUFFIXES = {".json", ".py", ".md", ".csv", ".txt", ".yaml", ".yml"}
HEAD = """# Thesis paper shared dependencies

The thesis paper lives under `thesis_paper/` so it can be handed over separately. It may READ the committed Paper A files below
(by path, never by a glob for "the most recent" file) and import nothing from the top-level `src/` package. Every file is pinned by
SHA256; text files are hashed with CRLF converted to LF, and the `.npz` is a Git LFS object (run `git lfs pull` in a fresh clone).

`thesis_paper/scripts/check_shared_dependencies.py` verifies this table and that no file under `thesis_paper/` imports `src.*`.
Run it before every thesis-paper commit. If a listed file changes, regenerate the table with `--write` in the same commit and say why.

| Path | SHA256 |
|---|---|
"""


def sha256(rel):
    """SHA256 of a repository file (LF-normalised for text)."""
    p = REPO / rel
    data = p.read_bytes()
    if p.suffix in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def wanted():
    """(path, hash) for every pinned file."""
    return [(p, sha256(p)) for p in (*pav.ARTIFACT_PATHS, SOURCE_DOCX)]


def src_imports():
    """Files under thesis_paper/ that import from the top-level src package."""
    bad = []
    for f in (REPO / "thesis_paper").rglob("*.py"):
        if re.search(r"(?m)^\s*(from|import)\s+src(\.|\s)", f.read_text(encoding="utf-8")):
            bad.append(str(f.relative_to(REPO)))
    return bad


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    cur = wanted()
    if args.write:
        MANIFEST.write_text(HEAD + "\n".join(f"| `{p}` | `{h}` |" for p, h in cur) + "\n", encoding="utf-8")
        print(f"wrote {MANIFEST} ({len(cur)} files)")
        return 0
    pinned = dict(re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", MANIFEST.read_text(encoding="utf-8")))
    bad = [f"{p}: {'not pinned' if p not in pinned else 'hash changed'}" for p, h in cur if pinned.get(p) != h]
    bad += [f"{p}: pinned but no longer read" for p in pinned if p not in dict(cur)]
    bad += [f"{f}: imports from src" for f in src_imports()]
    if bad:
        print("\n".join(bad))
        return 1
    print(f"OK: {len(cur)} pinned files match; no src imports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
