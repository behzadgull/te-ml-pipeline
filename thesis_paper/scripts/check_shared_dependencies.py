"""
Check (or write) thesis_paper/SHARED_DEPENDENCIES.md: everything outside thesis_paper/ that the thesis paper reads, each pinned by SHA256.

Three kinds of pinned file:
  - committed Paper A artifacts read by paper_a_values.py (results, reports, frozen hyperparameters, the pinned source docx);
  - the results of the thesis paper's own analyses that the manuscript uses (thesis_values.NA_RUNS: results.json and run_config.json);
  - the top-level src/ modules its scripts import, found by following the imports (ast) from every script under thesis_paper/scripts/.
Text files are hashed with CRLF converted to LF, so a Windows checkout with core.autocrlf hashes the same as the committed blob;
binary files (the Git LFS .npz) are hashed as they are, and need `git lfs pull` in a fresh clone.

Usage (from the repository root):
    python thesis_paper/scripts/check_shared_dependencies.py            # verify; exit 1 on a missing file or a changed hash
    python thesis_paper/scripts/check_shared_dependencies.py --write    # regenerate the manifest (after a deliberate change)
"""

import argparse
import ast
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_a_values as pav  # noqa: E402
import thesis_values as tv  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "thesis_paper" / "SHARED_DEPENDENCIES.md"
SOURCE_DOCX = "thesis_paper/source/Perovskite_Thermoelectric_Manuscript.docx"
TEXT_SUFFIXES = {".json", ".py", ".md", ".csv", ".txt", ".yaml", ".yml"}
HEAD = """# Thesis paper shared dependencies

The thesis paper lives under `thesis_paper/` so it can be handed over separately. It READS the files below by explicit path (never a glob for
"the most recent" file) and imports only the `src/` modules listed under "Shared code". Every file is pinned by SHA256; text files are hashed with
CRLF converted to LF, and the `.npz` files are Git LFS objects (run `git lfs pull` in a fresh clone).

`thesis_paper/scripts/check_shared_dependencies.py` verifies this table. Run it before every thesis-paper commit. If a listed file changes,
regenerate the table with `--write` in the same commit and say why.

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


def src_imports(path):
    """Names (relative paths, e.g. 'src/nested_cv.py') of the top-level src modules a Python file imports."""
    out = set()
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module == "src":
                out |= {f"src/{a.name}.py" for a in node.names}
            elif node.module.startswith("src."):
                out.add("src/" + node.module.split(".")[1] + ".py")
        elif isinstance(node, ast.Import):
            out |= {"src/" + a.name.split(".")[1] + ".py" for a in node.names if a.name.startswith("src.")}
    return {m for m in out if (REPO / m).exists()}


def shared_code():
    """The closure of src modules imported (directly or through each other) by the thesis scripts."""
    todo = set()
    for f in (HERE).glob("*.py"):
        todo |= src_imports(f)
    seen = set()
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        todo |= src_imports(REPO / m)
    return sorted(seen)


def wanted():
    """(path, hash) for every pinned file, in manifest order."""
    new = []
    for name in tv.NA_RUNS:
        new += [tv.na_path(name), f"{tv.NA_RUNS[name]}/run_config.json"]
    paths = [*pav.ARTIFACT_PATHS, SOURCE_DOCX, *new, *tv.NA7_FILES, *shared_code()]
    return [(p, sha256(p)) for p in dict.fromkeys(paths)]


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    cur = wanted()
    if args.write:
        code = set(shared_code())
        rows = []
        for p, h in cur:
            kind = "shared code" if p in code else ("thesis analysis result" if p.startswith("thesis_paper/results/") else "Paper A artifact / source")
            rows.append(f"| `{p}` | `{h}` |  <!-- {kind} -->")
        MANIFEST.write_text(HEAD + "\n".join(r.replace("|  <!--", "| <!--") for r in rows) + "\n", encoding="utf-8")
        print(f"wrote {MANIFEST} ({len(cur)} files)")
        return 0
    pinned = dict(re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", MANIFEST.read_text(encoding="utf-8")))
    bad = [f"{p}: {'not pinned' if p not in pinned else 'hash changed'}" for p, h in cur if pinned.get(p) != h]
    bad += [f"{p}: pinned but no longer read" for p in pinned if p not in dict(cur)]
    if bad:
        print("\n".join(bad))
        return 1
    print(f"OK: {len(cur)} pinned files match ({len(shared_code())} shared src modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
