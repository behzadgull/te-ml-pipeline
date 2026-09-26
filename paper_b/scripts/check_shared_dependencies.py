"""
Check paper_b/SHARED_DEPENDENCIES.md against the repository.

Two checks, run from the repository root:
  1. Every shared module listed in the manifest exists and its SHA256
     matches the manifest. The hash is taken over the file with CRLF
     converted to LF, so a Windows checkout with core.autocrlf=true hashes
     the same as the committed blob.
  2. No Python file under paper_b/ imports a top-level `src.*` module
     other than the ones the manifest lists.

Exits 1 with a message per failure, 0 if both checks pass.
"""

import ast
import hashlib
import re
import sys
from pathlib import Path

MANIFEST = Path("paper_b/SHARED_DEPENDENCIES.md")
ROW = re.compile(r"^\|\s*`(src/[A-Za-z0-9_]+\.py)`\s*\|\s*`([0-9a-f]{64})`")


def manifest_hashes():
    """Return {module path: SHA256} for the shared-code rows of the manifest."""
    found = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        match = ROW.match(line)
        if match:
            found[match.group(1)] = match.group(2)
    return found


def lf_sha256(path):
    """SHA256 of the file's bytes with CRLF normalised to LF."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def imported_src_modules(py_file):
    """Return the set of `src.<name>` modules a Python file imports."""
    tree = ast.parse(Path(py_file).read_text(encoding="utf-8"), filename=str(py_file))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
            if node.module == "src":
                names = [f"src.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        for name in names:
            if name == "src" or name.startswith("src."):
                modules.add(".".join(name.split(".")[:2]))
    return modules


def main():
    """Run both checks and return the process exit code."""
    problems = []
    hashes = manifest_hashes()
    if not hashes:
        problems.append(f"no shared-module rows found in {MANIFEST}")
    for module, expected in hashes.items():
        if not Path(module).exists():
            problems.append(f"{module}: listed in the manifest but missing")
        elif lf_sha256(module) != expected:
            problems.append(f"{module}: SHA256 differs from the manifest")
    allowed = {Path(module).stem for module in hashes}
    for py_file in sorted(Path("paper_b").rglob("*.py")):
        for module in imported_src_modules(py_file):
            if module.split(".")[1] not in allowed:
                problems.append(f"{py_file}: imports {module}, not in the manifest")
    for problem in problems:
        print("FAIL:", problem)
    if not problems:
        print(f"OK: {len(hashes)} shared modules match the manifest; no other src imports.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
