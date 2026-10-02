#!/usr/bin/env bash
# Build the Paper B supervisor preview: paper_b/paper/build/PaperB_preview.docx. See build_paper_b.ps1 for the details.
# Run from the repository root. ":" separates resource-path entries outside Windows (not run on Linux; checked with bash -n only).
set -euo pipefail

PANDOC="${PANDOC:-pandoc}"
PYTHON="${PYTHON:-python}"
mkdir -p paper_b/paper/build

PYTHONIOENCODING=utf-8 "$PYTHON" paper_b/scripts/make_tables_b.py --check
PYTHONIOENCODING=utf-8 "$PYTHON" paper_b/scripts/make_refs_b.py --check

"$PANDOC" paper_b/paper/paper.md --resource-path="paper_b/paper:." --citeproc --csl=paper_b/paper/csl/nature.csl \
  --reference-doc=paper_b/paper/templates/reference.docx --lua-filter=paper_b/paper/templates/frontmatter.lua \
  -o paper_b/paper/build/PaperB_preview.docx
