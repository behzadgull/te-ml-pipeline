#!/usr/bin/env bash
# Build a draft docx of the thesis paper: thesis_paper/paper/build/thesis_paper_draft.docx. See build_paper.ps1 for the details.
# Run from the repository root. ":" separates resource-path entries outside Windows (not run on Linux; checked with bash -n only).
set -euo pipefail

PANDOC="${PANDOC:-pandoc}"
PYTHON="${PYTHON:-python}"
mkdir -p thesis_paper/paper/build
PYTHONIOENCODING=utf-8 "$PYTHON" thesis_paper/scripts/make_thesis_values.py --check
PYTHONIOENCODING=utf-8 "$PYTHON" thesis_paper/scripts/check_shared_dependencies.py
PYTHONIOENCODING=utf-8 "$PYTHON" thesis_paper/scripts/import_paper_a_figures.py --check
"$PANDOC" thesis_paper/paper/paper.md --resource-path="thesis_paper/paper:." --citeproc --bibliography=thesis_paper/paper/refs.bib --csl=thesis_paper/paper/csl/nature.csl -o thesis_paper/paper/build/thesis_paper_draft.docx
