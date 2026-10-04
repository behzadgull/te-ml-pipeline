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
# Standing rule of 2026-10-04: the build fails on an unclassified number; DRAFT=1 prints the audit summary and builds anyway.
if [ "${DRAFT:-0}" = "1" ]; then PYTHONIOENCODING=utf-8 "$PYTHON" thesis_paper/scripts/audit_numbers.py --no-write; else PYTHONIOENCODING=utf-8 "$PYTHON" thesis_paper/scripts/audit_numbers.py --strict --no-write; fi
"$PANDOC" thesis_paper/paper/paper.md --resource-path="thesis_paper/paper:." --citeproc --bibliography=thesis_paper/paper/refs.bib --csl=thesis_paper/paper/csl/nature.csl -o thesis_paper/paper/build/thesis_paper_draft.docx
