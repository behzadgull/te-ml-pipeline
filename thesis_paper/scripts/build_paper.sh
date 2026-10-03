#!/usr/bin/env bash
# Build a draft docx of the thesis paper: thesis_paper/paper/build/thesis_paper_draft.docx. See build_paper.ps1 for the details.
# Run from the repository root. ":" separates resource-path entries outside Windows (not run on Linux; checked with bash -n only).
set -euo pipefail

PANDOC="${PANDOC:-pandoc}"
mkdir -p thesis_paper/paper/build
"$PANDOC" thesis_paper/paper/paper.md --resource-path="thesis_paper/paper:." -o thesis_paper/paper/build/thesis_paper_draft.docx
