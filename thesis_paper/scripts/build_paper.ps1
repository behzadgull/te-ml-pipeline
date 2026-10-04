# Build a draft docx of the thesis paper from thesis_paper/paper/paper.md (tables, equations, figures, citations). Output goes to
# thesis_paper/paper/build/ (gitignored). Before building, the generated values and tables are checked against the committed artifacts,
# the pinned dependencies are verified, and the reused Paper A figures are compared with their sources. The draft may still contain
# [[PENDING: NAx]] markers; make_thesis_values.py --list-pending prints them.
# Citations are pandoc keys resolved against thesis_paper/paper/refs.bib with a numbered style (Nature CSL copied from Paper A; the
# journal's own style replaces it at submission).
# Toolchain: pandoc (set $env:PANDOC if it is not on PATH) and python (set $env:PYTHON). Run from the repository root.
# ";" separates resource-path entries on Windows.
param([switch]$Draft)
$ErrorActionPreference = "Stop"

$pandoc = if ($env:PANDOC) { $env:PANDOC } else { "pandoc" }
$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force thesis_paper/paper/build | Out-Null

& $python thesis_paper/scripts/make_thesis_values.py --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python thesis_paper/scripts/check_shared_dependencies.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python thesis_paper/scripts/import_paper_a_figures.py --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
# Standing rule of 2026-10-04: every number is GENERATED, CITED or DESIGN. The build fails on an unclassified number; -Draft prints the audit
# summary and builds anyway (the manuscript is not submittable until the audit passes).
if ($Draft) { & $python thesis_paper/scripts/audit_numbers.py --no-write } else { & $python thesis_paper/scripts/audit_numbers.py --strict --no-write }
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pandoc thesis_paper/paper/paper.md --resource-path="thesis_paper/paper;." --citeproc --bibliography=thesis_paper/paper/refs.bib `
  --csl=thesis_paper/paper/csl/nature.csl -o thesis_paper/paper/build/thesis_paper_draft.docx
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
