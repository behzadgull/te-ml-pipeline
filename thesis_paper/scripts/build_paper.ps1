# Build a draft docx of the thesis paper from thesis_paper/paper/paper.md, to check that the converted manuscript renders
# (tables, equations, figures). Output goes to thesis_paper/paper/build/ (gitignored).
# Toolchain: pandoc (set $env:PANDOC to the binary if it is not on PATH). Run from the repository root.
# The in-text references are plain numbered text, not pandoc citations, so no bibliography or CSL is used yet; refs.bib is filled
# when the reference list is converted. ";" separates resource-path entries on Windows.
$ErrorActionPreference = "Stop"

$pandoc = if ($env:PANDOC) { $env:PANDOC } else { "pandoc" }
New-Item -ItemType Directory -Force thesis_paper/paper/build | Out-Null

& $pandoc thesis_paper/paper/paper.md --resource-path="thesis_paper/paper;." -o thesis_paper/paper/build/thesis_paper_draft.docx
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
