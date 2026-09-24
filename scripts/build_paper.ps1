# Build the manuscript outputs from paper/paper.md (docx and self-contained html).
# Toolchain: pandoc 3.9, installed via pypandoc_binary (pip install pypandoc_binary; the wheel bundles
# the pandoc binary). Set $env:PANDOC to that binary's path if pandoc is not on PATH.
# Run from the repository root. Outputs go to paper/build/ (gitignored).
# The resource path holds the directory of refs.bib (paper) and the repository root (figures/).
# ";" separates resource-path entries on Windows.
$ErrorActionPreference = "Stop"

$pandoc = if ($env:PANDOC) { $env:PANDOC } else { "pandoc" }
New-Item -ItemType Directory -Force paper/build | Out-Null

& $pandoc paper/paper.md --resource-path="paper;." --citeproc -o paper/build/paper.docx
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $pandoc paper/paper.md --resource-path="paper;." --citeproc --standalone --embed-resources -o paper/build/paper.html
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
