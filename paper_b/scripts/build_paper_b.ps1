# Build the Paper B supervisor preview from paper_b/paper/paper.md: paper_b/paper/build/PaperB_preview.docx
# (journal-neutral Word manuscript, Nature numbered citation style, reference.docx template).
# Mirrors scripts/build_paper.ps1 (Paper A); every input lives under paper_b/paper/ so Paper B can be handed over separately,
# except the figures (paper_b/figures/), which are referenced from the repository root.
# Toolchain: pandoc (set $env:PANDOC to the binary if it is not on PATH). Run from the repository root.
# Before building, the generated values, tables and references are checked against the committed artifacts.
# ";" separates resource-path entries on Windows.
$ErrorActionPreference = "Stop"

$pandoc = if ($env:PANDOC) { $env:PANDOC } else { "pandoc" }
$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
New-Item -ItemType Directory -Force paper_b/paper/build | Out-Null

$env:PYTHONIOENCODING = "utf-8"
& $python paper_b/scripts/make_tables_b.py --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python paper_b/scripts/make_refs_b.py --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pandoc paper_b/paper/paper.md --resource-path="paper_b/paper;." --citeproc --csl=paper_b/paper/csl/nature.csl `
  --reference-doc=paper_b/paper/templates/reference.docx --lua-filter=paper_b/paper/templates/frontmatter.lua `
  -o paper_b/paper/build/PaperB_preview.docx
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
