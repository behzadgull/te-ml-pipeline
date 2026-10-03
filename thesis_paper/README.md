# Thesis paper

The thesis manuscript (supervisor's decision, 2026-10-03) replaces Paper A as the main paper. Paper A's verified results move into it.
This folder follows the conventions of `paper_b/`: everything lives under `thesis_paper/` so that it can be handed over separately.

## Status

Conversion and inventory only. **`paper/paper.md` is the supervisor's manuscript as converted, with no text edits.** Nothing in it has
been replaced by a Paper A value yet; `reports/claim_inventory.csv` says which value replaces which.

## Layout

| Path | What |
|---|---|
| `source/Perovskite_Thermoelectric_Manuscript.docx` | The supervisor's annotated manuscript, byte-identical to the file received (SHA256 in `SHARED_DEPENDENCIES.md`). Black = thesis, blue = rewritten, red = issue notes, yellow highlight = text under revision. Never edited. |
| `paper/paper.md` | The conversion. Red notes are kept as HTML comments (`<!-- RED: ... -->`) at their location; yellow highlights are `<mark>`; blue text is plain text. |
| `paper/refs.bib` | Empty until the 62 numbered references are converted to citation keys. |
| `figures/source_media/` | The 12 images of the source, extracted unchanged. |
| `reports/claim_inventory.csv` | One row per number or factual claim, and one per red note, each with a status. |
| `reports/claim_inventory_summary.md` | Counts per status, the list of new analyses with inputs and effort, the drop candidates, the decisions that change the inventory. |
| `scripts/` | `convert_docx_to_md.py`, `build_claim_inventory.py`, `paper_a_values.py` (verified Paper A values, read from artifacts), `check_shared_dependencies.py`, `build_paper.ps1` / `.sh`. |
| `src/`, `results/` | Empty: code and results of new runs will go here. |
| `SHARED_DEPENDENCIES.md` | The committed Paper A files this folder reads, pinned by SHA256; nothing is imported from the top-level `src/`. |

## Regenerating

From the repository root, with pandoc on PATH (or `$PANDOC`) and Git LFS objects pulled:

```
python thesis_paper/scripts/convert_docx_to_md.py --check      # paper.md matches a fresh conversion (fails once the text is edited)
python thesis_paper/scripts/build_claim_inventory.py           # rewrites reports/claim_inventory.*
python thesis_paper/scripts/check_shared_dependencies.py       # artifact hashes and the no-import rule
```

## Rules carried over from the project

- No hand-typed values: every number in the paper, figures or configs comes from a committed artifact or a script's output. In the inventory,
  `verified_value` is read from the artifacts by `paper_a_values.py`.
- Every new run records the dataset SHA256, the code commit and whether the working tree was clean; run folders are named with a UTC timestamp.
- A result is not "confirmed" until its producing checkpoint directory or file is named alongside it.
- Commit only when asked.
