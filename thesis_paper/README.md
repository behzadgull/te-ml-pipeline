# Thesis paper

The thesis manuscript (supervisor's decision, 2026-10-03) replaces Paper A as the main paper. Paper A's verified results move into it.
This folder follows the conventions of `paper_b/`: everything lives under `thesis_paper/` so that it can be handed over separately.

## Status

`paper/paper.md` is no longer a plain conversion. Phase 1 replaced every PAPER_A value with a generated marker (numbers read from committed
artifacts, with assertions that tie the prose to them), steps 2 to 5 added the NA4, NA5, NA8, NA9 results and the pandoc citations, and every spot
that depends on an analysis that has not run is marked `[[PENDING: NAx]]` (`python thesis_paper/scripts/make_thesis_values.py --list-pending`).
Do not edit generated text by hand: change the script and rerun it.

| Analysis | State |
|---|---|
| NA4 dataset statistics, NA5 cleaning diagnostics, NA8 error metrics, NA9 ESTM counts | done, in the paper (`results/na4`, `na5`, `na8`, `na9`) |
| NA10 JARVIS | data prepared and featurised, seen/unseen flagged (`results/na10`); no predictions |
| NA11 Materials Project screening | code written and tested on a JARVIS stand-in; the perovskite connectivity test validated (`results/na11_validation`); the Materials Project query itself needs `MP_API_KEY` and has not run |
| NA12 literature oxides | two values extracted, the rest need figure digitisation (`reports/na12_measured_values.csv`) |
| NA7 direct versus derived under random validation | done on Kaggle T4 x2, in the paper (`results/na7`) |
| NA2, NA3, NA6, NA13 | scripts ready and smoke-tested on Linux (`scripts/kaggle/`, cells in `KAGGLE_CELLS.md`); not yet run |
| NA1 | needs the department V100S (`scripts/gpu/`) |

Citations: all 62 references were checked against Crossref (and publisher pages for the four without a DOI): `reports/citation_verification.csv`;
numbers claimed in the text were checked against abstracts: `reports/claim_citation_check.csv`; the judgements and the changes they caused are
in `reports/literature_verdicts.csv`.

## Layout

| Path | What |
|---|---|
| `source/Perovskite_Thermoelectric_Manuscript.docx` | The supervisor's annotated manuscript, byte-identical to the file received (SHA256 in `SHARED_DEPENDENCIES.md`). Black = thesis, blue = rewritten, red = issue notes, yellow highlight = text under revision. Never edited. |
| `paper/paper.md` | The conversion. Red notes are kept as HTML comments (`<!-- RED: ... -->`) at their location; yellow highlights are `<mark>`; blue text is plain text. |
| `paper/refs.bib` | The 62 references plus Ho et al. 2026 (cited in the introduction), built from DOIs by `scripts/build_refs_thesis.py`; the text cites them as `[@key]`. |
| `figures/source_media/` | The 12 images of the source, extracted unchanged. |
| `reports/claim_inventory.csv` | One row per number or factual claim, and one per red note, each with a status. |
| `reports/claim_inventory_summary.md` | Counts per status, the list of new analyses with inputs and effort, the drop candidates, the decisions that change the inventory. |
| `scripts/` | Conversion and inventory (`convert_docx_to_md.py`, `build_claim_inventory.py`); values (`paper_a_values.py`, `thesis_values.py`, `make_thesis_values.py`); analyses (`na4_*`, `na5_*`, `na8_*`, `na9_*`, `na10_*`, `na11_*`); figures (`make_figures_thesis.py`, `import_paper_a_figures.py`); citations (`verify_citations.py`, `build_refs_thesis.py`); `check_shared_dependencies.py`; `build_paper.ps1` / `.sh`; `gpu/` (V100S setup and calibration). |
| `src/` | Empty: the analyses import the pinned top-level `src` modules listed in `SHARED_DEPENDENCIES.md`. |
| `results/` | One timestamped UTC folder per run, each with `run_config.json` (input SHA256s, git head, dirty files, script SHA256). |
| `SHARED_DEPENDENCIES.md` | The committed Paper A files this folder reads, pinned by SHA256; nothing is imported from the top-level `src/`. |

## Regenerating

From the repository root, with pandoc on PATH (or `$PANDOC`) and Git LFS objects pulled:

```
python thesis_paper/scripts/make_thesis_values.py --check      # generated values and tables are current
python thesis_paper/scripts/check_shared_dependencies.py       # pinned artifact, result and shared-code hashes
python thesis_paper/scripts/import_paper_a_figures.py --check  # reused Paper A figures match their sources
python thesis_paper/scripts/make_figures_thesis.py             # Figures 6, 7 and 9
powershell thesis_paper/scripts/build_paper.ps1                 # draft docx with citations
```

## Rules carried over from the project

- No hand-typed values: every number in the paper, figures or configs comes from a committed artifact or a script's output. In the inventory,
  `verified_value` is read from the artifacts by `paper_a_values.py`.
- Every new run records the dataset SHA256, the code commit and whether the working tree was clean; run folders are named with a UTC timestamp.
- A result is not "confirmed" until its producing checkpoint directory or file is named alongside it.
- Commit only when asked.

## Running the pending analyses

- Materials Project query (needs a key; never put it in a file): `set MP_API_KEY=...`, then
  `PYTHONPATH=C:/Users/choha/py_extra/mp_api python thesis_paper/scripts/na11_mp_query.py`, then
  `python thesis_paper/scripts/na11_mp_filter_featurize.py --query data/external/mp/mp_query_<stamp>.json`. mp-api 0.46.5 is installed outside the
  project environment because it needs newer numpy, pandas and pymatgen than the pinned ones.
- JARVIS data preparation (already run): `PYTHONPATH=C:/Users/choha/py_extra/jarvis_2026.6.12 python thesis_paper/scripts/na10_jarvis_prep.py`.
- V100S: `scripts/gpu/setup_v100s.sh`, then `scripts/gpu/calibrate_v100s.sh` (see their headers).
