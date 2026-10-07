# Thesis paper

The thesis manuscript (supervisor's decision, 2026-10-03) replaces Paper A as the main paper. Paper A's verified results move into it.
This folder follows the conventions of `paper_b/`: everything lives under `thesis_paper/` so that it can be handed over separately.

## Status

State of 2026-10-07 (decisions in `docs/decisions.md`, 2026-10-04 to 2026-10-06). `paper/paper.md` is generated: every number is a marker filled from a committed artifact (GENERATED), a verified citation with
page or table (CITED, `docs/number_registry.csv`) or a registered design constant (DESIGN, `docs/design_constants.csv`); `scripts/audit_numbers.py` classifies every number and the build fails if the
unclassified count rises (111 now, to fall as the source PDFs arrive). Do not edit generated text by hand: change the script and rerun `scripts/make_thesis_values.py`.

| Analysis | State |
|---|---|
| NA4 dataset statistics, NA5 cleaning diagnostics, NA8 error metrics, NA9 ESTM counts, NA7 direct vs derived under random validation | done, in the paper |
| Final models, NA6 carrier-type classifier (and the same-folds sign comparison), NA10 JARVIS, NA3 SHAP (25 folds) and the per-row rerun for Figure 11 | done on Kaggle (`results/final_b`, `na6_*`, `na10_analysis`, `na3_a`, `na3_b`, `na3_rows_a`, `na3_rows_b`), in the paper |
| NA11 Materials Project screening | done: query, filters, 30 candidates (9 seen, 21 unseen clusters), the pre-registered shortlist rule applied, the thermal-limits check and the literature labels committed before the predictions were looked at (`results/na11_*`, `docs/candidate_literature_labels.csv`) |
| NA1 nested CV | done on Kaggle (`results/na1_a`, `na1_b`); the paired comparison with the frozen-hyperparameter rung is `results/na1_comparison`, in the paper |
| NA2 algorithm comparison | XGBoost (frozen), LightGBM (tuned) and the fixed random forest for all four targets are done and paired on identical folds (`results/na2_lightgbm`, `na2_rf_*`, `na2_comparison/20261006T132724`); Table 7 is filled except the stack. The mean of the three is reported. **The nested stack is running**: ten Kaggle CPU notebooks at commit `13a0efd` (`KAGGLE_CELLS.md` section 3c); results go through `scripts/na2_stacking_analysis.py` into Table 7 and its 8 pending markers |
| NA12 literature validation of the shortlist | the pre-registered design and `scripts/na12_literature_comparison.py` exist; `docs/na12_literature_values.csv` is empty until the source PDFs are in `source/pdfs/` (DOIs in its README); Table 12 and the Section 4.5.1 text are pending (2 markers) |
| NA13 feature selection (the thesis's Table 3 pipeline, inside each outer fold) | script ready and smoke-tested on Linux; **not yet run**; cells in `KAGGLE_CELLS.md` section 3d; no pending marker in the paper yet |
| Paper B | on hold (it resumes after the thesis paper); the department machine's scripts are in `scripts/dept/` and assume the 12-core V100S machine, which is not what the department VM currently shows (6 cores, no GPU visible) |

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
| `scripts/` | Conversion and inventory (`convert_docx_to_md.py`, `build_claim_inventory.py`); values (`paper_a_values.py`, `thesis_values.py`, `make_thesis_values.py`); analyses (`na4_*`, `na5_*`, `na8_*`, `na9_*`, `na10_*`, `na11_*`); figures (`make_figures_thesis.py`, `import_paper_a_figures.py`); citations (`verify_citations.py`, `build_refs_thesis.py`); `check_shared_dependencies.py`; `build_paper.ps1` / `.sh`; `gpu/` (V100S setup and calibration), `kaggle/` (the Kaggle sessions and `KAGGLE_CELLS.md`), `dept/` (the department machine). |
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
- Kaggle sessions: `scripts/kaggle/KAGGLE_CELLS.md` (every cell pinned to a clean commit, the uv Python 3.12 venv with the exact pins). Department machine: `scripts/dept/README.md`.
