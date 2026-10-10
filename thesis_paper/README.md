# Thesis paper

The thesis manuscript (supervisor's decision, 2026-10-03) replaces Paper A as the main paper. Paper A's verified results move into it.
This folder follows the conventions of `paper_b/`: everything lives under `thesis_paper/` so that it can be handed over separately.

## Status

State of 2026-10-10 (decisions in `docs/decisions.md`, 2026-10-04 to 2026-10-10). `paper/paper.md` is generated: every number is a marker filled from a committed artifact (GENERATED), a verified citation with
page or table (CITED, `docs/number_registry.csv`) or a registered design constant (DESIGN, `docs/design_constants.csv`); `scripts/audit_numbers.py` classifies every number. **57 numbers are
unclassified, all of them literature values (Sections 1 to 3.2.2) that wait for the source PDFs** (`reports/literature_value_verification.csv`, `reports/unclassified_numbers_proposal.csv`; `source/pdfs/` holds only its README);
`make_thesis_values.py --check` passes with 2 `[[PENDING]]` markers, both NA12. Do not edit generated text by hand: change the script and rerun `scripts/make_thesis_values.py`.

| Analysis | State |
|---|---|
| NA4 dataset statistics, NA5 cleaning diagnostics, NA8 error metrics, NA7 direct vs derived under random validation | done, in the paper |
| NA9 ESTM counts, NA10 JARVIS featurisation | rerun from a clean commit (`results/na9/20261010T173322`, `results/na10/20261010T173618`); outputs identical to the first runs, which had a dirty tree |
| Final models, NA6 carrier-type classifier (and the same-folds sign comparison), NA10 JARVIS analysis, NA3 SHAP (25 folds) and the per-row rerun for Figure 11 | done on Kaggle T4 (`results/final_a`, `final_b`, `na6_*`, `na10_analysis`, `na3_a`, `na3_b`, `na3_rows_a`, `na3_rows_b`), in the paper |
| External validation (ESTM, teMatDb) | rescored with the saved final models by `scripts/external_rescore.py`, run from a clean commit (`results/external_rescore/20261010T124804`): Table 8, Figure 9 and Table 8b (teMatDb strata a0, a, b, sample-level intervals); the earlier in-process refit (`results/external_snapfix`) is kept only for the fit-to-fit sentence of Section 4.2 |
| NA11 Materials Project screening | done: query, filters, 30 candidates (9 seen, 21 unseen clusters), the pre-registered shortlist rule applied, the thermal-limits check and the literature labels committed before the predictions were looked at (`results/na11_*`, `docs/candidate_literature_labels.csv`) |
| NA1 nested CV | done on Kaggle (`results/na1_a`, `na1_b`); the paired comparison is `results/na1_comparison`, in the paper |
| NA2 algorithm comparison | done: XGBoost, LightGBM, the fixed random forest and the nested stack (ten Kaggle CPU notebooks at `13a0efd`, `results/na2_stk_*`, `na2_stacking_preds`), paired on identical folds in `results/na2_comparison/20261009T190259`; the weights and the correlation of the base predictions are `results/na2_stack_weights/20261010T103738`. Table 7 is filled: the stack is not distinguishable from the unweighted mean of the three |
| NA13 feature selection (the thesis's Table 3 pipeline, inside each outer fold) | done on spcai3 (V100S, cuda) at `aad78f1`: `results/na13_S`, `na13_sigma`, `na13_kappa`, `na13_zT`; the paired analysis `results/na13_analysis/20261009T190036` (the selection lowers R2 on every target, intervals exclude zero), the convergence diagnostic `results/na13_convergence/20261008T171143`, the post hoc temperature_bin refit `results/na13_temperature/20261009T194925Z` (exploratory) and a CPU platform record `results/na13_platform_record`; in the paper (Section 4.1.5, Table 7b) |
| NA12 literature validation of the shortlist | **pending**: the pre-registered design (decisions 2026-10-05, B) and `scripts/na12_literature_comparison.py` exist; `docs/na12_literature_values.csv` is empty until the source PDFs are in `source/pdfs/` (DOIs in its README); Table 12 and Section 4.5.1 are placeholders (2 markers) |
| Paper B | on hold (it resumes after the thesis paper); the department machine's scripts are in `scripts/dept/` and assume the 12-core V100S machine (spcai3), which is not what the department VM currently shows (6 cores, no GPU visible) |

Citations: all 62 references were checked against Crossref (and publisher pages for the four without a DOI): `reports/citation_verification.csv`;
numbers claimed in the text were checked against abstracts: `reports/claim_citation_check.csv`; the judgements and the changes they caused are
in `reports/literature_verdicts.csv`.

## Layout

| Path | What |
|---|---|
| `source/Perovskite_Thermoelectric_Manuscript.docx` | The supervisor's annotated manuscript, byte-identical to the file received (SHA256 in `SHARED_DEPENDENCIES.md`). Black = thesis, blue = rewritten, red = issue notes, yellow highlight = text under revision. Never edited. |
| `paper/paper.md` | The manuscript: converted from the source, then rewritten, with generated values and table blocks (`make_thesis_values.py`). One red note (the AI-use declaration) and one TODO (an e-mail address the author supplies) remain as HTML comments; no yellow highlight (`<mark>`) remains. |
| `paper/refs.bib` | 74 entries, of which 72 are cited in `paper.md` as `[@key]` (`hira2018temperature` and `kanas2022tuning` are not). The first 62 are the manuscript's references, built from DOIs by `scripts/build_refs_thesis.py` and checked in `reports/citation_verification.csv`; the entries added since (Ho et al. 2026, the thermal-limit and teMatDb references and others) are entered by hand and are not in that report. |
| `figures/source_media/` | The 12 images of the source, extracted unchanged. |
| `reports/claim_inventory.csv` | One row per number or factual claim, and one per red note, each with a status. |
| `reports/claim_inventory_summary.md` | Counts per status, the list of new analyses with inputs and effort, the drop candidates, the decisions that change the inventory. |
| `scripts/` | Conversion and inventory (`convert_docx_to_md.py`, `build_claim_inventory.py`); values (`paper_a_values.py`, `thesis_values.py`, `make_thesis_values.py`, `number_list_na2_na13.py`); analyses (`na4_*`, `na5_*`, `na6_*`, `na8_*`, `na9_*`, `na10_*`, `na11_*`, `na12_*`, `na13_*`, `na2_*`, `model_comparison.py`, `external_rescore.py`); figures (`make_figures_thesis.py`, `import_paper_a_figures.py`); citations (`verify_citations.py`, `build_refs_thesis.py`); checks (`audit_numbers.py`, `check_shared_dependencies.py`, `thesis_carryover_check.py`); `build_paper.ps1` / `.sh`; `gpu/` (V100S setup and calibration), `kaggle/` (the Kaggle sessions and `KAGGLE_CELLS.md`), `dept/` (the department machine). |
| `src/` | Empty: the analyses import the pinned top-level `src` modules listed in `SHARED_DEPENDENCIES.md`. |
| `results/` | One timestamped UTC folder per run, each with `run_config.json` (input SHA256s, git head, dirty files, script SHA256). |
| `SHARED_DEPENDENCIES.md` | The committed Paper A files this folder reads, pinned by SHA256; nothing is imported from the top-level `src/`. |

## Regenerating

From the repository root, with pandoc on PATH (or `$PANDOC`) and Git LFS objects pulled:

```
python thesis_paper/scripts/make_thesis_values.py --check      # generated values and tables are current
python thesis_paper/scripts/check_shared_dependencies.py       # pinned artifact, result and shared-code hashes
python thesis_paper/scripts/import_paper_a_figures.py --check  # reused Paper A figures match their sources
python thesis_paper/scripts/make_figures_thesis.py             # Figures 4, 6, 7, 9, 10, 11 and 12 (5 and 8 come from import_paper_a_figures.py, 1 to 3 are source images)
python thesis_paper/scripts/audit_numbers.py                   # classify every number; the unclassified count is in the printed summary
powershell thesis_paper/scripts/build_paper.ps1                 # draft docx with citations
```

## Rules carried over from the project

- No hand-typed values: every number in the paper, figures or configs comes from a committed artifact or a script's output. In the inventory,
  `verified_value` is read from the artifacts by `paper_a_values.py`.
- Every new run records the dataset SHA256, the code commit and whether the working tree was clean; run folders are named with a UTC timestamp.
- A result is not "confirmed" until its producing checkpoint directory or file is named alongside it.
- Commit only when asked.

## Running the analyses

- Materials Project query (needs a key; never put it in a file): `set MP_API_KEY=...`, then
  `PYTHONPATH=C:/Users/choha/py_extra/mp_api python thesis_paper/scripts/na11_mp_query.py`, then
  `python thesis_paper/scripts/na11_mp_filter_featurize.py --query data/external/mp/mp_query_<stamp>.json`. mp-api 0.46.5 is installed outside the
  project environment because it needs newer numpy, pandas and pymatgen than the pinned ones.
- JARVIS data preparation (already run): `PYTHONPATH=C:/Users/choha/py_extra/jarvis_2026.6.12 python thesis_paper/scripts/na10_jarvis_prep.py`.
- Kaggle sessions: `scripts/kaggle/KAGGLE_CELLS.md` (every cell pinned to a clean commit, the uv Python 3.12 venv with the exact pins). Department machine: `scripts/dept/README.md`.
- External validation of the final models (ESTM and teMatDb; needs `final_a.tar.gz`, whose SHA256 is checked, and a clean tree):
  `python thesis_paper/scripts/external_rescore.py --models-tar <path>/final_a.tar.gz`; output in `results/external_rescore/<UTC stamp>/`.
- NA12 (pending): `python thesis_paper/scripts/na12_literature_comparison.py` stops with exit code 3 while `docs/na12_literature_values.csv` is empty. It compares the out-of-fold
  predictions of the committed chemistry-cluster CV (not the final models of Table 11), at the exact-composition and cluster levels (decisions 2026-10-05, B).
