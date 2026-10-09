# thesis_paper/tests

Self-tests of analysis scripts. They run on SYNTHETIC input: every number they print or write is synthetic, is not a result, and is never used by the paper.

| File | Tests |
|---|---|
| `na13_analysis_selftest.py` | `scripts/na13_analysis.py`: fake NA13 bundles built from the real folds and clusters with random predictions, a positive run, a rerun for determinism, and the refusal conditions pre-registered in `docs/decisions.md` (2026-10-09, item 5 and the analysis' header). |

Run from the repository root, for example `python thesis_paper/tests/na13_analysis_selftest.py` (about ten minutes; needs the snapfix CSV in `data/processed/`).
