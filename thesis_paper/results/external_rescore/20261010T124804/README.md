# External validation of the thesis paper's final models (ESTM and teMatDb)

Written by `scripts/external_rescore.py`; see its docstring and `docs/decisions.md` (2026-10-10). The saved boosters of `results/final_a` score ESTM (two passes never pooled, in-support rule) and teMatDb (strata a0, a, b; stratum b is small and is described qualitatively). Smear factors for the back-transform are the Duan factors of the committed out-of-fold predictions of the chemistry-cluster rung. Nothing is refitted.
