# NA13 convergence calibration run: one fold, from a dirty tree. NOT a result.

- **What**: the first run of `scripts/na13_convergence_check.py`, on one fold only (kappa, repeat 0, fold 0) with the ladder `max_iter` 2000 / 20,000 / 200,000. It showed 13 `ConvergenceWarning`s at 2000 iterations and none at 20,000 and 200,000, with the same chosen
  alpha and 37 of 39 final features shared between 2000 and 20,000 iterations.
- **Why it is kept**: the 2026-10-07 amendment in `docs/decisions.md` cites it as the record of (a) these single-fold observations, (b) the 2,865 s versus 508 s run times at 200,000 versus 20,000 iterations that led the 20-fold diagnostic to use the ladder 2000 / 20,000 only, and (c) the
  error it exposed in the script: its independent dual-gap check compared the gap returned by `lasso_path` (divided by the number of rows) with sklearn's tolerance without multiplying by the number of rows, so **every `n_nonconverged_by_gap` and `nonconverged_alpha_positions` value in `diagnostic.json` is void**.
  The warnings, chosen alphas, selected sets, fold R2 and timings in it are valid for that one fold.
- **Provenance**: `run_config.json` records `tree_clean` false (the script and `docs/decisions.md` were uncommitted when it ran), code `54ee709`, the snapfix CSV `d9fc1e5d...e489`. Nothing in the paper is computed from it.
- **The result run** is the 20-fold diagnostic (2000 and 20,000 iterations), rerun from the clean commit that contains the corrected script; it will be a separate folder.
