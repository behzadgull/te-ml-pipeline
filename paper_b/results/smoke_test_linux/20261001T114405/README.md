# Kaggle Linux smoke test, 20261001T114405

- **Platform**: Kaggle CPU notebook (no GPU), Linux.
- **Commit**: `6b57be05ad32fe6e9822bdf641746cb05c9046fd`.
- **Date**: 2026-10-01.
- **Result**: `all_passed: true`, all 17 checks.

This confirms the dependency-safe scheduling and crash-safe-session fix
(commit `6b57be0`) on an actual Linux host: check_9's accounting (fixed
the same commit, after the local smoke test itself caught it) passes,
check_16 proves the adversarial dependency-skip and the real
`--workers 2` crash scenario both resolve cleanly, and check_17 proves a
crash writes `status: "crashed"` with a recoverable manifest and that a
following `--restore-from` finishes the session.

`report_linux.json` is committed as downloaded from Kaggle, verified
byte-identical before committing (SHA256
`e4dd3368a3b95fbd74290cc96ad3581599c5a27fa503b54c65c24e5294f95a78`,
matching what Kaggle itself reported).
