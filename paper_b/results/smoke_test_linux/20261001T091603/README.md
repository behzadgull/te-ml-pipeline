# Kaggle Linux smoke test, 20261001T091603

- **Platform**: Kaggle CPU notebook (no GPU), Linux. The only full run of
  `smoke_test_lofo.py`'s 15 checks on a non-Windows host so far -- the
  local smoke test in `paper_b/results/smoke_test/` always runs on this
  project's Windows dev machine, which cannot reproduce any of the three
  Windows-vs-Linux bugs this run exists to confirm are fixed (the in-pair
  time budget's `time.time()` cross-process deadline, the splits
  manifest's Windows-CRLF-hashed SHA256s, and the `fork`-vs-`spawn`
  multiprocessing context mismatch -- see `paper_b/docs/
  Paper_B_Objectives_Methodology_v3_FROZEN.md` section 8.7 for all three).
- **Commit**: `95a6e540519fdd0a399f59b7a0e5f4b860b7d85c`.
- **Date**: 2026-10-01.
- **Result**: `all_passed: true`, all 15 checks, including check_15's real
  `--workers 2` multiprocessing.Pool run -- the one check that could
  actually hit the fork/spawn crash, and didn't.

Files:
- `report_linux.json` -- the run's own `report.json`, copied as-is.
- `smoke_test_linux_full.log` -- full stdout/stderr of the run, including
  Python's own `tarfile` `DeprecationWarning` on `extractall()` without a
  `filter` argument (PEP 706) -- fixed the same day this run's output was
  committed (`filter="data"` added to both `extractall()` call sites,
  `lofo_paperb.py`'s `restore_from` and this test's own tamper check).
