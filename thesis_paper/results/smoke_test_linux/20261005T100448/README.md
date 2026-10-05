# Kaggle Linux smoke test of code commit 7529e7c (the commit pinned for every remaining session)

- **Platform**: Kaggle CPU notebook (Accelerator None), Linux, Python 3.12.3 (uv venv).
- **Code commit**: `7529e7cfb7f14341994d1f57123e11edc75443e4`; the report records `tree_clean: true`, `allow_dirty: false`.
- **Date**: 2026-10-05 (folder timestamp is the download's modification time, UTC).
- **Result**: `all_passed: true`, all 44 checks ok.
- **Report**: `report_linux.json` is the file downloaded from Kaggle (saved by the browser as `smoke_report.txt`; the content is JSON), copied byte for byte. Its SHA256,
  `d6edbbf7bab89bc864cf3db8fa95ea322a37248d7eb7fcd2bc352a76ab566868`, equals the value Kaggle printed after the run and the value computed here on the downloaded file and on this copy.

Compared with the proof for `b2ad3cf` (`../20261004T085233`, 34 checks), this run also covers, on Linux and without `--allow-dirty`: the fixed-setting random forest of NA2 (3 units, no tuning
units, the fixed set recorded in the run identity: `na2_fixed_rf_has_no_tuning_units_and_records_the_set`), `na3_rows.py` (6 units) and `na1_nested_cv.py` (9 units), each with a manifest whose
SHA256 values match the files and a run_config with the dataset SHA256, git HEAD and `tree_clean` true; the earlier checks (resume, merge, tamper and parameter refusals, wrong-commit refusal,
zero time budget) pass again.

Not covered (CPU notebook): the `device="cuda"` path of any script, and the JARVIS scoring unit of the final-models script.
