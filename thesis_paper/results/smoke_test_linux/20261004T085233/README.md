# Kaggle Linux smoke test, 20261004T085233

- **Platform**: Kaggle CPU notebook (Accelerator None), Linux, Python 3.12.3 (uv venv, per the report's `environment.python`).
- **Code commit**: `b2ad3cf83ff33a15de294bf8db22d8f1d5e40ed5`; the report records `tree_clean: true`, `allow_dirty: false`.
- **Date**: 2026-10-04 (folder timestamp is the download's modification time, UTC).
- **Result**: `all_passed: true`, all 34 checks.

What it covers (`thesis_paper/scripts/kaggle/smoke_test_kaggle.py`, run without `--allow-dirty`, the way the real sessions run): NA7, NA2 random
forest and LightGBM, NA3, NA6, NA13 and the final models each run end to end on about 3,000 rows and write a complete status, a manifest whose
SHA256 values match the files, and a run_config with the dataset SHA256, git HEAD and `tree_clean` true; a script refuses to start without
`--expect-commit` and with a wrong one; a killed NA7 resumes from a directory and from a tar.gz with nothing recomputed and results identical to an
uninterrupted run; two NA7 shards merge into identical results; a restore with different parameters or a tampered unit is refused; a zero time budget
stops cleanly and is continued.

Not covered: the `device="cuda"` path (CPU notebook), the JARVIS scoring unit of the final-models script (its CSV is not in the clone, so that run
had 4 units, not 5), and the per-package versions inside the venv (the report records the Python version only; the pip probe printed them in Cell 1).
The Kaggle-side count of checks was reported as 33; the file holds 34 entries, all ok.

`report_linux.json` is the file downloaded from Kaggle (saved by the browser as `smoke_report.txt`; the content is JSON), copied byte for byte.
Its SHA256, `46dede140dde745cacca8fa2bd8c399a5a9a55cf58eac36bd28d48f64b6229cb`, equals the value Kaggle printed after the run and the value
computed here on the downloaded file and on this copy.
