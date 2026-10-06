# Department machine (Windows 10, 12-core Xeon Gold, 32 GB, V100S): nested stacking and the Paper B pooled chain

The scripts in this folder are self-contained: copy the whole folder to `C:\te_run\dept` (the clone is made by the setup script and sits at the pinned commit `13a0efd`, which predates these scripts).
Everything the runs write (venv, data, outputs, checkpoints, logs, backups) is under `C:\te_run` and **outside the clone**, because the harness refuses to start on a dirty tree.
All commands are Windows PowerShell 5.1 or PowerShell 7; the shell is opened in `C:\te_run`.

| File | Purpose |
|---|---|
| `setup_windows.ps1` | uv, the Python 3.12 venv with the Kaggle pins, the clone at `13a0efd`, the Paper B split files (git lfs), the CSV's size and SHA256, versions, nvidia-smi, a `device="cuda"` fit |
| `run_stacking.ps1` | `-Worker smoke`, `-Worker 1/2/3`, `-Worker all`: the nested stacking, 3 processes x 4 threads |
| `run_paper_b.ps1` | `-Mode smoke / pooled / specialist / control`: Paper B on the V100S |
| `verify_outputs.py` | before each restart: moves unreadable checkpoint files (possible after a power cut) to `C:\te_run\quarantine` so they are recomputed |
| `backup.ps1` | zips `out`, `ckpt` and `logs` to a second location, keeps the newest 7 |

## Work split and wall-time estimate

The 100 stacking units (4 targets x 5 repeats x 5 folds) go to three processes of about equal cost (`KAGGLE_CELLS.md` section 3c gives the per-fold model: about 58 / 48 / 36 / 40 minutes per outer fold for S / sigma / kappa / zT
at the Kaggle per-core speed, forest-dominated):

| Worker | Jobs (out-dir under `C:\te_run\out`) | Folds | Estimate at the Kaggle speed |
|---|---|---|---|
| 1 | `stk_S_all` (S, repeats 0-4) | 25 | 24 h |
| 2 | `stk_sigma_all` (sigma 0-4), `stk_kappa_r0`, `stk_zT_r0` | 25 + 5 + 5 | 20 + 3 + 3.3 = 26 h |
| 3 | `stk_kappa_r1_4`, `stk_zT_r1_4` | 20 + 20 | 12 + 13.3 = 25 h |

**Wall time: about 26 hours if one Xeon core with 4 threads is as fast as a Kaggle 4-core session; about 16 hours if it is as fast as the 8-core PC the timing note was measured on (1.64 times faster).** The Xeon's
clock is not known here; the first finished fold of each worker (printed in its log) gives the real number within an hour (compare with 58 / 48 / 36 / 40 min). The Paper B process shares the machine
(12 stacking threads already fill 12 cores), so both slow down by a few percent. This is an estimate from a model, not a measurement of this machine.

Memory: each stacking process holds the target's feature matrix, its outer-training copy and the forest being fitted, probably 6 to 8 GB at the peak (not measured; the Kaggle forest runs needed "a few GB").
Three of them plus Paper B (data loading about 2 GB) should fit in 32 GB, with little room to spare: watch `Get-Process python | Select Id, @{n='GB';e={[math]::Round($_.WorkingSet64/1GB,1)}}` during the first fold
of S (the largest); if the machine starts paging, stop worker 3 (`Ctrl+C` in its window; it resumes later) rather than let all of them thrash.

Paper B pooled (tuning_once + C0-C3, one worker on the V100S): the compute estimate prices it at about 53 GPU-hours on a T4 (`paper_b/reports/compute_estimate/20260929T111313/plans_comparison.csv`: 10.2 + 12.2 + 0.8 + 12.0 + 17.9)
plus about 4 CPU ridge-hours. The V100S speed relative to the T4 for these fits has not been measured, so the wall time is unknown; the first log lines give seconds per unit. At the T4 rate it would be about 2.2 days.
Paper B's units are ordered zT first, with the three device-control pairs first, so a partial run still holds the most valuable units.

## Steps, in order

1. **Prepare the machine.** Copy this folder to `C:\te_run\dept`, copy the snapfix CSV (`featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv`, 974,854,507 bytes) to `C:\te_run\data\`, and stop it sleeping:
   `powercfg /change standby-timeout-ac 0; powercfg /change hibernate-timeout-ac 0`.
2. **Setup:** `powershell -ExecutionPolicy Bypass -File C:\te_run\dept\setup_windows.ps1`. Expected last line: `== setup complete`. Send me the printed versions, the CSV line, nvidia-smi and the `device=cuda fit ok` line.
3. **Stacking smoke:** `powershell -ExecutionPolicy Bypass -File C:\te_run\dept\run_stacking.ps1 -Worker smoke`. Expected last line: `SMOKE OK` (about 2 minutes).
4. **Stacking, the real run:** `powershell -ExecutionPolicy Bypass -File C:\te_run\dept\run_stacking.ps1 -Worker all` (three minimised windows; logs `C:\te_run\logs\stk_*.log`; outputs `C:\te_run\out\stk_*`).
5. **Paper B GPU smoke, then the pooled chain, at the same time as step 4:**
   `powershell -ExecutionPolicy Bypass -File C:\te_run\dept\run_paper_b.ps1 -Mode smoke` (expected: `status: completed`), then
   `powershell -ExecutionPolicy Bypass -File C:\te_run\dept\run_paper_b.ps1 -Mode pooled` (log `C:\te_run\logs\paper_b_pooled.log`, progress `C:\te_run\ckpt\ckpt_gpu_pooled\status_worker0.json`).
6. **After a power cut or a closed window:** run steps 4 and 5 again, exactly the same commands (`-Worker all` and `-Mode pooled`). Finished jobs and units are skipped, unreadable files are quarantined first. Nothing restarts by itself after a reboot; log in and run them.
7. **Daily backup** to a second drive (replace `E:\te_backup`): once now, `powershell -ExecutionPolicy Bypass -File C:\te_run\dept\backup.ps1 -BackupDir E:\te_backup`; every day at 03:00:
   `schtasks /Create /TN te_backup /SC DAILY /ST 03:00 /TR "powershell -NoProfile -ExecutionPolicy Bypass -File C:\te_run\dept\backup.ps1 -BackupDir E:\te_backup"`.
8. **When the stacking is done** (all six `stk_*` folders have `status.json` with `"complete": true`; each also leaves a `stk_*.tar.gz` next to it): send me the six `.tar.gz` files and their SHA256 lines
   (`Get-FileHash -Algorithm SHA256 C:\te_run\out\*.tar.gz`), and I run `na2_stacking_analysis.py` and fill Table 7. Then start the Paper B specialists:
   `powershell -ExecutionPolicy Bypass -File C:\te_run\dept\run_paper_b.ps1 -Mode specialist` (4 workers, 1 thread each; its own directory `ckpt_cpu_specialist`), and, once the pooled chain is finished,
   `... -Mode control` (the three zT device-control pairs on the GPU; own directory `ckpt_gpu_control`).

## Notes

- `src/nested_cv.py` hard-codes `n_jobs=-1` for xgboost, LightGBM and the random forest; the thread limit is therefore set by the environment (`OMP_NUM_THREADS`, `LOKY_MAX_CPU_COUNT`, which joblib reads: checked locally that
  `n_jobs=-1` then resolves to 4) and by CPU affinity (4 distinct cores per worker). Each stacking log starts with the affinity that was applied.
- The stacking XGBoost runs on the CPU in this script, by design, so the V100S stays free for Paper B.
- The stacking bundles from this machine have `code commit 13a0efd`, the same as the Kaggle cells in section 3c; if a slice is later run on Kaggle instead, `na2_stacking_analysis.py` accepts the mix as long as
  the commit and the design parameters agree and every target, repeat and fold is covered once.
- The Windows `xgboost==2.0.3` wheel having CUDA support is expected, not checked here: step 2 prints the result of a real `device="cuda"` fit, and stops if it fails.
- A checkpoint written just before a power cut can come back empty or truncated on NTFS (no fsync in the harnesses); `verify_outputs.py` reads only files modified in the last 72 hours and moves the bad ones aside.
