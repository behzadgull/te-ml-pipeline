# Scripts provenance

Orchestration scripts that touch the model, produce a results artifact, or
compute a number cited in `CLAUDE.md`, but that don't belong in `src/`
(per-dataset glue: paths, column renames, dataset-specific dedup/scoring
sequencing). Each is listed with the exact output it produced, so the
provenance runs in both directions: `CLAUDE.md` cites a `results/`/`reports/`
path, and this file says which script under `scripts/` produced it.

None of these scripts were refactored, renamed, or otherwise modified beyond
adding the one-line purpose comment at the top of files that lacked one --
they reproduce the originals' outputs bit-exactly and are archival records of
the exact computation that ran, not general-purpose reusable tools.

## teMatDb, original run (File B)

| Script | Produced | Notes |
|---|---|---|
| `tematdb_scoring.py` | `results/20260910T123047_tematdb_external/metrics.json`, `run_config.json`, `tematdb_{a0,a,b}_predictions.npz` | The original 3-stratum (a0/a/b) scoring script. Recovered from scratchpad 2026-09-12 -- `CLAUDE.md`'s SCRIPT PROVENANCE note previously said this file no longer existed anywhere; that was wrong, corrected in commit (see CLAUDE.md). Its `bootstrap_r2_ci` uses `n_boot=1000, seed=0` -- this is the exact, now-identified reason the File A/control reimplementations' bootstrap CIs (below) don't match this script's CIs bit-for-bit: those use `N_BOOT=2000`. |
| `tematdb_N_step1_composition_match.py` | Section N1 (composition-identical pairing) numbers in `reports/tematdb_inventory/inventory_report.md` | Reads round-2/3 scratch artifacts (`_samples_with_doi_flag.csv`, `_step_g1_pair_median.csv`) that were never committed -- not independently re-runnable without them. |
| `tematdb_N_step2_matched_points.py` | Feeds step 3/4 (curve cache) | Reads step 1's output + raw training curves (`load_raw_curves()`, File B). |
| `tematdb_N_step3_matched_points_and_tep.py` | Feeds step 4 (matched points, both declared streams and recomputed zT_tep) | Reads step 2's pickle cache. Hardcodes a path to this session's own scratchpad directory for that cache -- an archival record, not turn-key re-runnable elsewhere. |
| `tematdb_N_step4_metrics.py` | Section N3 (`R2_agree`) numbers -- the values `CLAUDE.md`'s original (now-superseded) COMBINED CEILING digitization column cited | Reads step 3's output. `N_BOOT=2000`. |
| `tematdb_N_step5_n4_diagnostic.py` | Section N4 diagnostic (old temperature-proximity pairs restricted to the composition-DIFFER subset) | Reads step 1's output plus round-3's `_step_g1_best_match.csv`/`_step_l_matched_points.csv`. |
| `per_group_breakdown.py` | `results/20260910T123047_tematdb_external/per_group_breakdown.json` | Read-only, no model touch. Its stratum-b leave-one/two-group-out sigma finding is what `CLAUDE.md`'s CAVEATS (a) cites. |

## Canonical-dataset realignment (File A)

| Script | Produced |
|---|---|
| `noise_floor_fileA.py` | `results/noise_floor/20260911T114356/noise_floor_inputs.json` |
| `estm_fileA.py` | `results/20260911T114356_estm_external_fileA/` |
| `tematdb_C1_C2_fileA.py` | DOI overlap (C1) and cluster overlap (C2) portions of `results/20260911T114356_tematdb_inventory_fileA/inventory_fileA.json` |
| `tematdb_C3_fileA.py` | Composition-matched digitization agreement (section N, re-run against File A) portion of the same `inventory_fileA.json` |
| `consolidate_C.py` | Merges the two scripts above's scratch JSON fragments into the final `inventory_fileA.json` -- pure I/O, no new computation |
| `tematdb_D_scoring_fileA.py` | `results/20260911T114356_tematdb_scoring_fileA/` |

## Control runs (File B, same scripts as above, to isolate dataset effect from reimplementation effect)

| Script | Produced |
|---|---|
| `estm_control_fileB.py` | `results/20260911T134500_estm_control_fileB/` |
| `tematdb_D_control_fileB.py` | `results/20260911T134500_tematdb_control_fileB/` |

Both control scripts are `estm_fileA.py`/`tematdb_D_scoring_fileA.py` with
the File A override removed (unmodified `load_target_data`/
`dry_run_inventory` defaults, which resolve to File B) -- used to confirm the
reimplemented orchestration sequence reproduces the originals' point
estimates bit-exactly (`diff = 0.0` on every R2/RMSE/MAE/n/smear-factor and
every numeric per-row prediction field; see `CLAUDE.md`'s Canonical Dataset
section).

## Not included

`estm_B2_compare.py` (a scratchpad-only script that loaded the two already-
saved `estm_results.json` files -- original and File A -- and printed a
side-by-side R2/RMSE delta table) was not copied here: it produces no
`results/` artifact of its own and computes nothing that isn't already
directly re-derivable from the two `estm_results.json` files it read, both
of which are preserved in git.
