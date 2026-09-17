# External validation re-run against the snapfix training file (fixed chemistry_cluster_id)

Generated 20260917T160553. Training: `data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv`
(SHA256 d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489), fixed chemistry_cluster_id
SNAP(0.05) definition (commit c1c6873). Compared side by side against the File-A **pre-snapfix** run
(`results/20260911T114356_estm_external_fileA/`, `results/20260911T114356_tematdb_inventory_fileA/`,
`results/20260911T114356_tematdb_scoring_fileA/`), all preserved unchanged. Flag threshold: |delta| > 0.02.

**All 4 frozen-hyperparameter models and the sigma/kappa Duan smear factors were computed ONCE and
reused for both ESTM and teMatDb scoring** (both use the identical monkeypatched `load_target_data`
pointed at the snapfix file) -- not refit separately per database, since that would be redundant compute
for an identical training specification, not a methodological difference.

## Pre-flight

P1 (load_target_data row/cluster counts): PASS, all 4 targets exact match (185,064/7,810; 182,755/7,317; 121,110/5,908; 129,419/5,632). Mechanism: `functools.partial` rebind of `ev.load_target_data` inside `src.external_validation`'s own module namespace (process-local, no src/ edit) -- identical to `scripts/estm_fileA.py`'s existing pattern.

P2 (frozen hyperparameter SHA256): recorded current values for all 4 files (no independently-recorded prior SHA256 exists in CLAUDE.md to diff against). Confirmed by session history that nothing has written to `checkpoints/` since these files were produced -- unchanged by default, not independently verified against a stored hash.

## Smear factors

| | OLD (File A pre-snapfix) | NEW (snapfix) | delta | |
|---|---|---|---|---|
| smear_sigma | 1.2822 | 1.3817 | +0.0995 | **FLAG** |
| smear_kappa | 1.0442 | 1.0715 | +0.0273 | **FLAG** |
| smear_sigma OOF log10 R2 | 0.7594 | 0.7044 | -0.0551 | **FLAG** |
| smear_kappa OOF log10 R2 | 0.8457 | 0.8070 | -0.0387 | **FLAG** |

Both smear factors moved by more than the flag threshold -- the chemistry-cluster GroupKFold used to compute training OOF residuals now assigns different fold membership under the fixed cluster definition, which is expected (this is exactly the same fold mechanism the Five-Way Ladder's chemistry rung uses, already shown to move substantially under the snapfix regeneration).

## ESTM dedup counts

| | OLD n_dropped/n_surviving | NEW n_dropped/n_surviving |
|---|---|---|
| Pass (a) source-DOI | 1416/3123 | 1416/3123 |
| Pass (b) chemistry-cluster | 2592/1947 | 3091/1448 |

Pass (a) is DOI-based, unaffected by definition -- confirmed identical (1,416/3,123 both runs).
Pass (b) surviving set shrank (1,947 -> 1,448): unique ESTM clusters overlapping training actually DROPPED (401 -> 300 unique clusters), but more ROWS are dropped overall (2,592 -> 3,091). This is not a contradiction: SNAP merges near-duplicate stoichiometries into fewer, larger clusters, so each overlapping cluster now captures more ESTM rows on average, even though fewer distinct clusters overlap. Flagging this as a real, non-obvious effect of the cluster redefinition, not an error.

## ESTM scored R2, both passes

| Pass | Property | OLD R2 | NEW R2 | delta | |
|---|---|---|---|---|---|
| a | S |  0.5664 | 0.5664 | +0.0000 | |
| a | sigma |  0.3988 | 0.3988 | +0.0000 | |
| a | kappa |  0.6892 | 0.6892 | +0.0000 | |
| a | zT_direct |  0.6615 | 0.6615 | +0.0000 | |
| a | zT_derived |  0.2648 | 0.2064 | -0.0585 | **FLAG** |
| b | S |  0.4219 | 0.3536 | -0.0683 | **FLAG** |
| b | sigma |  0.2785 | 0.2746 | -0.0038 | |
| b | kappa |  0.6431 | 0.6184 | -0.0247 | **FLAG** |
| b | zT_direct |  0.5788 | 0.4982 | -0.0805 | **FLAG** |
| b | zT_derived |  0.1007 | -0.0959 | -0.1965 | **FLAG** |

Pass (a) direct-prediction properties (S, sigma, kappa, zT_direct) are bit-identical to the pre-snapfix run -- expected, since chemistry_cluster_id is not a model feature and pass (a)'s surviving row set is DOI-based (unaffected). Only zT_derived moves in pass (a), because it depends on the (changed) smear factors. Every property moves in pass (b), because the surviving TEST ROWS themselves differ under the new cluster-based dedup -- not purely a model-side effect there.

## teMatDb inventory strata sizes

| Quantity | OLD | NEW | delta |
|---|---|---|---|
| |a0| (DOI in training) | 176 | 176 | +0 |
| |a| (DOI not in training) | 96 | 96 | +0 |
| a0 parseable | 122 | 122 | +0 |
| a parseable | 62 | 62 | +0 |
| parsed | 184 | 184 | +0 |
| failed | 88 | 88 | +0 |
| unique clusters among successes | 156 | 130 | -26 | **FLAG**
| not-in-training clusters (stratum b size) | 36 | 27 | -9 | **FLAG**
| thinned rows a0 | 2027 | 2027 | +0 |
| thinned rows a | 903 | 903 | +0 |
| thinned rows b | 549 | 412 | -137 | **FLAG**

Stratum-b sample_id changes (old -> new): added=[], removed=[78, 91, 119, 141, 173, 201, 232, 379, 406] (9 removed, 0 added -- stratum b shrank from 36 to 27 samples because the SNAP fix now merges these 9 teMatDb compositions' host lattices into an existing training cluster that the old, unsnapped definition kept separate).

**Note on the 0.02 flag threshold for this section**: these are integer strata-size counts, not R2 values, so the |delta|>0.02 rule doesn't apply literally. Flagged (with the marker above) any count that moved by more than 2, to surface the same magnitude of finding qualitatively.

## Composition-matched digitization agreement R2 (300-800K)

| Property | OLD | NEW | delta | |
|---|---|---|---|---|
| S | 0.9639 | 0.9639 | +0.0000 | |
| sigma | 0.9840 | 0.9840 | +0.0000 | |
| kappa | 0.9833 | 0.9833 | +0.0000 | |
| zT (declared) | 0.9807 | 0.9807 | +0.0000 | |
| zT (recomputed TEP) | 0.9839 | 0.9839 | +0.0000 | |

**Confirmed invariant, as expected**: this agreement is computed from `composition_id`-matched raw curves (never touches `chemistry_cluster_id`), and `composition_id` was independently verified unchanged for every row by the snapfix regeneration task. All 5 values reproduce to 4 decimals; rerun rather than assumed unchanged, per the task's instruction.

## teMatDb 3-stratum scoring, all properties

### Stratum a0: OLD n_rows=2027 n_samples=122 n_clusters=110 | NEW n_rows=2027 n_samples=122 n_clusters=90

| Property | OLD R2 | NEW R2 | delta | |
|---|---|---|---|---|
| S | 0.9260 | 0.9260 | +0.0000 | |
| sigma_log10 | 0.7473 | 0.7473 | +0.0000 | |
| kappa_log10 | 0.9057 | 0.9057 | +0.0000 | |
| zT_direct_vs_declared | 0.8419 | 0.8419 | +0.0000 | |
| zT_direct_vs_tep | 0.8465 | 0.8465 | +0.0000 | |
| zT_derived_vs_declared | 0.6714 | 0.6544 | -0.0170 | |
| zT_derived_vs_tep | 0.6724 | 0.6532 | -0.0192 | |

### Stratum a: OLD n_rows=903 n_samples=61 n_clusters=52 | NEW n_rows=903 n_samples=61 n_clusters=49

| Property | OLD R2 | NEW R2 | delta | |
|---|---|---|---|---|
| S | 0.6992 | 0.6992 | +0.0000 | |
| sigma_log10 | 0.1746 | 0.1746 | +0.0000 | |
| kappa_log10 | 0.6531 | 0.6531 | +0.0000 | |
| zT_direct_vs_declared | 0.4957 | 0.4957 | +0.0000 | |
| zT_direct_vs_tep | 0.5074 | 0.5074 | +0.0000 | |
| zT_derived_vs_declared | 0.2718 | 0.2902 | +0.0185 | |
| zT_derived_vs_tep | 0.2756 | 0.2926 | +0.0170 | |

### Stratum b: OLD n_rows=549 n_samples=36 n_clusters=36 | NEW n_rows=412 n_samples=27 n_clusters=27

**Stratum b's underlying row/sample set itself changed (549->412 rows, 36->27 samples) -- the comparisons below are NOT a clean apples-to-apples model comparison for this stratum, they also reflect a different population.**

| Property | OLD R2 | NEW R2 | delta | |
|---|---|---|---|---|
| S | 0.6215 | 0.6049 | -0.0166 | |
| sigma_log10 | -0.2900 | 0.3260 | +0.6160 | **FLAG** |
| kappa_log10 | 0.6617 | 0.6605 | -0.0012 | |
| zT_direct_vs_declared | 0.5422 | 0.5344 | -0.0078 | |
| zT_direct_vs_tep | 0.5459 | 0.5354 | -0.0106 | |
| zT_derived_vs_declared | 0.0186 | -0.1334 | -0.1520 | **FLAG** |
| zT_derived_vs_tep | 0.0200 | -0.1381 | -0.1582 | **FLAG** |

## All flagged moves (|delta| > 0.02), consolidated

| Item | OLD | NEW | delta |
|---|---|---|---|
| smear_sigma | 1.2822 | 1.3817 | +0.0995 |
| smear_kappa | 1.0442 | 1.0715 | +0.0273 |
| smear_sigma OOF log10 R2 | 0.7594 | 0.7044 | -0.0551 |
| smear_kappa OOF log10 R2 | 0.8457 | 0.8070 | -0.0387 |
| (a) zT_derived | 0.2648 | 0.2064 | -0.0585 |
| (b) S | 0.4219 | 0.3536 | -0.0683 |
| (b) kappa | 0.6431 | 0.6184 | -0.0247 |
| (b) zT_direct | 0.5788 | 0.4982 | -0.0805 |
| (b) zT_derived | 0.1007 | -0.0959 | -0.1965 |
| sigma_log10 | -0.2900 | 0.3260 | +0.6160 |
| zT_derived_vs_declared | 0.0186 | -0.1334 | -0.1520 |
| zT_derived_vs_tep | 0.0200 | -0.1381 | -0.1582 |

12 R2/smear-factor values flagged.