# Thermoelectric ML Pipeline — Project Memory

## Goal
Rebuild a pipeline from scratch into two journal papers (general,
non-perovskite scope). Predict four thermoelectric properties — Seebeck
coefficient S, electrical conductivity σ, thermal conductivity κ, figure of
merit zT — from composition. Data source: Starrydata2. Two papers, Paper A
built first, Paper B's standalone-vs-fold status is a gated decision made
only after Paper A's core results exist (see Build Order below).

**Status: all open methodology questions are FROZEN decisions as of this
version, following three rounds of external review. Do not reopen or
re-litigate any item below without a specific, new, concrete reason —
implement as written.**

---

## Canonical Dataset (added 2026-09-11)

Two files share the filename `featurized_ThermoelectricMaterials_2026-08-15.csv`
but are NOT the same dataset -- different SHA256, different row counts,
from two separate Starrydata2 pulls against `src/data_acquisition.py`'s
unpinned GitHub "latest" release tag (regenerated daily from the live
database -- see that module's docstring). Found 2026-09-11 investigating
why `load_target_data("S")` gives 184,803 rows locally while the frozen
hyperparameters record `n_rows: 185064`.

**File A -- CANONICAL for every Paper A confirmed result**:
`checkpoints/saved_predictions/te-ml-pipeline/data/processed/
featurized_ThermoelectricMaterials_2026-08-15.csv`
- SHA256 `e97406fa5223efa466d1e3fe4ffb2b1bc0303dc94b6adac6a824a643a9156f8d`,
  975,062,560 bytes.
- Upstream pull dated 2026-08-22 (own `extraction_metadata.json`:
  `extraction_timestamp_utc 2026-08-22T07:05:52Z`, `upstream_db_snapshot
  "2026-08-22 02:00:02 UTC+0900 (JST)"`), despite the "2026-08-15"
  filename.
- Row counts (`load_target_data(target)`): S=185,064, sigma=182,755,
  kappa=121,110, zT=129,419.
- Produced: the confirmed Five-Way Ladder chemistry-cluster checkpoints
  (`checkpoints/saved_predictions/checkpoints/{S,sigma,kappa,zT}_chemistry/`
  -- every repeat's 5 outer test folds sum to these exact totals) and the
  frozen hyperparameter files
  (`checkpoints/saved_predictions/checkpoints/frozen_hyperparams/
  {S,sigma,kappa,zT}.json`).
  **Correction, 2026-09-22: this bullet was correct, but the "CANONICAL
  for every Paper A confirmed result" framing above it was read too
  broadly.** Only the chemistry-cluster rung was ever actually verified
  against File A's row counts (S=185,064 etc., matched exactly, as
  stated). The Five-Way Ladder's other four rungs at the time --
  composition, random 80/20, 5-fold, 10-fold, all produced from
  `checkpoints/ladder_regen_dl/` -- were never checked against File A's
  row counts at all, and turned out NOT to match them: 185,844 / 183,246
  / 121,535 / 129,851 rows (S/sigma/kappa/zT), a fourth, unidentified
  dataset snapshot with no recorded path or SHA256, larger than both File
  A and File B. Found 2026-09-22; see the Five-Way Ladder section's
  SUPERSEDED note for the correction and the rerun that fixed it
  (`results/ungrouped_snapfix/20260922T093243/`). As of that rerun, all
  five ladder rungs derive from the snapfix CSV (File A plus the fixed
  `chemistry_cluster_id` column, identical row counts to File A).
- Gitignored (`checkpoints/`), local-only, not on GitHub.
- Backed up as a private Kaggle dataset:
  `muhammadbehzadgull/te-ml-pipeline-canonical-dataset-a`, verified
  byte-identical 2026-09-11 (SHA256
  `e97406fa5223efa466d1e3fe4ffb2b1bc0303dc94b6adac6a824a643a9156f8d`,
  975,062,560 bytes). Kaggle input path:
  `/kaggle/input/datasets/muhammadbehzadgull/
  te-ml-pipeline-canonical-dataset-a/`.
- **Mislabeled figure, do not use**:
  `checkpoints/saved_predictions/te-ml-pipeline/figures/
  cleaning_funnel.png`/`.pdf` sit inside File A's own directory tree but
  render **File B's** cleaning funnel (`scripts/make_figures.py`'s
  `CLEANING_STEPS` constant was hardcoded from the 2026-08-15 pull's
  2026-08-17 re-run and never updated after File A became canonical).
  Confirmed 2026-09-14: the figure's final count (280,348) does not
  match File A's actual cleaned CSV (280,664). File A's own funnel is
  now computed at
  `results/cleaning_funnel/20260914T100914/funnel_counts.json`
  (verified against File A's cleaned CSV, gate passed) and
  `scripts/make_figures.py`'s `CLEANING_STEPS`/`RAW_CURVES` are updated
  to match; the old File-B-sourced figure files on disk here are left in
  place as a historical artifact, not deleted, but should never be cited
  as File A's funnel.

**File B -- used for ESTM/teMatDb external validation and the noise-floor
inputs, NOT what produced the confirmed ladder**: `data/processed/
featurized_ThermoelectricMaterials_2026-08-15.csv`
- SHA256 `19c983b96a3c48265916334a0dc35f1215cbbdbc9fdd7bd31a4dd291d243c286`,
  974,150,322 bytes.
- Upstream pull genuinely dated 2026-08-15 (own `extraction_metadata.json`).
- Row counts: S=184,803, sigma=182,530, kappa=120,894, zT=129,188.
- Produced: ESTM external validation
  (`checkpoints/external_validation/estm_results.json`, 2026-09-08) and
  teMatDb external validation
  (`results/20260910T123047_tematdb_external/`, 2026-09-10) -- both via
  refit-on-100%-training using File A's frozen hyperparameters, refit
  against File B. **Also produced the noise-floor inputs**
  (`results/noise_floor/20260910T134042/`): `sigma_total` and every other
  input in that artifact were computed from File B's cleaned CSV
  (`data/processed/cleaned_ThermoelectricMaterials_2026-08-15.csv`). The
  resulting R^2_max values were then compared, in this file's own
  "Confirmed Results -- Noise Floor" section (already committed), against
  a "Confirmed ceiling" computed from File A's chemistry-cluster
  checkpoints -- i.e. that table's headroom is itself a cross-dataset
  comparison (R^2_max from File B, the ceiling it's measured against from
  File A).
- Gitignored (`data/`), local-only, not on GitHub.

**Raw-pull row counts (source, before cleaning), from each copy's own
`data/raw/extraction_metadata.json`** -- the two upstream snapshots are
visibly different at the source, not only after processing:

| | papers | samples | curves |
|---|---|---|---|
| File A (2026-08-22 pull) | 9,494 | 55,261 | 156,101 |
| File B (2026-08-15 pull) | 9,481 | 55,166 | 155,758 |

**CLOSED (2026-09-11)**: ESTM external validation, the teMatDb inventory
and scoring, and the noise-floor inputs were all re-run against File A.
New output, each in its own timestamped directory, none overwriting the
File B-era artifacts named above:
- `results/noise_floor/20260911T114356/` (noise-floor inputs)
- `results/20260911T114356_estm_external_fileA/` (ESTM)
- `results/20260911T114356_tematdb_inventory_fileA/` (teMatDb DOI/cluster
  overlap + composition-matched digitization agreement, section N)
- `results/20260911T114356_tematdb_scoring_fileA/` (teMatDb 3-stratum
  scoring)

**Reimplementation verified faithful, not just dataset-swapped.** The
File A re-runs called `dry_run_inventory()` / `compute_training_smear_factors()`
/ `fit_frozen_external_model()` / `score_estm_pass()` directly (a
reimplemented orchestration sequence, not the original
`run_full_validation()` or the no-longer-on-disk `tematdb_scoring.py` --
see SCRIPT PROVENANCE below), so a control run pointed the SAME script at
File B (unmodified defaults, no override) and compared its output against
the original: `results/20260911T134500_estm_control_fileB/`,
`results/20260911T134500_tematdb_control_fileB/`. Result: every point
estimate bit-identical to the original (`diff = 0.0` exactly) -- smear
factors, dedup counts, R², RMSE, MAE, n, and every numeric per-row
prediction field across all 5,070 ESTM rows (both dedup passes) and all
3,479 teMatDb rows (all three strata). The one exception: teMatDb's
bootstrap-CI bounds differ by up to 0.020 -- an artifact of
`bootstrap_r2_ci` being freshly written for this task rather than
reproducing whatever the original (no-longer-on-disk) script used,
confined entirely to that one resampling diagnostic, never touching a
point estimate. Conclusion: every File A vs. File B delta reported below
is pure dataset effect, not reimplementation noise.

**Strata membership is invariant between snapshots.** teMatDb's DOI
overlap, cluster overlap, and composition-matched digitization strata are
identical whether measured against File A or File B: |a0|/|a| = 176/96,
a0-parseable/a-parseable = 122/62, parsed/failed = 184/88, unique clusters
among successes = 156, not-in-training clusters = 36 -- the exact same
36-sample stratum-b set, zero additions or removals. Section N's
composition-matched denominator is likewise identical: 96 of 176 samples,
identical matches-per-sample distribution. Only the *scored values* --
what the frozen model, refit on each snapshot, actually predicts -- move;
which rows belong to which stratum does not.

**Largest observed deltas, by category** (File A minus File B, or the
directional swing where sign flips):

| Category | Largest \|delta\| | Where |
|---|---|---|
| Noise-floor R2_max / headroom | 0.00015 | kappa headroom |
| ESTM R2 (S, sigma, kappa, zT_direct) | 0.017 | pass (b), kappa |
| ESTM zT_derived | 0.12, sign flip | pass (b): -0.019 -> +0.101 |
| teMatDb scoring R2 | 0.063 | stratum b, S |
| teMatDb digitization agreement (section N) | 0.023 | zT, recomputed TEP |

**File A results are now canonical for every Paper A number.** The File B
outputs listed above (`checkpoints/external_validation/`,
`results/20260910T123047_tematdb_external/`,
`results/noise_floor/20260910T134042/`) are retained on disk for
comparison and audit, not deleted, but are superseded as of this entry --
do not cite them as the current confirmed value for anything.

**Standing rule: `src/data_acquisition.py` must NOT be re-run for any
Paper A result.** It pulls Starrydata2's GitHub "latest" release tag,
regenerated daily from the live database, with no pin/version/date
parameter anywhere in `acquire()` -- a fresh pull on any day other than
2026-08-22 will not reproduce File A. If Phase 0 ever must be re-run
(e.g. for the Kaggle bit-identity gate), save the output under a
filename that does not reuse the "2026-08-15" label, and update this
section rather than silently overwriting either file in place.

**SCRIPT PROVENANCE -- CORRECTED AND RESOLVED 2026-09-12.** The claim
directly below this line, in the version of this note committed
2026-09-11, was wrong: it stated that the original `tematdb_scoring.py`
(which produced `results/20260910T123047_tematdb_external/metrics.json`)
"no longer exists on disk anywhere in this repo or its working
directories." It was never actually lost -- it survived in this
session's local scratchpad the whole time and has now been recovered and
committed to `scripts/tematdb_scoring.py`. Its recovery also identifies,
with certainty rather than as an unexplained artifact, why the File A and
control-run reimplementations' bootstrap CIs did not match its CIs
bit-for-bit (the discrepancy reported in "Reimplementation verified
faithful" above): `tematdb_scoring.py`'s `bootstrap_r2_ci` uses
`n_boot=1000, seed=0`; `tematdb_D_scoring_fileA.py` and
`tematdb_D_control_fileB.py` use `N_BOOT=2000`. Every point estimate
(R2, RMSE, MAE, n, smear factors) stays bit-exact regardless -- only the
CI bounds are affected, exactly as already stated, now with a confirmed
cause rather than a suspected one.

Every script that produced a number cited in this document is now
committed under `scripts/`: the original `tematdb_scoring.py`, its five
N-section digitization-floor siblings (`tematdb_N_step1_composition_match.py`
through `tematdb_N_step5_n4_diagnostic.py`), `per_group_breakdown.py`, and
the seven File A/control-run orchestration scripts (`noise_floor_fileA.py`,
`estm_fileA.py`, `tematdb_C1_C2_fileA.py`, `tematdb_C3_fileA.py`,
`consolidate_C.py`, `tematdb_D_scoring_fileA.py`, `estm_control_fileB.py`,
`tematdb_D_control_fileB.py`). See `scripts/README.md` for the full
script-to-output mapping, both directions. This resolves the "not yet
done" gap this note previously flagged.

**Addendum, 2026-09-19: `scripts/tematdb_scoring.py`'s `TRAINING_BOUNDS`
constant is computed from File B, not File A.** Found while reconstructing
the ESTM in-distribution bounds (see the External Validation section):
`TRAINING_BOUNDS` is populated by calling `load_training_data()` with no
override, which defaults to `src/external_validation.py`'s `TRAINING_CSV`
constant -- File B's path, not File A's. Confirmed directly: this
constant's sigma lower bound is 957.6, versus 958.78 computed fresh from
the snapfix (File A-derived) training set. teMatDb's own OOD tail is
0.12%, so the practical effect there is negligible -- but the constant
itself is wrong for File A and should not be reused elsewhere without
recomputing it against the correct training file.

**Standing rule: any script that produces a number cited in this
document must be committed to git (e.g. under `scripts/`), not left in
an untracked scratchpad.**

**CAVEATS (2026-09-11) -- state each plainly, do not overclaim precision
beyond what these numbers support:**

a. **Stratum b is small and unstable.** Only 36 samples. Its R² moves by
   up to 0.063 between the File A and File B snapshots, and the earlier
   leave-one-group-out diagnostic
   (`results/20260910T123047_tematdb_external/per_group_breakdown.json`)
   already showed stratum-b sigma's R² flips sign when a single 4-sample
   GROUP (Selenide) is removed. Report stratum-b findings qualitatively
   (direction, rough magnitude); do not quote stratum-b R² to two decimals
   as if it were a stable estimate.
b. **zT_derived is ill-conditioned.** A 0.7% change in the smear factor
   (1.2916 -> 1.2822, from refitting on a slightly different training
   snapshot) moved ESTM pass-b zT_derived's R² from -0.019 to +0.101 -- a
   swing of about 0.12 that crosses zero. The direct-beats-derived
   ORDERING is stable across both snapshots and is the finding to report;
   the zT_derived point value itself is not stable enough to quote without
   this caveat attached.
c. **The digitization floor carries snapshot sensitivity.** With the
   matched sample set and composition-identity pairing themselves
   unchanged between snapshots (see "Strata membership is invariant"
   above), the measured label-agreement R² still moved: zT +0.021
   (declared) / +0.023 (recomputed TEP), sigma +0.006, on File A vs.
   File B raw training curves. Quote the digitization floor to at most two
   decimals, and state this snapshot uncertainty alongside it wherever
   cited.

---

## Grouping Fixes (added 2026-09-19)

**BUG 1 -- chemistry_cluster_id did not snap near-integer host amounts.**
After dropping sub-threshold dopants, the function called `reduced_formula`
directly on the remaining host amounts. `reduced_formula` does not collapse
near-integer decimals onto their integer target -- Pb0.97Te1 stayed
Pb0.97Te1, not PbTe. Measured: 10,222 of 12,036 clusters carried decimal
coefficients, covering 79.3% of rows. The function's own docstring claimed
host amounts were "reduced to an integer ratio" -- that claim was never
true. Found 2026-09-14, when a worked example in a methods draft did not
reproduce.

Fix: SNAP(0.05), commit c1c6873. After dropping dopants, each host amount
within 5% of the nearest integer is rounded to that integer before
`reduced_formula` is called. Acceptance tests: merges Pb0.97Te1 with PbTe
and Co3.95Sb12 with CoSb3 (the intended collapse); leaves Bi2Te2.7Se0.3
distinct from Bi2Te3, elemental Te distinct from SnPbTe alloys, elemental Fe
distinct from Fe-based alloys, and a six-element high-entropy alloy distinct
from CoSb (the intended non-collapses).

Rejected alternative: rational approximation via
`get_integer_formula_and_factor`. Over-merged in exactly the cases SNAP(0.05)
keeps apart -- elemental Te with SnTe alloys, a high-entropy alloy with
CoSb -- and produced non-monotonic grouping across solid-solution series
(composition x and composition x+delta could land in different clusters
than x and x+2*delta).

**BUG 2 -- randomized_group_kfold pinned the largest n_splits groups to
fixed fold indices.** The function sorted groups by descending size
(shuffling first, to break ties among equal-sized groups), then greedily
bin-packed them via `argmin` over each fold's running total. Processing the
largest groups first, with every fold total starting at zero, means argmin
returns the lowest empty fold index every time -- so the k-th largest group
always landed in fold k-1, every repeat, regardless of the shuffle. The
shuffle only ever changed which group broke a size tie; it never changed
which fold the largest, unambiguously-ranked groups landed in. Repeats were
therefore not independent partitions for the rows that dominate the result,
and the across-repeat SD reported alongside every grouped number understated
the spread it was meant to measure. Measured: 6.30% of zT rows were in this
deterministic zone pre-SNAP, 13.57% post-SNAP (SNAP changes cluster sizes,
which changes which clusters rank among the top n_splits).

Fix: scheme S5, commit c1c6873. The top n_splits groups are each placed by
an independent uniform random draw with replacement, before the remainder is
greedy bin-packed exactly as before. Simulated over 20 repeats: fold-size SD
0.002% of mean (identical to the old code -- S5 does not trade balance for
randomness), 20 of 20 repeats gave distinct partitions where the old code
gave 1, and CoSb3 and TePb (two of the largest clusters) co-occurred in the
same fold in 4 of 20 repeats (never, under the old code).

Rejected alternatives: randomizing the argmin tie-break, which permutes
which fold receives which LABEL, not the partition itself, so the top groups
would still always land together in some fold, just an unpredictable one;
shuffling group order without sorting by size first, which cost up to 11%
fold-size imbalance; `GroupShuffleSplit`, which does not produce a partition
(test sets can overlap across folds, incompatible with pooled out-of-fold
R^2).

**CONSEQUENCE.** Both fixes make the honest anchor stricter, never looser --
so every previously reported grouped number was optimistic, and every gap
against an ungrouped rung was a lower bound on the true gap. The chemistry
rung dropped 0.037 to 0.058 across the four targets once the regeneration
was complete (see the Five-Way Ladder section for the new per-target
values). The regeneration required a NEW featurized CSV, not just a code
fix: `chemistry_cluster_id` is a column stored in the featurized CSV at
canonicalization time, not a value `src/nested_cv.py` computes at run time --
fixing the function alone does not change any number until the column
itself is regenerated from it. See the Canonical Dataset section for the
snapfix CSV this produced.

**Three regeneration failures found along the way, recorded because they
share one shape: an input was silently stale, and nothing raised an
error.**
- The first regeneration attempt reran the ladder against the still-stale
  `chemistry_cluster_id` column (the fixed function had not yet been used
  to regenerate it) and reproduced the old, pre-fix numbers almost exactly
  -- no error, no warning, just a number that looked plausible and was
  wrong.
- `src/direct_vs_derived_zt.py`'s `_fold_path` ignored the `checkpoint_dir`
  argument `run_direct_vs_derived()` accepted, and resolved every fold's
  checkpoint path against the old module-level constant instead. Passing a
  new checkpoint directory silently reused the pre-fix checkpoints from the
  old one and reported success.
- The same script loaded zT's hyperparameters from an untracked, ephemeral
  cache (`checkpoints/frozen_hyperparams/zT_xgboost.json`) rather than the
  canonical, git-tracked file -- that cache held a hyperparameter set tuned
  against File B, in August, before File A became canonical, silently
  different from every other confirmed result in this project.

**Standing rule this implies**: a regeneration must be verified by a value
that MUST differ if the inputs changed, not by the absence of an error --
absence of an error is exactly what all three failures above produced. The
REGENERATION GATE line added to `run_direct_vs_derived()` in commit fabc532
is the pattern to follow elsewhere: it prints the resolved checkpoint
directory, the resolved hyperparameters file path and its SHA256, and the
input subset's row count and unique group count, every run, so a rerun
against different inputs cannot print the identical line by accident.

---

## Build Order (execute top to bottom — later phases depend on earlier ones)

**Phase 0 — shared foundation, run once:**
1. Fresh Starrydata2 pull, version-dated at extraction.
2. Global data cleaning (see Data Cleaning Pipeline below) — clean once,
   use the identical cleaned dataset for everything downstream. Do NOT
   clean fold-locally as the primary path (see Grouping Key section for
   why).
3. Composition canonicalization.
4. Chemistry-cluster definition at the frozen 5 at% dopant threshold (see
   Grouping Key below). **This is the single most load-bearing decision
   in the whole project — nothing downstream is trustworthy until this is
   fixed.** Produce the 3-row sensitivity table (looser / 5 at% / stricter)
   before freezing.

**Phase 1 — Paper A's publishable spine:**
5. Five-way validation-inflation ladder (Section: Paper A, item 1).
6. Noise-floor anchor computed in per-property matched space (log10 for
   sigma and kappa, linear for S and zT, matching each property's
   confirmed R² scoring space) (Paper A, item 3).
7. External validation, two separate numbers (Paper A, item 6).

**Phase 2 — Paper A finish:**
8. Direct-vs-derived zT pathway, on the all-four-properties subset.
9. Screening rediscovery test, targeted holdout only (Paper A, item 7)
   -- DROPPED 2026-09-14, see Paper A item 7 below for the reason.
10. Temperature axis — write as ONE limitation paragraph only, do not build
    a dedicated temperature-extrapolation EXPERIMENT (cut per frozen
    decision, see below). This is a scoping decision about the
    experiment, not the feature set: temperature_bin stays in the model
    as a per-row input throughout, same as any other feature (see Paper
    A item 8).
11. PCA-split head-to-head against Athar/Jund (Paper A, item 9).

**Phase 3 — Paper B (only after Phase 1-2 complete):**
12. Stoichiometric-template family labeling, report "unassignable" bucket size.
13. Leave-one-family-out + BOTH controls (size-matched random +
    structured-removal of a different family).
14. Specialist-vs-pooled comparison (primary probe) + offset-stripped
    within-family correlation, across two model families (one high-capacity,
    one constrained) — report separately, do not average.
15. Apply the GO/NO-GO rule: if the specialist-vs-pooled result shows a
    clear coarse-vs-fine story, Paper B stands alone; if weak/ambiguous,
    fold it into Paper A as one additional section.

---

## Grouping Key — FROZEN definition

**Chemistry-cluster** (the coarsest, strictest grouping level, and the one
the "honest" ceiling is anchored to): defined as the **reduced host-lattice
stoichiometry with dopants below 5 at% collapsed into the parent** —
canonicalize each formula to a cluster identity based on the integer
stoichiometry of elements present above 5 at%; elements below that
threshold are dopants and do not split the cluster.

- Hierarchy: **Chemistry cluster ⊇ Composition ⊇ Sample.**
- All cross-validation (Paper A's ladder, Paper B's LOFO and both
  controls) groups at the chemistry-cluster level for the headline
  "honest" numbers. Composition-level and Sample-level splits may appear
  as intermediate, explicitly-labeled-as-less-strict rungs on Paper A's
  ladder, never as equivalent to the chemistry-cluster anchor.
- Run the ladder once more at a looser and a stricter clustering threshold
  as a 3-row sensitivity table before freezing 5 at% as primary.
- Terminology note (resolved 2026-08-15, after the fine-grained
  chemistry_cluster_id implementation was built and run against the real
  pull): the "~15-25 meaningful chemistry clusters" language used in
  earlier drafts of this file referred to broad structural families (the
  half-Heusler / skutterudite / PbTe-based kind of grouping, the same
  level Paper B's stoichiometric-template matching operates at) -- NOT
  the fine-grained chemistry_cluster_id defined above. chemistry_cluster_id
  correctly produces thousands of groups (12,454 measured on the
  2026-08-15 pull), most of them small (median size 1 sample, 70%
  singletons) -- that is the intended behavior of the 5 at% definition,
  not a bug, and it should not be conflated with Paper B's coarser family
  count.
  **SUPERSEDED 2026-09-19: "12,454".** Measured pre-fix. Under the fixed
  grouping (SNAP(0.05), commit c1c6873) and the snapfix featurized CSV:
  8,908 groups (`reports/grouping_key_remeasure/20260918T000232/`).
- **chemistry_cluster_id vs. the thesis's "parent chemical system"
  (sorted element set, no stoichiometry, no dopant threshold): not a
  finer/coarser pair of the same grouping, they cross-cut each other.**
  Measured 2026-08-19 by recomputing parent-system grouping on our own
  cleaned pull (17,207 formulas): 3,921 parent_system groups vs. 12,024
  chemistry_cluster_id groups -- fewer, bigger groups, which naively
  reads as "parent_system is just a coarser, stricter version." It
  isn't: 1,399 of 12,024 chemistry_cluster_id groups (11.6%) have
  members that span MORE THAN ONE parent_system, so the two definitions
  do not nest. Worst case -- **CoSb3, the single largest chemistry
  cluster (780 samples, see the repeated-CV note below), splits into 69
  different parent_system groups** purely because its filler/dopant
  species varies (La, Ce, Yb, Ba, In, Ga, Tl, K, Na, Br, I, Sn, O, ...).
  **SUPERSEDED 2026-09-19: the parent_system/chemistry_cluster_id counts
  and the CoSb3 figures in this bullet.** Remeasured against the fixed
  grouping and the snapfix CSV's 17,977 unique formulas: 3,909
  parent_system groups vs. 8,908 chemistry_cluster_id groups; 1,285 of
  8,908 (14.4%) span more than one parent_system; CoSb3 now has 959
  samples (grew 23%) and splits into 139 parent_system groups. The
  qualitative argument is unchanged (the two definitions still cross-cut,
  chemistry_cluster_id is still the more defensible anchor) -- only these
  counts moved.
  Under parent_system, a La-doped CoSb3 sample and a Yb-doped CoSb3
  sample -- chemically near-identical host-lattice measurements, the
  exact near-duplicate leakage case chemistry_cluster_id's 5 at%
  threshold exists to catch -- could legally land in different CV
  folds. On this axis parent_system is LOOSER than chemistry_cluster_id,
  not stricter. (Conversely parent_system is stricter on a different
  axis it isn't credited for: it merges compounds sharing an element set
  at very different stoichiometry, e.g. "Bi-Sb-Te" absorbs 109 distinct
  chemistry_cluster_id values regardless of how physically different
  they are -- an artifact of ignoring concentration, not a targeted
  leakage safeguard.) Conclusion: chemistry_cluster_id remains the more
  defensible anchor for the leakage mechanism this project actually
  cares about (dopant-variant near-duplicates of the same host lattice);
  parent_system's lower group count is not evidence it is a stricter
  superset.
- **Repeated grouped CV is still required, but the justification is
  group-SIZE imbalance, not group scarcity.** Measured directly (2026-08-15,
  sklearn 1.4.2 GroupKFold, n_splits=5, chemistry_cluster_id groups, real
  cleaned pull): fold ROW COUNTS come out essentially balanced (68,165-
  68,167 rows per fold, std < 0.01% of the mean) -- sklearn's GroupKFold
  greedily bin-packs groups by descending size, which balances total rows
  per fold even with a heavy-tailed group-size distribution. The actual
  problem is fold COMPOSITION: GroupKFold guarantees each group lands
  entirely in one fold, so the single largest cluster (CoSb3, skutterudite,
  780 samples / 9,898 rows) makes up 14.5% of whichever fold it lands in,
  and the next-largest (Ca3Co4O9, 548 samples / 5,027 rows) makes up 7.4%
  of its fold.
  **SUPERSEDED 2026-09-19: the fold-row-count range and both named
  clusters' figures.** Remeasured against the fixed grouping and the
  snapfix CSV (`reports/grouping_key_remeasure/20260918T000232/`): fold
  row counts 56,034-56,035 (still balanced to <0.01% of the mean -- the
  qualitative claim holds). **This comparison is confounded by a dataset
  change, not a clean grouping-only comparison**: 68,166 x 5 =~340,830
  matches this document's own pre-step8/9-fix cleaned-dataset row count,
  not the snapfix CSV's 280,173 rows -- do not read the row-count shift
  itself as a grouping effect. CoSb3 now has 959 samples / 13,185 rows,
  23.5% of its fold (grew 23% in sample count); Ca3Co4O9 now has 281
  samples / 3,239 rows, 5.8% of its fold (shrank 49% in sample count,
  the opposite direction from CoSb3 -- not investigated further here).
  A single grouped split is one arbitrary draw of which large,
  chemically distinct cluster gets held out in which fold; that fold's R^2
  is then partly a referendum on "how well does the model generalize to
  this one specific chemistry" rather than a representative average.
  Repeating with a different group-to-fold assignment is what separates
  that single-split composition effect from genuine model variance -- use
  a Nadeau-Bengio corrected significance test before claiming any
  inflation delta between ladder rungs is real, not noise.
  Implementation note for src/nested_cv.py: plain sklearn GroupKFold
  (as pinned, 1.4.2) has no shuffle/random_state parameter and is fully
  deterministic given a set of group labels -- calling it multiple times,
  or pre-shuffling row order first, both verified empirically to return
  the identical fold assignment every time (bin-packing depends only on
  group sizes, not row or group order). "Repeated" grouped CV therefore
  requires deliberately randomizing the group-to-fold assignment each
  repeat (e.g. shuffle the unique group list before a manual balanced
  assignment, or use repeated GroupShuffleSplit with distinct
  random_states), not simply looping GroupKFold.
- **n_repeats for the ladder's two UNGROUPED rungs (random 80/20,
  5-fold/10-fold): deliberately 1, not the grouped rungs' 5, decided
  2026-08-19.** The repeated-CV justification directly above is specific
  to GROUPED CV: GroupKFold forces an entire cluster onto one side of a
  split, so a single split is one arbitrary draw of which large,
  chemically distinct cluster gets held out, and repeats separate that
  composition effect from genuine model variance. Plain sklearn KFold/
  ShuffleSplit assign ROWS independently of any group, so there is no
  analogous whole-cluster-forced-onto-one-fold effect to average away --
  at this dataset's row count (~280K), a single shuffled split already
  gives a representative, low-variance partition. Additionally, the
  random-80/20 rung is already internally repeated: CLAUDE.md's own "~20
  times, pool" prescription is satisfied by `--n-outer-folds 20` (20
  independent ShuffleSplit draws within one outer-repeat pass, see Paper
  A item 1's implementation note) -- an extra outer n_repeats layer on
  top would just re-run that same 20-draw set again, adding compute
  without changing what's being measured (this ladder rung is deliberately
  the "how leaky is this naive baseline" measurement, not the calibrated
  honest-ceiling estimate that repeats are protecting elsewhere).
  src/nested_cv.py's `run_nested_cv(n_repeats=None)` (the default)
  therefore resolves to `N_OUTER_REPEATS_GROUPED` (5) for
  composition/chemistry and `N_OUTER_REPEATS_UNGROUPED` (1) for
  random/kfold -- pass `--n-repeats` explicitly to override either.
- **src/nested_cv.py compute budget: full XGBoost search space requires
  Kaggle/GPU, not local CPU.** Measured directly on the 2026-08-15
  featurized pull (142,997 zT rows, 396 MAGPIE+CBFV features,
  ~114k-126k rows per outer training fold): a single XGBoost fit costs
  3s at max_depth=3/n_estimators=100 vs 57s at max_depth=10/
  n_estimators=600 -- depth dominates cost, roughly 4x per doubling of
  the capped-vs-full range. One full calibration repeat (5 outer folds,
  6 Optuna trials/fold, 3 inner folds, search space capped to
  max_depth<=7/n_estimators<=350) took 1,615s (~27 min), giving mean
  outer R² = 0.5045, std = 0.0233 -- a real but intentionally
  under-tuned result (6 trials, 1 repeat), not a trustworthy estimate
  of the honest ceiling. **Also trained without temperature as a
  feature** (get_feature_columns() bug, fixed 2026-08-18 -- see Paper A
  item 8) -- doubly not representative of the frozen model, re-run
  after the fix before citing this number for anything beyond
  compute-time calibration.

  | | Capped (depth<=7, n_est<=350) | Full (depth<=10, n_est<=600) |
  |---|---|---|
  | Per outer fold, 20 trials | ~1,077s (~18 min), linear-scaled from measured | ~3,472-4,428s (~58-74 min), worst-case/avg-scaled from measured single-fit costs |
  | 5 repeats x 5 folds (25 outer folds) | ~7.5 hours | ~24-31 hours |
  | 10 repeats x 5 folds (50 outer folds) | ~15 hours | ~48-61 hours |

  Conclusion: the full intended search space (max_depth up to 10,
  n_estimators up to 600) across the full 5-10 repeat design is
  infeasible on local CPU (1-2.5+ days). **src/nested_cv.py's search
  space must stay at its full, originally-intended range and must never
  be silently narrowed for local-runtime convenience** -- a capped
  range is acceptable only for a clearly-labeled calibration/smoke-test
  run, never as the default the real reported numbers come from. The
  real run (5-10 repeats, full search space) runs on Kaggle/GPU; decide
  and document explicitly here if that changes.

**Global cleaning, not fold-local**: clean the full dataset once, globally,
before any split. The cross-row cleaning filters (multi-source CV
consistency, MAD outliers, rolling-median smoothing) leak only each
property's marginal distribution (mild/transductive), not the train→test
mapping. Add ONE sensitivity check — fold-local refit on the
chemistry-cluster split — and report whether the honest ceiling number
moves. Fold-local cleaning as the PRIMARY pipeline is rejected: it breaks
the apples-to-apples comparison the five-way ladder needs (each scheme
would clean the data differently).

---

## Data Cleaning Pipeline (11 steps — replicate this spec on the fresh pull)
1. **Property extraction & range filtering** (~1,114,628 rows). Bounds: S
   -1000 to +1000 μV/K; σ 10 to 10^7 S/m (semiconductor-insulator
   boundary, confirmed against thesis 2026-08-17 -- was implemented as 1
   before that, see resolved TODO below); κ 0.05 to 25 W/mK (Cahill-Pohl
   minimum); zT 0 to 4. Cite Snyder & Toberer 2008.
2. **Data integration & consolidation** (Python + pandas). Convert
   resistivity to conductivity.
3. **Temperature filtering** (300-800K), binned into 25K intervals.
4. **Pivot long→wide** (one row per sample per temperature bin).
5. **Formula cleaning** — remove formulas unparseable by pymatgen's
   Composition class (Ong et al. 2013).
   **TODO (found 2026-08-15, not blocking, fix before finalizing
   cleaning_funnel and cluster_size_distribution for the paper):**
   pymatgen's Composition parser does not reject placeholder/template
   element tokens (e.g. "M", "A", "Ln", "G") -- it silently accepts
   them as DummySpecies instead of raising, so formulas like
   "M0.125Ba0.125Sr0.5Yb0.25Co4Sb12.5H0.5" or "Ln0.949Lu0.05Sn0.001O3"
   (generic-site notation from source papers, not real chemistry)
   currently pass step 5 and get a composition_id / chemistry_cluster_id
   like any real formula. Found via src/featurization.py failing on 23
   such formulas (matminer raises KeyError, CBFV crashes outright).
   Step 5 needs an explicit DummySpecies check
   (isinstance(el, pymatgen.core.periodic_table.DummySpecies) for el in
   comp.elements) to reject these at the source, same place unparseable
   formulas are already dropped. Until fixed, cleaning_funnel's step 5
   count and cluster_size_distribution's cluster count both include a
   small number of bogus, non-chemistry clusters -- rerun
   scripts/make_figures.py after the fix and before those two figures
   are treated as final.
6. **zT self-consistency check** — remove rows where |reported zT −
   calculated S²σT/κ| relative error exceeds 50%.
7. **DFT data removal** — keyword search paper metadata ("DFT", "first
   principles", "ab initio", "VASP"), remove.
8. **Multi-source consistency filtering** — group by (formula,
   temperature), remove groups exceeding a per-property CV threshold.
   **Thresholds set by data-driven knee method** (tighten until further
   tightening removes data without changing held-out R²) — NOT via the
   round-robin source (that circularity was identified and closed; the
   round-robin source is used ONLY for the noise-floor anchor, Phase 1
   item 6, nowhere else in the pipeline).
9. **MAD outlier filter** — remove rows exceeding 3.5×MAD from the
   median, log scale for σ, κ. NOT applied to zT (judged against Step 1
   bounds instead).
10. **Minimum temperature coverage** — remove formulas with <3 distinct
    temperature measurements.
11. **Smoothness filter** — rolling median (window=3), flag spikes as NaN,
    remove rows where all properties became NaN.

Reference-scale sanity check (expect similar magnitude on the fresh pull,
not identical values): final dataset previously ~184,167 rows, ~13,605
unique formulas, ~2,834 parent chemical systems. Large deviations at any
step are a signal to stop and investigate, not to proceed.

**Reference per-step row counts (thesis)**, reconstructed from the
thesis's reported step deltas (absolute count given for steps 1, 3, 4, 5;
steps 6-11 given as -count (-%) relative to the previous step, chained
below):

| Step | Thesis row count | Delta from previous step |
|---|---|---|
| 1. Property extraction & range filtering | 1,114,628 | (starting point) |
| 2. Integration & consolidation | not reported | — |
| 3. Temperature filtering 300-800K | 397,651 | — (step 2 not separately reported, so this delta spans steps 2+3 combined) |
| 4. Pivot long→wide | 399,287 | +1,636 -- **inconsistent as given: a wide-pivot collapses multiple property-rows per (sample, temperature) into one row and should REDUCE row count, not increase it. This figure is flagged, not trusted as-is** (see note below) |
| 5. Formula cleaning | 244,834 | -154,453 (-38.7%) |
| 6. zT self-consistency check | 238,667 | -6,167 (-2.5%) |
| 7. DFT data removal | 236,887 | -1,780 (-0.8%) |
| 8. Multi-source consistency filtering | 201,777 | -35,110 (-14.8%) |
| 9. MAD outlier filter | 198,609 | -3,168 (-1.6%) |
| 10. Minimum temperature coverage | 184,687 | -13,922 (-7.0%) |
| 11. Smoothness filter (final) | 184,167 | -~500 (-0.3%) |

Note on step 4: the thesis figure (399,287) is larger than step 3's
(397,651), which is physically implausible for a long-to-wide pivot.
Coincidentally, 399,287 is also this pipeline's OWN step 4 row count on
the 2026-08-15 pull (see commit 413ec37) -- treat this reference value
as unverified/possibly mistranscribed, not as a trustworthy target, until
checked directly against the thesis text.

**Reference final-dataset per-property statistics (thesis)**, on the
184,167-row final dataset:

| Property | Coverage | Mean | Median | Std | Range |
|---|---|---|---|---|---|
| S (μV/K) | 91.4% | 17.3 | 61.5 | 173.9 | [-452, 577] |
| sigma (S/m) | 89.2% | 84,813 | 52,384 | 106,287 | [1,884, 1,200,000] |
| kappa (W/mK) | 63.8% | 2.51 | 2.02 | 1.88 | [0.32, 12.9] |
| zT | 68.0% | 0.44 | 0.33 | 0.39 | [0, 3.55] |

**RESOLVED 2026-08-17: steps 8 and 9 now drop rows, matching the thesis
mechanism.** Found 2026-08-17 comparing the 2026-08-15 pull's funnel
against the thesis's per-step counts: step 8 was removing 0 rows here
(NaN-out only) vs 14.8% in the thesis, and step 9 was removing 0 rows
here vs 1.6% in the thesis. Fixed in `step8_multi_source_consistency`
and `step9_mad_outlier_filter` (src/data_cleaning.py): both now drop
every row in a group/every row with a flagged value, rather than nulling
the offending property and keeping the row. The sigma lower bound was
also corrected from 1 to 10 S/m in the same pass (see item 1 above).

Funnel, same 2026-08-15 raw pull, before vs after the fix vs thesis:

| Step | Before fix | After fix | Thesis | Ratio after/thesis |
|---|---|---|---|---|
| 1. Extraction & range filter | 2,009,248 | 1,992,138 | 1,114,628 | 1.79x |
| 3. Temperature filtering | 1,098,084 | 1,093,377 | 397,651 | 2.75x |
| 4. Pivot long→wide | 399,287 | 397,791 | 399,287 (flagged) | — |
| 5. Formula cleaning | 394,419 | 392,927 | 244,834 | 1.61x |
| 6. zT self-consistency | 389,562 | 388,221 | 238,667 | 1.63x |
| 7. DFT removal | 388,576 | 387,235 | 236,887 | 1.63x |
| 8. Multi-source consistency | 388,576 | 308,656 | 201,777 | 1.53x |
| 9. MAD outlier filter | 388,576 | 289,318 | 198,609 | 1.46x |
| 10. Min. temperature coverage | 379,079 | 284,671 | 184,687 | 1.54x |
| 11. Smoothness filter (final) | 340,831 | 280,348 | 184,167 | **1.52x** |

Final row-count ratio improved from 1.85x to 1.52x. Step 8 now removes
20.3% (vs thesis's 14.8%) and step 9 removes 6.3% (vs thesis's 1.6%) --
we now OVER-remove relative to the thesis at both steps, the opposite
problem from before. This implementation drops a group/row if ANY of
the four properties trips its threshold (OR across properties); the
thesis's actual criterion may check fewer properties, use a looser
per-property threshold, or aggregate differently -- worth revisiting
once find_knee_threshold's knee-tuning (item 8's TODO) is unblocked in
Phase 1.

Final-dataset per-property statistics also moved closer to the thesis
table above (compare against it directly): coverage improved for S
(63.2%→66.0%, thesis 91.4%), sigma (60.3%→65.2%, thesis 89.2%), and zT
(42.0%→46.2%, thesis 68.0%); kappa was flat (43.5%→43.2%, thesis 63.8%).
Mean/median/std moved measurably closer to the thesis for sigma, kappa,
and zT; S's median moved slightly further away (58.4→50.7 vs thesis
61.5) even as its mean improved (25.4→19.5 vs thesis 17.3). Coverage
gaps of 20-25 points remain on all four properties -- consistent with
the still-unresolved step 3 and step 5 divergences below still inflating
the final row count relative to the thesis.

**Still unresolved, NOT touched by this fix:**
- **Step 3 (temperature filtering): 2.75x ratio, the largest remaining
  divergence in the funnel.** Step 2 isn't separately reported by the
  thesis, so this could partly be an unreported step 2 effect, but our
  own step 2 removes 0 rows, making step 3 the leading suspect. Check
  the thesis's exact 300-800K window and binning logic against
  `step3_filter_temperature`.
- **Step 5 (formula cleaning): 1.2% removed here vs 38.7% in the
  thesis.** `step5_clean_formulas` only drops pymatgen-unparseable
  formulas. A 38.7% cut is too large to be parse failures alone --
  likely additional composition-based scope filtering in the thesis
  (candidates: enforcing the "general, non-perovskite scope" from this
  file's Goal section, excluding pure elements/binaries, or
  deduplicating near-identical formulas). Not yet implemented here.

Resolve both before treating this pipeline's funnel or final dataset as
thesis-equivalent.

---

## Paper A — Validation-Inflation Ladder + Descriptor Saturation and the Label-Noise Ceiling (FROZEN)

**Renamed 2026-09-14** (from "Descriptor Ceiling"): the old name
predates the descriptor ablation and promised a descriptor result that
had not yet been measured at the time it was written. Both halves are
now measured and confirmed below: descriptor saturation (the
magpie/cbfv/full ablation) and the label-noise ceiling (the noise-floor
COMBINED CEILING work).

**Novelty framing**: the systematic multi-method quantification + noise-
floor anchor — NOT the leakage insight itself (already stated in Jia et
al. 2024 and the general ML-leakage literature; cite this explicitly to
preempt a reviewer citing it back).

1. **Five-way ladder, denominator-matched**: tune hyperparameters ONCE on
   the chemistry-cluster split via nested CV, freeze. Evaluate the
   identical frozen model under all five schemes (random 80/20, 5-fold,
   10-fold, composition-level, chemistry-cluster) using **pooled
   out-of-fold R²** for every scheme. Repeat the 80/20 split ~20 times and
   pool (it only covers 20% of rows per run, unlike k-fold schemes).
   State explicitly: this measured inflation is a LOWER BOUND on
   real-world practitioner inflation (a practitioner using a leaky split
   would also tune on it, compounding the effect).
   **Implemented 2026-08-19 in src/nested_cv.py** (both gaps closed):
   `tune_once(target=...)` runs the ONE chemistry-cluster-grouped
   Optuna search on the full dataset and saves it to
   `checkpoints/frozen_hyperparams/<target>_<model>.json`;
   `run_nested_cv(..., frozen_hyperparams_path=<that file>)` then reuses
   those hyperparameters unchanged for every outer fold instead of
   retuning, making `--split-strategy` the only varying factor across
   the five rungs. Every `run_nested_cv()` call (frozen or not) now
   computes and reports **pooled out-of-fold R²** — every held-out
   (y_true, y_pred) pair across all outer folds and repeats concatenated
   once and scored with a single `r2_score()` call — as the primary
   metric (`results_df.attrs["pooled_r2"]`), alongside the pre-existing
   per-fold mean/std as a secondary diagnostic (the two can diverge,
   especially for the random-80/20 rung where held-out sets overlap
   across draws). Per-fold predictions are checkpointed to
   `*_predictions.npz` so pooling reconstructs correctly across a
   resumed run, not just folds computed in the current process. Exact
   six-command workflow for one target's full ladder (xgboost, the
   default `--model`):
   ```
   python src/nested_cv.py --tune-once --target zT
   python src/nested_cv.py --split-strategy chemistry   --frozen-hyperparams checkpoints/frozen_hyperparams/zT_xgboost.json --target zT
   python src/nested_cv.py --split-strategy composition --frozen-hyperparams checkpoints/frozen_hyperparams/zT_xgboost.json --target zT
   python src/nested_cv.py --split-strategy kfold  --n-outer-folds 5  --frozen-hyperparams checkpoints/frozen_hyperparams/zT_xgboost.json --target zT
   python src/nested_cv.py --split-strategy kfold  --n-outer-folds 10 --frozen-hyperparams checkpoints/frozen_hyperparams/zT_xgboost.json --target zT
   python src/nested_cv.py --split-strategy random --n-outer-folds 20 --n-repeats 1 --frozen-hyperparams checkpoints/frozen_hyperparams/zT_xgboost.json --target zT
   ```
   Without `--frozen-hyperparams`, `run_nested_cv` still retunes fresh
   per outer fold (the original, pre-ladder behavior) — useful on its
   own, but not the frozen-model ladder comparison this item requires.

   **Model-comparison capability, added 2026-08-19: `--model` flag
   (MODEL_TYPES: `xgboost` default, `lightgbm`, `random_forest`,
   `ridge`).** Supports a model-selection figure (Figure 1) justifying
   XGBoost empirically under honest (chemistry-cluster) grouping, rather
   than asserting it, and doubles as the infrastructure Paper B item 4's
   "two model families... bracket capacity" comparison runs on. Every
   `--model` choice runs through the exact same nested-CV machinery,
   chemistry-cluster grouping, `--split-strategy` rungs,
   frozen-hyperparameter mode, and pooled-OOF-R² reporting above — only
   the Optuna search space and model constructor differ (see
   `MODEL_REGISTRY` in src/nested_cv.py), so `results_df` is directly
   comparable across models via the same schema (`model_type`,
   `pooled_r2`, `pooled_n` columns/attrs). `ridge` is wrapped in a
   `StandardScaler` pipeline (MAGPIE/CBFV/temperature features span very
   different scales; an unscaled Ridge fit would measure that, not real
   capacity) — it is the constrained/lower-capacity bracket model for
   Paper B item 4; `random_forest` is a structurally different
   high-capacity tree ensemble (bagged, not boosted) from `xgboost`/
   `lightgbm`. Hyperparameters are tuned and frozen per model_type
   separately (`tune_once(..., model_type=...)`), never shared or
   compared as values across models, only via each model's resulting
   R². Only `xgboost` has a GPU path in this module; `--device cuda`
   with any other `--model` runs on CPU with a one-time warning. Add
   `--model <name>` to any command in the six-command workflow above to
   run that rung for a different model family (remember to
   `--tune-once --model <name>` first — frozen files are per-model,
   loading the wrong one raises a clear error rather than silently
   reusing another model's hyperparameters).

   **Per-repeat pooling + Nadeau-Bengio test, added 2026-08-22.** The
   all-repeats `pooled_r2` above reports one bare number with no spread
   — cannot say whether a composition-vs-chemistry gap is real or noise.
   `run_nested_cv()` now also pools *within* each repeat (that repeat's
   `n_outer_folds` folds only), giving `n_repeats` separate R² values in
   `results_df.attrs["per_repeat_r2"]`, plus their
   `per_repeat_r2_mean`/`per_repeat_r2_std` — this is the number the
   ladder table should report (mean ± across-repeat SD), not the bare
   pooled figure. Validity of pairing composition's repeat *i* against
   chemistry's repeat *i* (needed for the NB test below) requires both
   rungs to draw identical per-repeat RNG seeds for the same `--seed` —
   verified empirically, not assumed, by `verify_repeat_seed_parity()`
   (runs two real `run_nested_cv()` calls, composition vs. chemistry,
   with `np.random.default_rng` instrumented to record every seed
   constructed; confirmed identical sequence for seed=0, the default
   every rung in the six-command workflow above uses). `nadeau_bengio_test
   (scores_a, scores_b, n_train, n_test)` implements the corrected
   paired t-test (Nadeau & Bengio 2003): inflates the naive paired
   t-test's variance estimate by `(1/k + n_test/n_train)` rather than
   `1/k` alone, correcting for the fact that repeated-CV repeats share
   overlapping training data and aren't independent. Takes two arrays of
   paired per-repeat R² (e.g. chemistry's and composition's
   `per_repeat_r2`, same seed) plus representative single-fold
   `n_train`/`n_test`; returns the mean diff, corrected t-statistic,
   `df=k-1`, and a two-sided p-value.
2. **Nested GroupKFold** for hyperparameter tuning — outer folds for
   reporting only, hyperparameters tuned on inner folds nested inside each
   outer training fold. Never tune and report on the same folds.
3. **Noise floor, computed in per-property matched space (log10 for
   sigma and kappa, linear for S and zT, matching each property's
   confirmed R² scoring space)** (relative uncertainties are
   multiplicative/heteroscedastic, not additive):
   `R²_max = 1 − σ²_noise(log) / σ²_total(log)`, with σ_noise(log) ≈ 0.17
   for zT (from Alleno et al. 2015, Rev. Sci. Instrum. 86:011301, DOI
   10.1063/1.4905250 — S ~6%, σ ~8%, κ ~11%, zT ~17-19%, ONE skutterudite
   compound, cite as inference/lower-bound analogy). σ_total(log) computed
   from this dataset's actual log-property variance. Report model R² in
   log space too, for direct comparability. State the round-robin figure
   is a lower bound on true database noise (excludes digitization error),
   so computed headroom is conservative/optimistic-in-the-paper's-favor —
   state this direction explicitly.
   **Decision (2026-08-20): sigma and kappa are trained on log10-
   transformed targets, not just evaluated in log space post-hoc.**
   sigma spans ~10³–10⁶⁺ S/m and kappa ~0.05–25 W/mK, both multiple
   orders of magnitude — raw-scale squared-error loss (what every model
   here optimizes) is then dominated by the largest-magnitude samples
   and effectively ignores relative error on low-conductivity/low-kappa
   materials; log10 converts this into a relative-error objective, which
   is what actually matters for a property spanning that range. S (can
   be negative, -1000 to 1000 μV/K — log10 undefined) and zT (0–4, not
   multiple orders of magnitude) stay on linear/raw scale. This decision
   also makes this item's "report model R² in log space" requirement
   internally consistent rather than needing a post-hoc conversion step:
   the model is already trained and evaluated in the same space the
   noise-floor ceiling is computed in, for sigma/kappa.
   Implemented in `src/nested_cv.py`: `LOG_TRANSFORM_TARGETS = ("sigma",
   "kappa")`; `_transform_target()` applies `np.log10` right after
   loading `y`, before any device conversion, in both `tune_once()` and
   `run_nested_cv()` — every downstream R² (`outer_r2`, `inner_cv_r2`,
   `pooled_r2`) is therefore computed in log10 space for sigma/kappa,
   linear space for S/zT, never mixed. `results_df.attrs["target_scale"]`
   and each checkpoint record's `"target_scale"` field record which
   explicitly, so this is never ambiguous downstream (e.g. in a
   cross-target comparison figure). `target_scale` is a FATAL resume key
   in `_check_run_config` — a sigma/kappa `checkpoint_dir` predating this
   decision cannot be silently resumed and have its old linear-scale
   predictions mixed into a new pooled log-space R²; it raises instead,
   same protection already in place for `model_type` mismatches.
   Positivity for `log10` is guaranteed by `data_cleaning.py` step 1's
   bounds (sigma ≥ 10, kappa ≥ 0.05), not re-checked in `nested_cv.py`.
4. **Unique-formula featurization** — compute descriptors once per unique
   formula, not per row (~13x compute reduction). CPU is sufficient, no
   GPU needed.
5. **Direct-vs-derived zT pathway** — direct zT prediction vs. derived
   S²σT/κ. Restrict BOTH pathways to the identical all-four-properties-
   present subset and identical splits (no training-set-size confound).
   Report component-error correlation structure, don't assume independence.
6. **External validation, TWO numbers, not merged**: (a) source-
   deduplicated (no shared DOI with training) — tests measurement
   transfer; (b) composition-cluster-deduplicated (no shared chemistry
   cluster with training) — tests chemistry transfer. Report dropped-row
   count for each. ESTM and teMatDb each touched exactly once, after the
   model is fully frozen.
7. **DROPPED (2026-09-14).** **Screening rediscovery, targeted holdout**: hold out ONLY specific
   known-good targets + their exact canonical duplicates (not all high-zT
   materials broadly) — retain other high-zT materials so the model keeps
   a performance signal to generalize from. Interpret result as caveated
   tail-extrapolation. If the high-zT region is too thin after holdout,
   report that finding directly rather than forcing a positive result.
   **Reason dropped**: the paper's four confirmed results (the Five-Way
   Ladder, the SHAP attribution negative control, the label-noise
   ceiling, and descriptor saturation) plus cross-database transfer
   (ESTM/teMatDb external validation) already constitute a full paper;
   this coda supports none of those contributions directly. Item text
   retained above, unedited, so the design survives if this is ever
   revived.
8. **Temperature axis — no dedicated extrapolation experiment; temperature
   STAYS a per-row model feature.** zT is typically non-monotonic in
   temperature (peaks then rolls over, often within 600-800K) — no
   extrapolator, tree-based or otherwise, can recover this from only the
   ≤600K rising branch. This is a data-coverage fact, not a
   generalization finding, so do not build a dedicated temperature-
   extrapolation EXPERIMENT (e.g. train on ≤600K / test on >600K, or a
   results section arguing temperature generalization) as a full
   experimental axis — write it as one limitation paragraph instead.
   **This is a decision about which EXPERIMENT to run, not about what
   goes into the model.** temperature_bin is part of every row already
   (each row is one formula at one temperature) and stays in the
   feature set for every experiment in this project, the same as any
   other per-row input.
   **RESOLVED 2026-08-18: this item's wording was ambiguous and had been
   misread as "exclude temperature from the feature set," not just "skip
   the extrapolation experiment."** src/nested_cv.py's
   get_feature_columns() dropped temperature_bin entirely (only
   MagpieData/CBFV_-prefixed columns were selected), so every model in
   this project was trained without ever seeing temperature. Found
   2026-08-18 comparing against the actual thesis dataset/feature list
   (C:\Users\choha\Downloads\archive\MASTER_DATASET_FINAL.xls +
   features_v2.json): the thesis's 335-feature set is 132 MAGPIE + 202
   non-Magpie descriptor columns + T_K itself, and its reported R^2
   (~0.70) is measured against models trained WITH temperature as an
   input; this pipeline's calibration run (~0.50, see the Grouping Key
   section's compute-budget note, "src/nested_cv.py compute budget")
   omitted it entirely -- not a like-for-like comparison, and a likely
   major contributor to the gap. Fixed in get_feature_columns() to
   include temperature_bin alongside the MagpieData/CBFV_ columns. Any
   nested_cv.py results computed before this fix (including that
   section's 0.5045 calibration number) were trained without
   temperature and should not be treated as representative of the
   frozen model going forward.
9. **PCA-split engagement, empirical not asserted**: run chemistry-cluster
   CV AND a PCA-based split (Athar/Jund's method) on the same data, report
   both honest numbers, state why chemistry-cluster grouping is stricter
   against near-duplicate leakage specifically.

---

**Process fix (2026-08-22): every confirmed number below must cite its
checkpoint directory (or frozen-hyperparameter file) as its source.**
The chemistry-rung sigma/kappa discrepancy below traced back to a
headline number with no surviving `run_config.json` or checkpoint
directory to verify it against — an orphaned number that could not be
reproduced or audited after the fact. Going forward, a result is not
"confirmed" in this document unless its producing checkpoint directory
(local path, or frozen-hyperparameter file) is named alongside it, so
an orphaned headline number cannot be created again.

## Confirmed Results — Five-Way Ladder (Paper A item 1, FINAL, updated 2026-09-19)

Pooled out-of-fold R², frozen hyperparameters per target (tuned once via
`tune_once`, reused unchanged across all five rungs), 397 features
(MAGPIE + CBFV + temperature_bin), XGBoost. sigma and kappa trained and
scored in log10 space per the frozen decision above; S and zT in linear
space. Composition and chemistry-cluster report mean ± across-repeat SD
(5 repeats, per-repeat pooled R², see the per-repeat pooling note
above); random 80/20 and 5-fold/10-fold report the single pooled R²
(n_repeats=1 for ungrouped rungs, per the frozen decision in the
Grouping Key section — no OUTER-repeat-to-repeat spread to report
there). **Corrected 2026-09-22**: this does not mean no spread at all —
random 80/20 pools 20 independent `ShuffleSplit` draws, and 5-fold/
10-fold each have their own per-fold spread; the tables below now
report that per-draw/per-fold SD alongside the pooled value. See the
SUPERSEDED note after the replacement table for why this section's
ungrouped columns needed regenerating, separately from the composition/
chemistry-cluster fix.

| Target | random 80/20 | 5-fold | 10-fold | composition | chemistry cluster |
|---|---|---|---|---|---|
| S | 0.9588 | 0.9588 | 0.9595 | 0.8331 ± 0.0034 | 0.8076 ± 0.0018 |
| sigma (log10) | 0.9152 | 0.9150 | 0.9175 | 0.7772 ± 0.0019 | 0.7600 ± 0.0008 |
| kappa (log10) | 0.9442 | 0.9444 | 0.9459 | 0.8562 ± 0.0010 | 0.8460 ± 0.0011 |
| zT | 0.9186 | 0.9184 | 0.9196 | 0.8174 ± 0.0019 | 0.7968 ± 0.0030 |

**SUPERSEDED 2026-09-19: the composition and chemistry-cluster columns
above.** Regenerated after two grouping-code fixes (SNAP(0.05)
`chemistry_cluster_id`, S5 `randomized_group_kfold`, both commit c1c6873)
and the resulting snapfix featurized CSV -- see the Grouping Fixes section
above. The random 80/20, 5-fold, and 10-fold columns are unaffected and
unchanged: they never call `chemistry_cluster_id` or
`randomized_group_kfold`, verified directly. Replacement table:

| Target | random 80/20 | 5-fold | 10-fold | composition | chemistry cluster |
|---|---|---|---|---|---|
| S | 0.9588 | 0.9588 | 0.9595 | 0.8322 ± 0.0044 | 0.7528 ± 0.0050 |
| sigma (log10) | 0.9152 | 0.9150 | 0.9175 | 0.7762 ± 0.0015 | 0.7020 ± 0.0020 |
| kappa (log10) | 0.9442 | 0.9444 | 0.9459 | 0.8565 ± 0.0013 | 0.8092 ± 0.0021 |
| zT | 0.9186 | 0.9184 | 0.9196 | 0.8178 ± 0.0009 | 0.7456 ± 0.0045 |

Provenance: `results/ladder_regen_snapfix/20260917T150000/
{S,sigma,kappa,zT}_chemistry_full/` (chemistry) and the same directory's
`{S,sigma,kappa,zT}_composition_full/` (composition), tabulated in
`reports/regen_snapfix/20260917T150000/ladder_metrics.json`. Random 80/20,
5-fold, and 10-fold columns carried over unchanged from the 2026-08-22
provenance below.

**SUPERSEDED 2026-09-22: the random 80/20, 5-fold, and 10-fold columns
above, and the "carried over unchanged" / "provenance-cited, 2026-08-22"
claims below for those three columns.** The `checkpoints/ladder_regen_dl/`
directory those columns actually came from has no recorded CSV path or
SHA256 anywhere in it (`progress.log` logs only row counts and per-fold
R², never a source file) — despite being described here as
"provenance-cited". Its row counts (S 185,844 / sigma 183,246 / kappa
121,535 / zT 129,851) do not match File A/snapfix (S 185,064 / sigma
182,755 / kappa 121,110 / zT 129,419) on any target, and are larger than
File B's too — an unidentified fourth dataset snapshot, not File A. The
composition column it also produced is superseded above already (by the
snapfix regeneration); only random 80/20, 5-fold, and 10-fold still
carried this stale, unidentified-dataset provenance until now. See the
Canonical Dataset section's correction to the "File A produced the whole
ladder" claim.

Rerun on the snapfix CSV (`ungrouped_snapfix.zip`, verified row counts
S 185,064 / sigma 182,755 / kappa 121,110 / zT 129,419 — File A's own
counts, confirmed against `load_target_data`). No run's pooled-vs-
per-fold-mean difference exceeds 0.0005; pooled R² reported below for
consistency with composition/chemistry's convention, with the per-fold/
per-draw SD shown alongside (previously omitted for these three columns
entirely). Replacement table, all five rungs now from the snapfix
dataset:

| Target | random 80/20 (20 draws) | 5-fold | 10-fold | composition | chemistry cluster |
|---|---|---|---|---|---|
| S | 0.9582 ± 0.0013 | 0.9585 ± 0.0013 | 0.9595 ± 0.0014 | 0.8322 ± 0.0044 | 0.7528 ± 0.0050 |
| sigma (log10) | 0.9150 ± 0.0009 | 0.9152 ± 0.0009 | 0.9174 ± 0.0024 | 0.7762 ± 0.0015 | 0.7020 ± 0.0020 |
| kappa (log10) | 0.9434 ± 0.0012 | 0.9436 ± 0.0013 | 0.9455 ± 0.0021 | 0.8565 ± 0.0013 | 0.8092 ± 0.0021 |
| zT | 0.9180 ± 0.0014 | 0.9181 ± 0.0029 | 0.9193 ± 0.0032 | 0.8178 ± 0.0009 | 0.7456 ± 0.0045 |

Provenance: `results/ungrouped_snapfix/20260922T093243/
{S,sigma,kappa,zT}_{random_f20,kfold_f5,kfold_f10}/`, tabulated in
`reports/ungrouped_snapfix/20260922T093243/metrics.json` and `table1.md`.
The random rung is **20 independent, overlapping 80/20 holdout draws via
sklearn `ShuffleSplit`, pooled** — not a single split; this description
applies everywhere the random rung is reported in this document, per the
Grouping Key section's own (already-correct) note on this point. Deltas
from the superseded `ladder_regen_dl` values are small (|delta| <=
0.0008 on every cell) and do not change any qualitative conclusion —
the dataset mismatch was real but its effect on the ungrouped rungs
specifically was negligible; see the report for the full delta table.

**All 20 cells are reproducible, provenance-cited, 2026-08-22.** Every
cell above comes from a checkpointed run with a surviving
`run_config.json` (seed=0, n_outer_folds/n_repeats as described above,
frozen hyperparameters from `checkpoints/saved_predictions/checkpoints/
frozen_hyperparams/{S,sigma,kappa,zT}.json`, the same files force-added
to git):

- chemistry cluster: `checkpoints/saved_predictions/checkpoints/
  {S,sigma,kappa,zT}_chemistry/`
- composition, random 80/20, 5-fold, 10-fold: `checkpoints/
  ladder_regen_dl/{S,sigma,kappa,zT}_{composition,random,kfold,kfold}
  _{f5,f20,f5,f10}/` (folder-name suffix is outer-fold count, e.g.
  `sigma_random_f20`, `kappa_kfold_f10`)

**SUPERSEDED 2026-09-22: the "provenance-cited" claim above, for the
`ladder_regen_dl` bullet only.** This was overstated: a `run_config.json`
existing and being internally self-consistent is not the same as the
underlying dataset being identified. Only the chemistry-cluster bullet
(File A, `checkpoints/saved_predictions/checkpoints/`) was genuinely
verified against a known, hashed dataset at the time. The
`ladder_regen_dl` bullet's random/5-fold/10-fold/composition runs had a
`run_config.json` (seed, fold count, hyperparameters path) but never a
dataset path or hash — "reproducible" was true in the narrow sense that
rerunning gave the same numbers, not in the sense that the dataset
itself could be identified or confirmed to be File A. It has since been
superseded by `results/ungrouped_snapfix/20260922T093243/` for the three
ungrouped columns; the composition column was already superseded by the
snapfix regeneration above.

This replaces every orphaned point estimate from the lost-config run
(prior chemistry-cluster: S=0.8083, sigma=0.7522, kappa=0.8226,
zT=0.7965; prior random/5-fold/10-fold/composition numbers were from
that same run) — none of which could be verified or reproduced (see
discrepancy diagnosis below). Nothing in this table is orphaned any
more.

The three ungrouped rungs (random/5-fold/10-fold) cluster tightly
within ~0.001-0.003 of each other for every target, then drop sharply
at composition-level grouping and drop again, more modestly, at the
chemistry-cluster anchor — the honest ceiling this project reports.
**SUPERSEDED 2026-09-19** (pre-regeneration gap, see the replacement table
above): the ungrouped-to-chemistry-cluster gap (random 80/20 minus chemistry
cluster) is largest for sigma (0.9152 to 0.7600, 15.5 points) and
smallest for kappa (0.9442 to 0.8460, 9.8 points).

Updated 2026-09-19, against the regenerated chemistry column: the gap
ranges from 0.135 (kappa) to 0.213 (sigma) across the four targets, mean
0.182 -- same ordering as before (largest for sigma, smallest for kappa),
substantially wider gap.

**SUPERSEDED 2026-09-22: the gap figures immediately above.** The random
80/20 side of that comparison was still the unidentified-dataset
`ladder_regen_dl` value at the time. Recomputed against the
`results/ungrouped_snapfix/20260922T093243/` rerun (same rows as the
chemistry-cluster column on both sides now): the gap ranges from
**0.134 (kappa) to 0.213 (sigma)**, mean **0.181** -- same ordering,
essentially unchanged, since the dataset mismatch moved the ungrouped
side by at most 0.0008 per target (see the replacement table's
provenance note above for the per-target deltas).

**Discrepancy diagnosis (2026-08-22): H1 (scale mismatch) is
UNRESOLVABLE, not rejected.** Tested whether the orphaned run's
sigma=0.7522/kappa=0.8226 were linear-space scores from before the
2026-08-20 log10 decision, by back-transforming the reproducible run's
log10-space predictions to linear space and rescoring: sigma landed at
0.6547 linear vs. 0.7600 log10 (orphaned value 0.7522 sits close to the
log10 number); kappa landed at 0.8015 linear vs. 0.8460 log10 (orphaned
value 0.8226 sits roughly between the two, closer to neither). This
neither confirms nor rules out H1: back-transforming a log10-TRAINED
model's predictions to linear space and rescoring is not equivalent to
what a model actually TRAINED on linear targets would have produced
(different loss function during training, not just different scoring),
so the test cannot distinguish "the orphaned run trained on linear
targets" from "the orphaned run trained on log10 targets with different
hyperparameters or data" — both remain consistent with the orphaned
numbers. The discrepancy was resolved not by identifying which of these
is true (the orphaned run's config no longer exists to check) but by
regenerating every rung from scratch on Kaggle with a preserved config,
per the process fix above.

**Composition-vs-chemistry Nadeau-Bengio corrected paired t-test**
(`nadeau_bengio_test()`, `src/nested_cv.py`), paired per-repeat
differences (composition minus chemistry), k=5 repeats,
seed alignment verified by `verify_repeat_seed_parity()` (both rungs'
`run_config.json` confirm `seed=0`, `n_repeats=5`, `n_outer_folds=5`).
Variance correction: `t = mean(d) / sqrt((1/k + n_test/n_train) *
var(d))`, `d` = the 5 paired (composition − chemistry) per-repeat
differences, `var(d)` with `ddof=1`, `n_train`/`n_test` = composition
rung's mean per-fold training/test set sizes (5-fold outer structure),
`df = k - 1 = 4`:

| Target | mean diff (comp − chem) | t | df | p |
|---|---|---|---|---|
| S | +0.0255 | 8.972 | 4 | 0.0009 |
| sigma | +0.0171 | 9.607 | 4 | 0.0007 |
| kappa | +0.0102 | 8.729 | 4 | 0.0009 |
| zT | +0.0206 | 8.550 | 4 | 0.0010 |

All four significant at p<0.001: composition scores significantly
higher (less strict) than chemistry-cluster for every property,
confirming chemistry-cluster grouping is a measurably stricter honest-
ceiling anchor, not just numerically different by chance or repeat-to-
repeat noise.

**SUPERSEDED 2026-09-19.** Computed against the pre-fix composition/
chemistry rungs. Regenerated against the fixed grouping (SNAP(0.05)
`chemistry_cluster_id`, S5 `randomized_group_kfold`, both commit c1c6873)
and the snapfix featurized CSV -- see the Five-Way Ladder table's
2026-09-19 update above. Replacement table, same method:

| Target | mean diff (comp − chem) | t | df | p |
|---|---|---|---|---|
| S | +0.0793 | 29.178 | 4 | <0.001 |
| sigma | +0.0742 | 41.511 | 4 | <0.001 |
| kappa | +0.0473 | 28.949 | 4 | <0.001 |
| zT | +0.0722 | 24.360 | 4 | <0.001 |

**Report the mean-diff effect size as the headline number, not t.** At
df=4 (k=5 repeats), the Nadeau-Bengio corrected t-statistic is unstable,
and a large t is not on its own remarkable -- a small number of repeats
lets t grow large without the underlying effect being any more
physically meaningful. The mean diff (composition minus chemistry,
+0.047 to +0.079 above) is the number that carries physical meaning; t
and p establish that the direction is not noise, not how large the
effect is.

## Confirmed Results — Noise Floor (Paper A item 3, FINAL)

R²_max computed per-property matched space (log10 for sigma/kappa,
linear for S/zT — matching each property's confirmed R² scoring space,
see item 3 above), Alleno et al. 2015 round-robin relative uncertainties
as the noise reference. Implemented in `src/noise_floor.py`.
Inputs (relative_uncertainty, sigma_noise, sigma_total, n_used per
property/space, plus the input dataset path/SHA256 and checkpoint
provenance) persisted at `results/noise_floor/20260910T134042/
noise_floor_inputs.json`, computed against `data/processed/
cleaned_ThermoelectricMaterials_2026-08-15.csv`.

| Target | Scale | R²_max | Confirmed ceiling | Headroom |
|---|---|---|---|---|
| S | linear | 0.9974 | 0.8076 | 0.190 |
| sigma | log10 | 0.9968 | 0.7600 | 0.237 |
| kappa | log10 | 0.9776 | 0.8460 | 0.132 |
| zT | linear | 0.9785 | 0.7968 | 0.182 |

Confirmed ceiling = the chemistry-cluster mean from the reproducible
2026-08-22 checkpoint set at `checkpoints/saved_predictions/checkpoints/
{S,sigma,kappa,zT}_chemistry/` (see the Five-Way Ladder section above).
Headroom inherits that set's across-repeat SD (S ±0.0018, sigma ±0.0008,
kappa ±0.0011, zT ±0.0030) as its ceiling-side uncertainty, but the
dominant uncertainty remains R²_max's optimistic-bound caveat below, not
this repeat-to-repeat spread. This table is FINAL: its only dependency
(the chemistry-cluster ceiling) is now reproducible and provenance-clean
— unlike the Five-Way Ladder table above, which stays non-final until
its four orphaned rungs are regenerated.

**R²_max column verified unchanged, 2026-09-19.** R²_max depends only on
the cleaned dataset's own property variance and the Alleno et al. noise
reference, neither of which the grouping fixes touch. Rerun directly
against the identical cleaned CSV (byte-identical SHA256): exactly zero
delta on every property/space cell
(`results/noise_floor/20260917T172251/`). The R²_max column above is
unaffected and remains current.

**SUPERSEDED 2026-09-19: the Confirmed ceiling and Headroom columns
above, and the FINAL claim's stated dependency.** Both were tied to the
chemistry-cluster ladder, itself superseded by the grouping fixes
(SNAP(0.05) `chemistry_cluster_id`, S5 `randomized_group_kfold`, commit
c1c6873) -- see the Five-Way Ladder table's 2026-09-19 update.
Replacement table:

| Target | Scale | R²_max | Confirmed ceiling | Headroom |
|---|---|---|---|---|
| S | linear | 0.9974 | 0.7528 | 0.245 |
| sigma | log10 | 0.9968 | 0.7020 | 0.295 |
| kappa | log10 | 0.9776 | 0.8092 | 0.168 |
| zT | linear | 0.9785 | 0.7456 | 0.233 |

Confirmed ceiling = the chemistry-cluster mean from
`results/ladder_regen_snapfix/20260917T150000/
{S,sigma,kappa,zT}_chemistry_full/` (see the Five-Way Ladder section's
2026-09-19 update). Headroom's ceiling-side across-repeat SD: S ±0.0050,
sigma ±0.0020, kappa ±0.0021, zT ±0.0045.

R²_max is a best-case upper bound: Alleno et al. is a single-compound
(skutterudite) round-robin measurement excluding digitization error, so
true database noise is higher than this and true headroom is smaller
than shown — state this direction explicitly wherever these numbers are
cited, per item 3's frozen instruction.

**COMBINED CEILING (measurement + digitization noise), added 2026-09-10 --
SUPERSEDED 2026-09-11, digitization column below used File B; see the
updated table after it. Retained for audit, not deleted.**

| Target | R2_max (Alleno, measurement) | Digitization ceiling | Combined | Confirmed | Headroom |
|---|---|---|---|---|---|
| S                                  | 0.9974 | 0.963-0.981 | 0.960-0.979 | 0.8076 | 0.153-0.171 |
| sigma                              | 0.9968 | 0.978-0.989 | 0.975-0.986 | 0.7600 | 0.215-0.226 |
| kappa                              | 0.9776 | 0.981-0.990 | 0.958-0.968 | 0.8460 | 0.112-0.122 |
| zT (vs ZT_author_declared)         | 0.9785 | 0.960-0.980 | 0.938-0.958 | 0.7968 | 0.141-0.162 |
| zT (vs recomputed alpha^2*T/(rho*kappa)) | 0.9785 | 0.961-0.980 | 0.939-0.959 | 0.7968 | 0.142-0.162 |

Digitization ceiling from composition-matched cross-database label
agreement, teMatDb vs Starrydata2, restricted to 300-800 K, section N of
reports/tematdb_inventory/inventory_report.md, commit 27ac09f. Per-property
support: S 824 points / 86 samples; sigma 798 / 79; kappa 722 / 86; zT
885 / 87 (declared) and 853 / 86 (recomputed). Each property's agreement
was computed in the same space as its PAPER_SCALE entry (linear for S and
zT, log10 for sigma and kappa); this was verified before combining, since
the noise fractions are not additive across differing spaces. Lower bound
of each range is raw agreement R^2; upper bound is (1+R2_agree)/2, which
corrects for both labels being noisy under an equal-noise assumption.
Combined via additive noise fractions: (1-R2_comb) = (1-R2_meas) +
(1-R2_dig), valid because the Alleno round-robin measures inter-lab
instrumental scatter and explicitly excludes digitization error, so the
two terms are independent. Two zT rows are reported because teMatDb
carries both an author-digitized zT and one recomputed from its own
alpha, rho, kappa; they differ by 0.001, so target construction does not
drive the result. The digitization fraction was measured on a
narrower-variance subset (79-87 of 176 DOI-overlap samples) than the full
database, so it overstates the penalty and the combined ceiling is
conservative. Both terms remain lower bounds on total label noise: neither
captures synthesis-to-synthesis variation.

**Also SUPERSEDED 2026-09-19** (Confirmed/Headroom columns, same reason
as the table below): see the replacement table after the File A/UPDATED
table below, which supersedes both this table's and that one's
Confirmed/Headroom columns identically.

**COMBINED CEILING, UPDATED 2026-09-11 (File A digitization values).**

| Target | R2_max (Alleno, measurement) | Digitization ceiling | Combined | Confirmed | Headroom |
|---|---|---|---|---|---|
| S | 0.9974 | 0.964-0.982 | 0.961-0.979 | 0.8076 | 0.154-0.172 |
| sigma | 0.9968 | 0.984-0.992 | 0.981-0.989 | 0.7600 | 0.221-0.229 |
| kappa | 0.9776 | 0.983-0.992 | 0.961-0.969 | 0.8460 | 0.115-0.123 |
| zT (vs ZT_author_declared) | 0.9785 | 0.981-0.990 | 0.959-0.969 | 0.7968 | 0.162-0.172 |
| zT (vs recomputed alpha^2*T/(rho*kappa)) | 0.9785 | 0.984-0.992 | 0.962-0.971 | 0.7968 | 0.166-0.174 |

Same formulas as the File B table above (lower bound = raw R2_agree,
upper bound = (1+R2_agree)/2, combined via additive noise fractions),
digitization values from
`results/20260911T114356_tematdb_inventory_fileA/inventory_fileA.json`
section N3, 300-800K. **Snapshot sensitivity: the digitization column
alone moves by up to +/-0.02 between File A and File B (largest for zT,
see CAVEATS (c) above) -- treat this table's bounds as carrying that
additional uncertainty on top of the stated range, not as a tighter
estimate than the File B table above.**

**R2_max and Digitization ceiling columns verified unchanged, 2026-09-19.**
Neither depends on `chemistry_cluster_id` or `randomized_group_kfold`:
R2_max comes from the cleaned dataset's own property variance (confirmed
byte-identical rerun, exact zero delta -- see the simple Noise Floor
table's 2026-09-19 update above); Digitization ceiling comes from
composition-matched teMatDb-vs-Starrydata2 label agreement
(`composition_id`-based, confirmed byte-identical to 4 decimals in
`results/external_snapfix/20260917T160553/`). The Combined column, being
derived from only these two, is therefore also unchanged.

**SUPERSEDED 2026-09-19: the Confirmed and Headroom columns above** (both
in this table and in the File B digitization table before it) --
superseded by the fixed grouping's chemistry rung, see the Five-Way
Ladder table's 2026-09-19 update. Replacement table:

| Target | R2_max (Alleno, measurement) | Digitization ceiling | Combined | Confirmed | Headroom |
|---|---|---|---|---|---|
| S | 0.9974 | 0.964-0.982 | 0.961-0.979 | 0.7528 | 0.2085-0.2266 |
| sigma | 0.9968 | 0.984-0.992 | 0.981-0.989 | 0.7020 | 0.2787-0.2868 |
| kappa | 0.9776 | 0.983-0.992 | 0.961-0.969 | 0.8092 | 0.1517-0.1601 |
| zT (vs ZT_author_declared) | 0.9785 | 0.981-0.990 | 0.959-0.969 | 0.7456 | 0.2137-0.2233 |
| zT (vs recomputed alpha^2*T/(rho*kappa)) | 0.9785 | 0.984-0.992 | 0.962-0.971 | 0.7456 | 0.2168-0.2249 |

Confirmed = the chemistry-cluster mean from
`results/ladder_regen_snapfix/20260917T150000/`. Full arithmetic and
provenance in
`results/noise_floor/20260917T172251/noise_floor_inputs.json` and
`results/noise_floor/20260917T172251/combined_ceiling_snapfix.md`.

## Confirmed Results — Descriptor Ablation (2026-09-12)

**Method**: chemistry-cluster split only (the honest-ceiling rung, not
the full five-way ladder), frozen hyperparameters -- the SAME ones tuned
once on the full 397-feature set
(`checkpoints/saved_predictions/checkpoints/frozen_hyperparams/
{S,sigma,kappa,zT}.json`) -- applied UNCHANGED to the 133-feature
(MagpieData only) and 265-feature (CBFV_ only) subsets, via
`src/nested_cv.py`'s `--feature-set` flag. **Stated explicitly as a
methods choice**: hyperparameters are NOT retuned per feature set, so any
R² difference between magpie/cbfv/full isolates the descriptor set's own
effect, not a confound from re-tuning on a smaller feature space. 5
repeats x 5 outer folds each (25 outer evaluations), identical protocol
to the confirmed ladder.

**Full table** (per-repeat pooled R² = mean +/- across-repeat SD, k=5;
sigma/kappa scored in log10 space, S/zT in linear space, matching each
target's own `target_scale`):

| Target | magpie (133 feat) | cbfv (265 feat) | full (397 feat) |
|---|---|---|---|
| S | 0.8038 +/- 0.0031 | 0.8060 +/- 0.0025 | 0.8076 +/- 0.0018 |
| sigma (log10) | 0.7497 +/- 0.0010 | 0.7585 +/- 0.0013 | 0.7600 +/- 0.0008 |
| kappa (log10) | 0.8419 +/- 0.0009 | 0.8436 +/- 0.0013 | 0.8460 +/- 0.0011 |
| zT | 0.7940 +/- 0.0016 | 0.7945 +/- 0.0042 | 0.7968 +/- 0.0030 |

**SUPERSEDED 2026-09-19: the full table above.** Computed against the
pre-fix chemistry rung. Regenerated against the fixed grouping (SNAP(0.05)
`chemistry_cluster_id`, S5 `randomized_group_kfold`, commit c1c6873) and
the snapfix featurized CSV. Replacement table:

| Target | magpie (133 feat) | cbfv (265 feat) | full (397 feat) |
|---|---|---|---|
| S | 0.7464 +/- 0.0039 | 0.7502 +/- 0.0029 | 0.7528 +/- 0.0050 |
| sigma (log10) | 0.6919 +/- 0.0011 | 0.7018 +/- 0.0020 | 0.7020 +/- 0.0020 |
| kappa (log10) | 0.8028 +/- 0.0025 | 0.8048 +/- 0.0026 | 0.8092 +/- 0.0021 |
| zT | 0.7406 +/- 0.0024 | 0.7443 +/- 0.0042 | 0.7456 +/- 0.0045 |

**Deltas, in absolute R², not SD units:**

| Target | full - magpie | full - cbfv |
|---|---|---|
| S | +0.0038 | +0.0017 |
| sigma | +0.0103 | +0.0015 |
| kappa | +0.0042 | +0.0025 |
| zT | +0.0028 | +0.0023 |

**SUPERSEDED 2026-09-19: full-minus-magpie deltas above** (same reason).
New full-minus-magpie deltas: S +0.0065, sigma +0.0101, kappa +0.0064,
zT +0.0049.

**Do not report these as multiples of the across-repeat SD.** The SD
here measures fold-reshuffling precision (how much the pooled R² moves
if the same model/data is re-split into different chemistry-cluster
folds with a different seed) -- it is a precision estimate, not an
effect-size yardstick. At this dataset's row count (~185K-280K per
target), the SD is small enough (0.0008-0.0031 across all three feature
sets) that almost any systematic difference clears 2x SD: full-vs-magpie
clears it on S/sigma/kappa (2.1x/12.6x/3.8x), full-vs-cbfv clears it only
on kappa (2.2x). Reporting "N times the SD" alongside these small
absolute deltas overstates how material the difference is -- it answers
"is this distinguishable from fold-reshuffling noise" (usually yes,
given enough rows), not "does this matter for the paper's claims" (see
the headroom framing below instead).

**Fraction-of-headroom framing** (full-minus-magpie as a percentage of
the File A COMBINED CEILING table's headroom range, the canonical
combined-ceiling table above -- headroom is the honest ceiling's
remaining gap to the noise/digitization-limited ceiling, so this states
how much of that theoretically-recoverable gap the full descriptor
set's extra 264 features, over magpie-only, actually close):

- S: 0.003840 / [0.154, 0.172] = **2.23%-2.49%**
- sigma: 0.010292 / [0.221, 0.229] = **4.49%-4.66%**
- kappa: 0.004184 / [0.115, 0.123] = **3.40%-3.64%**
- zT: 0.002804 / [0.162, 0.172] (declared) or [0.166, 0.174]
  (recomputed TEP) = **1.61%-1.73%**

**SUPERSEDED 2026-09-19: the fraction-of-headroom figures above.** Both
the delta (full-minus-magpie) and the headroom denominator changed --
see the Five-Way Ladder and COMBINED CEILING tables' 2026-09-19 updates.
New fractions, against the NEW headroom: S 2.86%-3.10%, sigma
3.52%-3.63%, kappa 3.97%-4.19%, zT 2.21%-2.31% (declared) or
2.20%-2.28% (recomputed TEP).

Full descriptor coverage over magpie-only closes under 5% of headroom
on every target; the conclusion is the same one reached against the
now-superseded File B combined-ceiling table (that table's parenthetical
comparison has been dropped here rather than kept alongside the
canonical File A numbers, to avoid inviting a reader to cite the wrong
table).

**cbfv-alone saturates too, in both directions**: full-minus-cbfv is
0.0015-0.0025 across all four targets -- the same order of magnitude as
full-minus-magpie's smallest value (zT, 0.0028) and well inside the
noise-floor's own optimistic-bound caveat. Descriptor headroom in this
dataset is small under either restriction, not only under the smaller
(magpie) one.

**Provenance**: `results/descriptor_ablation/20260912T184318/` (8 run
directories -- `{S,sigma,kappa,zT}_chemistry_{magpie,cbfv}` -- plus
`environment.txt`, copied from the Kaggle P100 run that produced them;
per-row predictions saved for all 25 folds x 8 runs).
`environment.txt` records Python 3.12.13, numpy==1.26.4,
optuna==3.6.1, xgboost==2.0.3, scikit-learn==1.4.2, CUDA 12.8.1, Tesla
P100-PCIE-16GB driver 580.159.04. Computation method (pooled and
per-repeat-pooled R² from saved per-row predictions, identical code for
magpie/cbfv/full) reported in
`reports/descriptor_ablation/20260912T184318/ablation_table.md` and
`ablation_metrics.json`.

**SUPERSEDED 2026-09-19: the provenance above, for the new table/deltas/
fractions only** (the OLD 2026-09-12 provenance remains accurate for the
OLD, superseded numbers it always described). New provenance:
`reports/ablation_snapfix/20260918T000111/ablation_table.md` and
`ablation_metrics.json`, computed from the snapfix descriptor ablation at
`results/descriptor_ablation_snapfix/20260917T184650/` and the snapfix
chemistry rung at `results/ladder_regen_snapfix/20260917T150000/`.

## Confirmed Results — SHAP Attribution Comparison (2026-09-13)

**Method**: zT only, chemistry-cluster split (the honest-ceiling rung) and
random 80/20 split, repeat 0, all 5 outer folds each (10 fits total), the
SAME frozen hyperparameters as the confirmed ladder
(`checkpoints/saved_predictions/checkpoints/frozen_hyperparams/zT.json`).
Fold construction reused unchanged from `src/nested_cv.py`'s own
`outer_splits()` (see `scripts/shap_attribution_zT.py`), not
reimplemented. Attributions computed via native XGBoost exact TreeSHAP
(`Booster.predict(dmatrix, pred_contribs=True)`) -- not the `shap`
package, which requires numpy>=2 against this project's pinned
numpy==1.26.4 -- with the trailing bias/expected-value column dropped
before normalizing. Per fold: a fixed, seeded 20,000-row random
subsample of that fold's own test set (seed 0 throughout; every one of
the 10 folds had a test set larger than 20,000 rows, so the full 20,000
was used every time, never truncated). Per-feature mean(|SHAP|) is
normalized to shares over the full **397-feature** set (confirmed via
`summary.json`'s `n_features` field -- 397, not 398, so the bias column
was correctly excluded before normalizing). Device: cuda (Kaggle GPU
runtime, xgboost==2.0.3, numpy==1.26.4, scikit-learn==1.4.2,
optuna==3.6.1, CUDA toolkit 12.8.1, per the run's own `environment.txt`).

**R² per split, pooled across the 5 folds**:

| split_strategy | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | pooled |
|---|---|---|---|---|---|---|
| random | 0.9155 | 0.9156 | 0.9187 | 0.9188 | 0.9164 | 0.9170 |
| chemistry | 0.8018 | 0.7791 | 0.7981 | 0.8181 | 0.7911 | 0.7973 |

Pooled gap (random - chemistry) = 0.9170 - 0.7973 = **0.1197**, reproducing
the Five-Way Ladder's own zT gap (random 80/20 0.9186 vs chemistry
cluster 0.7968, gap 0.1218 -- see the Five-Way Ladder table above) to
within 0.002, using an independent 5-fold refit rather than reading the
ladder's checkpoints directly.

**SUPERSEDED 2026-09-19.** Computed under the pre-fix grouping, 5 folds
(repeat 0 only). Regenerated against the fixed grouping (SNAP(0.05)
`chemistry_cluster_id`, S5 `randomized_group_kfold`, commit c1c6873), the
snapfix CSV, and generalized to all 25 folds (5 repeats), since repeat 0
alone left several attribution deltas too close to the fold-SD noise
floor to call -- see below. Pooled R2: random 0.9180, chemistry 0.7456,
gap 0.1724, reproducing the Five-Way Ladder's own new zT gap (0.9186 vs
0.7456, gap 0.1730 -- see the Five-Way Ladder table's 2026-09-19 update)
to within 0.0006.

**Coarse group comparison** (source: MagpieData / CBFV_ / temperature_bin):

| group | random mean share +/- SD | chemistry mean share +/- SD | delta (chem-rand) | delta / pooled SD |
|---|---|---|---|---|
| cbfv | 0.5385 +/- 0.0052 | 0.5297 +/- 0.0191 | -0.0088 | -0.63 |
| magpie | 0.2647 +/- 0.0062 | 0.2731 +/- 0.0212 | +0.0083 | +0.54 |
| temperature | 0.1968 +/- 0.0016 | 0.1972 +/- 0.0047 | +0.0004 | +0.12 |

**SUPERSEDED 2026-09-19** (same reason). New coarse shares, 25 folds:

| group | random mean share +/- SD | chemistry mean share +/- SD | delta (chem-rand) | delta / pooled SD |
|---|---|---|---|---|
| magpie | 0.2654 +/- 0.0056 | 0.2759 +/- 0.0228 | +0.0105 | +0.63 |
| cbfv | 0.5377 +/- 0.0049 | 0.5288 +/- 0.0251 | -0.0089 | -0.49 |
| temperature | 0.1969 +/- 0.0014 | 0.1953 +/- 0.0058 | -0.0015 | -0.36 |

**Fine descriptor-semantic group comparison** (see the script's
`FINE_GROUP_MEMBERS` mapping; 0 of 397 columns fell into "other"):

| group | random mean share +/- SD | chemistry mean share +/- SD | delta (chem-rand) | delta / pooled SD |
|---|---|---|---|---|
| atomic_radius | 0.1942 +/- 0.0042 | 0.1859 +/- 0.0164 | -0.0083 | -0.69 |
| dft_groundstate | 0.0218 +/- 0.0014 | 0.0289 +/- 0.0146 | +0.0071 | +0.68 |
| electronegativity | 0.0708 +/- 0.0033 | 0.0668 +/- 0.0070 | -0.0040 | -0.74 |
| periodic_position | 0.1116 +/- 0.0024 | 0.1152 +/- 0.0071 | +0.0036 | +0.68 |
| thermodynamic_bulk | 0.1535 +/- 0.0023 | 0.1561 +/- 0.0067 | +0.0026 | +0.52 |
| valence_electron_config | 0.1917 +/- 0.0043 | 0.1906 +/- 0.0193 | -0.0011 | -0.08 |
| atomic_mass | 0.0174 +/- 0.0019 | 0.0166 +/- 0.0023 | -0.0008 | -0.35 |
| temperature | 0.1968 +/- 0.0016 | 0.1972 +/- 0.0047 | +0.0004 | +0.12 |
| metal_class | 0.0231 +/- 0.0006 | 0.0234 +/- 0.0030 | +0.0004 | +0.16 |
| melting_point | 0.0191 +/- 0.0008 | 0.0194 +/- 0.0014 | +0.0002 | +0.21 |

**SUPERSEDED 2026-09-19** (same reason). New fine-group shares, 25 folds:

| group | random mean share +/- SD | chemistry mean share +/- SD | delta (chem-rand) | delta / pooled SD |
|---|---|---|---|---|
| valence_electron_config | 0.1939 +/- 0.0051 | 0.1814 +/- 0.0219 | -0.0125 | -0.78 |
| atomic_radius | 0.1945 +/- 0.0046 | 0.1802 +/- 0.0277 | -0.0143 | -0.72 |
| electronegativity | 0.0700 +/- 0.0026 | 0.0671 +/- 0.0058 | -0.0029 | -0.64 |
| melting_point | 0.0185 +/- 0.0009 | 0.0199 +/- 0.0025 | +0.0014 | +0.76 |
| periodic_position | 0.1120 +/- 0.0021 | 0.1257 +/- 0.0337 | +0.0137 | +0.57 |
| metal_class | 0.0226 +/- 0.0010 | 0.0250 +/- 0.0058 | +0.0024 | +0.57 |
| dft_groundstate | 0.0206 +/- 0.0022 | 0.0277 +/- 0.0184 | +0.0071 | +0.54 |
| thermodynamic_bulk | 0.1533 +/- 0.0036 | 0.1587 +/- 0.0160 | +0.0053 | +0.46 |
| atomic_mass | 0.0177 +/- 0.0015 | 0.0189 +/- 0.0042 | +0.0012 | +0.37 |
| temperature | 0.1969 +/- 0.0014 | 0.1953 +/- 0.0058 | -0.0015 | -0.36 |

**Finding, stated as negative**: no group, coarse or fine, shows a delta
exceeding 0.74 pooled fold-SD (electronegativity's -0.74 is the largest
magnitude of any row above). At this fold count, attribution shares are
statistically indistinguishable between the random and chemistry-cluster
split strategies. The ~0.12 R² inflation documented above is **not**
accompanied by any detectable shift in which descriptor families the
model relies on.

**SUPERSEDED 2026-09-19, in a way worth stating carefully rather than
just replacing the number.** A 5-fold-per-arm regeneration against the
fixed grouping (repeat 0 only, `results/shap_attribution/20260917T122537/`)
found THREE groups above 1 pooled fold-SD (melting_point +1.88,
electronegativity -1.38, valence_electron_config -1.33) -- the null
result did not hold at 5 folds. That run was underpowered, not wrong: at
k=5 the SD estimate itself is noisy. Generalizing to all 25 folds (5
repeats per arm, `results/shap_attribution/20260917T134930/`) resolved
it -- max |delta| is 0.78 pooled fold-SD (valence_electron_config),
nothing approaches 1 SD, the null result the 2026-09-13 run originally
reported holds again, on a firmer footing (25 observations, not 5).

**Secondary finding, new**: the chemistry arm's fold-to-fold SD runs 2x
to 16x the random arm's SD, across every coarse and fine group, and does
NOT narrow as fold count increases from 5 to 25 -- because grouped CV's
fold composition genuinely varies more than row-random CV's does (which
groups, and how many rows each carries, differs every repeat), not
because of an estimation artifact that more folds would average away.

**Interpretation**: this is consistent with the inflation being a
property of test-set composition -- the random split's test fold
contains near-duplicates (same or very similar chemistry_cluster_id) of
its own training rows, which the model can score well without learning
anything different -- rather than a change in the model's learned
internal structure. The frozen hyperparameters and descriptor set are
identical across both split strategies by construction; only which rows
land in train vs. test differs.

**Caveat** (verbatim from `scripts/shap_attribution_zT.py`'s own
docstring and `summary.json`): "The random-split and chemistry-cluster
models are trained on DIFFERENT training sets by construction, in every
one of the 5 folds (the random-split model's training rows include
near-duplicates of its own test rows -- that is the phenomenon under
study). This compares two different fitted models per fold, not an
ablation of one fixed model. Averaging over 5 folds addresses
fold-to-fold sampling variation within each split_strategy; it does not
eliminate this train-set-composition confound, which is inherent to
comparing the two split strategies at all."

**Provenance**: `results/shap_attribution/20260913T082733/`
(`comparison_table.md`, `summary.json`, `shap_arrays.npz` -- per-fold,
per-feature mean|SHAP|/share/gain arrays plus train/test/subsample
indices, enough to rebuild a figure without refitting -- and
`environment.txt`), copied from the Kaggle run that produced them.
Computed by `scripts/shap_attribution_zT.py`.

**SUPERSEDED 2026-09-19: the provenance above, for the numbers this
section now supersedes.** New provenance: `results/shap_attribution/
20260917T122537/` (5-fold-per-arm intermediate, underpowered, see above)
and `results/shap_attribution/20260917T134930/` (25-fold-per-arm,
current), both computed by `scripts/shap_attribution_zT.py` after it was
generalized to loop all repeats in commit 25507a8.

## Confirmed Results — Direct-vs-Derived zT (Paper A item 5, FINAL)

Direct-vs-derived zT, all-four-properties-present subset (55,948 rows,
5,786 chemistry clusters), chemistry-cluster grouped CV, one shared
frozen XGBoost hyperparameter set (from a single zT `tune_once` run)
reused across all four models so the comparison isolates pathway choice,
not per-property tuning. Implemented in `src/direct_vs_derived_zt.py`.

- Direct zT: pooled R² = 0.7931
- Derived S²σT/κ: pooled R² = 0.6659
- Gap: 0.127 (direct beats derived)

**SUPERSEDED 2026-09-19: subset size and all R2 values above.** This
section also used the WRONG hyperparameters until commit 37506aa:
`get_or_tune_zt_hyperparams()` loaded from an untracked, ephemeral
`tune_once` cache (`checkpoints/frozen_hyperparams/zT_xgboost.json`,
tuned against File B's row count in August, before File A became
canonical) instead of the canonical, git-tracked frozen hyperparameters
every other confirmed result in this document cites -- found by audit,
fixed to load the canonical file, `_fold_path`'s checkpoint_dir bug fixed
alongside it (commit fabc532). New subset: 56,088 rows, 4,139 chemistry
clusters (both the fixed grouping AND the snapfix CSV changed cluster
membership; see the Grouping Fixes section). New pooled R2, canonical
hyperparameters: direct zT = 0.7262, derived S2*sigma*T/kappa = 0.5244,
gap = 0.2018 (direct beats derived, same ordering, substantially larger
gap).

Component models feeding the derived pathway, same subset/splits/frozen
hyperparameters: S = 0.8735, sigma (log10) = 0.7719, kappa (log10) =
0.8693. Each component model is individually decent on its own, but
combining three imperfect predictions through S²σT/κ compounds their
errors multiplicatively rather than additively — direct prediction
clearly wins. Consistent with Paper A item 5's requirement to report
component-error correlation structure rather than assume independence.

**SUPERSEDED 2026-09-19** (same reason). New component models: S =
0.8172, sigma (log10) = 0.6856, kappa (log10) = 0.8221. Same qualitative
picture: each is individually decent, direct prediction still clearly
wins.

**Back-transform bias check (Duan smearing correction, 2026-08-22)**:
sigma and kappa are predicted in log10 space then exponentiated back for
S²σT/κ, which can introduce Jensen's-inequality retransformation bias
(E[10^X] > 10^E[X] for any X with nonzero spread) — checked whether that
bias, not genuine error propagation, explains the 0.127 gap above.
Implemented in `src/backtransform_check.py`; figure at
`figures/fig_backtransform_check.pdf`/`.png`.

- Gap survives Duan correction: derived R² = 0.6563 (corrected) vs
  0.6659 (uncorrected) — correction slightly worsens, not improves.
- Tail-SSE shares nearly identical (direct 24.7% vs derived 24.0% for
  the top 1% of rows) — no concentrated outlier blow-up from
  exponentiation; the residual distribution is broadly wider for derived
  across its whole range, not just in the tails.
- sigma/kappa residual correlation = 0.42 (moderate, positive) — errors
  do not cancel, they compound.
- **Conclusion: the 0.127 gap is genuine multiplicative error
  propagation, not a back-transform artifact.**

**SUPERSEDED 2026-09-19: this whole back-transform check block, and its
conclusion.** The claim that the gap "survives Duan correction" no
longer holds cleanly. New numbers (canonical hyperparameters, fixed
grouping): corrected derived R2 = 0.5279 vs uncorrected 0.5244 -- a
difference of 0.0035, and the two land on OPPOSITE sides of each other
in the two most recent runs, which differ only in which hyperparameters
were used (the wrong-hyperparameter snapfix run gave corrected 0.5239 <
uncorrected 0.5379; the canonical-hyperparameter rerun gives corrected
0.5279 > uncorrected 0.5244). New tail-SSE shares: direct 23.0% vs
derived 21.3% (top 1% of rows) -- still nearly identical, no concentrated
outlier blow-up. New sigma/kappa residual correlation = 0.4305 (moderate,
positive, essentially unchanged from 0.42).

**Replacement claim**: the direct-derived gap is roughly 0.20 either way
(0.2018 uncorrected here), and the Duan smearing correction moves
derived-zT by under 0.005 in a direction that is not consistent across
runs -- it neither explains the gap as a back-transform artifact nor
rescues the derived pathway. The gap is best read as genuine
multiplicative error propagation, same conclusion as before, but the
specific "survives correction" framing should be dropped: at this
magnitude the correction is noise relative to hyperparameter choice, not
a directional finding.

**Provenance**: `results/direct_vs_derived_snapfix/20260918T200058/`
(`direct_vs_derived_results.json`, `backtransform_check_results.json`)
and `checkpoints/direct_vs_derived_zt_snapfix_canonical/` (25 per-fold
predictions, git-tracked). Superseded intermediate (wrong hyperparameters,
same fixed grouping): `results/direct_vs_derived_snapfix/20260918T103933/`,
kept locally, not committed.

## Confirmed Results — External Validation, ESTM (Paper A item 6, ESTM COMPLETE)

Protocol: refit-on-full-training-then-predict (no saved/serialized model
exists anywhere in this repo — "frozen" means frozen hyperparameters,
`checkpoints/saved_predictions/checkpoints/frozen_hyperparams/
{S,sigma,kappa,zT}.json`). All four models refit on 100% of training
(no holdout). sigma/kappa's Duan smearing factors come from ONE
chemistry-cluster GroupKFold pass over full training data (frozen
hyperparameters, seed=0, never retuned), calibrated and frozen before
ESTM was touched. Implemented in `src/external_validation.py`.

**Stated honestly: ESTM was touched by the frozen model twice, not
once.** RUN1 (the intended single touch) only saved aggregate R²/n —
`score_estm_pass()` discarded every per-row prediction. RUN2 was a
logging-only re-run, adding per-row prediction saving with the model,
hyperparameters, dedup, and ESTM set all held identical, and was
certified to reproduce RUN1's aggregate R² to 4 decimals on every
property/pass (bit-exact, 0.00e+00 diff on all 10 property/pass cells).
RUN1's original output is retained as
`checkpoints/external_validation/estm_results_RUN1_backup.json` for
that audit trail. All numbers below are RUN2's (identical to RUN1's).

Provenance: `checkpoints/external_validation/estm_results.json`
(aggregate numbers), `checkpoints/external_validation/
estm_predictions_pass{a,b}.npz` (per-row predictions, both passes).

**Two dedup passes, never merged:**

| Pass | Dropped rows | Surviving rows | Unique chemistry clusters surviving |
|---|---|---|---|
| (a) source-DOI | 1,416 | 3,123 | 505 |
| (b) composition-cluster | 2,592 | 1,947 | 359 |

**SUPERSEDED 2026-09-19: pass (b)'s row/cluster counts.** Pass (a) is
DOI-based, unaffected by the grouping fix: rows unchanged (1,416 dropped,
3,123 surviving), but the SAME rows' chemistry_cluster_id labels changed
under SNAP(0.05), so their own cluster-diversity count moves from 505 to
357 (recomputed directly from the saved per-row predictions). Pass (b)'s
dedup is itself chemistry-cluster-based, so its row set changes: 3,091
dropped (not 2,592), 1,448 surviving (not 1,947), 228 unique clusters
among survivors (not 359), 300 unique ESTM clusters overlapping training
(not 401). Replacement table:

| Pass | Dropped rows | Surviving rows | Unique chemistry clusters surviving |
|---|---|---|---|
| (a) source-DOI | 1,416 | 3,123 | 357 |
| (b) composition-cluster | 3,091 | 1,448 | 228 |

**Per-property R², full-set vs. in-distribution** (in-distribution =
within training's per-property support on S, sigma, kappa, and
temperature simultaneously):

| Pass | Property | Full R² | n | In-dist R² | n | OOD fraction |
|---|---|---|---|---|---|---|
| a | S | 0.5647 | 3,123 | 0.7501 | 2,709 | 2.4% |
| a | sigma | 0.3986 | 3,123 | 0.6088 | 2,709 | 12.0% |
| a | kappa | 0.6779 | 3,123 | 0.7240 | 2,709 | 2.9% |
| a | zT_direct | 0.6658 | 3,123 | 0.6742 | 2,709 | — |
| a | zT_derived | 0.2233 | 3,123 | 0.3166 | 2,709 | — |
| b | S | 0.4217 | 1,947 | 0.6124 | 1,655 | 2.5% |
| b | sigma | 0.2764 | 1,947 | 0.4471 | 1,655 | 13.5% |
| b | kappa | 0.6264 | 1,947 | 0.6870 | 1,655 | 3.7% |
| b | zT_direct | 0.5793 | 1,947 | 0.6053 | 1,655 | — |
| b | zT_derived | −0.0188 | 1,947 | −0.0070 | 1,655 | — |

**SUPERSEDED 2026-09-19: pass (b)'s full-set row, and both passes'
zT_derived full-set cell.** Pass (a)'s direct-prediction full-set cells
(S, sigma, kappa, zT_direct) are BIT-IDENTICAL, confirmed directly --
chemistry_cluster_id is not a model feature, and pass (a)'s row set is
unaffected. New full-set values: pass (a) zT_derived R2 = 0.2064, n=3,123
(smear-factor dependency only, row set unchanged); pass (b) S=0.3536,
sigma=0.2746, kappa=0.6184, zT_direct=0.4982, zT_derived=-0.0959, all
n=1,448.

**Open gap, not recomputed in this pass: the In-dist R2/n columns above.**
"In-distribution" restricts to rows within training's per-property
support (S, sigma, kappa, temperature simultaneously) -- the exact bounds
used to produce the original 0.6124/0.4471/0.6870/0.6053 figures were
never committed as a reusable script, only the resulting numbers. Rather
than reconstruct a bounds definition that might silently differ from the
original and produce a false-comparable number, this is flagged as an
open item: the saved per-row predictions
(`results/external_snapfix/20260917T160553/estm_predictions_pass{a,b}.npz`)
support this recompute whenever the original bounds definition is
located or re-derived.

**RESOLVED 2026-09-19.** Bounds reconstructed from the snapfix training
set's own per-property min/max, joint across S/sigma/kappa simultaneously
plus the 300-800K temperature window: S (-461.0258, 562.5), sigma
(958.7831, 1656678.2772), kappa (0.2830364, 13.7753). **Validated, not
just assumed defensible**: this reconstruction yields exactly n=2,709
in-distribution rows for pass (a) -- bit-identical to the original
table's row count above, despite being derived independently via a
different path (this session never located the original bounds script;
see the SCRIPT PROVENANCE addendum below). The resulting R2 values differ
from the original in-distribution figures by 0.004 to 0.01 (e.g. pass a
S: 0.7404 new vs 0.7501 old) -- attributed to the training set and
grouping having changed (fixed chemistry_cluster_id, snapfix CSV, see the
Grouping Fixes section), not to the bounds rule itself, since the row
COUNT match this precisely implies a materially identical bounds
definition. Replacement table (joint in-distribution, all three
properties + temperature simultaneously):

| Pass | Property | Full R² | n | In-dist R² | n | OOD fraction |
|---|---|---|---|---|---|---|
| a | S | 0.5664 | 3,123 | 0.7404 | 2,709 | 2.37% |
| a | sigma | 0.3988 | 3,123 | 0.6132 | 2,709 | 11.98% |
| a | kappa | 0.6892 | 3,123 | 0.7314 | 2,709 | 2.91% |
| a | zT_direct | 0.6615 | 3,123 | 0.6686 | 2,709 | — |
| a | zT_derived | 0.2064 | 3,123 | 0.3286 | 2,709 | — |
| b | S | 0.3536 | 1,448 | 0.5115 | 1,196 | 2.28% |
| b | sigma | 0.2746 | 1,448 | 0.4085 | 1,196 | 15.54% |
| b | kappa | 0.6184 | 1,448 | 0.6565 | 1,196 | 4.49% |
| b | zT_direct | 0.4982 | 1,448 | 0.5343 | 1,196 | — |
| b | zT_derived | -0.0959 | 1,448 | -0.0814 | 1,196 | — |

OOD fractions above are per-property, independent. Joint OOD (all three
properties + temperature simultaneously, the actual denominator behind
the In-dist n column): pass (a) 13.26% (2,709 of 3,123 survive), pass
(b) 17.40% (1,196 of 1,448 survive).

Temperature contributes 0% OOD in either pass — `step3_filter_temperature`
enforces training's exact 300-800K window on ESTM before anything else
runs, so no temperature-driven exclusion happens later. Sigma is the
dominant OOD contributor (12.0% pass a, 13.5% pass b) — ESTM's sigma
tail reaches down to 4e-4 S/m, far below training's cleaned floor of
~958 S/m.

**SUPERSEDED 2026-09-19: the sigma OOD figures in the sentence above.**
New independent per-property OOD fractions: sigma 11.98% (pass a), 15.54%
(pass b) -- still the dominant OOD contributor by a wide margin over S
(2.37%/2.28%) and kappa (2.91%/4.49%), same qualitative finding.

**Headline finding: a second, distinct inflation gap, beyond the
random-vs-grouped gap the Five-Way Ladder already documents.** Pass
(b) in-distribution — the strictest reading (novel chemistries,
never-seen-conductivity-regime rows excluded) — against internal
chemistry-cluster grouped CV:

| Property | Internal grouped CV | ESTM pass (b), in-distribution | Drop |
|---|---|---|---|
| S | 0.8076 | 0.6124 | 0.1952 |
| sigma | 0.7600 | 0.4471 | 0.3129 |
| kappa | 0.8460 | 0.6870 | 0.1590 |
| zT_direct | 0.7968 | 0.6053 | 0.1915 |

**SUPERSEDED 2026-09-19: the Internal grouped CV column.** Replacement,
from the fixed-grouping chemistry rung:

| Property | Internal grouped CV | ESTM pass (b), in-distribution | Drop |
|---|---|---|---|
| S | 0.7528 | 0.6124 (unverified, see open gap above) | pending |
| sigma | 0.7020 | 0.4471 (unverified, see open gap above) | pending |
| kappa | 0.8092 | 0.6870 (unverified, see open gap above) | pending |
| zT_direct | 0.7456 | 0.6053 (unverified, see open gap above) | pending |

The Drop column cannot be recomputed until the ESTM-side in-distribution
column is -- both sides of the subtraction must be current at once, not
mixed old-ESTM/new-ladder.

**RESOLVED 2026-09-19**, now that the ESTM-side in-distribution gap above
is closed. Replacement table, both sides current:

| Property | Internal grouped CV | ESTM pass (b), in-distribution | Drop |
|---|---|---|---|
| S | 0.7528 | 0.5115 | 0.2413 |
| sigma | 0.7020 | 0.4085 | 0.2935 |
| kappa | 0.8092 | 0.6565 | 0.1527 |
| zT_direct | 0.7456 | 0.5343 | 0.2113 |

Even chemistry-cluster grouped CV — this project's own honest ceiling,
already measurably stricter than composition/random/k-fold per the
Nadeau-Bengio test above — is itself optimistic relative to genuine
cross-database transfer. This is a SECOND, independent inflation
mechanism from the one the Five-Way Ladder documents: grouped CV
controls for near-duplicate leakage within Starrydata2, but says
nothing about whether the model generalizes to a differently-curated
external database with its own measurement conventions, compound
coverage, and digitization error.

**sigma out-of-distribution handling**: report **0.4471 (in-support
only, pass b)** as the sigma transfer number, not the 0.2764 full-set
figure — the full-set number is dominated by extrapolation failure,
not transfer failure. The 263 sub-floor rows (13.5% of pass b) carry
**79.8% of total sigma SSE** (1,444.5 of 1,810.1, log10 space); their
mean log-residual is **−2.04** (predictions ~2 orders of magnitude too
HIGH), vs. +0.05 for in-support rows — the model extrapolates toward
training's much higher typical conductivity when shown a
near-insulating composition it never trained on. **State explicitly:
the model does not extrapolate below its training conductivity floor
of ~958 S/m — this is a scope limitation, not a silent failure mode.**

**SUPERSEDED 2026-09-19, not independently recomputed: this whole
paragraph.** Every figure here (0.4471, 263 sub-floor rows, 79.8% SSE
share, -2.04 mean log-residual) is pass-(b)-in-distribution-specific and
shares the open gap above -- pass (b)'s surviving row set changed
(1,947 -> 1,448), so these figures must be recomputed against the new
surviving set once the in-distribution bounds definition is available,
not silently carried forward.

**RESOLVED 2026-09-19, in part.** The R2 split is now closed: sigma's
full-set R2 (0.27-0.40 across the two passes: pass b 0.2746, pass a
0.3988) improves to 0.41-0.61 in-distribution (pass b 0.4085, pass a
0.6132, see the replacement table above) -- meaning a LARGE SHARE of
sigma's apparent external-validation failure is extrapolation below the
training conductivity floor, not a failure to generalize to
chemistries/regimes the model could in principle handle. The split IS
materially informative for ESTM: joint OOD is 13.26% (pass a) / 17.40%
(pass b), two orders of magnitude larger than teMatDb's 0.12% OOD tail
(which was correctly judged negligible and not worth reporting there) --
this one should not be dropped the same way. The finer sub-floor-row/SSE-
share diagnostic (263 rows, 79.8% SSE, -2.04 mean log-residual) is a
narrower, separate item, NOT recomputed here against the new pass-b
surviving set -- still open, smaller in scope than the R2 split this
resolves.

**zT_derived: report the protocol-consistent frozen-smear number,
−0.0070 (pass b, in-distribution), as the result — NOT the naive
+0.1681 figure.** Frozen smear factors (smear_sigma=1.2916,
smear_kappa=1.0446) come from training's own chemistry-cluster
out-of-fold residuals, calibrated and frozen BEFORE ESTM was touched,
specifically so the correction never uses ESTM's own true values (using
them would be test-set leakage on the external validation this
calibrates). That protocol choice has a real, measured cost here:
training's OOF residuals wanted a +0.111 log10 sigma correction, but
ESTM's own in-support residuals only need about +0.058 — roughly half
as much. Applying training's larger correction to ESTM overshoots, and
the mis-calibrated correction makes derived-zT WORSE than no correction
at all (naive back-transform R²=+0.1681; frozen-smear R²=−0.0070). Both
are far below direct-zT's 0.6053 either way, so the pathway conclusion
is unchanged. **This is the accepted trade-off of the leakage-avoidance
protocol, not a bug**: retransformation bias is distribution-specific
and does not transfer across databases, so a smear factor honestly
calibrated without touching test labels will not be optimally
calibrated for the test distribution. State this trade-off explicitly
rather than silently reporting whichever number looks better.

**SUPERSEDED 2026-09-19: the smear factors, and the -0.0070/+0.1681
pass-b-in-distribution result.** New smear factors (chemistry-cluster
GroupKFold OOF pass, fixed grouping, snapfix training):
smear_sigma=1.3817, smear_kappa=1.0715 -- both larger than before
(1.2916, 1.0446), consistent with the chemistry-cluster fold assignment
itself changing under S5. The pass-b-in-distribution zT_derived result
this paragraph reports shares the open gap above and needs the same
in-distribution recompute before restating.

**RESOLVED 2026-09-19.** New pass-b, in-distribution, frozen-smear
zT_derived R2 = -0.0814 (n=1,196; see the replacement table above) --
still clearly worse than direct-zT's in-distribution 0.5343, so the
pathway conclusion (direct beats derived) is unchanged. The naive
(uncorrected back-transform) comparison point is not recomputed here;
only the protocol-consistent frozen-smear number this section reports as
the result.

**ESTM external validation: COMPLETE.** teMatDb: PENDING — a separate
dataset, not yet downloaded (item 6 requires both, each touched exactly
once).

## Paper B — Cross-Family Generalization (FROZEN)

**Core claim = two falsifiable questions, not vague "granularity"
language:**
(a) Does a pooled multi-family model's within-distribution accuracy
overstate its accuracy on a held-out family, and by how much per family?
(b) How much of within-family accuracy is family-level mean offset vs.
finer intra-family composition signal?

1. **Family labels via stoichiometric-template matching** from composition
   alone (Starrydata2 has no structure data) — e.g., ABX half-Heusler, AB₃
   skutterudite templates. Unmatched compositions go to an explicit
   "unassignable" bucket (report its size, don't silently drop). Restrict
   study to families above the a priori sample-size threshold. State
   explicitly that this limits coverage to clean-template compounds,
   likely the LEAST representative of real (often doped/defected)
   thermoelectrics — report what fraction of the dataset this covers.
2. **Leave-one-family-out + TWO controls**: (a) size-matched
   random-removal control (scattered rows, same total reduction); (b)
   structured-removal control (a DIFFERENT whole family of similar size,
   removed the same contiguous way as true LOFO). Comparing both against
   true LOFO separates "lost a structured region generically" from "lost
   this specific family."
3. **Primary probe = specialist-vs-pooled comparison**, NOT the
   family-label ablation (family label is largely redundant with
   composition, so ablation alone is weak/misleading as primary
   evidence — demote to secondary check). Train one specialist model per
   qualifying family, compare to the pooled model's within-family
   performance. Also report offset-stripped within-family predicted-vs-
   actual correlation.
4. **Two model families**, chosen to bracket capacity (one high-capacity
   GBDT, one constrained/lower-capacity model). Report per-family results
   for each SEPARATELY — do not average.
   `src/nested_cv.py --model` (see Paper A item 1's implementation note)
   provides this directly: `xgboost`/`lightgbm` as high-capacity GBDTs,
   `ridge` as the constrained/lower-capacity bracket, `random_forest` as
   a third, structurally different high-capacity option (bagged, not
   boosted) if a non-boosting high-capacity comparison is wanted instead
   of/alongside a GBDT.
5. **Sample-level grouping applies identically** across true LOFO and
   both controls — not only the LOFO condition. This is a common silent-
   divergence point; verify explicitly.
6. **Independence from Paper A is a GATED decision**: build Paper A first.
   After Paper A's ladder exists, run Paper B's analysis. If
   specialist-vs-pooled shows a clear coarse-vs-fine story → standalone
   paper. If weak/ambiguous → fold into Paper A as one section. Do not
   decide this in advance.
7. State explicitly, as an acknowledged limitation, that even the
   specialist-vs-pooled probe cannot fully separate "shortcut learning"
   from "genuinely family-dependent physics" (if physics really differs by
   family, family-correlated signal is legitimate, not a shortcut). Do not
   oversell this as a resolved binary.

**Prior art to cite (full list, frozen)**: Meredig et al. (2018, Mol.
Syst. Des. Eng. — introduced LOCO-CV); Durdy et al. (LOCO-CV as general
baseline); RSC Digital Discovery, DOI 10.1039/d2dd00004k; "Scaffold
Splits Overestimate Virtual Screening Performance" (arXiv:2406.00873);
"Scaffold splits hide structural-frontier failures in ADMET models"
(arXiv:2607.10729 — ENGAGE this directly as a counterpoint/limitation, it
argues grouped splits can still hide failures — don't cite only
supportively); MD-HIT (npj Comput. Mater. 2024, dataset-redundancy
control). Novelty claim: the METHOD (LOCO-CV) is established prior art —
the novel part is the thermoelectric multi-family application plus the
specialist-vs-pooled decomposition. Do not claim "no prior work does this
test" without this qualifier.

---

## Competitive landscape (verified by DOI/arXiv — cite and differentiate)
- Jia, Aziz, Hashimoto & Li (2024), Sci. China Materials 67(4):1173-1182,
  DOI 10.1007/s40843-023-2777-2 — composition-CV on Starrydata2, single method.
- Barua, Lee, Oliynyk & Kleinke (2025), ACS Appl. Mater. Interfaces
  17(1):1662-1673, DOI 10.1021/acsami.4c19149 — closest competitor,
  ~160K rows, 3 external test sets, honest R² 0.67-0.80.
- Sun et al. (2026), Cell Reports Physical Science, DOI
  10.1016/j.xcrp.2025.103093 — TabPFN, 10-fold CV (leaky), zT only.
- Athar, Mecibah & Jund (Feb 2026), arXiv:2602.01149 — PCA split,
  half-Heusler only, zT only, plus a 6.6×10⁸-composition screen. SAME
  authors as the Starrydata2 curation critique (Materials Today Physics
  2025, 59:101948) and the Jan 2026 generalizability review (arXiv:2601.06571)
  — 3 papers in 4 months, fast-moving group, treat competitive window as
  narrowing.
- Ma & Poon, arXiv:2509.00299 — verified NOT a scoop despite title
  ("Reexamining ML Models on Predicting Thermoelectric Properties"):
  physics-based feature engineering, no split-strategy comparison. Cite
  and explicitly state why it doesn't overlap.
- Wang, Zhong, Zhang et al. (2025), Materials & Design 249:113552, DOI
  10.1016/j.matdes.2024.113552 — R²=0.970 stacking ensemble. CV
  methodology UNVERIFIED (paywalled) — do not assert it's leaky, soften
  to "should be interrogated before treating as a benchmark."

---

## Local dev environment
- Neither Python nor git is on PATH in this terminal — call by full path:
  Python at `C:\Users\choha\AppData\Local\Programs\Python\Python312\python.exe`,
  git at `C:\Program Files\Git\cmd\git.exe`.
- This machine has no GPU (integrated Intel Iris Xe only, no NVIDIA/CUDA,
  no cupy installed). GPU code paths (`device="cuda"` in
  `src/nested_cv.py`) cannot be run or timed locally — verify them on
  Kaggle. Locally, only smoke-test such paths on `device="cpu"` for
  behavior-preservation, and say explicitly that GPU behavior/performance
  is unverified until run on Kaggle.

## Code conventions
- Python, `src/` module structure (not notebooks) — this repo goes on
  GitHub for a PhD portfolio, needs to read as engineered, not exploratory.
- Structure: `src/data_cleaning.py`, `src/canonicalization.py` (chemistry-
  cluster definition), `src/featurization.py`, `src/validation_ladder.py`,
  `src/nested_cv.py`, `src/noise_floor.py`, `src/screening.py`,
  `src/external_validation.py`, `src/family_labels.py`,
  `src/lofo_paperb.py`, `scripts/run_pipeline.py`, `figures/`.
- `config.yaml` for paths, seeds, fold counts, extraction date, the 5 at%
  cluster threshold (and its sensitivity-table alternates).
- `.gitignore` raw/processed data and any API keys — never commit data files.
- Docstrings on all public functions, pinned versions in requirements.txt.

## Style
- No em-dashes in written text (docstrings, README, comments) — use
  commas, colons, or semicolons instead.
- Be direct about methodology flaws or shortcuts — flag them, don't smooth
  over them.
