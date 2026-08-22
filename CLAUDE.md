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
9. Screening rediscovery test, targeted holdout only (Paper A, item 7).
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
  of its fold. A single grouped split is one arbitrary draw of which large,
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

## Paper A — Validation-Inflation Ladder + Descriptor Ceiling (FROZEN)

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
7. **Screening rediscovery, targeted holdout**: hold out ONLY specific
   known-good targets + their exact canonical duplicates (not all high-zT
   materials broadly) — retain other high-zT materials so the model keeps
   a performance signal to generalize from. Interpret result as caveated
   tail-extrapolation. If the high-zT region is too thin after holdout,
   report that finding directly rather than forcing a positive result.
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

## Confirmed Results — Five-Way Ladder (Paper A item 1, FINAL)

Pooled out-of-fold R², frozen hyperparameters per target (tuned once via
`tune_once`, reused unchanged across all five rungs), 397 features
(MAGPIE + CBFV + temperature_bin), XGBoost. sigma and kappa trained and
scored in log10 space per the frozen decision above; S and zT in linear
space.

| Target | random 80/20 | 5-fold | 10-fold | composition | chemistry cluster |
|---|---|---|---|---|---|
| S | 0.9586 | 0.9585 | 0.9594 | 0.8314 | 0.8083 |
| sigma (log10) | 0.9539 | 0.9533 | 0.9550 | 0.7791 | 0.7522 |
| kappa (log10) | 0.9615 | 0.9611 | 0.9625 | 0.8380 | 0.8226 |
| zT | 0.9138 | 0.9132 | 0.9148 | 0.8164 | 0.7965 |

All four targets show the same qualitative pattern: the three ungrouped
rungs (random/5-fold/10-fold) cluster tightly within ~0.001-0.002 of each
other, then drop sharply at composition-level grouping and drop again,
more modestly, at the chemistry-cluster anchor — the honest ceiling this
project reports. The ungrouped-to-chemistry-cluster gap is largest for
sigma (0.9539 to 0.7522, 20.2 points) and smallest for S (0.9586 to
0.8083, 15.0 points).

**Chemistry-cluster spread across repeats (2026-08-22, per-repeat pooled
R², see the per-repeat pooling note above)**, computed from a Kaggle
checkpoint set supplied 2026-08-22:

| Target | chemistry cluster mean ± SD | per-repeat values |
|---|---|---|
| S | 0.8076 ± 0.0018 | 0.8062, 0.8064, 0.8077, 0.8071, 0.8108 |
| sigma (log10) | 0.7600 ± 0.0008 | 0.7608, 0.7599, 0.7608, 0.7598, 0.7588 |
| kappa (log10) | 0.8460 ± 0.0011 | 0.8463, 0.8463, 0.8470, 0.8465, 0.8441 |
| zT | 0.7968 ± 0.0030 | 0.7973, 0.7915, 0.7985, 0.7984, 0.7983 |

**Discrepancy, flagged not reconciled**: this checkpoint set's all-repeats
pooled R² (S=0.8076, sigma=0.7600, kappa=0.8460, zT=0.7968 — the mean of
the per-repeat column above) does not exactly match the chemistry-cluster
column in the table above (S=0.8083, sigma=0.7522, kappa=0.8226,
zT=0.7965). S and zT are close; sigma is off by 0.008 and kappa by 0.023
— more than rounding. This checkpoint set is very likely a different
(re-run) Kaggle pass than whatever produced the table above's numbers,
not yet identified which. The table above's numbers are kept as the
confirmed figures pending that reconciliation; treat the per-repeat
spread here as informative about repeat-to-repeat variance, not as a
replacement point estimate.

**Composition-rung spread and the composition-vs-chemistry Nadeau-Bengio
significance test are BOTH PENDING**: no composition-rung checkpoint
predictions exist locally (only the chemistry-rung set above has been
supplied) — `nadeau_bengio_test()` is implemented and ready
(`src/nested_cv.py`, see the per-repeat pooling note above) but has
nothing to run against yet. Complete once composition-rung
`repeatN_foldM_predictions.npz` + `run_config.json` files (same layout
as the chemistry set) are available, for all four properties.

## Confirmed Results — Noise Floor (Paper A item 3, FINAL)

R²_max computed per-property matched space (log10 for sigma/kappa,
linear for S/zT — matching each property's confirmed R² scoring space,
see item 3 above), Alleno et al. 2015 round-robin relative uncertainties
as the noise reference. Implemented in `src/noise_floor.py`.

| Target | Scale | R²_max | Confirmed ceiling | Headroom |
|---|---|---|---|---|
| S | linear | 0.9974 | 0.8083 | 0.189 |
| sigma | log10 | 0.9968 | 0.7522 | 0.245 |
| kappa | log10 | 0.9776 | 0.8226 | 0.155 |
| zT | linear | 0.9785 | 0.7965 | 0.182 |

R²_max is a best-case upper bound: Alleno et al. is a single-compound
(skutterudite) round-robin measurement excluding digitization error, so
true database noise is higher than this and true headroom is smaller
than shown — state this direction explicitly wherever these numbers are
cited, per item 3's frozen instruction.

## Confirmed Results — Direct-vs-Derived zT (Paper A item 5, FINAL)

Direct-vs-derived zT, all-four-properties-present subset (55,948 rows,
5,786 chemistry clusters), chemistry-cluster grouped CV, one shared
frozen XGBoost hyperparameter set (from a single zT `tune_once` run)
reused across all four models so the comparison isolates pathway choice,
not per-property tuning. Implemented in `src/direct_vs_derived_zt.py`.

- Direct zT: pooled R² = 0.7931
- Derived S²σT/κ: pooled R² = 0.6659
- Gap: 0.127 (direct beats derived)

Component models feeding the derived pathway, same subset/splits/frozen
hyperparameters: S = 0.8735, sigma (log10) = 0.7719, kappa (log10) =
0.8693. Each component model is individually decent on its own, but
combining three imperfect predictions through S²σT/κ compounds their
errors multiplicatively rather than additively — direct prediction
clearly wins. Consistent with Paper A item 5's requirement to report
component-error correlation structure rather than assume independence.

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
