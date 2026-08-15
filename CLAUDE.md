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
6. Noise-floor anchor computed in log space (Paper A, item 3).
7. External validation, two separate numbers (Paper A, item 6).

**Phase 2 — Paper A finish:**
8. Direct-vs-derived zT pathway, on the all-four-properties subset.
9. Screening rediscovery test, targeted holdout only (Paper A, item 7).
10. Temperature axis — write as ONE limitation paragraph only, do not build
    as a full experimental axis (cut per frozen decision, see below).
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
   -1000 to +1000 μV/K; σ 1 to 10^7 S/m (semiconductor-insulator boundary
   — confirm exact lower bound during implementation); κ 0.05 to 25 W/mK
   (Cahill-Pohl minimum); zT 0 to 4. Cite Snyder & Toberer 2008.
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
9. **MAD outlier filter** — 3.5×MAD from median, log scale for σ, κ. NOT
   applied to zT (judged against Step 1 bounds instead).
10. **Minimum temperature coverage** — remove formulas with <3 distinct
    temperature measurements.
11. **Smoothness filter** — rolling median (window=3), flag spikes as NaN,
    remove rows where all properties became NaN.

Reference-scale sanity check (expect similar magnitude on the fresh pull,
not identical values): final dataset previously ~184,167 rows, ~13,605
unique formulas, ~2,834 parent chemical systems. Large deviations at any
step are a signal to stop and investigate, not to proceed.

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
2. **Nested GroupKFold** for hyperparameter tuning — outer folds for
   reporting only, hyperparameters tuned on inner folds nested inside each
   outer training fold. Never tune and report on the same folds.
3. **Noise floor, computed in log space** (relative uncertainties are
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
8. **Temperature axis — CUT, one paragraph only.** zT is typically
   non-monotonic in temperature (peaks then rolls over, often within
   600-800K) — no extrapolator, tree-based or otherwise, can recover this
   from only the ≤600K rising branch. This is a data-coverage fact, not a
   generalization finding. Do not build this as a full experimental axis.
9. **PCA-split engagement, empirical not asserted**: run chemistry-cluster
   CV AND a PCA-based split (Athar/Jund's method) on the same data, report
   both honest numbers, state why chemistry-cluster grouping is stricter
   against near-duplicate leakage specifically.

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
