# Paper A: Validation-Inflation Ladder + Descriptor Ceiling
## v3 — FROZEN. Every open question below is a committed decision, not a discussion point.

**Status: closed for review. Implement as written.** This version converts
every prior "resolution" that still had a remaining "it depends" into one
concrete, unambiguous decision. Do not reopen items marked frozen below
without a specific, new reason — general re-review is done.

---

## 1. Objective / Core Claim (unchanged from v2 — do not reopen)
Quantify how much different validation strategies inflate reported
thermoelectric property prediction accuracy (S, σ, κ, zT), anchor the
honest ceiling against an independently measured noise floor, validate on
properly deduplicated external data.

## 2. Novelty framing (unchanged from v2 — do not reopen)
Novelty is the systematic multi-method quantification + noise-floor
anchor, not the leakage insight itself (already stated in Jia et al. 2024
and the general ML-leakage literature). Ma & Poon arXiv:2509.00299 is
cited and explicitly distinguished (not a scoop — physics-based feature
engineering, no split comparison). Wang et al.'s R²=0.970 (*Materials &
Design* 249:113552) is described as "should be interrogated," never
asserted as leaky. Competitive landscape table from v2 stands unchanged.

---

## 3. FROZEN DECISIONS — Load-bearing (everything downstream depends on these)

### 3.1 Chemistry-cluster boundary — DEFINED
A cluster is defined as: **the reduced host-lattice stoichiometry with
dopants below 5 at% collapsed into the parent** — canonicalize each
formula to a cluster identity based on the integer stoichiometry of
elements present above 5 at%; elements below that threshold are treated
as dopants and do not split the cluster.

- The honest ceiling (Section 4) is anchored at this cluster level.
- Run the full ladder once more at a looser threshold and once at a
  stricter threshold than 5 at%, reported as a 3-row sensitivity table.
  This sensitivity table is itself a reported result, not a discarded
  intermediate step — it preempts the "why this boundary" question
  directly.
- After the sensitivity table is produced, the 5 at% choice is frozen as
  the paper's primary reported number.
- **Statistical consequence to report explicitly**: with an estimated
  ~15-25 meaningful chemistry clusters, 5-fold grouped CV puts only a
  handful of clusters per test fold, so the standard deviation across
  outer folds will be large. Use **repeated grouped CV** (multiple
  random outer-fold assignments, not a single 5-fold run) or a corrected
  resampled significance test (e.g., Nadeau-Bengio correction) before
  claiming any inflation delta between rungs of the ladder is
  statistically real, not noise.

### 3.2 Cleaning-pipeline vs. identical-dataset conflict — RESOLVED
**Decision: clean the full dataset once, globally.** Use this single
fixed, cleaned dataset across all five validation schemes in Section 3.3
below.

- Rationale: the cross-row cleaning filters (multi-source CV consistency,
  MAD outlier bounds, rolling-median smoothing) operate on each target
  property's marginal distribution, not on the train→test mapping
  directly — this is mild, transductive leakage (test-set distribution
  shape mildly informs which rows survive cleaning), categorically
  different from train/test contamination.
- Add **one sensitivity check**: refit the cleaning filters fold-locally
  on the chemistry-cluster split only, and report whether the honest
  ceiling number moves. If it doesn't move materially, this confirms the
  global-cleaning shortcut was safe. Report this check in the paper
  regardless of outcome.
- Fold-local cleaning as the *primary* pipeline is rejected — it would
  make the cleaned dataset a function of the split, breaking the
  apples-to-apples comparison the five-way ladder (Section 3.3) requires.

### 3.3 Noise-floor conversion — computed in log space
The round-robin uncertainty figures (Alleno et al. 2015: S ~6%, σ ~8%, κ
~11%, zT ~17-19%) are **relative** (multiplicative, heteroscedastic)
uncertainties, not additive ones. The R²_max conversion is therefore done
entirely in log space:

```
R²_max = 1 − σ²_noise(log) / σ²_total(log)
```

- `σ_noise(log)` = the relative uncertainty taken directly as the
  log-space standard deviation (≈0.17 for zT; this is the standard
  small-relative-uncertainty approximation, log(1+x)≈x).
- `σ_total(log)` = the actual variance of the log-transformed property in
  this dataset (computed from data, not assumed).
- Model R² is **also computed and reported in log space**, so the model
  performance and the noise-floor ceiling are on the same scale and
  directly comparable. If the model itself is trained on log-transformed
  targets, this is already consistent; if not, report a log-space R² for
  this comparison specifically, separate from the primary reported metric
  if the primary metric is on raw values.
- Retain the stated limitation from v2: the round-robin figure is a lower
  bound on true database noise (excludes digitization/transcription
  error), so the computed R²_max, even done correctly in log space, likely
  overstates true achievable headroom. State this explicitly as a
  conservative-direction limitation.

---

## 4. FROZEN DECISIONS — Secondary (mechanical, implement as specified)

### 4.1 Five-way validation comparison, denominator-matched
- Tune hyperparameters **once**, on the chemistry-cluster split (Section
  3.1), using the nested-CV procedure. Freeze this hyperparameter set.
- Evaluate the identical frozen model under all five split schemes
  (random 80/20, 5-fold, 10-fold, composition-level, chemistry-cluster),
  using **pooled out-of-fold (OOF) R²** for every scheme so the metric
  computation is identical across rungs.
- The 80/20 random-split rung only scores 20% of rows in a single run,
  unlike the k-fold schemes which pool across 100% of rows. **Repeat the
  80/20 split ~20 times** (different random seeds) and pool results, so
  its effective coverage is comparable to the k-fold schemes.
- State once, explicitly, in the paper: this measured inflation is a
  **lower bound** on real-world practitioner inflation, since a real
  practitioner using a leaky random split would also tune hyperparameters
  on that same leaky split, producing even more optimistic (and more
  inflated) results than this frozen-hyperparameter design measures. This
  is a conservative, defensible framing, not a weakness to hide.

### 4.2 Cleaning thresholds — decoupled from round-robin
Step-8 consistency-filtering thresholds (per-property CV cutoffs) are set
by a **data-driven knee method**: tighten the threshold progressively
until further tightening removes additional data without changing the
held-out honest R². The round-robin uncertainty source (Alleno et al.
2015) is used **only** for the noise-floor anchor (Section 3.3) — nowhere
else in the pipeline. This closes the circularity risk without reverting
to unjustified hand-picked thresholds.

### 4.3 External validation — two numbers, not one merged set
Report **two separate external-validation numbers**, not a single merged
deduplicated set:
1. **Source-deduplicated**: remove any ESTM/teMatDb row sharing a source
   DOI with training data. This tests measurement-transfer (same
   chemistry, independent measurement) — a realistic deployment-relevant
   regime.
2. **Composition-cluster-deduplicated**: remove any row whose chemistry
   cluster (Section 3.1 definition) appears in training. This tests
   chemistry-transfer specifically.
- Report the dropped-row count for each deduplication separately.
- Do not merge these into one over-deduplicated set — doing so conflates
  two different, both-legitimate questions and risks shrinking the
  external set to statistical meaninglessness.

### 4.4 Screening rediscovery test — targeted holdout, not blanket exclusion
- Hold out **only the specific known-good target materials and their
  exact canonical duplicates** from training — not all high-zT materials
  broadly.
- Retain other representative high-zT materials in training, so the model
  retains a high-performance signal to generalize from.
- Interpret the resulting rediscovery result explicitly as **caveated
  tail-extrapolation**, not as unqualified validation.
- If, after this targeted holdout, the remaining high-zT training region
  is too thin to support meaningful prediction, **report that finding
  directly** (i.e., "the high-performance region is too sparse to support
  reliable tail extrapolation") rather than forcing or overstating a
  positive rediscovery result.

### 4.5 Temperature-extrapolation axis — CUT
**Decision: remove this axis from the paper, reduce to a single
limitation paragraph.** Reason: thermoelectric zT is typically
non-monotonic in temperature (rises to a peak, then rolls over), and this
peak commonly falls within the 600-800K range being held out. A model
trained only on the ≤600K rising branch cannot recover a non-monotonic
curve's behavior beyond that range — this applies to any extrapolator,
tree-based or otherwise, not only to the tree-specific limitation
previously identified. This makes the axis fundamentally a **data-coverage
statement**, not a generalization finding, and it does not survive as a
supporting result. Cutting it tightens the paper's scope (consistent with
Section 5's spine-vs-extension framing in v2).

### 4.6 PCA-split (Athar/Jund) engagement — empirical, not asserted
Rather than asserting that chemistry-cluster grouping and PCA-based
splitting solve "different problems," **run both on the same dataset and
report both resulting honest numbers directly**, alongside a stated
explanation of why the chemistry-cluster approach is stricter against
near-duplicate leakage specifically (PCA-splitting optimizes train/test
representativeness/coverage but does not prevent near-duplicate
compositions from landing on opposite sides of the split). This converts
the differentiation claim from an assertion into an empirical result a
reviewer from that camp cannot dismiss as hand-waving.

---

## 5. Scope (unchanged from v2 — do not reopen)
Core spine: inflation ladder (4.1) + noise-floor anchor (3.3) + two-number
external validation (4.3). Supporting extension: direct-vs-derived pathway
(subset-matched, per v2 Section 3.7 — unchanged). Applied coda: targeted
screening rediscovery (4.4). Temperature extrapolation is removed per 4.5,
not retained even as an extension.

---

## 6. Build order for this paper (Phase 0-2 of the shared build plan)

**Phase 0 (shared with Paper B, run once):** fresh Starrydata2 pull →
global cleaning (Section 3.2) → composition canonicalization → chemistry-
cluster definition at 5 at% (Section 3.1). This phase gates everything
downstream in both papers — nothing after this point is trustworthy until
the cluster definition is fixed and the sensitivity table is produced.

**Phase 1 (this paper's publishable spine):** five-way ladder (4.1) +
noise floor in log space (3.3) + two-metric external validation (4.3).

**Phase 2 (finish):** direct-vs-derived on the all-four-properties subset
(unchanged from v2) → screening as caveated coda (4.4) → temperature axis
reduced to one paragraph (4.5) → PCA head-to-head (4.6).
