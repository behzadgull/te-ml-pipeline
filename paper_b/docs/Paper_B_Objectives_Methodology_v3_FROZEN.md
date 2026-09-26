# Paper B: Cross-Family Generalization in Thermoelectric ML
## v3 — FROZEN. Every open question below is a committed decision, not a discussion point.

**Status: closed for review. Implement as written.** Independence from
Paper A is a **gated decision**, not decided here — see Section 5.

---

## 1. Core claim — two crisp, falsifiable questions (replaces "granularity" framing)

The v2 framing ("measurements of the granularity of the learned
composition-property relationship") was correct to drop the
shortcut-vs-physics binary, but was too hedged to be a falsifiable claim.
**Frozen replacement — this paper answers exactly two questions:**

**(a)** Does a pooled, multi-family model's within-distribution accuracy
overstate its accuracy on a held-out family, and by how much, per family?
(The cross-family analog of Paper A's core thesis.)

**(b)** How much of the pooled model's within-family accuracy is
attributable to a family-level mean offset, versus finer intra-family
composition signal?

These two questions are the paper's thesis. Do not retreat to
"we report measurements" as the framing — commit to (a) and (b) as stated
hypotheses and report results against them directly.

---

## 2. FROZEN DECISIONS — Methodology

### 2.1 Primary probe: specialist-vs-pooled comparison (replaces label ablation as primary)
**The family-label ablation test (v2's primary probe) is demoted to a
secondary, supporting check.** Reason: family identity is largely a
deterministic function of composition, so a model with composition
features already implicitly infers family. Adding an explicit family
label therefore only measures *incremental* signal beyond what
composition already encodes — a small measured effect would be
misread as "family doesn't matter," when it may actually mean "family is
redundant with signal the model already has." This down-side
interpretive trap is more likely to occur than the previously-noted
up-side ambiguity, and makes the ablation alone nearly uninformative as
primary evidence.

**Frozen primary probe — two components:**
1. **Specialist-vs-pooled comparison**: train one within-family specialist
   model per qualifying family (its own dedicated cross-validation, no
   other families' data). Compare specialist performance against the
   pooled multi-family model's performance, evaluated within that same
   family. If specialists sharply outperform the pooled model
   within-family, the pooled model is not resolving intra-family
   variation well — direct evidence bearing on question (b) above.
2. **Offset-stripped within-family correlation**: measure the correlation
   between predicted and actual values *within* each family after
   removing the family-level mean offset, isolating how well the model
   captures intra-family variation specifically, independent of getting
   the family's average right.

The family-label ablation (v2 Section 3.6) is retained only as a
secondary, supporting check — not the paper's primary evidentiary claim.

### 2.2 Family labels — stoichiometric-template matching (no structure data required)
Starrydata2 is composition-only (extracted from published plots); most
rows carry no structural information, so a structure-based family
definition is infeasible on this data.

**Frozen decision**: define family membership via **stoichiometric-
template matching from composition alone** (e.g., ABX half-Heusler
template, AB₃ skutterudite template, etc. — computable directly from
parsed formulas).
- Compositions that do not cleanly match any defined template are
  assigned to an explicit **"unassignable" bucket**, not silently
  dropped. Report the size of this bucket.
- Restrict the study to families whose sample count exceeds the a priori
  threshold set in Section 2.4.
- **State explicitly, as a reported limitation, that this approach
  restricts study coverage to clean-template compounds** — real
  thermoelectrics are often doped, defected, or solid-solution
  compositions that won't match a clean template, meaning the families
  this study can test are likely the least representative of the field
  as actually practiced. Report what fraction of the total dataset falls
  into clean, testable templates versus the unassignable bucket.

### 2.3 Two controls, not one — isolating structured vs. random removal
**Original size-matched control (v2) is necessary but not sufficient.**
Leave-one-family-out removes a contiguous chemistry region (an entire
family); a size-matched random-removal control removes scattered rows
from all families instead. Even at matched size, these differ in more
than "family F present vs. absent" — they differ in *regional* versus
*random* removal structure, so part of any observed LOFO deficit could
reflect "a structured region of chemistry space is gone" rather than
"family F specifically is gone."

**Frozen decision — run two controls:**
1. **Size-matched random-removal control** (as in v2): randomly remove
   rows matching the LOFO training-set reduction, scattered across all
   families.
2. **Structured-removal control** (new): remove a *different* whole
   family of similar sample size, in the same contiguous, regional
   manner as the actual LOFO condition.

Comparing both controls against the true LOFO result separates "effect of
losing a structured chemistry region generically" from "effect of losing
this specific family."

### 2.4 Family-size threshold and reporting
- Set a minimum per-family sample-size threshold **a priori**, before
  running any analysis.
- Report every family meeting the threshold, with confidence intervals on
  every per-family score.
- List and flag (do not silently omit) families below the threshold.
- Report per family, not only pooled — the paper's claim concerns the
  pattern of transfer difficulty across families.

### 2.5 Model families — bracket the mechanism, don't average
Run **at least two structurally different model families**, chosen
specifically to bracket capacity: one high-capacity model prone to
memorization (e.g., a gradient-boosted tree ensemble) and one more
constrained/lower-capacity model. Report both models' per-family results
**separately** — do not average across model families, since transfer
magnitude is expected to vary with model capacity, and that variation is
itself an informative result, not noise to be averaged away.

### 2.6 Prior art — full citation list (frozen)
Cite the complete set below in the introduction, engaging rather than
only citing supportively where noted:
- Meredig et al. (2018), *Molecular Systems Design & Engineering* —
  introduced leave-one-cluster-out CV for extrapolation to unseen
  material families.
- Durdy et al., "Random projections and kernelised leave one cluster out
  cross validation" — positions LOCO-CV as a general baseline.
- RSC *Digital Discovery*, "Limitations of ML models when predicting
  compounds with completely new chemistries" (DOI: 10.1039/d2dd00004k).
- "Scaffold Splits Overestimate Virtual Screening Performance"
  (arXiv:2406.00873) — molecular-ML analog of this paper's core concern.
- "Scaffold splits hide structural-frontier failures in ADMET models"
  (arXiv:2607.10729) — **engage directly, not only supportively**: this
  paper argues even grouped/scaffold splits can hide certain failure
  modes, which is directly relevant to this paper's own claims and should
  be addressed as a limitation or counterpoint, not cited as if it only
  supports the grouping approach.
- MD-HIT (*npj Computational Materials*, 2024) — dataset-redundancy
  control for material property prediction; relevant to both this paper's
  and the companion paper's near-duplicate/grouping problem.

---

## 3. Grouping key and shared infrastructure (unchanged from v2 — do not reopen)
Sample-level grouping (per companion Paper A's chemistry-cluster
hierarchy, frozen at the 5 at% dopant threshold) applies identically
across every condition in this paper — the true LOFO condition and both
controls (Section 2.3) — not only the leave-one-family-out condition.

---

## 4. Interpretation logic (unchanged from v2 — do not reopen)
The three-cause ambiguity (shortcut learning vs. covariate shift vs.
genuinely family-dependent physics) is not fully resolved by any single
test in this design, including the specialist-vs-pooled probe (Section
2.1). This is reported as an acknowledged interpretive limitation, not
hidden or oversold as resolved.

---

## 5. Independence from Paper A — GATED decision, not decided here

**This is not resolved by this document.** Post-reframe, this paper's
distinct contribution rests entirely on whether the specialist-vs-pooled
analysis (Section 2.1) produces a clear, interesting coarse-vs-fine
result. The rule for deciding is fixed; the outcome is not:

- **Build Paper A first.** This paper imports Paper A's cleaned dataset,
  chemistry-cluster definition, and (for the frozen-model comparisons)
  Paper A's modeling infrastructure. This paper's entire interpretation is
  relative to Paper A's within-distribution honest ceiling.
- **After Paper A's ladder is complete**, run this paper's LOFO analysis,
  both controls, and the specialist-vs-pooled comparison.
- **Decision rule**: if the specialist-vs-pooled analysis shows a clear,
  well-evidenced coarse-vs-fine story, publish this as a standalone paper.
  If the result is weak or ambiguous, fold this analysis into Paper A as
  a single additional section (e.g., "cross-family generalization") rather
  than publishing separately.
- Do not attempt to resolve standalone-vs-fold before this analysis
  exists — it cannot be determined in advance.

---

## 6. Build order for this paper (Phase 3 of the shared build plan)

Runs only after Paper A's Phase 0 (shared cleaning/cluster-definition) and
Phase 1 (five-way ladder) are complete:
1. Stoichiometric-template family labeling (2.2), report unassignable
   bucket size.
2. LOFO condition + both controls (2.3), per-family with CIs (2.4).
3. Specialist-vs-pooled comparison + offset-stripped correlation (2.1),
   across two model families (2.5).
4. Apply the Section 5 decision rule: standalone paper, or fold into
   Paper A.

---

## 7. v4 amendments (2026-09-26)

Recorded before any Paper B family count was computed. Sections 1 to 6
above are unchanged; where an amendment below is more specific than the
text above, the amendment governs. Items (a) to (e) are frozen on the
same terms as the rest of this document.

**(a) A priori family-size threshold (fills the open value in 2.4).**
A family qualifies for a target only if it has at least 30 chemistry
clusters AND at least 1,000 rows for that target. The threshold is
applied per target, so a family can qualify for one property and not
another. It is stored in `config.yaml` as `paper_b.min_family_sample_size`
(`min_clusters: 30`, `min_rows_per_target: 1000`). Reasons: within-family
5-fold grouped CV puts whole chemistry clusters in folds, so 30 clusters
gives at least 6 per fold; 1,000 rows gives about 200 test rows per fold,
enough for a stable three-term MSE decomposition (see (b)). The threshold
was committed before any family counts were computed, so it cannot have
been tuned to them.

**(b) Primary metric for question (b).** Per held-out family, the Murphy
decomposition of the mean squared error,

    MSE = (mean(y_hat) - mean(y))^2 + (s_yhat - r s_y)^2 + (1 - r^2) s_y^2

that is, an offset term, a scale term and an unexplained-variance term,
with each reported as a share of the MSE. Alongside it: the within-family
Pearson correlation r, and skill against a constant baseline. Raw R^2 is
reported next to these, not instead of them. Reason: R^2 computed on a
single family folds the family-level offset and the intra-family signal
into one number, which is the confound question (b) asks about; the
decomposition separates them. The definition of the constant baseline is
fixed in the step that implements the metric, before any held-out-family
result is computed.

**(c) Hyperparameters are retuned inside each LOFO fold, on the remaining
families only.** Paper A's frozen hyperparameters were tuned with every
family present, so they carry information about the held-out family;
reusing them for LOFO would let the held-out family influence the model
it is meant to be foreign to. The same rule applies to both controls in
2.3 and to the specialists in 2.1, each tuned on its own training data
only.

**(d) Family labels come from the host formula of the snapfix
`chemistry_cluster_id`, not from the raw formula.** Reason: the label must
be constant within a chemistry cluster, so that family and the grouping
key nest (Section 3) and no cluster can straddle a train/test boundary
by carrying two labels. A raw formula would put La-doped and Yb-doped
CoSb3 in the same family only if the matcher tolerated the dopants; the
cluster host has already removed them at the frozen 5 at% threshold.
Source dataset: the snapfix featurized CSV recorded in CLAUDE.md
(SHA256 `d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489`).

**(e) Prior evidence for question (a).** Ho et al. (2026,
`ho2026physicsinspired`) is acknowledged as prior evidence bearing on
question (a) and must be cited in the introduction. The paper must state
exactly which held-out unit that study used; CLAUDE.md records its
verified split as composition-wise (all temperature records of a
composition held out), so any statement that it is family-wise must be
checked against the paper before it is written.

### 7.1 Further amendments (2026-09-26, after the first commit of section 7)

These are additions and corrections dated after items (a) to (e) above were
committed (`6f88f9d`). Items (a) to (e) are left as committed; where a
sub-item below is more specific, it governs.

**(e) correction.** Ho et al. (2026, `ho2026physicsinspired`) compared random,
composition-wise and family-wise single splits on a 3,879-row ESTM subset.
The family-wise split gave negative R^2 for all targets, and a constant-mean
baseline was also negative in every seed-target cell. Verified from the paper
on 2026-09-24. This is the check that item (e) required, and it confirms the
study has a family-wise result; the composition-wise result recorded in
CLAUDE.md comes from the same study. The introduction states this as prior
evidence for question (a), alongside the composition-wise result.

**(b) completion.** The skill baselines are defined here, before any
implementation:
- `skill_train = 1 - MSE_model / MSE_c`, where `c` is the mean of the target
  over the training families of that LOFO fold (the constant a practitioner
  could have used without seeing the held-out family), and `MSE_c` is the mean
  squared error of predicting `c` for every row of the held-out family;
- R^2, whose baseline is the held-out family's own mean (an oracle no
  practitioner has), is reported alongside `skill_train`, not instead of it.
Both are computed in each target's Paper A scoring space (log10 for sigma and
kappa, linear for S and zT). This replaces the sentence in (b) that left the
constant's definition to the implementing step.

**(f) code layout.** All Paper B code, configuration, documents, results and
reports live under `paper_b/`, so the paper can be handed over separately.
Paper B imports only the top-level modules declared, with SHA256, in
`paper_b/SHARED_DEPENDENCIES.md`, and loads the snapfix CSV by explicit path
with a SHA256 check. The threshold in (a) is stored in
`paper_b/config/paper_b.yaml`, not in the top-level `config.yaml` as (a)
states; its values are unchanged. Introduced in commit `0009abf`.

### 7.2 Family rules revised once, v1 to v2 (2026-09-26)

Family rules revised once (v1 -> v2) after reviewing the unassignable hosts and
audit sample of the v1 run, before any model training. Reasons per change
follow. The v1 rules (`paper_b/config/families_v1.yaml`, unchanged) and the v1
run (`paper_b/reports/family_labels/20260926T154406/`) are retained. The
a priori threshold in (a) is unchanged and applies to the v2 families exactly
as it did to v1; no model has been trained on either version.

1. **Tolerance 10% -> 15% for every site-ratio rule** (Cu2X / Ag2X stays at
   20%). Removing a substituted dopant makes its site look deficient, and
   cation-deficient hosts (Ge1-xTe) are normal. Applied uniformly, not per
   family.
2. **iv_vi_rocksalt split into snse_type, gete_type and the remaining
   iv_vi_rocksalt.** SnSe/SnS and GeTe are structurally distinct from the
   rocksalt IV-VI compounds. For alloys the majority cation decides (a tie
   has no majority and falls through to iv_vi_rocksalt). snse_type takes Sn
   with Se or S only; gete_type takes Ge with Te only.
3. **The oxide rule split by cation set** into layered_cobaltite (Co with Ca,
   Na, Bi or Sr), perovskite_titanate (Ti with Sr, Ca or Ba), manganite (Mn
   with Ca, La or Sr), zno_based (Zn the majority cation), in2o3_based (In the
   majority cation) and other_oxide. O >= 20% is kept and every oxide rule is
   checked after all non-oxide rules. One oxide bucket mixed chemically
   unrelated compounds. The cation conditions test presence, not layered
   structure, so the label layered_cobaltite is a name for the rule, not a
   structural claim.
4. **New rules for families that were unassignable in v1:** zinc_antimonide
   (Zn4Sb3 or ZnSb, two variants); zintl_14_1_11 (A14MX11, M optional because
   one M per 26 atoms is below the 5 at% dopant threshold); tm_silicide (Mn,
   Cr, Fe, Ru or Re with 1.7 to 2.0 Si per metal, range not widened by the
   tolerance); sige (Si and/or Ge only); i_v_vi2 (Ag or Cu, Sb or Bi, Te, Se
   or S, 1:1:2); gete_sb2te3_pseudobinary ((Ge, Pb or Sn)Te with Sb2Te3 or
   Bi2Te3, tested by charge balance because the tie line has no fixed ratio,
   and placed after tetradymite and the IV-VI rules); diamond_like_cu (1:1:2,
   3:1:4 and 2:1:3 variants); full_heusler (X2YZ); mgagsb (1:1:1).
5. **Choices not specified in the request and open to review:** the element
   sets of full_heusler and of the optional M site of zintl_14_1_11.

### 7.3 Family rules v2.1: defect fixes, held-out set, super-families (2026-09-26)

**Family rules v2.1 correct three defects found in the v2 audit; they add no
family.** The v2 rules (`paper_b/config/families_v2.yaml`) and their run
(`paper_b/reports/family_labels/20260926T155416/`) are retained, as are v1 and
its run (section 7.2). No model has been trained on any version. Reasons per
fix:

1. **layered_cobaltite renamed cobaltite; Co versus Mn decided by the majority
   transition metal.** The cation test (Co with Ca, Na, Bi or Sr) never
   checked layering and admits perovskite Sr/Ba cobaltites, so the name
   overclaimed; the label names the cation rule and makes no structural claim.
   A host containing both Co and Mn now goes to cobaltite if it has strictly
   more Co than Mn, and to manganite if it has strictly more Mn than Co (both
   rules keep their own cation conditions); a tie matches neither and falls to
   the later oxide rules. This is the same principle as the IV-VI majority
   cation. In v2 such a host went to whichever rule came first.
2. **gete_sb2te3_pseudobinary requires (Sb + Bi) to be at least 1/3 of all
   cations** (Ge, Pb, Sn, Sb, Bi). v2 accepted any charge-balanced
   (Ge,Pb,Sn)(Sb,Bi)Te, which swept up GeTe- and SnTe-type hosts carrying 5 to
   10% Sb or Bi; those are not pseudobinary compounds. A host that fails the
   1/3 test is not taken by gete_type or iv_vi_rocksalt either, because Sb and
   Bi are not on those rules' sites, so it is unassignable.
3. **tm_silicide follows the uniform 15% principle.** v2 left this rule out of
   it. The 15% tolerance now applies to the metal site, so the effective Si
   per metal runs from 1.7 up to 2.0 / 0.85 = 2.353. The lower limit is
   unchanged.

**Known remaining over-reach.** zintl_14_1_11 captures other Yb-Sb (and Ca-Sb,
Ca-Bi) stoichiometries, because 15% on a 14:11 count accepts a wide A:X range.
It qualifies for no target under the a priori threshold, so it is never held
out and this over-reach does not affect a held-out-family result.

**Held-out family set.** A family is held out for a target only if it is a
named family that passes the a priori threshold of (a) for that target. The
unassignable bucket and other_oxide are never held out; they always stay in
the training pool.

**Pre-registered super-family sensitivity analysis.** In addition to the
primary family-level analysis, which is unchanged, the leave-one-family-out
analysis is repeated with these groups treated as single families:
- IV-VI = {iv_vi_rocksalt, gete_type, snse_type, gete_sb2te3_pseudobinary};
- oxides = {cobaltite, manganite, perovskite_titanate, zno_based, in2o3_based};
- zintl = {zintl_122, mg3x2_zintl, zintl_14_1_11}.

All other families stand alone in this analysis. The primary analysis stays at
family level.

### 7.4 Family rules v2.2 and super-family qualification (2026-09-26)

**v2.2 implements the intent of the v2.1 pseudobinary fix.** The v2.1
instruction assumed that hosts failing the 1/3 test would fall through to
gete_type or iv_vi_rocksalt by majority cation. The rules could not do that: Sb
and Bi were not on those rules' sites, so such hosts became unassignable. v2.2
gives gete_type and iv_vi_rocksalt an optional Sb/Bi share of the cation site,
pooled with it in the cations:anions ratio (within the uniform 15% tolerance)
and required to be strictly below 1/3 of the total cations, the exact
complement of the pseudobinary rule's at-least-1/3. GeTe and IV-VI hosts with
Sb + Bi below 1/3 of the cations therefore belong to those families, and hosts
at or above 1/3 to gete_sb2te3_pseudobinary. The majority cation among Ge, Pb
and Sn still decides between gete_type and iv_vi_rocksalt, and snse_type is
unchanged. No other rule changed; only Sb and Bi were added. The v2.1 rules
(`paper_b/config/families_v2_1.yaml`) and their run are retained. The run
asserts, before writing anything, that the only hosts whose label differs from
v2.1's are unassignable hosts that moved to gete_type or iv_vi_rocksalt and
contain an element of each of Ge/Pb/Sn, Sb/Bi and Te/Se/S. The v2.2 rules are
frozen from the commit that records them; any further change needs a dated
amendment.

**Super-family qualification (defines the open point of 7.3).** A super-family
qualifies for a target under the same a priori threshold as a family (at least
30 clusters and at least 1,000 rows for that target, clusters counted as those
holding a row for the target), applied to the union of its members. Members
that fall below the threshold on their own still count toward the union. The
definitions are in `paper_b/config/super_families.yaml` and the calculation in
`paper_b/scripts/super_family_qualification.py`.

---

## 8. Modelling harness (pre-registered) (2026-09-26)

Design only; no harness code exists yet. It applies to the held-out set defined
in 7.3 (named families that pass the threshold of (a) for the target, never
unassignable or other_oxide) and, as the sensitivity analysis, to the
super-families of 7.3 and 7.4. The rules and labels are those of the final
family-label run recorded in section 7.4.

### 8.1 Design

For each target t and each held-out unit F (a family in the primary analysis; a
super-family in the sensitivity analysis, where the test set is the union of
its members):

- **Test folds.** F's rows for t are split into 5 chemistry-cluster folds, R
  times (R repeats, each a different randomised assignment, using the module of
  `paper_b/SHARED_DEPENDENCIES.md`). The test fold is F's rows in that fold. All
  conditions and the specialist use the same folds.
- **C0, pooled.** Train on all rows for t except the test fold.
- **C1, LOFO.** Train on all rows for t except all of F. Its training set does
  not depend on the test fold, so it is fitted once per (t, F, model) and its
  predictions are used for every fold and repeat.
- **C2, size-matched random.** C0 minus a random, scattered sample of non-F rows
  (uniform over rows, not clusters), drawn so that the training size equals C1's,
  that is |F| minus the test-fold size rows are removed. Seeded by a
  deterministic function of (t, F, repeat, fold); the seed and the removed row
  identities are recorded.
- **C3, structured.** C0 minus one whole family G, so the training set loses a
  contiguous chemistry region other than F. G is chosen by rule, per (t, F): the
  held-out unit for t closest to F in row count for t, outside F's own
  super-family (F itself and its co-members in the primary analysis; F itself in
  the sensitivity analysis), never unassignable or other_oxide; ties go to the
  name first in alphabetical order. G is recorded.
- **Hyperparameters.** Tuned once per (t, F, model), on the rows for t excluding
  F, with an inner 3-fold chemistry-cluster CV, the search space and TPE/median
  pruning setup of Paper A (`src/nested_cv.py`), and used unchanged by C0 to C3.
  This is the LOFO fold of (c): tuning never sees F. Pooled tuning uses 20 Optuna
  trials (added 2026-09-26).
- **Specialist.** Trained on F's training folds only, on the same test folds, with
  nested tuning inside each outer training fold (inner 3-fold chemistry-cluster
  CV, the same search space, on F's training rows), using **10** Optuna trials,
  against 20 for the pooled tuning (added 2026-09-26). Compared with C0 of the
  same model and folds. Direction of bias: an under-tuned specialist can only
  understate the specialist advantage, so any positive specialist-minus-C0 result
  is conservative; a null result must be reported with this caveat.
- **Models.** XGBoost and ridge (a StandardScaler pipeline; alpha tuned in the
  same way over Paper A's ridge search space), reported separately and never
  averaged (2.5).

### 8.2 Metrics

Per (t, F, condition, model), scored in each target's Paper A space (log10 for
sigma and kappa, linear for S and zT), computed within F only and never pooled
across held-out units:

- R^2, per repeat on F's five pooled test folds, then mean and SD over repeats;
  with the oracle baseline of (b) (F's own mean);
- `skill_train = 1 - MSE_model / MSE_c`, where c is the mean of that condition's
  training targets (the specialist's, its own training folds');
- the Murphy decomposition of (b): offset, scale and unexplained shares of the
  MSE;
- the within-family Pearson correlation r between prediction and target.

**Confidence intervals (added 2026-09-26).** Every metric above carries a 95%
percentile interval from a cluster bootstrap over F's chemistry clusters: 1,000
resamples, seeded, per (t, F, condition, model), computed on the saved
predictions. In each resample, F's clusters are drawn with replacement; for each
repeat, that repeat's test predictions of the drawn clusters are pooled (a
cluster drawn twice counts twice), the metric is computed, and the metric is
averaged over repeats. Condition differences (C0 - C1, C0 - C2, C0 - C3 and
specialist - C0) use the same resample for both conditions of the model (paired
resamples), and the interval is taken on the difference. The SD across repeats is
reported separately, as split-to-split variability; it is not a confidence
interval.

### 8.3 Provenance

Every run_config records: the snapfix dataset SHA256 and size; the labels run
folder and the SHA256 of its `host_family_labels.csv`; the SHA256 of
`families.yaml`, `super_families.yaml` and `paper_b.yaml`; the seeds; the
tuned-hyperparameter file for each (t, F, model); the code git HEAD; and
`tree_clean`.

### 8.4 Compute and the values still to fix

The number of tuning trials is fixed in 8.1 (20 pooled, 10 per specialist fold).
The number of repeats R is fixed from the compute estimate
(`paper_b/reports/compute_estimate/`) and recorded, before any run, in a dated
amendment. The estimate prices the design above from Paper A's recorded fit
times; its assumptions are stated in that folder's `run_config.json`.
