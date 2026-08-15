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
