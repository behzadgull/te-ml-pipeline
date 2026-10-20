# DRAFT: reframing the paper as screening and prediction first, validation in support

Written 2026-10-20 (numbers filled 2026-10-20). No number is changed and no result or caveat is removed: every number below is the same marker as in `paper/paper.md` (filled by `make_thesis_values.py`); sections move, they are not deleted.
Every number below is a marker filled from a committed artifact by `scripts/export_reframe_draft.py` (the same value dictionary as `paper.md`); the numbers of the S and kappa view (section 4) come from `scripts/screening_s_kappa_view.py`, run on a clean tree.

---

## 1. Outline (new section order)

**1 Introduction** (screening aim first). The need for lead-free thermoelectrics and for screening; the aim: ranked, lead-free perovskite-type candidates from composition-only models trained on experiments, and a measured statement
of how far each predicted property can be trusted. Contributions, reordered: (i) the screening and its ranked list, with the shortlist rule fixed before the predictions were seen; (ii) per-property reliability (S, *σ*, *κ*, zT) from
cross-validation, external databases and literature measurements; (iii) the validation methodology that gives (ii); (iv) the data curation; (v) direct versus component-wise zT; (vi) explainability.

**2 Background and related work.** 2.1 Thermoelectric fundamentals; 2.2 Perovskite thermoelectrics; 2.3 Computational methods and materials databases; 2.4 ML for thermoelectric property prediction, with the old 2.5 (validation and leakage) and 2.6
(interpretability) condensed into it; 2.5 Research gaps, with the screening gap first.

**3 Data and models.** 3.1 Research framework (the pipeline, screening as its output); 3.2 Data curation (the eleven steps; long detail to Supplement S2); 3.3 Features; 3.4 Models (hyperparameter search, frozen sets); 3.5 Candidate selection (the old 3.6
virtual-screening protocol, moved up, including the pre-registered shortlist rule); 3.6 Validation protocols (condensed: chemistry-cluster CV, ESTM and teMatDb, JARVIS, the literature comparison; the protocol text of the literature comparison and the
amendments stay in Supplement S1).

**4 How far the predictions can be trusted** (all validation, condensed). 4.1 Chemistry-cluster cross-validation: the headline R² per property and the validation inflation (the old 4.1.1 and 4.1.2); 4.2 External validation: ESTM, teMatDb, JARVIS (old 4.2);
4.3 Literature measurements (the old 4.5.1: the pre-registered comparison, its result, the context paragraph; the per-paper Table 12 and Table 13 go to Supplement S3, the independent-paper exact-level rows stay in the main text); 4.4 Direct versus component-wise
zT (old 4.4); 4.5 What the models use (SHAP, old 4.3); 4.6 Reliability by property: one table that sets, per property, the chemistry-cluster R², the ESTM stratum-b R² and the NA12 exact-level error against the rung (existing numbers only). Algorithm comparison, stacking,
feature selection and the ladder details (old 4.1.3 to 4.1.5, the ladder tables) go to Supplement S2, each with its result and caveat intact.

**5 Screening results** (the main section). 5.1 Candidate pool and filters (the counts); 5.2 The ranked list (Table 11), seen and unseen clusters reported separately; 5.3 The shortlist and its physical limits (Table 11b, the thermal limits, the 600 K view); 5.4 The sign override and the
JARVIS check of the Seebeck model; 5.5 [proposed, added after NA12, see 4] a view by predicted S and *κ*.

**6 Discussion: the candidates and the reliability of each property.** How to read the list given 4.6 (S and *κ* are the better-validated properties, *σ* and zT the poorer ones, with the numbers); the candidates by status (seen or unseen cluster, studied before, thermal limit);
limitations (old 4.7); the published metrics (old 4.6, condensed); future work.

**7 Conclusions** (screening first, then reliability, then the methodological findings).

**Supplement.** S1 the pre-registration and amendments 1 to 17 (exists); S2 validation details (ladder, algorithm comparison, stacking, feature selection, label-noise ceiling, the cleaning funnel); S3 the literature comparison per paper (Tables 12 and 13) and the protocol text.

| New place | From the current paper |
|---|---|
| 1, 2.1 to 2.4, 2.5 | 1, 2.1 to 2.7 (2.5 and 2.6 merged into 2.4) |
| 3.1 to 3.5 | 3.1, 3.2, 3.3, 3.4, 3.6 (3.2 detail in S2) |
| 3.6 | 3.5, 3.7 (protocols condensed) |
| 4.1 to 4.6 | 4.1.1, 4.1.2, 4.2, 4.5.1, 4.4, 4.3 and a new reliability table |
| S2 | 4.1.3, 4.1.5, parts of 4.1.1 and 3.2.2 |
| 5.1 to 5.4 | 4.5 (virtual screening), 4.1.4 (sign classifier) |
| 6 | 4.6, 4.7, parts of 4.5 and 5 |
| 7 | 5 |

---

## 2. Titles and keywords

Current title: *Machine learning-assisted materials screening and prediction of thermoelectric properties of lead-free perovskite materials*.

1. *Machine-learning screening of lead-free perovskite thermoelectrics from experimental data, with property-by-property validation*
2. *Predicting and screening lead-free perovskite thermoelectric materials with composition-based machine learning*
Alternatives:
3. *Screening lead-free perovskite thermoelectrics with composition-based machine learning: ranked hypotheses and the reliability of each predicted property*
4. *Hypotheses for lead-free perovskite thermoelectrics from experimentally trained machine learning, with property-by-property validation*
5. *Which predicted properties can be trusted? Composition-based machine-learning screening of lead-free perovskite thermoelectrics*

Keywords: Thermoelectric materials ; Lead-free perovskites ; Virtual screening ; Machine learning ; Figure of merit ; Seebeck coefficient ; Thermal conductivity ; Cross-validation ; XGBoost ; SHAP

(The current keywords "Validation bias" and "Data leakage" are dropped from the list; "Cross-validation" keeps the validation theme.)

---

## 3. Abstract, rewritten

Lead-free thermoelectric materials are needed, and screening candidates with machine learning (ML) is useful only if the predictions can be trusted. Here, gradient-boosted (XGBoost) models trained on a pinned snapshot of Starrydata2 (<!--v:n_clean-->280,664<!--/v--> cleaned rows; <!--v:n_S-->185,064<!--/v-->, <!--v:n_sigma-->182,755<!--/v-->, <!--v:n_kappa-->121,110<!--/v--> and <!--v:n_zT-->129,419<!--/v--> rows with a Seebeck coefficient, electrical conductivity, thermal conductivity and figure of merit), described by <!--v:n_feat-->397<!--/v--> composition-based features (MAGPIE, CBFV and temperature), predict the Seebeck coefficient (S), electrical conductivity (*σ*), thermal conductivity (*κ*) and figure of merit (zT) and are applied, as hypothesis generation, to lead-free, perovskite-type compounds of the Materials Project. The screening gives <!--v:mp_f6-->30<!--/v--> candidates, <!--v:nv_main_seen-->9<!--/v--> of them in chemistry clusters present in the training data and <!--v:nv_main_unseen-->21<!--/v--> in clusters that are not. Under a rule fixed before the predictions were seen, the shortlist is the <!--v:sl_n-->4<!--/v--> candidates that are on the convex hull and in chemistry clusters present in the training data (<!--v:sl_1_name-->CsSnI~3~<!--/v-->, <!--v:sl_2_name-->BaSnO~3~<!--/v-->, <!--v:sl_3_name-->LaRhO~3~<!--/v--> and <!--v:sl_4_name-->LaCoO~3~<!--/v-->, in the order of their predicted zT); all <!--v:sl_lit_yes-->4<!--/v--> have a published thermoelectric measurement and a chemistry cluster present in the training data, so they are not blind recoveries. The ranked list is a set of hypotheses to test, not a set of predictions to rely on, and how far each predicted property can be trusted was measured. Chemistry-cluster grouped cross-validation lowered R^2^ by <!--v:gap_lo-->0.134<!--/v-->--<!--v:gap_hi-->0.213<!--/v--> relative to random splits, giving composition-only R^2^ values of <!--v:chem_S-->0.753<!--/v-->, <!--v:chem_sigma-->0.702<!--/v-->, <!--v:chem_kappa-->0.809<!--/v--> and <!--v:chem_zT-->0.746<!--/v--> for S, *σ*, *κ* and zT, respectively. External validation on the independent ESTM database lowered R^2^ further: for zT it was <!--v:estm_b_zT-->0.513<!--/v--> on rows whose chemistry cluster is absent from training and <!--v:estm_a_zT-->0.666<!--/v--> on rows sharing no source publication with training. Against DFT-computed Seebeck coefficients (JARVIS) the Seebeck model shows at most marginal sign transfer (agreement <!--v:jvc_all-->0.52<!--/v-->) and a negative rank correlation for semiconductors (<!--v:jv_rho_semi-->-0.08<!--/v-->). In the pre-registered comparison of out-of-fold predictions with published measurements, <!--v:na12_ind_S_exact_in-->4<!--/v--> of <!--v:na12_ind_S_exact_n-->16<!--/v--> Seebeck values and <!--v:na12_ind_nonS_exact_in-->0<!--/v--> of <!--v:na12_ind_nonS_exact_n-->36<!--/v--> *σ*, *κ* and zT values from <!--v:na12_ind_exact_npapers-->3<!--/v--> independent papers lie inside the measurement band at the exact level (the inter-laboratory measurement uncertainty, <!--v:na12_eps_min_pct-->6<!--/v-->--<!--v:na12_eps_max_pct-->19<!--/v--> %). Direct zT regression outperformed the component-wise route S^2^*σ*T/*κ* (R^2^ = <!--v:dvd_direct-->0.727<!--/v--> vs <!--v:dvd_derived-->0.540<!--/v-->) because component errors compound, and SHAP analysis ranked temperature first for the zT model and a CBFV statistic first for the S, *σ* and *κ* models. No predicted zT of any other candidate is presented as a finding. Composition-based models are useful for triage, but chemistry-aware validation is essential to avoid overstating their accuracy.

(Every sentence, number and caveat of the current abstract is kept; the order is now screening, shortlist, "hypotheses, not predictions", then the validation that says how far to trust each property, then the two methodological findings. The section reference "Section 4.5.1" is dropped from the abstract and becomes "Section 4.3" in the body when the sections move.)

---

## 4. Proposal (added after NA12, not pre-registered): a screening view by predicted S and *κ*

**Why.** The pre-registered ranking is by predicted zT. The validation evidence says that S and *κ* are the better-validated properties and *σ* and zT the poorer ones: chemistry-cluster R² of <!--v:chem_kappa-->0.809<!--/v--> (*κ*), <!--v:chem_S-->0.753<!--/v--> (S),
<!--v:chem_zT-->0.746<!--/v--> (zT) and <!--v:chem_sigma-->0.702<!--/v--> (*σ*); on ESTM stratum b (cluster absent from training) *κ* and zT hold up better than S and *σ*; and in the literature comparison the exact-level error of S is
<!--v:na12_ctx_S_ratio-->0.7<!--/v--> times the rung's typical error, that of *κ* <!--v:na12_ctx_kappa_ratio-->1.9<!--/v--> times, of *σ* <!--v:na12_ctx_sigma_ratio-->3.2<!--/v--> times and of zT <!--v:na12_ctx_zT_ratio-->11.8<!--/v--> times, on small samples. A view
that lists the candidates by S and *κ* alone, next to the zT ranking, would show which candidates are favourable on the properties we can better trust, and how much of the zT order rests on the property we trust least.

**What it is (design, fixed before the script was run).** Using only committed files (`results/na11_ranked/20261005T053426/`, the per-temperature predictions behind it and the thermal limits of Table 11b, `docs/shortlist_thermal_limits.csv`): (a) a two-dimensional figure of |S| against *κ* for the <!--v:sk_n_main-->30<!--/v--> main-list candidates, seen and unseen clusters marked, with the position # of Table 11 written beside each point; (b) Table 11c, with each candidate's zT rank (the position # of Table 11, the rank within its cluster group), its rank by |S| and its rank by *κ* (low is better), ordered by the sum of the last two; (c) no combined score and no new threshold. S and *κ* are taken at the grid temperature of the maximum predicted zT within each compound's thermal limit: CsSnI~3~ (melting at 724 K; <!--v:sk_n_limited-->2<!--/v--> of the <!--v:sk_n_main-->30<!--/v--> entries) is limited to <!--v:sk_T_limit-->700<!--/v--> K, the other <!--v:sk_n_800-->28<!--/v--> use <!--v:sk_T-->800<!--/v--> K, and LaCoO~3~'s conditional limit (reducing atmosphere) is flagged, not applied, as in Table 11b. Limits were researched only for the shortlist, so no other compound has one applied; this is a stated gap. Nothing is refitted and no new prediction is made. It is labelled "added after the literature comparison, not pre-registered", the pre-registered shortlist and its rule are unchanged, and every number is generated by a script from the committed files. (A first run took 800 K for all 30 candidates, although CsSnI~3~ is limited and sits at positions 1 and 2 of Table 11c; it was corrected the same day and is recorded in `docs/decisions.md`.)

**What it shows (computed by `scripts/screening_s_kappa_view.py`, results/screening_s_kappa/20261020T053227, clean tree; added after NA12, not pre-registered).** In the <!--v:sk_n_main-->30<!--/v-->-candidate main list, the rank correlation (Spearman) of predicted zT at that temperature with |S| is <!--v:sk_rho_zT_S-->0.06<!--/v--> and with low *κ* is <!--v:sk_rho_zT_kappa-->0.45<!--/v-->, and |S| and *κ* are almost uncorrelated (<!--v:sk_rho_S_kappa-->−0.05<!--/v-->): the zT order follows *κ* more than S. Of the <!--v:sk_top_n-->10<!--/v--> best by zT, <!--v:sk_top_zT_kappa-->5<!--/v--> are among the <!--v:sk_top_n-->10<!--/v--> lowest in *κ* and <!--v:sk_top_zT_S-->4<!--/v--> among the <!--v:sk_top_n-->10<!--/v--> largest in |S|; <!--v:sk_top_S_kappa-->4<!--/v--> candidates are in the top <!--v:sk_top_n-->10<!--/v--> for both |S| and low *κ*. <!--v:sk_seen_zT-->2<!--/v--> of the <!--v:sk_top_n-->10<!--/v--> best by zT are in chemistry clusters present in the training data, against <!--v:sk_seen_S_kappa-->5<!--/v--> of the <!--v:sk_top_n-->10<!--/v--> best by the rank sum of |S| and *κ*, so the S and *κ* view moves weight toward candidates whose chemistry the models have seen. Several entries of the list (CaVO~3~, CsSnI~3~, MnBO~3~, YCoO~3~) occur twice with identical predictions, which Table 11 already notes.

![Predicted |S| against predicted *κ* at the maximum of the predicted zT within each compound's thermal limit (Table 11b) for the main list; the number beside each point is its position # in Table 11 (the rank by predicted zT within its cluster group). Added after the literature comparison (NA12); not pre-registered.](../results/screening_s_kappa/20261020T053227/fig_s_kappa.png)

**What it does not show.** It does not test any prediction against a measurement; |S| without *σ* does not make a thermoelectric, so the view is a way of reading the list, not a replacement for it; the values of <!--v:sk_n_800-->28<!--/v--> of the <!--v:sk_n_main-->30<!--/v--> entries are at the grid boundary (<!--v:sk_T-->800<!--/v--> K), where the training data are sparsest, so the view inherits the boundary caveat of the zT list; the sign of S is set by the classifier and was overridden for <!--v:sk_n_override-->3<!--/v--> of the <!--v:sk_n_main-->30<!--/v-->; and it uses the models' own predictions, so it cannot say more than their reliability allows (S: ESTM stratum b R² <!--v:estm_b_S-->0.337<!--/v-->, a moderate value).

**Status.** Added to `paper/paper.md` (Section 4.5, after Table 11b: a paragraph, Figure 13 and Table 11c; the decision is recorded in `docs/decisions.md`, 2026-10-20). The version in the paper uses the Figure 13 and Table 11c of the run named above.
