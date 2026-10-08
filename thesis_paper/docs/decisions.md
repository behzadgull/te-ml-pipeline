# Thesis paper: decisions

## 2026-10-04: no number from the thesis is trusted

Standing rule. This includes thresholds, constants and quoted literature values.

1. Every number in `paper/paper.md` is exactly one of:
   - **GENERATED**: an inline marker `<!--v:key-->value<!--/v-->` filled by `scripts/make_thesis_values.py` from a committed artifact, or a value in a generated
     table block that traces to a value of the generator's dictionary.
   - **CITED**: an inline marker `<!--c:id-->value<!--/c-->`; the value was verified against the source itself, and `docs/number_registry.csv` has a row with
     class CITED, `verified` = yes, the source, and the page, table or equation it was read from. A value taken from an abstract, a search result or a secondary
     paper is not verified.
   - **DESIGN**: an inline marker `<!--d:id-->value<!--/d-->`; a constant that a committed config or a code constant holds, with its justification in the text;
     the registry row names the config key and the justification. `docs/design_constants.csv` is the working list, with the literature check for each constant
     (status JUSTIFIED, PARTLY JUSTIFIED, UNJUSTIFIED, OWN DESIGN, UNVERIFIED).
2. `scripts/audit_numbers.py` finds every number in `paper.md` (text, tables, captions) and classifies it; `--strict` and the build fail on any unclassified
   number or any marker without a valid registry row. The only exclusions are section numbers, cross-references to tables, figures, sections, steps and
   equations, equation numbers, list enumerators, citation keys and reference numbers, link targets, digits inside sub- and superscripts (chemical formulas and
   unit exponents) and the digits inside LaTeX math; each exclusion is counted in the audit summary so that it can be reviewed. Reference-list entries live in
   `refs.bib`.
3. `scripts/thesis_carryover_check.py` lists every number of `paper.md` whose value also occurs in the original thesis docx, with both contexts, for review.
4. A DESIGN constant that came from the thesis needs a literature justification or is flagged as unjustified in `docs/design_constants.csv`; an unjustified
   constant stays in the paper only as a stated choice (and, where it matters, with a sensitivity analysis).

First audit, 2026-10-04 (before any number was changed): `reports/number_audit/`, `reports/thesis_carryover/`, `docs/design_constants.csv`.

## 2026-10-05: sign override and JARVIS framing (before the G3 GPU session)

1. **Classifier inputs checked (leakage):** the NA6 classifier's 397 inputs are the 132 MAGPIE and 264 CBFV composition descriptors and `temperature_bin` (the `feature_columns` of
   `results/na6/20261004T102905/units/final.json`); no measured S, sigma, kappa or zT column is among them.
2. **Sign override decision:** rule fixed in `scripts/na6_sign_override.py` before the same-folds run: the classifier's sign replaces the regressor's only if the lower 95% cluster-bootstrap
   bound of the paired sign-accuracy difference (classifier minus regressor) is above zero. Because the Kaggle classifier and the S regressor use different fold partitions (the classifier
   dropped the 8 S = 0 rows), the comparison was repeated with the classifier refitted on the regressor's own folds (`results/na6_sign_override/20261004T194811_same_folds`): 0.949 against
   0.941, difference +0.0083 [0.0043, 0.0121]. **Decision: keep the override**, stated as a small gain (about 0.8 points, mostly rows with |S| < 20 uV/K). The regressor's sign stays in the
   output (`S_reg`).
3. **JARVIS framing:** the sign test covers mostly metals and semimetals (JARVIS gap < 0.05 eV: 91% of the entries that enter it), because a semiconductor's n and p values have opposite
   signs by construction. The paper reports sign agreement and rank correlation by gap class and by seen / unseen cluster, and states that the result is a description of transfer to
   computed values, not a validation (`results/na10_analysis/20261004T190444`). The temperature of the JARVIS Seebeck values (600 K) is still UNVERIFIED (D26).
4. **BaZrSe3:** MP mp-998350 has E_hull 0.0775 eV/atom (not stable, PBE gap 0.20 eV): it is in the E_hull <= 0.10 lists only and in no list at E_hull <= 0.05. The paper must not call it a
   main-list candidate; the thesis's earlier claim of BaZrSe3 as the leading, stable candidate is not reproduced.

## 2026-10-05: two pre-registrations, made before any G3 output is unpacked and before any prediction for the MP candidates or the literature compounds has been looked at

No prediction of the final models (`final_b`), no screening prediction and no out-of-fold prediction for a literature compound was inspected when these rules were written. The shortlist below
uses only the committed sensitivity and training-cluster tables (`results/na11_sensitivity/20261004T135205`, `results/na11_candidate_novelty/20261004T185101`).

### A. Candidate-highlighting rule (screening, Section 4.5 and the abstract and conclusion)

1. **Shortlist** = the compounds (Materials Project ids) of the main list (E_hull <= 0.05 eV/atom, PBE gap <= 0.6 eV, main toxic-element list) that
   (i) are in the candidate set of every E_hull tightening of the sensitivity run (<= 0.025 and <= 0, i.e. they are on the convex hull), and
   (ii) have a chemistry cluster that occurs in the training data (any target; the `cluster_seen_any` flag of the candidate-novelty run).
   Currently four ids: BaSnO3 (mp-3163), CsSnI3 (mp-614013), LaCoO3 (mp-1288145) and LaRhO3 (mp-5163). The set is fixed by the committed tables; a different set after any rerun
   of the screening is a change of this pre-registration and must be recorded here before it is used.
2. **Ordering of the shortlist**: by the maximum over the screening grid (300 to 800 K in 100 K steps) of the predicted zT (direct model, final models of `final_b`), highest first; the
   temperature of the maximum is reported with it (ties: the lower temperature). The ordering applies to the shortlist only.
3. **Literature label**: every candidate of the full list is labelled "previously studied as a thermoelectric" or not, from a literature check that is done independently of the predictions.
   Criterion: at least one published report of measured thermoelectric transport (S, sigma, kappa or zT) of the compound itself, undoped or doped; each "yes" carries a citation (DOI).
   The labels and their evidence are recorded in `docs/candidate_literature_labels.csv` before the predictions are inspected. A shortlist member that was previously studied is
   reported as a recovery, not as a discovery.
4. **Highlighting**: all other candidates appear only in the full ranked list with their seen / unseen and stability labels. No candidate outside the shortlist is highlighted in the
   abstract or the conclusion; the abstract and conclusion may name the shortlist, with the label and the limits (JARVIS semiconductor result, Section 4.2).

### B. NA12 literature validation: design (done before the values are read and before any prediction is looked at)

1. **Compounds, fixed now**: shortlist members that have experimental thermoelectric literature (CsSnI3 and LaCoO3 are expected to; BaSnO3 and LaRhO3 are checked, and are included only if
   such literature exists, the check being recorded in the literature ledger), plus La-doped and Nb-doped SrTiO3 and doped CaMnO3. The exact compositions, including the dopants and their
   amounts, are those of the papers selected for them; the papers (DOI) are listed in `source/pdfs/README.md` and committed before the comparison is run; temperatures are 800 K or below.
2. **Predictions compared**: the out-of-fold predictions of the committed chemistry-cluster CV (`results/ladder_regen_snapfix/20260917T150000/{S,sigma,kappa,zT}_chemistry_full`; the
   cluster of the row is held out in that prediction), five repeats; NOT the final models. Two levels, reported separately and labelled: (a) exact: the row of the same composition_id at
   the same temperature bin as the literature value (the cleaning pipeline's 25 K binning), if the training data contain it; (b) cluster-level: the mean over the rows of the same
   chemistry cluster at that bin (so a dopant level absent from the training data is still compared with held-out predictions of its cluster). If neither exists, "no out-of-fold
   prediction" is reported and no refit is made.
3. **For each literature value two facts are reported**: whether its DOI is among the DOIs of the Starrydata2 data (the DOI is a training source) and whether its composition (and its cluster)
   is in the training data.
4. **Error**: the error of the prediction relative to the literature value is reported next to the measurement uncertainty band of the Alleno round-robin (relative standard uncertainties S, sigma, kappa, zT as
   in `src/noise_floor.py`, cited in the paper); sigma and kappa are compared in log10 and linear space (the model predicts log10; back-transformed without smearing). No pass/fail threshold is set beyond
   "inside the band" or not.
5. **Values** are read from the PDFs only, with page and figure or table; values digitised from a figure are marked as digitised with the method (tool, axis calibration); values from a
   table or the text are marked as such. The values file is `docs/na12_literature_values.csv`; the comparison script (`scripts/na12_literature_comparison.py`) is run only after it is
   filled and committed.

### 2026-10-05 addendum to B (before any prediction for the literature compounds was looked at)

- **Papers (DOI, Crossref-checked)** are listed in `source/pdfs/README.md`: CsSnI3 10.1021/acsaem.2c01936; LaCoO3 10.1080/14786435.2016.1263404; LaRhO3 10.1007/s11664-009-0666-x; La-doped BaSnO3
  10.1039/d0ce00702a; La- and Nb-doped SrTiO3 10.1063/1.1847723; doped CaMnO3 10.1006/jssc.1995.1384 and 10.1109/ict.2006.331291. BaSnO3 and LaRhO3 stay in only if the paper gives the
  values (the check is the values file itself).
- **Value selection rule** (README section): all tabulated or stated values at 300 K up to but not including 800 K; plotted-only properties digitised at 300, 400, 500, 600 and 700 K. A value at
  800 K is recorded and not compared, because the pipeline's 25 K bins cover 300 K to 800 K exclusive.
- **Script**: `scripts/na12_literature_comparison.py` (tested on synthetic rows for PbTe and Bi2Te3 only, never for a pre-registered compound). It compares the out-of-fold predictions at two
  levels (exact composition; cluster-level) and refits nothing; with an empty values file it stops with exit code 3.

## 2026-10-05: NA2 random forest: one fixed, pre-stated hyperparameter set (design change, made before any random-forest test-fold result exists)

**State when this was written.** The first random-forest session (C1) was stopped by its budget after 15 of 20 tuning trials for S (10.7 h, status incomplete; bundle `na2_random_forest.tar.gz`,
SHA256 fe755d4b701ebcb6734ccb1658c69bcc0c5fdd39b78e45cfb83283bdbefcc0dc, verified). Its listing holds only tuning units (`units/tune_S_trial000..014.json`), no outer-fold unit: no
random-forest test-fold prediction or score exists. The values of the 15 trials had not been read when the set below was chosen; the set comes from the literature.

**Decision.**
1. Random forest is run with ONE fixed hyperparameter set for all four targets, no tuning: `n_estimators` 500, `max_features` 1/3 (a third of the 397 features, 132), `min_samples_leaf` 5,
   `max_depth` None, bootstrap sampling with replacement of full size, `min_samples_split` 2, squared-error criterion, `random_state` 0. sigma and kappa are fitted in log10 as in Paper A.
2. Justification (verified in the text of Probst, Wright & Boulesteix 2019, WIREs Data Min. Knowl. Discov. 9:e1301, DOI 10.1002/widm.1301, arXiv:1804.03515v2, PDF pages given):
   - the typical software defaults for regression are mtry = p/3 and a node size of 5, with 500 or 1000 trees and sampling with replacement (Table 1, page 2; text on page 3);
   - the number of trees is "not tunable in the classical sense but should be set sufficiently high" (page 4), so a fixed 500 is not a tuned value;
   - RF "works reasonably well with the default values" (abstract, page 1), and the average gain from tuning the defaults over 38 datasets was small: an AUC increase of 0.010 for all hyperparameters
     together, with 0.006 for mtry, 0.004 for sample size, 0.001 for node size and 0.002 for the replacement rule, per hyperparameter (page 7, citing Probst et al. 2018).
   Limits of this source, to be stated in the paper: those gains are for classification (AUC) on 38 datasets, not for regression on composition features; and the same review (page 3) reports that for
   high-dimensional data higher mtry values gave lower error (Genuer et al. 2008; Goldstein et al. 2011), so mtry = p/3 may sit below the optimum for these 397 features.
3. The 15 completed S tuning trials are reported as evidence of tuning insensitivity: the range of their inner-CV R2 (grouped 3-fold CV on all S rows), no trial excluded, with the search space
   (`src/nested_cv.py`, `_random_forest_search_space`). They are not used to choose the fixed set and are not part of the RF test-fold result.
4. **Asymmetry, stated in the paper:** XGBoost is tuned once on the chemistry-cluster split (20 Optuna trials, frozen) and random forest is not tuned. Direction: if tuning would improve the random
   forest, the fixed setting understates it, so the comparison is biased in favour of XGBoost and an XGBoost advantage over this random forest is an upper bound on its advantage over a tuned one; a random
   forest that matches or beats XGBoost would be a result that survives the asymmetry. The size of the bias is unknown; the 15 trials bound it only for S.
5. LightGBM keeps its tuning (20 trials, the XGBoost protocol). The random-forest rung is a new analysis (new identity): no `--restore-from` of the tuning bundle.
6. The tuning bundle is committed as a record (`results/na2_random_forest_tuning/`), marked incomplete and not an accepted result.

## 2026-10-05: shortlist physics check, per-row SHAP rerun and NA1 on Kaggle (after the G3 predictions had been seen)

1. **Shortlist physics check (annotation, not a change of rule A).** Added at the maintainer's request after the predictions were seen; the shortlist, its order by the maximum over the grid and the
   highlighting rule are unchanged. Thermal limits are in `docs/shortlist_thermal_limits.csv` with their sources: CsSnI3 melts at 451 C (724 K; from two open-access papers that cite the primary
   study, which was not read), so its predictions are reported only up to 700 K (maximum 0.53 at 700 K instead of 0.58 at 800 K); no melting or decomposition temperature at or below 800 K was found for
   BaSnO3 (doped films stable in air at 530 C), LaRhO3 (calculated decomposition 1728 K in air) or LaCoO3 (stable in air; reduced in a reducing atmosphere from about 500 C = 773 K, flagged, not applied).
   The MP structure of CsSnI3 (cubic Pm-3m) is the phase above 440 K; the orthorhombic phase is stable below 362 K. A **secondary view** at 600 K is added as a labelled second ranking because 406 of
   409 maxima sit at the 800 K edge; it gives the same order as the primary ranking.
2. **Figure 11 (beeswarm)**: the G3 NA3 run saved only mean absolute SHAP per feature and fold. A new analysis `na3_rows` (script `kaggle/na3_rows.py`) repeats repeat 0 of the same chemistry-cluster folds
   with the same frozen hyperparameters and saves the SHAP values and feature values of a fixed seeded subsample of 1,000 test rows per fold (generator `default_rng(seed + fold)`), 5,000 rows per
   target, checked by SHAP additivity and by the fold R2 against the committed rung. Nothing else of the committed NA3 changes.
3. **NA1 on Kaggle GPU** (`kaggle/na1_nested_cv.py`): same design as the planned department-machine run (re-tuning inside each outer training fold, 20 trials, 3 inner folds, the Paper A chemistry-cluster
   folds, 5 repeats x 5 folds, four targets), with checkpointed trials and units.

## 2026-10-06: feature labels and the layout of Figures 10 to 12

1. Every feature is named from one committed mapping table, `docs/feature_labels.csv` (built by `scripts/make_feature_labels.py`, read by `scripts/feature_labels.py`): "Elemental <property>, <statistic> (MAGPIE or CBFV)",
   "Temperature" for the temperature bin; source typos are fixed in the label only. Figure 10, Figure 11, Table 9 and the text use it; nothing is typed per plot. The statistic names were checked in the code of the packages:
   the CBFV "dev" statistic is the atomic-fraction-weighted MEAN ABSOLUTE deviation about the weighted mean (CBFV composition.py), not a standard deviation, and is labelled so; MAGPIE "avg_dev" is the same quantity.
2. Captions and the Table 9 note state that every feature other than temperature is a statistic of an ELEMENTAL property of the formula, not a measured property of the material.
3. Panels and columns are in the order S, sigma, kappa, zT. Figure 10 shows the top 15 features per target instead of 20 (D23 and `shap_top_n`) so that the labels, which need two lines, do not overlap; Figure 11 shows 10.
4. An earlier sentence that the third zT feature "governs carrier transport in transition-metal compounds" was an uncited physical claim and is removed.

## 2026-10-06: Figure 10 and Table 9: non-negative intervals across the folds, and only supported orderings are interpreted

1. The error bars of Figure 10 and the intervals of Table 9 are the 2.5th to 97.5th percentile of the per-fold mean |SHAP| across the 25 folds (`thesis_values.INTERVAL`), never mean +/- SD, so no bar can extend below zero (asserted
   in the figure code). The 25 folds are five repeats of five grouped folds and are not independent; the caption says so.
2. Table 9 gives, per feature, the interval and the number of folds in which it is among the three most important features. The text interprets only what the folds support: temperature first in all 25 folds for zT; the
   leading feature of S and sigma in the top three in 24 of 25 folds; temperature in the top five in all 25 folds for sigma, kappa and zT. The intervals of the features ranked second to fifth overlap in every model
   (asserted in the value hook, which stops the build if that stops being true), so their order is not interpreted; for kappa the feature that leads on average is in the top three in only 14 folds.
3. X-axis labels state the units: S in uV/K, sigma and kappa in log10, zT unitless.

## 2026-10-06: NA2 stacking is nested; the mean of three is reported as the leak-free comparison; the random forest ties XGBoost on S

1. The random forest for S is in (`results/na2_rf_S`, code 7529e7c, 25 of 25 folds). Paired against XGBoost on the same folds (cluster bootstrap, `results/na2_comparison/20261006T132724`) its difference for S
   is +0.0003 with an interval that spans zero, so XGBoost is highest on three of the four targets (sigma, kappa, zT), not on every target. The text and the hook (`thesis_values.na2_na1_values`, which stops the build
   if the prose stops being true) say so; the earlier "highest on every target" is withdrawn.
2. Stacking design, stated before any nested result exists: a non-negative ridge meta-learner (alpha 1, intercept) on the out-of-fold base predictions of XGBoost (frozen Paper A set), LightGBM (frozen tuned set) and the
   fixed forest, where those base predictions are made by INNER 3-fold chemistry-cluster CV inside each outer training fold, never in-fold; asserted in the code (`assert_nested` in `scripts/kaggle/na2_stacking_nested.py`).
   The stack's outer prediction applies the weights to the committed outer-fold predictions. Reported: stack against the best single model per target (the best by mean R2 on the same folds, which favours the single
   model), against XGBoost, and against the unweighted mean of the three, each paired with a cluster-bootstrap interval; a null result is reported as one.
3. The earlier local `na2_stacking.py` (cross-fitted over the outer folds; the base predictions of the meta-training folds come from models that saw the held-out fold) is not that design. Its run
   (`results/na2_stacking_crossfit_interim`) is kept with a README and is not quoted in the paper. It did not give a null: the stack and the plain mean were both 0.002 to 0.005 above XGBoost.
4. The unweighted mean of the three base models involves no fitted weights, so it needs no nesting; it is computed in `scripts/model_comparison.py` from the same predictions and is reported now (Table 7): above XGBoost on
   all four targets, with intervals that exclude zero. The nested stack can only be judged as an addition to that.
5. The nested stack costs about 76 CPU hours on Kaggle (inner forest fits dominate); the cells are in `scripts/kaggle/KAGGLE_CELLS.md` section 3c, pinned to code commit 13a0efd582fe9abae95a00d29c677b70eb8ff103
   (the single-commit rule 7529e7c is broken for these sessions only, because the two stacking scripts did not exist at that commit; the base-model code is unchanged between the two).

## 2026-10-07: NA13 before the Kaggle run: leakage check, convergence fix with pre-registered acceptance criteria, reporting

Status of this entry: written 2026-10-07, committed 2026-10-08. Seen before writing it: two timing folds (kappa repeat 0 fold 0, S repeat 0 fold 0, `na13_feature_selection.py` at 13a0efd, `max_iter` 2000), which printed 13 and 15 `ConvergenceWarning` lines, 39 and 25 selected features, fold R2 0.770 and 0.755, 251 and 253 survivors of the Pearson filter; no chosen alpha, no selected-feature list and no comparison of settings had been looked at. Seen before the commit (after writing it): the single calibration fold (kappa repeat 0 fold 0 at 2000, 20,000 and 200,000 iterations, first amendment) and the logs of the diagnostic runs that were stopped (kappa folds 1 and 2 and S folds 0 and 1: warning counts at 2000 iterations, no warning at 20,000 iterations in the folds seen, chosen-alpha positions 18 and 19). Criterion A was tightened, not loosened, after that.

1. **Leakage check, from the code** (`select()` and `main()`). The outer training rows `tr` are the only rows any selection step receives: `select(X[tr], y[tr], groups[tr], ...)`. Inside it: the Pearson filter computes its standardisation and correlations on `Xtr`; `StandardScaler().fit_transform` is fitted on the outer training rows; LassoCV gets `cv=GroupKFold(3).split(Xs, groups=groups_tr)`, so its inner folds are grouped by chemistry cluster and lie inside the outer training rows; the mutual-information ranking uses a 20,000-row subsample drawn from the outer training rows (`rng.choice(len(Xtr), ...)`); the XGBoost refit uses `X[tr][:, sel]`; the outer test rows enter only `model.predict(X[te][:, sel])` and the R2. One weaker point, not a leak to the outer test fold: the scaler is fitted once on the whole outer training set before the inner CV, so each inner validation fold's rows contributed to the mean and variance used to scale it. No outer-test row enters any fit; the inner CV is mildly optimistic; the outer score is affected only indirectly, through the selected features. The Pearson filter is also fitted on all outer training rows, not per inner fold, which is the same level of looseness.
2. **Convergence: pre-registered acceptance criteria.** Diagnostic: repeat 0, all five folds, all four targets (20 outer folds), the same folds as the rung, the thesis's settings otherwise unchanged (Pearson 0.95, 20 alphas, `eps` 1e-3, `tol` 1e-4, 3 inner folds), `max_iter` ladder 2000 (as run), 20,000, 200,000; `tol` is not changed. Per fold and setting the diagnostic records: chosen alpha, the number of `ConvergenceWarning`s, the Lasso non-zero count, the final selected set (top k by mutual information) and the fold R2 of the frozen XGBoost on it; and, by an independent dual-gap check (`sklearn.linear_model.lasso_path` per inner fold on the same alpha grid, converged when the dual gap is at most `tol` times the centred training sum of squares of y, which is sklearn's own criterion), which alphas of the grid did not converge, as grid positions relative to the chosen alpha (the grid is descending, a smaller alpha means weaker regularisation).
   - **A (adoption).** Adopt the smallest `max_iter` on the ladder at which no diagnosed fold prints a `ConvergenceWarning` and the dual-gap check finds no non-converged fit at any alpha, and the final refit converges (second amendment, item 3). If no setting passes, the largest setting tried is adopted (not 2000) and its remaining non-convergence is reported as a limitation; `tol` is not loosened to make the warnings go away. "Tried" includes 200,000 iterations, which is run on the folds where 20,000 still warns (first amendment). If only 200,000 passes, it is adopted for all 100 units. Its cost, estimated from the calibration fold only: 2,865 s for kappa fold 0 on the maintainer's 8-core PC with 4 threads at 200,000 iterations (`results/na13_convergence/20261007T114756`), a time that also contains the diagnostic's own repeat of the inner paths, so the script alone costs between about half of it and all of it; scaled by the training rows (S 1.53, sigma 1.51, kappa 1.00, zT 1.07 times kappa's 96,888; mean 1.28) and by Kaggle being 1.26 to 1.64 times slower than that PC in the other measurements of this work, this is about 64 to 167 Kaggle CPU hours for the 100 folds (per target roughly 12 to 33 h for kappa and 19 to 50 h for S), far more than one session. The run would then be split into per-target sessions run in parallel, each continued across sessions with `--restore-from` until its 25 units are done (the script has no `--repeats` option, so a target is not split further without changing it). These figures are an estimate from one fold with linear scaling in rows, not a measurement of the full run.
   - **B-D (what is said about the effect, not adoption).** B: the chosen alpha of the adopted setting is within one grid step (log10(1/0.001)/19 = 0.158 decades) of the 2000-iteration one in at least 90% of the folds that are counted. Folds in which the chosen alpha is the grid minimum (position 19) under both settings are counted and reported separately and are not counted for B: an identical position at the edge of the grid says only that the inner-CV curve was still falling at the edge under both, not that the choice is stable, and B is not cited as stability evidence for them. C: the Jaccard similarity of the final selected sets is at least 0.80 in the median and at least 0.60 in every fold, and the Lasso non-zero count changes by at most 10% in the median. D: the fold R2 differs by at most 0.005 in every diagnosed fold and the mean absolute difference is at most 0.002 (the across-repeat SD of the committed chemistry-cluster rung is 0.0020 to 0.0050 over the four targets: `per_repeat_r2_std` in `reports/regen_snapfix/20260917T150000/ladder_metrics.json`). If B to D all hold, the statement is "the two timing folds' warnings did not matter and the converged fit selects essentially the same features"; if B has no counted folds (every fold is excluded at the grid minimum under both settings), the stability statement rests on C and D alone and says so; if any fails, the old setting's selections are described as dependent on the iteration limit and only the converged setting is used. These thresholds are own design, not tuned; the observed values are reported whatever they are.
   - **E (position, reported without a threshold).** Where the non-converged alphas sit relative to the chosen one under 2000 iterations (all smaller than the chosen alpha, or some at or above it).
   - **F (collinearity, reported without a threshold).** The pairs of the 397 columns with |r| above 0.9999 (exact duplicates), whether MAGPIE `avg_dev` and CBFV `dev` of the same elemental property are among them and which column the Pearson filter drops; the largest |r| among the survivors; the condition number of the standardised survivor matrix; the number of survivors per fold.
3. **Reporting (item 5).** Per outer fold the units record `n_after_pearson`, `n_after_lasso` (the data-driven size, before the thesis's k is imposed), `n_selected`, the chosen alpha and the number of convergence warnings. The results report, per target, the mean, SD, minimum and maximum of each count over the 25 folds, and the selection frequency of every feature (fraction of the 25 folds, full list, not the top 30), plus the features selected in every fold. Because k is imposed (`n_selected = min(k, n_after_lasso)`), `n_selected` is constant unless the Lasso keeps fewer; the data-driven count to set against the thesis's 25 / 44 / 39 / 32 is `n_after_lasso`, and the paper says so instead of comparing the imposed k with itself.
4. **Smoke step.** Section 3d of `KAGGLE_CELLS.md` runs a Linux smoke of the changed script on the new commit before the real run (as section 3c does), not relying on S0 at 7529e7c, which tested the old script. The code commit that contains the changed script is pinned in the cells after it exists; the cells are a later documentation-only commit.

### Amendment, 2026-10-07 (after one calibration fold, before the 20-fold diagnostic): the dual-gap check's scale was wrong in my implementation, not in the criterion

A calibration run of the diagnostic on one fold only (kappa, repeat 0, fold 0, ladder 2000 / 20,000 / 200,000; folder `results/na13_convergence/20261007T114756`) showed 13 `ConvergenceWarning`s at `max_iter` 2000 and none at 20,000 and 200,000, with the same chosen alpha (0.000163, the LAST, smallest position of the 20-alpha grid) and 188 non-zero Lasso coefficients, and 37 of the 39 final features shared (Jaccard 0.90) between 2000 and 20,000 iterations, fold R2 0.7701 against 0.7743. The same run's independent dual-gap check reported 0 non-converged fits at 2000 iterations although each inner path raised 4 warnings: `sklearn.linear_model.lasso_path` returns the dual gap divided by the number of samples, so the comparison with tol times the centred sum of squares of y (sklearn's criterion, which I had specified correctly in item 2) was made on the wrong scale. Checked on one inner fold afterwards: gap x n_samples reproduces the four warnings (gaps 2.05, 4.83, 30.4, 25.9 against a tolerance of 0.652, at the last four alpha positions 16 to 19). The check now multiplies by n_samples. The criteria A to F are unchanged. The calibration folder's `n_nonconverged_by_gap` values are void (its warnings, alphas, selected sets, R2 and timings are valid), and the folder is kept as the record of this correction. Because 200,000 iterations took 2,865 s on that fold against 508 s for 20,000 with an identical outcome, the 20-fold diagnostic runs the ladder 2000 / 20,000 only; 200,000 is run on the folds where 20,000 still warns, if there are any (the adoption rule "smallest max_iter with no warning in all 20 folds" does not need it otherwise).

### Second amendment, 2026-10-08 (before the 20-fold diagnostic; written in the same uncommitted change): the warning count is split into the inner paths and the final refit, and one more report-only quantity is added

1. **Why the dual-gap check was one short of the warnings** (kappa repeat 0 fold 2: 13 warnings, 12 non-converged inner-path fits; S fold 1: 19 and 18). Checked directly on kappa fold 2 at 2000 iterations: LassoCV raised 13 warnings in all; the independent inner-path check finds 12 (four at each of the three inner folds, the alpha positions 16 to 19 of the 20-alpha grid, the chosen alpha being position 19); a separate `Lasso` refit at the chosen alpha on all the outer training rows (precompute off, as LassoCV refits when its own setting is 'auto') raises 1 warning with duality gap 125.3 against a tolerance of 1.005. 12 + 1 = 13: the missing warning is LassoCV's final refit at the chosen alpha on the full outer training set, which an inner-path check does not cover. (The S fold 1 difference of one, 19 warnings against 18 by the dual-gap check, is inferred from the log counts of the stopped run, not re-run.)
2. **Implementation, both scripts.** `na13_feature_selection.py` now records per fold `n_convergence_warnings` (all), `n_convergence_warnings_inner_paths` (all minus the refit's), and the refit's `final_refit_converged`, `final_refit_dual_gap`, `final_refit_tolerance` and `final_refit_n_iter`, read from the fitted LassoCV (`dual_gap_`, `n_iter_`; `dual_gap_` is stored divided by the number of rows, so it is multiplied back before it is compared with tol times the centred sum of squares of y). `na13_convergence_check.py` repeats the refit independently (`final_refit_check`: a plain `Lasso` at the chosen alpha) and records whether it agrees with the value read from LassoCV, and whether the warning total equals the inner-path non-converged fits plus the refit's. The first version of the LassoCV-side record forgot the factor of the number of rows and the independent check disagreed with it on kappa fold 2; that disagreement showed the error in a test run whose output was deleted, so the missing factor was caught before any run whose output was kept, and it is fixed (after the fix both give 125.29 against 1.0046, not converged, and 13 = 12 + 1).
3. **Criterion A is unchanged in intent, now stated for the refit too**: a setting is adopted only if there is no ConvergenceWarning, no non-converged inner-path fit by the dual-gap check, AND the independent check finds the final refit converged, in all diagnosed folds; the summary also reports whether the accounting 13 = 12 + 1 holds in every fold and setting.
4. **G (report-only, like E and F): the position of the chosen alpha in the grid.** Per fold: the position of the chosen alpha in the descending 20-alpha grid (0 is the largest alpha, 19 the smallest, the weakest regularisation), and the share of folds in which it is the smallest alpha of the grid. The diagnostic reports it for its 20 folds at each max_iter (`E2_alpha_position` in `summary.json`); the Kaggle run reports it for its 100 units, per target and overall (`lasso_alpha_grid_position`, `share_folds_alpha_at_grid_minimum` per target and `_overall` in `results.json`). Seen before this item was written: the chosen alpha was position 19 in kappa folds 0 to 2 and position 18 or 19 in S folds 0 and 1, i.e. at or next to the end of the grid; no threshold is attached, and the paper reports the Lasso step's own `n_after_lasso` from the Kaggle run instead of describing the selection as a reduction to 25 to 44 features.
5. The 20-fold diagnostic (2000 and 20,000 iterations) is rerun from the clean commit that contains this change; the earlier partial runs (two processes stopped when the session closed) wrote nothing and were removed, and the calibration folder `20261007T114756` remains as described in the first amendment.

### Result, 2026-10-08: the 20-fold diagnostic (2000 and 20,000 iterations), run from the clean commit 8ac332f

Runs: kappa `results/na13_convergence/20261008T093143`, zT `...093146`, S `...093150`, sigma `...093154`; each `run_config.json` records commit `8ac332f`, `tree_clean` true, the ladder 2000 / 20,000 and the snapfix CSV `d9fc1e5d...`. The evaluation of the criteria above is `results/na13_convergence/20261008T171143/` (`summary.json` is `na13_convergence_check.summarize` on the four folders; `per_fold.csv`, `report.json` and `README.md` are written by `scripts/na13_convergence_report.py`).

- **A passes at 20,000, which is adopted.** At 2000 iterations all 20 folds warn (258 warnings = 239 non-converged inner-path fits + 19 non-converged final refits; the accounting holds in every fold and setting, and the independent refit check agrees with the value read from LassoCV everywhere); at 20,000 no fold has a warning, a non-converged inner-path fit or a non-converged final refit. 200,000 iterations were therefore not needed.
- **B holds on the counted folds.** 6 folds are excluded because the chosen alpha is the grid minimum (position 19) under both settings (S fold 1; kappa folds 0, 1, 2, 4; zT fold 2) and no stability claim is made for them. Of the 14 counted folds, 13 are within one grid step (threshold 13); S fold 2 moved from position 16 to 19 (0.474 decades). The chosen alpha position is identical in 19 of the 20 folds; only S fold 2 changes.
- **C fails.** The Jaccard similarity of the selected sets has median 0.931 but minimum 0.5625, in S fold 2 (threshold 0.60); the relative change of the Lasso non-zero count has median 1.2%.
- **D fails.** Six folds differ by more than 0.005 in fold R2 (S fold 0 -0.0085, S fold 3 +0.0062, kappa fold 2 +0.0063, zT fold 2 -0.0057, zT fold 3 +0.0068, zT fold 4 +0.0060); the mean absolute difference is 0.0034 (threshold 0.002) and the mean signed difference (20,000 minus 2000) is +0.0007.
- **Consequence, as pre-registered:** the 2000-iteration selections are described as iteration-dependent, and only the 20,000-iteration setting is used (`LASSO_MAX_ITER` = 20000, in a separate code commit).
- **Caveat on D, stated without changing the verdict.** D's thresholds (0.005 per fold, 0.002 for the mean) were set by reference to the across-repeat SD of the rung's pooled per-repeat R2 (0.0020 to 0.0050), but the dR2 compared with them is a per-fold difference on the same test rows under two settings. These are different quantities, so the thresholds are not calibrated to dR2; the verdict stands as pre-registered.
- **E, G, F (report-only).** E: of the 239 non-converged inner-path fits at 2000 iterations, 157 lie at alphas larger than or equal to the chosen alpha, and 82 at smaller alphas; in 19 of 20 folds the chosen alpha lies inside the non-converged range (the exception is S fold 3). G: the chosen alpha is the grid minimum in 6 of 20 folds at 2000 iterations and 7 of 20 at 20,000, and its position is never below 14 of 19 at either setting. F: the exact-duplicate column pairs (|r| > 0.9999, over all rows) number 59 for S, 105 for sigma, 96 for kappa and 102 for zT; the standardised Pearson survivors (241 to 255 per fold) are exactly singular in every fold (condition number about 7e150 to 8e150).
