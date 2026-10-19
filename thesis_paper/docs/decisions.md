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

### Pre-registration, 2026-10-09 (written before any code change): the NA13 claim is a paired comparison on one machine and one device

What was seen before writing it: the Linux timing runs on spcai3 (commit `47339cf`, final XGBoost on the CPU, outer fold 0 of repeat 0, 20,000 iterations). For kappa the selection is identical to the Windows diagnostic's (`20261008T093143`, setting 20,000): the same set in the same order, chosen alpha position 19, `n_after_lasso` 188; the fold R2 is nevertheless 0.7681 on spcai3 against 0.7743 on the Windows PC. On the Windows PC the same CPU fit gave the same R2 at every thread count from 1 to 8 and on repeated runs, and the selection was the same with 2 and with 4 threads. The fold-0 R2 differences between the two machines (spcai3 minus Windows) are -0.0062 (kappa), +0.0032 (S), +0.0019 (zT) and +0.0002 (sigma). So the difference lies in the XGBoost fit across platforms, not in the selection (one fold; the other targets' selected sets were not compared).

1. **Paired design.** In every unit (target, repeat, fold) the target's frozen XGBoost is fitted twice on the same outer training rows, once on all 397 features and once on the selected features, and both are scored on the same outer test rows, on the same machine and the same device. The quantities are dR2 = R2(selected) - R2(all 397) per fold, and the same difference of the pooled per-repeat R2 (five values per target), with mean and SD over the repeats. The NA13 claim (the selected features against all features, the thesis's Table 3 question) rests only on this paired difference. The committed 397-feature rung values (cuda, Kaggle) are reported alongside in the results and are not used for the claim.
2. **Device.** Both XGBoost fits run on `cuda` (the V100S of spcai3); the Pearson filter, LassoCV and the MI ranking stay on the CPU. `na13_feature_selection.py` gets its `--device` option passed to both fits and records it in the run parameters, `run_config.json` and every unit; each unit records `r2_all397` and `r2_selected` (and their difference). The V100S is not the Kaggle T4s the rung ran on, and a GPU fit may not be bit-reproducible (stated from the general behaviour of GPU histogram building; NOT verified here, and the first cell of section 3d tests it: one fold is run twice on `cuda` and both R2 are printed), which is why the all-397 refit on the same device, not the committed values, is the reference. If the two runs agree, nothing follows beyond that statement; if they differ, the difference is reported next to the interval of item 5 as the run-to-run noise of a `cuda` fit, no per-fold dR2 smaller than it is interpreted on its own, and the claim and its interval are unchanged.
3. **For the caveat on D (2026-10-08 Result, above), added as a fact and without changing any verdict:** platform alone moved the kappa fold-0 R2 by 0.0062 (0.7743 on Windows against 0.7681 on Linux) with an identical selection, one fold, which is the same size as D's per-fold threshold of 0.005. D's verdict is unchanged: D was evaluated on one machine, comparing 2000 with 20,000 iterations.
4. **Cost.** The timings of the Linux runs above are for the selected-feature fit only. The new design adds one 397-feature fit per unit (a few tens of seconds on the GPU for the S model, from the NA6 timings of this work), so the wall-time estimate of section 3d grows a little; the extra is stated there and is re-timed at the new pin before the run.
5. **How the claim is judged.** The NA13 claim is judged by a paired cluster-bootstrap interval on the pooled dR2 (selected minus all 397), resampled over chemistry clusters: the same procedure as Table 7 (`scripts/model_comparison.py`; decisions 2026-10-06 item 2): the pooled out-of-fold R2 of each repeat is computed from per-cluster sums, the clusters are resampled with replacement and every row of a drawn cluster is carried along in every repeat, 2000 draws, seed 0, the statistic is the mean over the five repeats of the pooled R2 difference, the interval is the 2.5th to the 97.5th percentile. The interval is reported whatever it is, and an interval that contains zero is reported as a null result, not as a trend. The script that does this for NA13 does not exist yet (`model_comparison.py` reads the NA1 and NA2 bundle layouts, not the NA13 units); it will be written, and committed, before the results of the run are read. The folds are rebuilt from the snapfix CSV with the same seed and fold code, as `model_comparison.py` does, because the units hold the predictions but not the clusters.

**Fact, 2026-10-09 (before any NA13 result):** section 3d, Cell 1, on spcai3 at commit `aad78f1` ran kappa repeat 0 fold 0 twice on `cuda`. `r2_selected` 0.7682163607797026 and `r2_all397` 0.8120420292255622 were IDENTICAL in both runs, with the same selected set in the same order. So, per item 2 above, no run-to-run noise term is reported. The two unit files (`~/runs/na13_timing2_kappa` and `~/runs/na13_det_kappa` on spcai3) will be committed with the results as a record, not as a result.

## 2026-10-09, after the NA13 and nested-stacking results were seen; input handling and provenance only: two analysis scripts, and one convergence fact

Made after the results of `na13_analysis.py` (run on the four NA13 tarballs) and of `na2_stacking_analysis.py` / `model_comparison.py na2` had been seen. None of these changes touches an estimator, a threshold, an interval, the resampling or a number: they concern which input files a script accepts, when it records the git state, and what it reports about warnings. The reruns from clean trees are compared number by number with the first runs; any difference stops the process and is reported before anything is committed.

1. **`na13_analysis.py`: one file outside the manifest is allowed, `README.md`, and nothing else.** The first run refused the committed bundle folders (`results/na13_<T>/<stamp>/`) because each carries a `README.md` outside its manifest (written when the bundle was committed; the bundle's manifest covers only what the run wrote). The analysis was therefore run on the original `.tar.gz` files, which are the same bytes (their SHA256 is the recorded bundle identity). The script now accepts exactly the top-level `README.md` as an extra file; any other extra file, or a `README.md` in a subfolder, is still refused (two new tests in `tests/na13_analysis_selftest.py`). The analysis is then rerun on the committed folders and must reproduce the tarball run.
2. **`na2_stacking_analysis.py`: the git state is recorded at the start, before anything is written.** It took the state after writing its own output folder, so its `run_config.json` said `tree_clean` false because of its own output. It is rerun from a clean tree and must reproduce the first output.
3. **Fact about the NA13 bundles.** Of the 100 units exactly one records a ConvergenceWarning: S repeat 2 fold 0, in an inner path (1 warning); its final refit converged. The spcai3 logs showed none because `select()` captures warnings (`catch_warnings(record=True)`) instead of printing them, so a log cannot show a warning at all; the units' `n_convergence_warnings` are the record (section 3d's hand-back check is changed accordingly). `na13_analysis.py` now names the units with a warning in `analysis.json` and in its README.
4. **Temperature in the selected features (report only, after the results were seen).** `temperature_bin` is one of the 397 features and is never selected, in any of the 100 folds of any of the four targets. It passes the Pearson filter in all 100 outer training sets. Recomputed with the pipeline's own functions for zT repeat 0 folds 0 and 1 only (the other targets and folds were not recomputed): the Lasso keeps it (standardised coefficients +0.168 and +0.161, non-zero), but the mutual-information ranking places it 98th of 147 and 110th of 170 non-zero features (MI 0.087 and 0.086 against 0.194 and 0.193 at rank k = 32), so for those two folds it is dropped at the third step, the MI ranking. This is a fact about the result; whether it explains the size of the zT difference is not tested here and is not part of the pre-registered claim.

### Exploratory follow-up to NA13, 2026-10-10 (post hoc, written before any code of it): the selected features plus `temperature_bin`

Status: this is added AFTER the NA13 results were seen and is exploratory. It does not change the verdict of item 5 of the 2026-10-09 pre-registration, which was the paired cluster-bootstrap interval of the selected features against all 397 features and stands as computed (`results/na13_analysis/20261009T190036`).

What is already known (2026-10-09 note, item 4): the NA13 selection never kept `temperature_bin`, in any of the 100 folds of any target (0 of 100); it passes the Pearson filter in all 100 outer training sets, and in the two folds recomputed (zT repeat 0, folds 0 and 1) the Lasso keeps it and the mutual-information ranking drops it.

The check: for every one of the 100 committed NA13 units, the target's frozen XGBoost (the same frozen hyperparameters as everywhere in NA13) is refitted on that unit's selected features plus `temperature_bin`, on the same outer training rows, scored on the same outer test rows, on `cuda` on spcai3 (the machine and device of the committed `r2_selected` and `r2_all397`). It is reported next to the committed `r2_selected` and `r2_all397` of the same unit.
1. **Reported as exploratory and descriptive.** Per unit and per target: the new R2, its difference from `r2_selected` and from `r2_all397`, per fold and pooled per repeat, with mean, SD, minimum and maximum. No threshold, no hypothesis test and no interval is attached or claimed; nothing here is a result of the pre-registered design.
2. **Inputs.** The selected feature lists come from the committed NA13 bundle folders (`results/na13_<T>/<stamp>/units`, manifests verified, pin `aad78f1`); the folds are rebuilt from the snapfix CSV with the same seed and fold code and checked against each unit's saved `y_true`. A unit whose selected list already contains `temperature_bin` stops the script (none should: 0 of 100).
3. **Control.** For the first unit of each target the selected-only fit is repeated and its R2 compared with the committed `r2_selected`, to show that the machine and device reproduce the committed fits; the result is recorded, not used to drop anything.
4. **Provenance.** The script refuses to run unless the clone is at a clean commit and the device is `cuda`; its outputs are written outside the clone; `run_config.json` records the CSV and bundle identities, the commit, the tree state and the script SHA256. A CPU smoke on one unit per target is for the plumbing only and is never a result.

**Result, 2026-10-10 (exploratory, descriptive only; the item-5 verdict is unchanged).** Run on spcai3, cuda, at code `39e5b26` (clean tree), 100 units; results in `results/na13_temperature/20261009T194925Z/` (bundle SHA256 `160bc86ab2dd1ef612ce8c64928392609bcd560d8aa4ac8f36edc885a17a5202`; every control refit equals the committed `r2_selected` exactly). Adding `temperature_bin` to the selected features recovers, in pooled per-repeat means, this share of the gap between the selected-feature fit and the all-397 fit: S 34%, sigma 61%, kappa 70%, zT 97%. Selected + `temperature_bin` stays below the all-397 fit in 25, 24, 24 and 20 of the 25 folds (S, sigma, kappa, zT). No threshold, test or interval is attached to these figures, and they do not change the paired result of the pre-registration (`results/na13_analysis/20261009T190036`).

## 2026-10-10: device provenance audit (a fact, written after the results were seen): what each script records about the device, and what shows that the committed cuda results ran on a GPU

Scope: reading the scripts, the committed run configs and logs; one local test on the GPU-less Windows PC (xgboost 2.0.3). No script that produced a result was changed, and no number or verdict changes. The audit was prompted by the 2026-10-10 finding that `na13_temperature_check.py` (first version `ed78db7`) started a `--device cuda` run on a machine with no GPU.

**Library behaviour (tested here, xgboost 2.0.3):** `device="cuda"` with no visible GPU does not raise. XGBoost warns "No visible GPU is found, setting device to CPU" and fits on the CPU; the booster's `save_config()["learner"]["generic_param"]["device"]` then reads `cpu`. A booster that is on cuda and predicts on host arrays emits, once, "Falling back to prediction using DMatrix due to mismatched devices"; a fallback (CPU) booster predicting on host arrays does not (tested: no warning). `ncv._to_device` returns the numpy array unchanged when cupy cannot be imported (it prints one warning), and the model keeps `device="cuda"`. A saved model JSON (`models/<t>.json`) carries no device field.

**1. What each script records.** Answer: none records the device the booster actually used. Every one records the REQUESTED device (or, for Paper A, the requested device after `_resolve_device_for_model`, which only downgrades non-XGBoost models to cpu).
- The harness (`kaggle/harness.py`, used by `na13_feature_selection.py`, `na7_random_dvd.py`, `na6_classifier.py`, `na_final_models.py`, `na3_shap.py`, `na3_rows.py`, `na1_nested_cv.py`) writes `device` = `args.device` (the request; forced to cpu by `--smoke`) and `gpu` = the output of `nvidia-smi --query-gpu=name,driver_version,memory.total` (the GPUs present on the machine, null if the command fails) into each session's `run_configs/session_*.json`. It does not refuse `cuda` when `gpu` is null. The units of na3, na3_rows, na6, na7, na1 and the final models carry no device field; the units of `na13_feature_selection.py` carry `"device": args.device` and the log prints it as `[cuda]` (the request again).
- `src/nested_cv.py` (Paper A: ladder, ablation, SHAP, ungrouped, direct-vs-derived): `device` in `run_config.json`, in the checkpoint records and in the first `progress.log` line (`device=cuda`) is the resolved request; `environment.txt` carries an `nvidia-smi` name and driver listing and the installed `cupy-cuda12x==14.0.1` and `xgboost==2.0.3`.
- `na2_trees.py` and `na2_stacking_nested.py` hard-code or default to the CPU (`device: cpu`, `gpu` null in their session configs); their record is true by construction.
- `na13_temperature_check.py` at `39e5b26`: the only script that reads the booster's own device (a tiny probe fit, `generic_param.device`, refuses unless it starts with `cuda`). The value read is not written to any output; `run_config.json` and the units record the requested `cuda`, and the per-unit `[cuda]` in the log is that string. The booster's device does appear once in the run's log, in xgboost's own warning (see the table row for this run and correction (1) below).
- `na7`: `session_01.json` and `session_01_restored1.json` record `cuda` (shard 0 on GPU 0, shard 1 on GPU 1, run concurrently); `session_03.json` records `cpu` because it is the merge (`--restore-from` both shards, `--max-units 0`) and computed no fit; `gpu` lists both T4s in all three.

**2. Evidence per committed result that claims cuda.** "Requested" below means the config field above; it is not evidence by itself.

| Result (folder) | Recorded hardware | Evidence beyond the request |
|---|---|---|
| Paper A: `descriptor_ablation/20260912T184318`, `shap_attribution/20260913T082733` | `environment.txt`: Tesla P100-PCIE-16GB, driver 580.159.04 | none independent of the request |
| Paper A: `regen_postfix/20260916T161906`, `ladder_regen_snapfix/20260917T150000`, `descriptor_ablation_snapfix/20260917T184650`, `shap_attribution/20260917T134930`, `ungrouped_snapfix/20260922T093243` | `environment.txt`: 2 x Tesla T4, driver 580.159.04; `progress.log` first line `device=cuda` (the nested_cv runs; shap_attribution has `device: cuda` in `summary.json`) | the chemistry rung of `ladder_regen_snapfix` is reproduced by NA13 and NA3 (below) |
| Paper A: `direct_vs_derived_snapfix/20260924T122148_zt_shared_cuda`, `.../20260924T124138_per_target_cuda` | no GPU name in either folder (the P100 named in CLAUDE.md is not recorded there) | both `dvd_*.log` contain the same warning, "XGBoost is running on: cuda:0, while the input data is on: cpu.", which a CPU booster does not emit |
| `final_a`, `final_b`, `na1_a`, `na1_b`, `na3_a`, `na3_b`, `na3_rows_a`, `na3_rows_b`, `na6`, `na7` (shards) | `gpu`: 2 x Tesla T4, driver 580.178.04, 15360 MiB | na3 and na3_rows: every unit's `fold_r2` equals its `committed_fold_r2` (the rung above), 120 units, maximum absolute difference 0 |
| `na13_S`, `na13_sigma`, `na13_kappa`, `na13_zT` (`aad78f1`) | `gpu`: Tesla V100S-PCIE-32GB, 580.178.04, 32768 MiB | `r2_all397` against the committed T4 rung, per fold: maximum absolute difference 1.1e-16, exactly equal in 91 of 100 units (S 25, sigma 17, kappa 24, zT 25) |
| `na13_determinism_record` (`aad78f1`) | `gpu`: Tesla V100S-PCIE-32GB | kappa repeat 0 fold 0 run twice: `r2_selected` and `r2_all397` identical |
| `na13_temperature/20261009T194925Z` (`39e5b26`) | none recorded in the folder (the run config has no `gpu` field); the spcai3 V100S per section 3d | the `39e5b26` gate (it could not start with a non-cuda probe booster; verified here to refuse on a fallback); line 2 of `spcai3_run.log`, xgboost's own warning at the first prediction: "Falling back to prediction using DMatrix due to mismatched devices. ... XGBoost is running on: cuda:0, while the input data is on: cpu." (a CPU-fallback booster never emits it, tested above); the 4 control refits equal the committed `r2_selected` exactly |

**3. What this evidence shows and does not.**
- A fit that reproduces the committed rung to 1e-16 is the same computation as the rung's. By itself that does not distinguish cuda from CPU, because the rung's own device rests on the table's first rows (a request and an `nvidia-smi` listing of installed GPUs). The independent point is that a CPU fit of the same fold is measurably different: kappa repeat 0 fold 0 with the selected set the cuda unit used (identical set and order to the Windows diagnostic's at 20,000 iterations, checked here) gives R2 0.7682163607797026 on cuda (`na13_kappa`), 0.774260 on the Windows CPU (`na13_convergence/20261008T093143`) and 0.7681 on the spcai3 CPU (recorded in the 2026-10-09 entry; that run's artifact is not committed, and its value is at four decimals). So for that fold the cuda value differs from both CPU values by at least 0.00006. One fold, one target, one feature set.
- Therefore: NA13 (V100S) and NA3 and NA3_rows (T4s) are supported by an outcome that matches the rung exactly, whereas the CPU fits did not reproduce the cuda value in the one fold where a CPU fit was compared (a selected-feature fit; no CPU all-397 fit is recorded, so the contrast is indirect). The mismatched-devices warning (direct-vs-derived logs and `na13_temperature`, naming `cuda:0`) shows the booster was on a cuda device in those runs. For the Paper A results of 2026-09-12 to 2026-09-22 and for `final`, `na1`, `na6` and `na7` the evidence is the recorded request plus the `nvidia-smi` listing of the installed GPUs plus, for the first group, the reproduction by later cuda runs; none of it records the booster's device.
- Two corrections to examples quoted when this audit was requested: (1) the 2026-10-10 log's `[cuda]` per-unit marker is the requested string, but the same log does print `cuda:0` in xgboost's warning (line 2, quoted in the table); that warning is the booster's own device, not the request. (An earlier draft of this entry said the log did not print `cuda:0`; that was wrong, corrected before commit.) The evidence for that run is the gate, the warning and the exact control refits listed above. (2) "Kaggle sessions printed nvidia-smi with T4s" is true for the listing (committed in `gpu` and `environment.txt`) and shows the GPUs were present, not which device the booster used.
- The "tiny `device="cuda"` fit ... cuda fit ok" line in every Kaggle cell of `KAGGLE_CELLS.md` (sections 3a to 3d, and Cell 1 of 3d on spcai3) has the same hole as the `ed78db7` check: it succeeds on the CPU fallback and prints "ok". Only the `nvidia-smi` line beside it, which lists installed GPUs, supports the hardware statement; the fit line does not.

**4. What would close the gap (not done; no code of a script that produced results is changed here).** Record `booster.save_config()["learner"]["generic_param"]["device"]` of the fitted model in each unit (or once per session), and have the harness refuse `--device cuda` when `gpu` is null or the probe booster's device does not start with `cuda`. `na13_temperature_check.py` already refuses; it does not record the value.

### Correction, 2026-10-10, to the 2026-10-07 NA13 entry, item 2 (what "the thesis's settings" meant)

Item 2 of the 2026-10-07 entry says the diagnostic used "the thesis's settings otherwise unchanged (Pearson 0.95, 20 alphas, `eps` 1e-3, `tol` 1e-4, 3 inner folds)". That wording is imprecise and is corrected here. The thesis (its section 3.3 and Table 3) names the three steps (Pearson filtering, LassoCV, mutual-information ranking) and the number of features kept per target (25, 44, 39, 32), and states no setting of the steps (its only stated |r| > 0.95 threshold belongs to a different step, the removal of intra-CBFV features). Per `docs/design_constants.csv` D28 to D30, the Pearson threshold 0.95, the 20 alphas, the 3 inner chemistry-cluster folds, `eps` and `tol` (scikit-learn's LassoCV defaults: `select()` passes neither) and the 20,000-row mutual-information subsample are this work's choices, fixed before any result of the comparison was seen and not tuned; only k comes from the thesis (D27, unjustified there). The one change made after the first runs is `max_iter`, 2000 to 20,000 (Result 2026-10-08). The pre-registered criteria, the diagnostic and the paired comparison are unchanged by this correction; the paper (section 4.1.5) attributes the settings accordingly and says that, because the earlier work did not state them, the comparison tests the procedure as instantiated here and other settings could change the numbers.

## 2026-10-10: external validation is rescored with the saved final models; teMatDb scoring is added to the paper; where the smear factors come from

Written after the audit of the external-validation results (the same day) and before the rescoring script was committed or run from a clean tree. What was already known when this was written: the Paper A results of the ESTM and teMatDb scoring on the snapfix CSV (`results/external_snapfix/20260917T160553`), and, from one throw-away test run of the new script on a dirty tree (outside the clone, never a result), that the saved models give external R2 values that differ from the in-process refit by up to about 0.03 for S, sigma, kappa and zT-direct, and by much more for the derived zT (a stated limitation of that quantity, see item 3).

1. **ESTM and teMatDb are rescored with the saved final models** (`results/final_a/20261004T102855`; `models/{S,sigma,kappa,zT}.json`, SHA256 in its README and manifest), not with the in-process refit. Reason: the script that produced `results/external_snapfix/20260917T160553` is not in the repository, and the folder records no commit or tree state, so the folder cannot be reproduced or audited; and its models are a different fit (CPU, in-process) of the same frozen hyperparameters than the models that produce the screening predictions and the JARVIS results (T4). The new script `scripts/external_rescore.py` loads the four boosters, checks them against the committed manifest, and runs the unchanged data path of `src/external_validation.py` (temperature window, canonicalisation, featurisation, the 397-column assertion, the two deduplication passes against the snapfix CSV) and the in-support rule of `scripts/insupport_share.py`. The old folder is kept, not deleted, and is no longer cited.
2. **teMatDb scoring (strata a0, a, b) is added to the paper (Section 4.2).** Decided with its Paper A results already known (snapfix R2 by stratum, `tematdb_D_scoring_snapfix.json`). All three strata are reported, with their sample-level bootstrap intervals, whatever the new values are; nothing is selected by outcome. Stratum b has 27 samples and is described qualitatively (direction and rough size), not as a stable estimate, as in Paper A caveat (a). Strata: a0 = the sample's DOI is in the training data, a = it is not, b = the chemistry cluster (fixed definition) is absent from training; the sample ids of b are those of `tematdb_inventory_snapfix.json` (the cached `_scoring_df.csv` carries the pre-fix flag, 549 rows; the rebuilt set is 412 rows and 27 samples).
3. **Smear factors for the sigma and kappa back-transform (derived zT).** The factors used so far (sigma 1.3817, kappa 1.0715) were computed in the same in-process pass as the refit: one chemistry-cluster 5-fold pass of fresh fits with `default_rng(0)`. Those out-of-fold predictions were not saved and the fold seed is not that of any repeat of the committed rung, so the factors cannot be reproduced from committed files. The committed out-of-fold predictions of the chemistry-cluster rung (`results/ladder_regen_snapfix/20260917T150000/{sigma,kappa}_chemistry_full`, 5 repeats x 5 folds, frozen hyperparameters, training rows only) give per-repeat Duan factors of 1.352 to 1.385 for sigma and 1.066 to 1.072 for kappa; their mean (1.3717 and 1.0683) is used from now on and is computed, with every file's SHA256, by the rescoring script. The old values lie inside those per-repeat ranges. Consequence stated here: the derived-zT columns of `results/final_a` and `final_b` (`predictions_*.csv`) carry the old factors; no table or sentence of the paper quotes them (Table 11 and the JARVIS analysis use the direct zT and S), and the ESTM and teMatDb derived-zT figures are not quoted in the paper either. The derived zT of an external set is ill-conditioned (a 1% change of a smear factor, or a different fit of the same hyperparameters, moves its R2 by tenths), so it is reported in the output files only.
4. **The fit-to-fit difference is stated in the paper.** The old in-process CPU refit and the saved T4 models are two fits of the same frozen hyperparameters on the same rows, and their external R2 values differ by up to about 0.03 (S, sigma, kappa and direct zT, both ESTM passes; largest for kappa in the cluster-disjoint pass). This is one pair of fits: an indication of how much a refit of the same specification can move an external R2, not a variance estimate. Section 4.2 says so in one sentence whose figures are generated from the two committed result files (the old folder stays committed and pinned for that purpose); the derived zT is not offered as an external result.

## 2026-10-10 (later): rounding rule for stated bounds; teMatDb release identity; the approved handling of the unclassified numbers

1. **Rounding rule.** A generated value used in an "at most", "up to", "within" or "no more than" claim is rounded UP, one used in an "at least" claim is rounded DOWN, and an exact value ("the maximum is", an observed range "by X to Y") is rounded to nearest. `thesis_values.py` carries `_up` and `_down`; keys used in both roles have two variants (`a2_m3_dmax` for the observed range, `a2_m3_dmax_ub` for "at most"). `spread_max` states an exact maximum and stays at nearest rounding.
2. **teMatDb data identity (checked 2026-10-10).** The SHA256 of the three data files used, of the same files at repository commit `86a9bf979e35e2b9b0a4927bc50ad031edf84ad7`, and of the same files in the Zenodo archive v1.1.6 (`teMatDb-v1.1.6a.zip`, root `byungkiryu-teMatDb-68b6ec6`, record 10.5281/zenodo.15518036) are equal: `teMatDb_samples.csv` f65d0717864bee5702ad1e7815c658105d0e6e0e8397e35db19ae1d9a5195503 (34,995 bytes), `teMatDb_rawTEPs.csv` ed907cd16ec9c7be1cc36e8e242f152b41485247fd581009db96b9b0b95d690c (537,427), `teMatDb_collocatedTEPs.csv` cc0cc7d363106a3849c266a8230721e2657534c97a56ffbe9c8a3eb1aaaf6fd9 (4,893,361). The repository commit and the Zenodo release are therefore interchangeable for the data scored, and both are cited; the arXiv preprint is marked as a preprint.
3. **Unclassified numbers (`reports/unclassified_numbers_proposal.csv`), implemented as approved.** GEN: the Step-1 bounds `b_sigma_hi` and `b_zT_lo`, the example temperatures (asserted to share one bin of the cleaning code), the MAD consistency constant (computed, 1/Phi^-1(3/4)), the zT cut and the top fraction (read from the literals of `na4_dataset_statistics.py` and `na8_error_metrics.py`, which keep them as literals in the computing statement because no named constant exists; asserted to be unique), the temperature feature count, the selenium fraction of the worked example (with the grouping assertions of the same sentence), the screening grid step, and the repeats and folds of Table 10. DESIGN: the dopant threshold and the snap tolerance of the chemistry cluster and the two ends of the screening grid, with `docs/design_constants.csv` D31 (PARTLY JUSTIFIED by Ma & Poon), D32 (UNJUSTIFIED, own design) and D33 (OWN DESIGN). LAB: exact-pattern exclusion rules in `audit_numbers.py` (the space-group symbol, the resistivity conversion, the "(steps a--b)" headings, this study's own protocol phrases, and PENDING placeholders), each with its own count in the audit summary; the protocol names "random 80/20", "5-fold" and "10-fold" are tied by assertions to the committed run configurations and to `RANDOM_HOLDOUT_TEST_SIZE`. The LaCoO3 "S of 200 to 600 uV/K" (not found in the source that could be read, ledger row L06) was rephrased to "large Seebeck coefficients". The 57 literature numbers (CIT) and the uncited "up to about 0.2" stay pending, and so does the Materials Project count (L16), because that sentence describes the database and not this study's query.

## 2026-10-11: Figures 1 to 3 are to be replaced; the carrier-concentration number of Figure 2

1. **Figures 1 to 3** (`figures/source_media/image1.png` to `image3.png`, from `source/Perovskite_Thermoelectric_Manuscript.docx`) are AI-generated images (maintainer's statement). Nothing in the repository records a source for them: the alt text of the docx repeats the captions and the PNGs carry no metadata. They are not to be submitted. They must be replaced by author-made figures drawn by a committed script (no hand-typed values in the image), and each caption must name its source if it is adapted from a published figure.
2. **Figure 2 prints "optimum ~10^19-10^20 cm^-3" inside the image** (and the axis labels 10^16 to 10^21). The caption no longer states the number: the value was attributed to Snyder & Toberer 2008, whose PDF is not in `source/pdfs/` and whose abstract has no carrier-concentration value (ledger L20, NUMBER REMOVED). Until the PDF is read and a page recorded, the image and the caption disagree; the image is the item to fix (redraw without the label, or restore the number with the page).
3. Related, same date: the year 1821 was removed from Section 2.1 (ledger L23: the supplied "Rowe" PDF is lecture notes by K. Iltanen, 2011); the Step-1 zT bound of 4 is a registered design choice (D34), no longer attributed to Snyder & Toberer.

## 2026-10-13: Snyder & Toberer read; Flahaut and Ohta numbers; NA12 PDF inventory

1. **Snyder & Toberer 2008** (`snyder2008complex.pdf`, Nature Materials 7:105-114) states a zT peak "typically ... between 10^19 and 10^21 carriers per cm^3" (article pages 105 and 107; Figure 1, axis 10^18 to 10^21), not the 10^19 to 10^20 printed in Figure 2. Figure 2 is to be redrawn by the maintainer; its caption states no number.
2. **Flahaut 2006** (IEEE ICT, pp. 103-106) gives zT = 0.2 at 1000 K for Ca0.95Yb0.05MnO3, S of CaMnO3 of about -350 uV/K and a factor of 4 over CaMnO3; it does not give zT < 0.1 for CaMnO3, nor a list Dy, Y, Nd, Yb. Section 2.2 now states only what the source states (ledger L05).
3. **Ohta 2005** (J. Appl. Phys. 97:034106) covers heavily La- or Nb-doped SrTiO3 crystals only: it gives no S, kappa or zT of undoped SrTiO3 and finds the doped crystals' kappa similar to undoped, not reduced. Section 2.2 was rewritten for the doped crystals with Table II values (ledger L07).
4. **NA12 PDFs** are inventoried (structure only, no value read) in `docs/na12_pdf_inventory.md`.

## 2026-10-13: NA12, amendment 1 to pre-registration B (how discrete measurements are read); Sebastia-Luna et al. 2022

Made before any NA12 value is written to `docs/na12_literature_values.csv`. **No out-of-fold prediction for any NA12 compound has been looked at**: the extraction of Sebastia-Luna's Fig. 2a was
validated only against the maxima that the same paper prints in its own Table 2 (eight cells over Figs. 2a and 3d, all inside the printed uncertainty, in a scratch run that wrote nothing to the repository).
The amendment changes only the reading rule below; the compounds, the two comparison levels, the error measure and the Alleno band of pre-registration B are unchanged.

1. **Discrete measurements.** Where a paper reports discrete measurements (markers, no curve), each marker is read at its stated measurement temperature and mapped by the pipeline's floor bins
   (`step3_filter_temperature`: a bin is labelled by its lower edge, [300, 325) is 300; values below 300 K or at or above 800 K are recorded and not compared). No value is interpolated onto the
   100 K grid; that rule stays for continuous curves.
2. **Stated temperature.** The temperature the paper states is used (here the nominal 25, 50, 75 and 100 C of the series, 298.15, 323.15, 348.15 and 373.15 K). Where a marker is drawn
   at another position (the Seebeck series of Fig. 2a sits about 2 C to the right of the other three), the drawn offset is recorded in the notes and not used.
3. **Compounds.** Compositions that are not one of the pre-registered compounds are excluded (Cs2SnI6 and Cs2SnI4 of the same paper).
4. **Uncertainties.** An uncertainty is recorded only where the paper prints one (Table 2 of Sebastia-Luna); none is attached to a digitised marker.
5. **Application to Sebastia-Luna et al. 2022** (DOI 10.1021/acsaem.2c01936): CsSnI3 (the black phase) only; Table 2 values where tabulated (sigma and kappa at 25 C, S at 100 C, zT at 50 C, with the
   printed uncertainty), the Fig. 2a markers otherwise; the 25 C values are below the window and are recorded, not compared. Extraction script `scripts/na12_extract_sebastialuna.py` (with
   `scripts/pdf_vector.py`), run from a clean commit into `results/na12_extraction/<UTC>/` with the PDF SHA256, the code commit and `tree_clean`; the rows are appended to the values file by the same
   script (`--append-values`). The comparison is not run before the values are reviewed.


## 2026-10-14: NA12, application of amendment 1 to Flahaut et al. 2006 and Singh & Pandey 2017 (before their rows are written)

No out-of-fold prediction has been looked at. Both papers are read as vector figures with `scripts/pdf_vector.py`, `scripts/na12_plot_tools.py` and one script each.

1. **Flahaut et al. 2006** (training source: its curves are in Starrydata2; every row says so, a comparison restates the cross-validation error). Resistivity curves (Fig. 2) are read on the 100 K
   grid (continuous curves); the curve of a label is the one nearest to the tip of the arrow that starts at the label, accepted only where it is unique (nearest within 6 pt and at least twice
   as close as the second). Seebeck markers (Fig. 3) are read only for the series whose marker shape is unique in the legend (CaMnO3 and x = 0.15); x = 0.05/0.4 and x = 0.1/0.5 share a shape
   and are not read. Thermal conductivity (Fig. 4): the open markers (the closed ones are the electronic part), x = 0.05, 0.1, 0.15; the x = 0 series is the data of Ohtaki et al. (said in the
   paper) and is not read. zT is read as Z (Fig. 5) times the nominal temperature. The paper's measurement temperatures (S at 373 to 973 K in 100 K steps) are used for S and Z (the markers
   are drawn about 5 K higher; recorded). Compositions are the nominal Ca1-xYbxMnO3 of the paper.
2. **Singh & Pandey 2017**: the experimental Seebeck coefficient only (the markers of Fig. 1 and the 300 K value of the abstract); the DFT and Boltzmann-transport results are excluded. Each
   marker is a row at its drawn temperature; the markers of the inset (the same data, enlarged) are not read.
3. Checks: the extraction of each paper is validated against the numbers the paper states (asserted in the scripts); where S, rho (or sigma), kappa and zT are all read at one temperature, the zT
   is recomputed and the ratio recorded (flag outside 0.8 to 1.25).

## 2026-10-14: NA12, amendment 2 to pre-registration B (aggregation and reporting); to be implemented before the comparison is run

Made before the comparison has been run for any NA12 value and with no out-of-fold prediction for an NA12 compound looked at.

1. **Aggregation.** Literature values of one paper, one composition (composition_id) and one property that fall in the same 25 K bin of the pipeline (floor bins, [300, 325) is 300) are averaged
   into one value (arithmetic mean in the units of the pipeline, as step 4 of the cleaning does) before the comparison; the number n of values that were averaged and their identifiers are recorded.
   Values outside the window [300, 800) K are recorded and not compared, as before.
2. **Reporting per paper.** Results are reported per paper. Any overall summary weights papers equally, not rows: the fraction inside the band and the median absolute relative error are
   computed within each paper and then averaged (median) over papers.
3. **Training-source papers.** A paper whose DOI is among the DOIs of the training data (Flahaut et al. 2006; its curves are in Starrydata2) is a training-source paper: it is reported
   separately, restates the cross-validation error, and is never pooled with the independent papers in an overall summary.
4. Implemented in `scripts/na12_literature_comparison.py` (`aggregate_bins`, `summarise`, and a self-test that needs no dataset and no prediction: `--self-test`). The pooled-over-rows summary of the
   earlier version of the script is removed.

## 2026-10-14: NA12, amendment 3 to pre-registration B (Ohta 2005 and Ohtaki 1995); made before any of their values is digitised

No out-of-fold prediction for any NA12 compound has been looked at. The figures of `ohta2005.pdf` and `ohtaki1995.pdf` are bitmaps and are to be digitised by hand (WebPlotDigitizer).

1. **Room temperature** is 298.15 K (25 C), below the window: such values are recorded and not compared. This applies to Table II of Ohta et al. (all four samples, the text values copied), to the first
   thermal-conductivity marker of Fig. 3 of Ohta et al. (about 300 K; the same measurement as Table II, so it is not digitised separately), and to Fig. 3 of Ohtaki et al. (sigma at room temperature
   against the ionic radius; not digitised).
2. **Ohta et al. compositions are not derived from the carrier concentration.** Ohta is compared at the cluster level only (B.2(b)): every sample is SrTiO3 cluster, because any doping of these
   crystals is below 5 at%. Each sample is recorded as the paper states it ("La-doped SrTiO3, n = 8.4e19 cm^-3" and so on, samples (a) to (d) of its Table I). In the values file the composition
   column holds the host formula SrTiO3 (the column has to parse), the sample as stated is in `compound_label`, and a row marked cluster-level is not compared at the exact level.
   Amendment 2's aggregation is read within one sample: its key includes the sample as stated (`compound_label`), so the four samples are never averaged together. Both changes are made to
   `na12_literature_comparison.py`, with their self-test, before the comparison is run.
3. **Cross-checks.** Fig. 2 (log sigma T against 1/T) and Fig. 5 (power factor) of Ohtaki et al. are cross-checks of Fig. 1 and Fig. 4 and are not rows.
4. The reading rules of amendment 1 apply to the markers of both papers: each marker is read at its drawn temperature; the lines drawn through the Ohta markers are fits, not data.

## 2026-10-14: NA12, amendment 4 to pre-registration B (marker temperatures of figures with several series: column consensus); made before any comparison, with no out-of-fold prediction for an NA12 compound looked at

Prompted by the digitisation of Fig. 1 of Ohta et al. (Seebeck coefficient of four SrTiO3 crystals). The clicked temperatures of one measurement scattered by up to about 14 K between series, and
for some columns the clicks of one measurement fell in two different 25 K bins. Measured on the figure image (not on any prediction): the clicks lie on a coarse lattice of the digitiser's canvas
(7.75 K in temperature, 4.8 uV/K in S), the two clicked axis-calibration points sit about 3 px (2 K) off the printed ticks, and the marker centres of the four series agree with each other at every
temperature to within about 1.5 K, i.e. the series were measured at common temperatures; those common temperatures are not a regular 50 K grid (the steps between consecutive columns are 44 to 54 K).
Snapping to a nominal 350 + 50k K was therefore rejected: the figure does not support it, and the paper states no measurement temperatures.

1. **Rule.** Where several series of one figure are measured at common temperatures, the temperature of each marker is the consensus of its column: the mean, over the series that have a marker
   there, of the marker-centre temperatures read from the image on a calibration through the figure's own printed tick marks (a straight line through all of them; the calibration uncertainty is its
   largest residual). The S, sigma, kappa or zT value of the marker comes from the digitised click, as before. The value is then assigned to its 25 K bin by the floor rule of amendment 1 (a).
   A column that only one series reaches uses that series' own image-centre temperature. The extraction records, per marker, the column temperature, its standard deviation across the series and the
   calibration uncertainty.
2. **Pre-declared sensitivity.** A row whose column temperature lies within the calibration uncertainty of a 25 K bin edge (an "edge row") is also run in the neighbouring bin; both results are
   reported and the headline uses the floor bin of the column temperature. Edge rows are marked in the `notes` column with the token `[edge-sensitivity: also bin B]` (B = the neighbouring bin); a row
   that is compared only at the cluster level (amendment 3) carries the token `[cluster-level]`. These tokens are what `na12_literature_comparison.py` reads.
3. **Other figures of the same paper.** Each figure's columns are computed from that figure's own image and ticks; the agreement with the columns of Fig. 1 is reported as a check, not assumed
   (the thermal conductivity was measured in separate runs, so its temperatures need not coincide).
4. **Sample (b) of Ohta et al.** Its doping is printed 7.9e20 cm^-3 in the captions of Figs. 1 and 3 and 8.4e20 cm^-3 in Tables I and II (the S, sigma and n of Table II agree with the latter). The
   label records the doping as in the tables; the discrepancy is noted in every row of that sample.
5. **Implementation.** `scripts/na12_extract_ohta.py` (reads the digitised CSVs and the WPD project, the figure image, and the S values of Table II from the PDF text; the SHA256 of every input, the
   commit and the tree state are recorded; figure bitmaps are not committed). The comparison script gets the `[cluster-level]` and `[edge-sensitivity]` handling and the per-sample aggregation key of
   amendment 3, with a self-test, before the comparison is run; that is not done yet.

## 2026-10-16: NA12, amendment 5 to pre-registration B (one temperature per measurement column across the figures of one run; one marker-centre method); made before any comparison, with no out-of-fold prediction for an NA12 compound looked at

Prompted by the digitisation of Fig. 2 of Ohta et al. (electrical conductivity of the same four crystals). The S and sigma markers are the same measurement temperatures (the columns of Fig. 2 agree
with those of Fig. 1 within 1.9 K, inside the figures' calibration uncertainty of 1.15 K and 2.1 K), yet amendment 4, which computes each figure's columns from its own image, gave the same
measurement different bins and edge flags in the two figures (Fig. 1: columns 6 to 8; Fig. 2: columns 1 and 7 to 10).

1. **One temperature per column across figures.** Where several figures report the same measurement run (Ohta et al. Figs. 1 and 2, and Fig. 4 if its columns match), a column's temperature is ONE
   value: the mean of the figures' column consensus temperatures (amendment 4) weighted by 1/u^2, where u is the figure's calibration uncertainty (the largest residual of the straight line through
   all printed ticks of its temperature axis, taken as the standard uncertainty of that axis); the combined uncertainty is 1/sqrt(sum of 1/u^2). Two columns of different figures are the same
   measurement if they differ by at most 2 sqrt(u1^2 + u2^2) and are nearest to each other; a column of a figure that matches no other keeps its own temperature and uncertainty. Fig. 3
   (thermal conductivity, separate runs) is not combined.
2. **Edge flags and bins use the combined values.** The floor bin is that of the combined temperature, and a row is an edge row (amendment 4, item 2) when the combined temperature lies within the
   combined uncertainty of a 25 K bin edge. All rows of one column, in every figure, therefore carry the same T_K, bin and flag.
3. **One marker-centre method.** For all figures the marker centre is the one that does not depend on the line joining the markers: the centroid of the enclosed white interior of an open symbol,
   or of the body of a filled symbol after a morphological opening that removes the line. The left-right symmetry fit used for Fig. 1 on 2026-10-14 is kept only as a recorded cross-check: it is
   biased by the connecting line at several markers of Fig. 2 (checked on enlarged crops), and the two methods agree within 2.2 px on the 40 markers of Fig. 1. The 40 Fig. 1 rows of commit 8f6bfc5
   are replaced by the rows of the new run (the old ones are not kept).
4. **File names.** The four Fig. 2 CSVs were named top to bottom by the digitiser (a, b, c, d for the series b, d, c, a of the figure); they are renamed to the true sample letters, and the
   `--series-map` of the extraction is no longer used. The project file of Fig. 2 holds only one dataset, whose name (d_Nb_...) predates the rename and whose points are those of sample (a); the
   extraction matches a project dataset to a CSV by its points, not its name.
5. **Implementation.** `scripts/na12_extract_ohta.py` takes several figures in one run (`--figure N CSV_DIR IMAGE`, repeated) and writes the combined column table to extraction.json;
   `--append-values <dir> --replace-prefix OH2005_` replaces the earlier rows of the paper.

## 2026-10-16: NA12, amendment 6 to pre-registration B (edge-flag tolerance 2u); made before any comparison, with no out-of-fold prediction for an NA12 compound looked at

Amendment 4, item 2, flags a row as an edge row when its temperature lies within the calibration uncertainty u of a 25 K bin edge. Taken as a standard uncertainty (amendment 5), u covers
about 68 %, so a marker 1.1 to 1.6 K from an edge (one to one and a half u) was left unflagged although its bin is no better than a coin flip at that margin.

1. **Tolerance.** A row is an edge row when the temperature used for its bin (the combined column temperature of amendment 5, or the figure's own column temperature where there is only one figure)
   lies within 2u of a 25 K bin edge, u being the uncertainty of that temperature (about 95 % coverage). The neighbouring bin and the `[edge-sensitivity: also bin B]` token are as in amendment 4.
2. **Scope.** Every NA12 extraction in which temperatures are read from a figure axis. Values at temperatures the paper states (Sebastia-Luna 2022, Flahaut 2006, Singh 2017: nominal or tabulated
   temperatures) are unaffected, as no edge flag applies to them.
3. **Ohta et al. Figs. 1 and 2.** The extraction is re-run with the new tolerance, and the earlier rows of the paper (amendment 5) are replaced.

## 2026-10-16: NA12, amendment 7 to pre-registration B (Ohta et al. Fig. 3, thermal conductivity: only legible markers); made before any comparison, with no out-of-fold prediction looked at

In Fig. 3 the four series overlap at most temperatures, so a marker centre cannot always be placed.

1. **Rule, fixed before any comparison.** Only markers whose centre can be placed (the symbol is recognisable and its interior or body is not covered by another) are digitised. Every other
   marker of the figure inside the window (300 K <= T < 800 K) is recorded as "not legible" with its sample and column in `not_legible.csv` of the run, and the count is reported with the
   results, so that which markers are in the comparison depends on the figure's legibility and not on the values or on any prediction.
2. **What is not data.** The dotted curve and the markers near 0 to 0.5 W/mK are the electronic thermal conductivity estimated from sigma with the Wiedemann-Franz law (derived, not measured),
   the solid line is a fit, and the inset repeats the same data; none of them is read. The marker near 300 K is the room-temperature value of Table II (amendment 3).
3. **Columns.** The thermal conductivity was measured in separate runs, so its columns (about 383, 481, 583, 684 and 788 K) are their own, from the figure's own ticks, and are not combined with
   those of Figs. 1, 2 and 4. No zT consistency check from markers is possible (no common temperatures).
4. Fig. 4 (ZT) is digitised first; its columns are combined with those of Figs. 1 and 2 only if each matches within the tolerance of amendment 5.

## 2026-10-17: NA12, amendment 8 to pre-registration B (approximate zT consistency check for Ohta et al.); made before any comparison, with no out-of-fold prediction looked at

The rules below (log-log interpolation, no extrapolation, reported not asserted) were approved on 2026-10-17 before the Fig. 3 files were delivered; this text was written after the files'
names and values had been listed (to check their format), and before any extraction or comparison.

Amendment 1, item 3, asks for a zT check where S, sigma, kappa and zT are read at one temperature. For Ohta et al. kappa (Fig. 3) is measured in separate runs at other temperatures than S, sigma
and zT (Figs. 1, 2 and 4), so the exact check is impossible; an approximate one is made.

1. **Check.** For each sample and each column of the S, sigma and zT run that has all three values: zT_calc = S^2 sigma T / kappa, with S in V/K, sigma in S/m, T the combined column temperature
   (amendment 5) and kappa interpolated at that T from the sample's legible Fig. 3 markers; the ratio zT_calc / zT of Fig. 4 is recorded.
2. **Interpolation.** Linear in log kappa against log T between the sample's legible Fig. 3 markers (each at its own column temperature, amendment 7). No extrapolation: a column outside the range of
   the sample's legible markers has no check, and the gap is listed with its reason (also a column where the sample lacks S, sigma or zT).
3. **Reporting.** The check is labelled approximate (kappa at other temperatures, from separate runs, and only the legible markers), is reported and not asserted (nothing is excluded or changed on its
   account), and the ratio is flagged outside 0.8 to 1.25. It goes to `extraction.json` of the run (`approx_zt_check`: every column, its inputs, ratio and flag, and the gaps), and into the notes of
   the zT rows concerned. The ratios combine the reading errors of four figures (a few per cent each) with the interpolation error, so they are a plausibility check, not an acceptance criterion.
4. **Not-legible markers.** The run writes `not_legible.csv`: every column of Fig. 3 inside the window, with each sample that has no digitised marker in it (amendment 7); the list is derived from
   the digitised files, not typed.

## 2026-10-17: NA12, amendment 9 to pre-registration B (Ohtaki et al. 1995: legible markers only, extending amendment 7); made before any Ohtaki marker is digitised, with no out-of-fold prediction looked at

Ohtaki et al. (J. Solid State Chem. 120:105-111) report sigma (Fig. 1), S (Fig. 4), kappa (Fig. 6) and Z and ZT (Fig. 7) of (Ca0.9M0.1)MnO3 (M = Bi, Ce, Y, Sm, Sb, La, Pb, In, Sn) and CaMnO3 in a 1-bit scan
where many symbols overlap. An enlarged-crop survey (made before any digitising, no value read) found: Figs. 6 and 7 isolated and legible; Fig. 4 and Fig. 1 legible for some series (Ca, In, Ce, Pb, Bi; Sn,
Sb in part) and overlapped for others (Y, La, Sm).

1. **Rule of amendment 7 applies.** Only markers whose centre can be placed (the symbol is recognisable and its interior or body is not covered by another) are digitised. Every other marker of the
   figure inside the window (26.85 <= T < 526.85 deg C, i.e. 300 <= T < 800 K; Fig. 1: 3.33 > 1000/T > 1.25) is recorded as "not legible" with its series and column in `not_legible.csv` of the run,
   derived from the digitised files (every column inside the window with each series that has no digitised marker in it), not typed; the count is reported with the results. The first marker of every
   series (about 25 deg C, 298.15 K) is room temperature and is recorded, not compared (amendment 3).
2. **Sm and Sn.** The symbols of Sm (triangle with speckled interior) and Sn (solid triangle) differ only by the speckle. A triangle whose speckle cannot be told apart is skipped (not legible); it is never
   assigned to Sm or Sn by its position or its value.
3. **Which figures are rows.** Fig. 1 (sigma), Fig. 4 (S), Fig. 6 (kappa) and the ZT of Fig. 7 (right axis). Fig. 2 (log sigma T), Fig. 5 (power factor) and the Z of Fig. 7 (left axis) are checks only and not
   rows (amendment 3, item 3): Z with ZT = Z T(K), and sigma T, S^2 sigma with the digitised sigma and S, reported and not asserted.
4. **Axes and columns.** Temperatures are read on the deg C axes with the figure's own printed ticks and converted to K (T_K = T_C + 273.15) after the column consensus (amendments 4 and 5, 2u edge tolerance of
   amendment 6); on Fig. 1 the calibration is on the 1000/T axis and T = 1000/x, with the uncertainty of T propagated from that of x. Figures of one measurement run (Figs. 1, 4 and 7) share columns only if
   their columns match under amendment 5, which is decided after digitising and not assumed; Fig. 6 (kappa; its temperatures differ between its two samples) has its own columns, as Fig. 3 of Ohta et al.
5. **Compositions and levels.** The composition of a row is the formula as the paper states it (Ca0.9M0.1MnO3 with the dopant, or CaMnO3), compared at the exact and the cluster level as for the other papers
   (not a [cluster-level] value; amendment 3 concerns Ohta et al. only); the training-source status of the paper is decided by the comparison script from its DOI.
6. **Order.** Fig. 7, Fig. 6, Fig. 4, Fig. 1, each digitised before any comparison is run.

## 2026-10-17: NA12, amendment 10 to pre-registration B (Ohtaki et al. Fig. 7: column-grouping gate 6 deg C); made after the Fig. 7 digitising, before any comparison, with no out-of-fold prediction looked at

The extraction script stops when the markers of one column differ by more than a gate (a data-quality check against mis-grouping, not a result; 5 deg C for the Ohta figures, where the largest spread was 4.1 K).
In Fig. 7 of Ohtaki et al. the Z and ZT markers of one column are plotted up to 5.6 deg C apart (columns at about 500, 600 and, within 5.0, 200 deg C), while the columns are 90 to 100 deg C apart and the
temperature axis is good to 1 deg C: the offsets are in the figure, not a grouping error. The gate is therefore 6 deg C for this figure only; this value was chosen after seeing the 5.6, which is stated
here. The column temperature is the mean of the Z and ZT markers (amendment 4); the columns above 5 deg C are flagged in the notes of their rows. Using the ZT marker's own temperature instead would move
T by up to 2.8 K and change one floor bin (column 2, 373.9 against 375.4 K), a row that is already an edge row run in both bins (amendment 6). ZT against Z T(K) agrees to within 5.4 % at all 8 columns
(reported, not asserted). The abstract's statements on Z and ZT are not checked against Fig. 7 (the markers beyond 690 deg C are not digitised): decided 2026-10-17, Fig. 7 stays as digitised.

## 2026-10-17: NA12, amendment 11 to pre-registration B (Ohtaki et al. Fig. 4: unplaceable clicks, columns of the not-legible list, estimator settings); made after a test run of the Fig. 4 files, before the official run and any comparison, with no out-of-fold prediction looked at

Ohtaki et al. Fig. 4 (S, ten series, one WebPlotDigitizer project with one calibration for all) was digitised under amendment 9 (nine series; Sm not, its speckled triangle cannot be told from Sn's solid one, item 2). A test run from a dirty tree
(discarded, not a result) showed the points below. The rules and the parameter choices were made after seeing that run, which is stated here.

1. **Calibration.** The project holds one dataset (Sn); its 6 clicks fix the affine map (pixel to temperature and S) exactly (the axes are rotated), which reproduces the project's four calibration points to 1e-12 and is
   inverted to place the clicks of the other series. The S ticks are read at the right frame (the left-frame ticks are covered by markers and sit up to 6 px, 2 uV/K, away); the clicked S calibration must agree with them to 5 uV/K.
2. **Columns.** The series are measured at their own temperatures (the clicked temperatures differ by tens of deg C, Bi has twice as many columns as the others), so, as for Fig. 6, every marker is its own column and
   has its own temperature (amendment 9, item 4); Fig. 4 shares no column with Fig. 7. The temperature axis is the straight line through all six printed bottom ticks (200 to 1200 deg C; the 1200 tick sits 6 px off
   the equidistant position of the other five), giving u = 2.1 K, so the 2u edge tolerance of amendment 6 is 4.2 K and flags 27 of the 49 rows. Reported, not used: interpolating between the bottom and the top ticks moves
   any marker by at most 2.8 K, inside u.
3. **Unplaceable clicks.** A clicked marker whose centre the estimator cannot place (no enclosed interior or opened body near the click; or a symmetry fit in which fewer than 3 of the 5 settings agree) is skipped, not
   digitised, and listed in `not_legible.csv` with its clicked temperature and reason (amendment 9, item 1: the centre cannot be placed). Three clicks: Y at 214 deg C (hidden behind the Bi circle), Sb at 108 deg C (its
   interior is occupied by an Sm triangle), and CaMnO3 at 389 deg C (an isolated plus whose symmetry fit agrees in 1 of 5 settings; it looks legible, and is skipped because the rule is applied as written, not by eye).
4. **Not-legible columns.** The series have no common columns, so the columns of the list are derived by single-linkage clustering of the digitised markers of the figure (a gap of 20 K starts a new column), and a cluster
   is a column of the figure only if it holds markers of at least half of the series (5 of 10, Sm counted as a series with no digitised marker). A cluster of one or two series is those series' own extra temperature
   (Bi and In), not a column at which the other series were expected. Each column inside 300 <= T < 800 K lists the series with no digitised marker in it.
5. **Estimator settings, chosen after seeing the failures.** The enclosed-interior search window is 40 px (was 34) for every Fig. 4 series, and the largest accepted interior area is 3200 px (was 1500) for Sb, whose open squares
   have interiors of about 1800 px; `hole_centre` in `na12_extract_ohta.py` takes both as optional arguments whose defaults are the old values, so no earlier extraction changes. Symmetry fit for the symbols with no enclosed white interior
   (Pb hatched, In dotted, Ca plus), as for CaMnO3 in Fig. 6.
6. **Reported, not asserted** (in `extraction.json`): the interior area of every marker (partly covered interiors: Sb at 576 K, La at 378 and 722 K, Y at 900 K), the difference between the symmetry fit and the interior centre,
   and the click against the image temperature.
7. **Records.** Rows are written for every digitised marker, including those at or above 800 K (recorded, not compared), as for Figs. 6 and 7. The official run takes Figs. 7, 6 and 4 together (the append replaces all
   `OHTAKI1995_` rows); the test run reproduced the 17 rows of Figs. 6 and 7 byte for byte.

## 2026-10-17: NA12, amendment 12 to pre-registration B (Ohtaki et al. Fig. 1: the 1000/T axis, tilt, units, room temperature); made after a test run of the Fig. 1 files, before the official run and any comparison, with no out-of-fold prediction looked at

Fig. 1 (log10 sigma against 1000/T, ten series; Sm not digitised, amendment 9 item 2) is read as Fig. 4 is (one project and one calibration for all series, amendment 11), with the changes below, made after seeing a test run from a dirty tree
(discarded, not a result). Items 7 to 10 were added the same day after the test run showed the points they answer (file problems, the column derivation, and doubtful markers), still before the official run.

1. **Calibration.** x: the printed ticks 1.0 and 4.0, entered as 1 and 4 (linear in 1000/T, not temperatures); y: the printed ticks 2 and 0, entered as 2 and 0 (linear; the axis is log10 of sigma in S/cm as plain numbers, not a log
   scale). The project's single dataset (Ce) must equal its CSV; the affine map fitted on its 6 clicks reproduces the four calibration points to better than 1e-12 and is inverted to place the clicks of the other series.
   The ticks 3 and -1 are the top and bottom frame lines and are used from the frame.
2. **Temperature and its uncertainty.** x is read on the printed ticks (every 0.5, 1.0 to 4.0) and T = 1000/x. The uncertainty of x is the largest residual of the straight lines through the bottom and through the top ticks
   (about 0.0066 in the test run); u(T) = T^2/1000 u(x) is computed per marker (about 0.6 K near 330 K to 5 K near 870 K), is the u of that marker's column in amendments 5 and 6 (`_combine_run` takes an optional per-marker `u_K`;
   without it the figure's single u is used, so no earlier extraction changes), and flags the edge rows with the 2u tolerance of amendment 6.
3. **Tilt.** The figure is tilted: the top ticks sit 6.5 to 8 px (of 432 px per unit of 1000/T) left of the bottom ticks. x of a marker is therefore interpolated between the bottom and the top tick lines at the marker's
   height. On the bottom ticks alone T would differ by up to 11 K (reported in `extraction.json`, not used). This differs from Figs. 4, 6 and 7, which use the bottom ticks only (for Fig. 4 the top-tick alternative is at most
   2.8 K, reported); those are not changed.
4. **Values.** The click value is log10(sigma / S cm^-1); the row holds sigma in S/m = 100 x 10^y, 3 significant digits (the unit of the other papers' sigma rows).
5. **Room temperature.** The markers at 1000/T about 3.35 to 3.45 (T below 300 K) were not digitised, as in Fig. 4, with one exception: `sb_sigma.csv` holds the Sb marker at 1000/T = 3.389 (T 295 K), which is recorded below the
   window and not compared (amendment 9 item 1; the row's notes say so).
6. **Estimator and unplaceable clicks.** As amendment 11 (items 3 and 5): the symbol decides the centre method (Pb hatched, In dotted and Ca plus by the symmetry fit), the interior search window is 40 px, Sb's largest interior
   is 3200 px, a click whose centre cannot be placed is skipped and listed in `not_legible.csv`. `sm_sigma.csv` in the folder is not a Fig. 1 file (its values are temperatures in deg C and S in uV/K) and is not read; nor is the old `sn_sigma.csv`, which held Sb points. The Sn series is `sn-redo_sigma.csv`.
7. **Not-legible columns (the Fig. 4 rule does not carry over).** Clustering all markers with a 20 K gap chained different columns of this figure into one (9 markers over 70 K) and hid the gaps of a real column
   with fewer than half the series. The series of Fig. 1 share measurement columns, so the columns are those of a reference series, Bi (the series with most digitised in-window markers), plus any column where another
   series has a marker more than 20 K from every Bi column (such markers are clustered with a 20 K gap; the column temperature is their mean). Each column inside 300 <= T < 800 K lists every series (Sm included, as it has
   no file) that has no marker within 20 K of the column temperature. Columns use the markers that remain after item 8.
8. **Image-derived legibility gates (Fig. 1 only).** A placed marker is not legible, and is not a row, if any of: (a) its enclosed-interior area is below 0.8 of the median area of its series (the interior is partly covered);
   (b) for the interior-centre estimator, the symmetry-fit centre and the interior centre differ by more than 10 px; (c) its image temperature differs from the temperature of its click by more than 15 K (the click and the
   marker are not the same marker, or the centre is biased by neighbours). The gates apply at once to all placed markers, inside or outside the window, with the series median taken before any marker is removed, and each
   removed marker is listed in `not_legible.csv` with its reason. Thresholds fixed after seeing the flags of the test run (stated here). In the test run they remove 12 markers (Ce 417 K; Y 911 K; Sb 570 K; La 426, 526, 572 and
   675 K; Pb 871 K; In 700 and 805 K; Sn 1068 and 1174 K); the two In markers at 700 and 721 K were 21 K apart where the clicks were 50 K apart.
9. **Fig. 4 is not changed.** Fig. 4 was extracted under amendment 11 without these gates and its rows stay. Applied to it they would have removed 5 of its 49 rows (Y 900 K, Sb 576 K, La 378 and 722 K by item 8a; CaMnO3 585 K
   by item 8c; computed from the saved clicks, not applied). The difference between the two figures is stated, not hidden.
10. **Sn calibration.** The Sn clicks were made with their own calibration (`sn_project.json`), which is not in the folder; the Sn markers are located with the shared project's calibration (positions only), and the Sn values
   are the clicks. Check, reported: the value read from the body centre row of each Sn marker agrees with its click to 0.042 in log10 sigma at most (the centroid of a solid triangle sits below its centre), the same as the
   other interior-centre series (0.009 to 0.042).

## 2026-10-17: NA12, amendment 13 to pre-registration B (the legibility gates of amendment 12 apply to every digitised scanned figure); made before the comparison, with no out-of-fold prediction looked at

Found on review: amendment 12 (item 8) applied three image-derived gates to Ohtaki et al. Fig. 1 only, and item 9 left Fig. 4 under amendment 11. Two rules for the same kind of marker in the same two papers is arbitrary, and the
gates were chosen after seeing the digitised Fig. 1, not because Fig. 1 is special.

1. **Scope.** The gates of amendment 12, item 8, apply unchanged to every digitised scanned figure: Ohta et al. Figs. 1 to 4 and Ohtaki et al. Figs. 1, 4, 6 and 7 (the figures read as vector graphics, Sebastia-Luna, Flahaut
   and Singh, have no marker centres to place). The thresholds are not changed: (a) enclosed-interior area below 0.8 of the median of its series; (b) symmetry-fit and interior centres more than 10 px apart (interior-centre
   markers); (c) image temperature more than 15 K from the clicked temperature. This replaces the "Fig. 1 only" scope of amendment 12 item 8 and its item 9 (Fig. 4 not changed).
2. **How they apply to Ohta et al.** Gate (a) uses the interior area of the markers whose centre is the enclosed interior or opened body; gate (b) the markers with an interior centre; gate (c) the markers with an image
   temperature, and is not applied to a figure whose project labels a printed temperature tick with the wrong value (Fig. 3: the clicked temperatures are not recorded there, and three of its markers have no located centre).
   The Table II identity checks (series order, symbol fill, the first marker against room temperature) use the markers before gating; the column temperatures are computed from the markers that remain.
3. **Pixels.** The limit of 10 px is in pixels of the digitised bitmap, as for amendment 12. The bitmaps are 3 times (Ohta) and 4 times (Ohtaki) the native scan, so the same limit is 3.3 and 2.5 native pixels; it was not
   rescaled (stated here, so that the figures are not read as equally strict in native pixels).
4. **Effect, per figure and gate** (from the test run of the changed code from a dirty tree, discarded; the official run's `extraction.json` is the record). The thresholds were fixed by amendment 12; the effect on the figures
   other than Ohtaki Fig. 1 was not looked at before they were applied to them. Ohta Fig. 1: none; Fig. 2: 2 markers (gate b: sample (a) at 750 K, 19.3 px; sample (c) at 545 K, 11.6 px); Fig. 3: none; Fig. 4: 1 marker
   (gate b: sample (c) at 350 K, 13.2 px). Ohtaki Fig. 1: 12 (amendment 12: a 2, b 5, c 6, one marker fails a and b); Fig. 4: 5 (gate a: Y 900 K, Sb 576 K, La 378 K and 722 K; gate c: CaMnO3 585 K); Figs. 6 and 7: none.
   In Ohta et al. the combined column temperatures of the remaining markers move by at most 0.1 K, with no change of floor bin or edge flag.
5. **Records.** Each removed marker is listed in `not_legible.csv` of its run with its reason; the rows of Ohta et al. (137 before, 134 after) and of Ohtaki et al. (132 before, 127 after) are replaced by the rows of the
   official runs. The gate thresholds are module constants (`GATES` in `na12_extract_ohta.py`, read by `na12_extract_ohtaki.py`), as are the column gap (`COLUMN_GAP_K`), the 2u edge factor (`EDGE_FACTOR`), the room
   temperature (`ROOM_TEMPERATURE_K`) and the curve grid step (`GRID_STEP_K`); the number registry reads them from there.

## 2026-10-19: NA12, amendment 14 to pre-registration B (Shibasaki et al. 2009 and Rajasekaran et al. 2020: which values are rows); made before any value of the two papers is read into the values file and before the comparison, with no out-of-fold prediction looked at

The two last PDFs were inventoried (`docs/na12_pdf_inventory.md`, 2026-10-19). **Disclosure:** while skimming the text layer for that inventory, a few values came into view that were not meant to be read: in Shibasaki et al.
statements of resistivity and S at 300 K and a mobility in the running text; in Rajasekaran et al. the n, mobility, sigma, S and power-factor entries of the BSOLa5 and BSOLa10 rows of Table 2; the rendered Fig. 10 was also looked at to judge
its legibility. Nothing was recorded from them; the values are read again under the rules below. No prediction was looked at, so the guard of pre-registration B (values read before predictions are seen) is intact; the
instruction "read no values" was not met to the letter.

1. **Shibasaki et al., compositions.** The paper-specific note of the pre-registration governs ("LaRhO3 (undoped member of the B-site series)"): only undoped LaRhO3 (x = 0) is compared. The Ni- and Mg-substituted members are recorded in the
   values file and marked `[not-compared: doped member, amendment 14]`; the comparison script treats that token like a value outside the window (kept in the output with the reason, never aggregated, never compared). Ni at x = 0.3
   (6 at%) is a cluster of its own and is recorded too.
2. **Shibasaki et al., which figure.** The rows come from Fig. 5, whose caption states 300 K: sigma (the conductivity panel, converted to S/m) and S, one marker per composition and series, T = 300 K as stated (amendment 1, item 2: the
   stated temperature is used). 300 K is the lower edge of the window and a stated temperature, not the room-temperature value (298.15 K, amendment 3), so these rows are compared and carry no edge-sensitivity token. The
   end markers of Figs. 2 and 4 (drawn at 297.9 to 306.6 K, the text gives 4.2 to 300 K) are a cross-check only: the extraction reports, for each composition, the Fig. 2 and Fig. 4 marker nearest 300 K against the Fig. 5 value
   (reported, not asserted); the power factor of Fig. 5(c) is derived and is a check at most. If Fig. 5 has two x = 0 markers (one in the Ni series, one in the Mg series), both are rows with the same stated sample ("LaRhO3, x = 0, Fig. 5"),
   so that amendment 2 aggregates them (n and identifiers kept); the series each came from is in the notes. Whether the paper is a training source is decided by the comparison script from its DOI.
3. **Rajasekaran et al., Table 2.** Table 2 states no temperature for its sigma and S. It is recorded here and not read into the values file (a row needs a temperature), is not compared, and no temperature is inferred for it (in particular
   not that of the last marker of Fig. 10). The rows come from Fig. 10(a) (sigma) and 10(b) (S): BSOLa5 and BSOLa10, six markers each at the temperatures of the figure, by hand in WebPlotDigitizer under amendments 4, 5, 6, 9, 12 and 13 (raster,
   legible markers only, the gates). Fig. 10(c), the power factor, is derived and is not read. Undoped BaSnO3 has no sigma or S in the paper and contributes no values. La-doped BaSnO3 is compared at the cluster level
   (La is below 5 at% in both samples); each row carries `[cluster-level]`.
4. **Implementation.** `scripts/na12_literature_comparison.py` reads the token `[not-compared: reason]` (self-test extended); `scripts/na12_extract_shibasaki.py` (vector extraction, as for Sebastia-Luna) writes the Shibasaki rows from a
   clean tree with the SHA256 of the PDF, the commit and the tree state.

## 2026-10-19: NA12, amendment 15 to pre-registration B (a value the stated-temperature figure cannot resolve is taken from the figure that does); made before the comparison, with no out-of-fold prediction looked at

Found when the Shibasaki et al. rows of amendment 14 were extracted and cross-checked, before any prediction was looked at. The undoped LaRhO3 conductivity of Fig. 5 (the figure that states 300 K) is plotted on a linear axis of 0 to 40 ohm^-1 cm^-1
about 0.2% of the axis range above zero, so its reading precision is far below the property's spread; the end markers of the x = 0 series of Figs. 2 and 4, on logarithmic resistivity axes, resolve it, and they differ from the Fig. 5 value by
a factor of about 1.6 to 1.8 (a difference of 0.2 dex, the size of the model's own sigma error). This is a reading-precision reason known before any comparison; it is not a choice made after seeing a prediction.

1. **Rule.** Where the figure that states the temperature places a value within 1% of its axis range (below 0.01 x the range of the labelled ticks), the value of the same sample at that temperature is taken from another figure of the
   paper that resolves it. The extraction applies the test to every Fig. 5 marker and stops if it triggers anywhere other than the one case below.
2. **Application.** It triggers for the undoped LaRhO3 conductivity (x = 0, both series) of Fig. 5(a) and for nothing else (the smallest doped conductivity is above 1% of the range; S and the other values are far from it). The conductivity of
   undoped LaRhO3 is therefore taken from the end markers of the x = 0 series of Fig. 2 (Ni series) and Fig. 4 (Mg series): the marker nearest 300 K of each (drawn at 300.3 and 300.2 K; the text gives 4.2 to 300 K, so the stated 300 K is used and
   the drawn offset is recorded, not used, as in amendment 1, item 2), read as the resistivity in ohm cm (the unit of the axes) and converted to sigma by the comparison script. The two rows carry one stated sample ("LaRhO3, undoped, x = 0, as stated
   for Figs. 2 and 4"), so amendment 2 aggregates them (n = 2, identifiers kept). The S of undoped LaRhO3 stays from Fig. 5 (the Fig. 2 and 4 end markers agree with it, the cross-check of amendment 14).
3. **Records.** The Fig. 5 conductivity of the two x = 0 markers stays in the values file with the token `[not-compared: unreadable in Fig. 5 (within 1% of the axis range), replaced by Figs. 2 and 4, amendment 15]`; the notes of the Fig. 2 and
   Fig. 4 rows state the Fig. 5 value, its share of the axis range and this reason. The cross-check of amendment 14 is unchanged and still reports the difference.

## 2026-10-19: NA12, amendment 16 to pre-registration B (Rajasekaran et al. Fig. 10: values from the printed ticks, marker opening radius, the abstract as a check); made after the test run of the digitised files, before the official run and the comparison, with no out-of-fold prediction looked at

Fig. 10(a) and 10(b) were digitised by hand (amendment 14, item 3). The test run of `scripts/na12_extract_rajasekaran.py` (a dirty-tree run, discarded) showed the points below.
**Disclosure:** while looking up the composition formula in the abstract, its text values came into view: the Seebeck coefficient of BSOLa5 at 573 K and the two electrical conductivities (BSOLa5 and BSOLa10, mantissas 1.19 and
1.03, temperature not stated in the sentence). They are used only as checks (item 4); no prediction was looked at.

1. **The clicked value axes do not match the printed ticks, so the clicked values are not used.** In both projects the two value-calibration points were entered on a 0 to 1000 scale across the frame height: in panel (a) the top
   frame line (printed 1.5e3) was entered as 1000, and in panel (b) the frame height (0 at the top, -70 uV/K at the bottom) was entered as 0 to 1000 (the test run reports the clicked value against the printed tick at both calibration
   points). The clicked temperature axis is correct (300 and 600 on the first and last printed tick). The pixel position of every click is exact (the projects' linear calibrations are inverted; the series not held in a project are placed by the
   inverse of its map), so each value is read at the click's pixel row on the printed ticks of the panel (sigma: 0, 500, 1000, 1500 S/m; S: 0 to -70 uV/K in steps of 10), after correcting the frame's small tilt. This replaces "the value comes from
   the clicked value" of amendment 4 for this figure; the click row, not the marker centre row, is used, as the click is the digitiser's reading.
2. **Marker centre.** The markers of Fig. 10 (squares and circles about 20 to 26 px across, joined by a line about 4 px wide) are filled; their centre is the centroid of the opened body (amendment 5, item 3) of the series' own colour (black
   or red), with an opening disc of radius 7 px (`hole_centre` takes the radius as an optional argument whose default, 11 px, is unchanged, so no earlier extraction changes): 11 px removes the squares. The radius was chosen after the 11 px
   failed on the test run, which is stated here.
3. **Compositions.** BSOLa5 and BSOLa10 are Ba1-xLaxSnO3 with x = 0.05 and 0.10 (the paper: La at the Ba site), written Ba0.95La0.05SnO3 and Ba0.9La0.1SnO3; La is 1 and 2 at%, below 5 at%, so both are compared at the cluster level
   (`[cluster-level]`). The two samples are two stated samples, the two panels one measurement run (one combined column set, amendment 5).
4. **The abstract as a check.** The Seebeck coefficient of BSOLa5 at 573 K stated in the abstract is compared with the digitised 573 K marker (asserted, to 3 uV/K; the test run agrees to 0.1 uV/K); the ratio of the two printed conductivities
   is compared with the ratio of the two 573 K conductivity markers (reported; agreement 0.7%). No row is taken from the abstract (the 573 K marker is the same measurement, and Table 2 stays unread, amendment 14, item 3).
5. **Gates and unplaceable clicks (amendments 11 to 13) apply.** The first test run of this figure removed three markers by gates (the 323 K S marker of both samples, and sigma of BSOLa10 at 575 K) and skipped BSOLa5 sigma at 424 and
   476 K (the black squares are covered by the red circles); that run exposed an artefact of the symmetry cross-check and of the search window, which amendment 17 corrects for every scanned figure.

## 2026-10-19: NA12, amendment 17 to pre-registration B (the symmetry cross-check on the opened body, and a window centred on the click; every scanned figure); made before the comparison, with no out-of-fold prediction looked at

**Disclosure:** this amendment was prompted by the end markers of Rajasekaran et al. Fig. 10, where the gates of amendment 13 removed markers for a reason that is not about legibility. It changes rules that were already applied to Ohta et al. and
Ohtaki et al., so every scanned extraction is re-run and the changes are shown (below, and in the run folders).

1. **What went wrong.** The left-right symmetry fit of amendments 5 and 12 is computed on the raw column profile. A joining line (Ohta et al. Figs. 1 to 4, Rajasekaran et al. Fig. 10) lies on one side only at the first and last marker of a series and biases
   that fit. Two things followed: gate (b), which compares the fit with the interior centre, fired on those end markers; and the interior search window, which was centred on the biased fit, was cut across the marker (the first S marker of both
   samples of Fig. 10(b): the biased fit lay 24 px from the click, the 68 px window clipped the marker, and its interior area was measured as 260 and 307 px instead of 365 and 426), so gate (a) fired too.
2. **Image check of the area gate (from the image only).** The opened bodies of those markers, measured with a window centred on the marker, have the same size as the others (365 and 426 px against 344 to 373 and 422 to 432 px; 21 x 21 and 23 x 23 px).
   The two series' markers touch each other at the first column (their 3 px halos share 17 and 22 px) but neither is covered. The area-gate failure was therefore the window artefact, not a genuine loss of marker, and the markers stay in.
3. **Rule.** (i) The symmetry cross-check of gate (b) is computed on the opened body: the symbol (an open one with its interior filled) is opened with the disc that removes the joining line (radius 11 px, 7 px for Fig. 10), the component at the found centre is
   pasted into an empty image and fitted as before; the raw fit is recorded as `centre_px_symmetry_raw`, not used by the gate. (ii) The enclosed interior or opened body is searched in two passes: a window centred on the click, then a window centred on what the first pass
   found, so that a marker is not clipped when the click is off-centre. The centre of a marker is still the centroid of its enclosed interior or opened body (amendment 5, item 3); where the old window was centred well, the centre is unchanged. The gates and their
   thresholds (amendment 13) are unchanged; the area is now the area of the unclipped body. The symbols with no enclosed interior (hatched, dotted, plus) keep the raw fit as their centre (amendment 11, item 5).
4. **Scope.** Every scanned figure: Ohta et al. Figs. 1 to 4, Ohtaki et al. Figs. 1, 4, 6 and 7, Rajasekaran et al. Fig. 10(a) and (b); the vector figures have no marker centres to place. Implemented in `body_find` and `body_symmetry` of
   `na12_extract_ohta.py`, used by the three extraction scripts.
5. **Effect.** In the test run of the changed code (a dirty-tree run, discarded; the official runs' `extraction.json` are the record) the three Ohta markers removed under amendment 13 return; Ohtaki Fig. 1 goes from 12 to 8 removed markers and Ohtaki Fig. 4 keeps its 5 and
   places the Sb marker at 108 deg C that was unplaceable (the window artefact again); the Rajasekaran end markers stay. The per-figure comparison with the rows of amendment 13 is in the reply that introduced this amendment and in the run folders.
