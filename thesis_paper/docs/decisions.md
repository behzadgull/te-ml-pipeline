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
