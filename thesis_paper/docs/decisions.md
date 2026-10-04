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
