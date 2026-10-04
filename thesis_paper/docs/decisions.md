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
