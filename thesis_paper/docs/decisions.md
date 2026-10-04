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
