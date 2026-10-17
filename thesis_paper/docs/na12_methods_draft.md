# DRAFT (not committed): NA12 methods text for Section 4.5.1 (methods part only)

Written 2026-10-17 from `docs/decisions.md` (pre-registration B and amendments 1 to 12), `scripts/na12_literature_comparison.py` and the committed extraction runs. No comparison has been run; the text contains no
result of it. Numbers are plain here; the list at the end says which need a marker (DESIGN, GENERATED or CITED) before the number audit.

---

### 4.5.1 Literature validation: protocol

**Papers and compounds.** The literature check was fixed before any value was read and before any prediction for these compounds was looked at (pre-registration of 5 October 2026, `docs/decisions.md`). It uses <!--v:na12_papers-->7<!--/v-->
papers, each matched by DOI to its title, first author and journal through Crossref: CsSnI~3~ (vacuum-deposited films) [@sebastialuna2022], LaCoO~3~ [@singh2016], LaRhO~3~ [@shibasaki2009], La-doped BaSnO~3~ [@rajasekaran2020],
La- and Nb-doped SrTiO~3~ single crystals [@ohta2005temperature], doped CaMnO~3~ [@ohtaki1995; @flahaut2006effect]. The compositions, dopants included, are those the papers state. BaSnO~3~ and LaRhO~3~ stay in only if their paper
gives measured values. The test is not blind: the compounds were chosen because they have published measurements and, for the pre-registered shortlist, a chemistry cluster present in the training data.

**What is compared.** The predictions are the out-of-fold predictions of the chemistry-cluster cross-validation of Section 4.1.1 (frozen hyperparameters, five repeats of five folds), in which the chemistry cluster of
the row was held out. They are not the predictions of the final models, and not those of Table 11, which are final-model predictions for Materials Project compounds. Nothing is refitted. Each literature value is
compared at two levels that are reported separately: *exact*, the training rows with the same composition at the same <!--d:clean_bin_width-->25<!--/d--> K temperature bin as the value (mean over those rows and over the five repeats), if the training
data contain the composition; and *cluster level*, the mean over the training rows of the same chemistry cluster at that bin, so that a dopant level absent from the training data is still compared with held-out
predictions of its cluster. Where a level has no row the value is reported as having no out-of-fold prediction. *σ* and *κ* are predicted in log~10~ and compared in log~10~ and in linear units (back-transformed
without smearing).

**Reading the literature.** Values are read from the PDFs only, with page and figure or table, and marked as table, text or digitised. A value read from a figure carries the tool (vector extraction from the PDF where the figure is
vector, WebPlotDigitizer, version 5, clicks on the scan otherwise [@rohatgiwebplotdigitizer]) and the axis calibration; no uncertainty is attached to a digitised marker, and an uncertainty is recorded only where the paper prints one. Discrete
markers are read at their own measurement temperature and assigned to the <!--d:clean_bin_width-->25<!--/d--> K floor bins of the cleaning pipeline (the bin [<!--v:t_min-->300<!--/v-->, 325) K is labelled <!--v:t_min-->300<!--/v--> K); curves, and only curves, are read on a <!--d:na12_grid_step_K-->100<!--/d--> K grid (<!--v:t_min-->300<!--/v--> to
<!--d:na12_grid_max_K-->700<!--/d--> K). Values below <!--v:t_min-->300<!--/v--> K (room temperature, <!--d:na12_room_temperature_K-->298.15<!--/d--> K) or at or above <!--v:t_max-->800<!--/v--> K are recorded and not compared. Derived curves are not read (Wiedemann-Franz electronic conductivity, fitted lines, DFT and Boltzmann-transport
results). Units are converted to those of the pipeline (S: µV K^-1^, *σ*: S m^-1^, *κ*: W m^-1^ K^-1^; *zT* as stated, or *Z* multiplied by the temperature). Where S, *σ* (or resistivity), *κ* and *zT* are all read at one
temperature, *zT* is recomputed from the other three and the ratio is reported against the printed value; where the temperatures differ (*κ* of the SrTiO~3~ crystals) an interpolated, labelled approximate check is
made. These checks are reported, never used to exclude or change a value.

**Marker temperatures in figures with several series.** In a figure whose series were measured at common temperatures, the temperature of a marker is the consensus of its column: the mean of the marker-centre
temperatures of the series, read on a straight line through the printed ticks of the axis (the calibration uncertainty *u* is the largest residual of that line); where several figures report one measurement run, a column has
one temperature, the mean of the figures' consensus weighted by 1/*u*^2^. The marker centre is the centroid of the enclosed white interior of an open symbol or of the opened body of a filled one; a symmetry fit is used
only for symbols with no enclosed interior (hatched, dotted, plus). Where the series do not share temperatures (Ohtaki et al., Figs. 4 and 6), every marker has its own temperature. For a 1000/T axis (Ohtaki et al.,
Fig. 1), *T* = 1000/*x*, *u*(*T*) = *T*^2^/1000 *u*(*x*), and *x* is interpolated between the bottom and the top tick lines because the scan is tilted.

**Edge sensitivity.** A value whose temperature lies within <!--d:na12_edge_factor-->2<!--/d--> *u* of a bin edge is compared in its floor bin (the headline) and also in the neighbouring bin; both are reported, together with the overall summary
under the neighbouring-bin assignment and the values whose result changes.

**Legibility.** Only markers whose centre can be placed are read. A marker overlapped by another symbol is recorded as not legible, with its series and column, in a list derived from the digitised files and not
typed; the count is reported with the results. A click whose centre the estimator cannot place is skipped. Three image-derived gates apply, with the same thresholds, to every digitised scanned figure (Ohta et al.,
Figs. 1 to 4; Ohtaki et al., Figs. 1, 4, 6 and 7): a marker is removed if its enclosed interior is smaller than <!--d:na12_gate_area-->0.8<!--/d--> of the median of its series (interior partly covered), if its symmetry-fit and interior centres are more than
<!--d:na12_gate_sym_px-->10<!--/d--> px apart, or if its image temperature is more than <!--d:na12_gate_click_K-->15<!--/d--> K from the temperature of its click. The columns of a figure for the not-legible list are derived from a reference series (and a
<!--d:na12_column_gap_K-->20<!--/d--> K gap) where the series share columns, and from the clustering of all markers where they do not.

**Aggregation and error.** Values of one paper, composition, sample and property in the same 25 K bin are averaged into one value before the comparison (*n* and the identifiers are kept). The doped SrTiO~3~ crystals of
Ohta et al. have dopant levels below <!--d:grp_dopant_pct-->5<!--/d--> at% given only as carrier concentrations, so each is compared at the cluster level only, as SrTiO~3~, and the four samples are never averaged together. The error is
(prediction - literature)/|literature| and, for *σ* and *κ*, the log~10~ ratio, set beside the relative standard uncertainty of the round-robin measurements of Alleno et al. (Section 4.1.1; S, *σ*, *κ*, *zT*), the only
threshold used: a prediction is inside the band if its relative error does not exceed it.

**Reporting.** Results are reported per paper. Any overall summary weights papers equally: the fraction inside the band and the median absolute relative error are computed within each paper and then summarised by the
median over papers. A paper whose DOI is among the DOIs of the training data (Flahaut et al.) is a training-source paper: it is reported separately, restates the cross-validation error, and is never pooled with the
independent papers. For every value the table states whether its DOI is among the training DOIs, whether its composition and cluster are in the training data, and whether the training data hold a row with the same DOI and
composition (the value is then probably a training label, although its cluster was held out).

**Pre-registration and its amendments.** The comparison script was run only after the values file was filled and committed. The rules above are the pre-registration plus <!--v:na12_amendments-->13<!--/v--> dated amendments
(`docs/decisions.md`; the full list is in the Supplement [TODO: table]), each written before the comparison and with no out-of-fold prediction for these compounds looked at. The thresholds of the legibility gates (amendments 10 to 12)
were set after viewing the digitised figures, and amendment 13 applies them to every scanned figure; this is stated, not hidden, and no prediction was involved. Every extraction was run from a committed script on a clean tree, with the SHA256 of each input recorded.

---

## Notes for the maintainer (not part of the text)

- Numbers are now marked: DESIGN rows `na12_gate_area`, `na12_gate_sym_px`, `na12_gate_click_K`, `na12_column_gap_K`, `na12_edge_factor`, `na12_room_temperature_K`, `na12_grid_step_K`, `na12_grid_max_K` are in
  `docs/number_registry.csv` and read from the code constants; "7 papers" and "13 amendments" are GENERATED (`na12_values()` in `scripts/thesis_values.py`, from the PDF README table and the amendment headings).
  Existing keys reused: `clean_bin_width`, `t_min`, `t_max`, `grp_dopant_pct`. "Five repeats of five folds" and "twelve" style number words are not audited as numbers; the Alleno values are not restated.
- References added and verified against Crossref (`scripts/verify_added_citations.py`, positions 75 to 79): `sebastialuna2022`, `singh2016` (year 2016: the Crossref record is December 2016, issue 97(6)), `shibasaki2009`,
  `rajasekaran2020`, `ohtaki1995`. WebPlotDigitizer (version 5, online, used October 2026) is in `refs.bib` as `rohatgiwebplotdigitizer` (a `@misc`, year 2026 = the year it was used; no DOI, so it is not in the Crossref verification).
- The test is not blind, and the thresholds of 10 to 12 were set after viewing the figures: both stay in the text (the second in the amendment paragraph).
