# NA12: inventory of the source PDFs (2026-10-13)

Structure only. No value was read for NA12 and no prediction was looked at. The comparison rules are those of `docs/decisions.md`, pre-registration B (2026-10-05).
"Vector" means the plotted curves are path operators in the PDF page and can be extracted without image digitising; "raster" means a bitmap that has to be digitised by hand.
The count of path operators and embedded images per page was taken from the PDF files themselves (`scripts` are not involved; the numbers are descriptive, not cited).

| # | Paper | File in `source/pdfs/` | Published version? | Properties given | Compositions | Form |
|---|---|---|---|---|---|---|
| 1 | Sebastia-Luna et al. 2022 | `sebastialuna2022.pdf` | Yes: ACS Appl. Energy Mater. 5:10216-10223, DOI 10.1021/acsaem.2c01936 (8 pages) | sigma, kappa, S and ZT against temperature (Fig. 2a for B-gamma-CsSnI3; Fig. 3d for Cs2SnI6); Hall coefficient, carrier concentration and mobility (Fig. 2b) | B-gamma-CsSnI3 (the NA12 compound); Cs2SnI6 and Cs2SnI4 are also reported (the latter has only a conductivity) | Table 2 (text numbers): the maximum of each property per phase with its temperature; abstract: a ZT range. Curves: vector (many thousand path operators on PDF pages 3 and 4, with small embedded images), error bars drawn. Supporting Information not in the file |
| 2 | Singh & Pandey 2017 | `singh2016.pdf` | Yes: Philosophical Magazine 97(6):451-463, DOI 10.1080/14786435.2016.1263404 (T&F cover and typeset pages) | **Experimental S(T) only** (Fig. 1, 300 to 600 K, a polycrystalline sample prepared by solution combustion) and one S value at 300 K in the text. Everything else (S against chemical potential, calculated S(T), power factor, ZT) is calculated by DFT and Boltzmann transport, not measured | LaCoO3 only | Fig. 1: vector (PDF page 5); the 300 K value is a text number. The calculated curves (Figs. 4 to 7) are not experimental data and are outside the NA12 rule |
| 3 | Shibasaki, Takahashi & Terasaki 2009 | `shibasaki2009.pdf` (full text, supplied 2026-10-19; SHA256 `b7991a80577f3c264addc3ad687ea6cb13b259b87eb01beed0ab7cd0bade7a31`) | Yes: J. Electron. Mater. 38(7):1013-1016 (2009), Special Issue Paper, DOI 10.1007/s11664-009-0666-x (4 pages = article pages 1013 to 1016, with the references; received 14 July 2008, accepted 8 January 2009) | Resistivity and S (Figs. 2 and 4: temperature; Fig. 5: doping dependence at 300 K, with the power factor); no kappa, no zT; no data table | LaRh1-xNixO3 (x = 0, 0.03, 0.05, 0.1, 0.2, 0.3) and LaRh1-xMgxO3 (x = 0, 0.03, 0.05, 0.1, 0.2); the undoped LaRhO3 is the x = 0 series | Figs. 2 and 4: vector (about 27,000 and 15,000 path operators on PDF pages 2 and 3), dense filled markers (about 70 to 190 per series), no connecting curve; Fig. 5: vector, 11 markers per panel at 300 K; Fig. 3 (low-temperature variable-range-hopping replot) is outside the window |
| 4 | Ohta et al. 2005 | `ohta2005.pdf` | Yes: J. Appl. Phys. 97:034106 (2005), DOI 10.1063/1.1847723 (AIP page and typeset pages) | sigma (Fig. 2), S (Fig. 1), kappa (Fig. 3), ZT (Fig. 4), 300 to 1050 K; Table I (doping levels); Table II (room-temperature sigma, n, mobility, S, power factor, kappa, Z and ZT) | Four heavily doped crystals: two La-doped, two Nb-doped SrTiO3. **No undoped SrTiO3 data** | Table II: text numbers at room temperature only. Figs. 1 to 4: raster (bitmaps about 745 pixels wide on PDF pages 3 and 4), manual digitising |
| 5 | Ohtaki et al. 1995 | `ohtaki1995.pdf` | Yes, as a scan: J. Solid State Chem. 120:105-111 (1995), pages 105 to 111, DOI not printed on the page (10.1006/jssc.1995.1384 per `source/pdfs/README.md`). Seven pages, 1-bit CCITT bitmaps of about 1730 x 2160 pixels, **no text layer** | sigma (log sigma against 1/T, Fig. 1; log sigma T, Fig. 2; sigma at room temperature against the ionic radius, Fig. 3), S (Fig. 4), power factor (Fig. 5), kappa (Fig. 6), Z and ZT (Fig. 7); temperatures in degrees Celsius, to about 1000 C | (Ca0.9M0.1)MnO3, M = Y, La, Ce, Sm, In, Sn, Sb, Pb, Bi, and undoped CaMnO3 (sigma and S); kappa only for the Bi member and CaMnO3; Z and ZT only for the Bi member | No data tables. A few numbers in the abstract and text (the Bi member's maxima). Curves: raster scan, manual digitising; small markers, several overlapping series |
| 6 | Flahaut et al. 2006 | `flahaut2006.pdf` | Yes: 2006 25th Int. Conf. on Thermoelectrics (ICT), pp. 103-106, IEEE copy, DOI 10.1109/ICT.2006.331291 (4 pages) | Resistivity (Fig. 2), S (Fig. 3), kappa (Fig. 4), Z (Fig. 5); zT only as a text number at 1000 K | Ca1-xYbxMnO3, x = 0 to 0.5 (S and resistivity), x = 0, 0.05, 0.1, 0.15 (kappa and Z); Table 1 gives nominal and measured composition, not properties | Figs. 2 to 5: vector (thousands of path operators on PDF pages 2 and 3). A few text numbers (best ZT, S of CaMnO3, the lowest resistivity). Flahaut is a training source (Starrydata2), so its comparison restates CV error |
| 7 | Rajasekaran et al. 2020 | `rajasekaran2020.pdf` (supplied 2026-10-19; SHA256 `e6c902b095a36a4f85615628ddbe612229f7bb0bcc7f5e8b264a22097e50e60b`) | Yes: CrystEngComm 22(32):5363-5374 (2020), DOI 10.1039/d0ce00702a (12 pages = article pages 5363 to 5374; received 11 May 2020, accepted 13 July 2020) | Fig. 10: sigma, S and the power factor against temperature; Table 2: n, mobility, sigma, S, PF per sample | La-doped BaSnO3: BSOLa5 and BSOLa10 (Fig. 10 and Table 2); undoped BSO has a Table 2 row with no sigma or S entry | Fig. 10: raster (a CMYK JPEG of 1657 x 1245 pixels, three panels, filled square and circle markers joined by a line); Figs. 1 to 9 are not thermoelectric data |

## Other files present in `source/pdfs/`

- `013-High-temperaturethermoelectricpropertiesofW-substitutedCaMnO3.pdf`: Alfaruq et al., Mater. Res. Soc. Symp. Proc. 1490 (2013), DOI 10.1557/opl.2013.85. It is **not an NA12 source** (the seven NA12 papers were fixed in pre-registration B, 2026-10-05) and is not in `paper/refs.bib`.
- `snyder2008complex.pdf`: Snyder & Toberer 2008, Nature Materials 7:105-114 (published). Read for the carrier-concentration claim (ledger L20).
- `iltanen2011_notes.pdf`: lecture notes of K. Iltanen (2011), not Rowe's handbook (ledger L23).

## File names

Renamed on 2026-10-13 to the names `source/pdfs/README.md` asks for: `flahaut2006.pdf`, `ohta2005.pdf`, `ohtaki1995.pdf`, `snyder2008complex.pdf`.

## Inventory of the last two papers (2026-10-19)

Structure only, from the PDF text layer (captions, legends, axis labels, tick positions) and the path operators and images of the pages. No value was recorded for NA12 and no out-of-fold prediction was looked at.
**Disclosure:** the first pass printed running text, and with it a few numbers came into view that were not meant to be read: in Shibasaki et al., resistivity and S statements at 300 K in the discussion and a mobility;
in Rajasekaran et al., the n, mobility, sigma, S and power-factor entries of the BSOLa5 and BSOLa10 rows of Table 2. The rendered Fig. 10 was also looked at, to judge its legibility. Nothing was recorded; the values are read
again under the rules at extraction. No prediction was involved, so pre-registration B is not touched, but the instruction "read no values" was not met to the letter.

**Files.** Both were supplied as `*.pdf.pdf` and renamed to `rajasekaran2020.pdf` and `shibasaki2009.pdf`; the one-page `shibasaki2009.pdf` of 2026-10-11 is kept as `shibasaki2009_onepage_old.pdf` (SHA256 `caeada5463daa9a25a97213ef480b598a90ea49a864fc204719999740db62112`, not used).
Both are the published versions of the right papers: title, journal, volume, pages and DOI match the Crossref records checked in `docs/decisions.md` (2026-10-05) and `refs.bib`.

### Shibasaki et al. 2009 (LaRhO3 and its Ni- and Mg-substituted members)

- **Measured range.** The text gives 4.2 K to 300 K. The axes of Figs. 2 and 4 run to about 310 K; the last drawn marker of a series lies between 297.9 and 306.6 K (tick-calibration residual 0.06 K). Nothing is drawn beyond 307 K.
  The window 300 <= T < 800 K therefore holds only the 300 K end of each series. By drawn position, series with at least one marker at or above 300 K: Fig. 2a: x = 0.0, 0.1 and 0.3; Fig. 2b: every series; Fig. 4a: every series but
  one (the red); Fig. 4b: every series (one or two markers each). By the caption of Fig. 5: one value per composition at 300 K.
- **Figures.** Fig. 1: X-ray patterns and lattice constants (not a property). Fig. 2: resistivity (a, logarithmic axis) and S (b) of LaRh1-xNixO3, six series. Fig. 3: ln rho and S replotted for the variable-range-hopping test
  (axes in (T/K) to the powers -1/4 and 1/2, so below about 64 K): outside the window. Fig. 4: resistivity (a), S (b) and, in an inset, the Hall coefficient (not a property) of LaRh1-xMgxO3, five series.
  Fig. 5: Ni- and Mg-content dependence of the conductivity (a), S (b) and the power factor (c) at 300 K.
- **Form.** All five figures are vector. Markers are filled paths; the series are told apart by colour (Fig. 2 legend: black x = 0.0, blue 0.03, red 0.05, green 0.1, cyan 0.2, white 0.3; Fig. 4 shares the scheme, to be confirmed on the
  image). Figs. 2 and 4 have 70 to 190 markers per series at a spacing of a few kelvin, and no line through them (Fig. 3 has guide lines). Overlap is no problem for vector paths; the image-derived gates (amendment 13) do not apply.
- **Properties.** Resistivity (converted to sigma) and S; no kappa, no zT. The power factor of Fig. 5 is derived (S^2/rho) and is not one of the four properties.
- **Compositions and clusters.** x = 0 is LaRhO3. For x <= 0.2 the substituent is below 5 at% (x = 0.2: 4 at%), so these members belong to the LaRhO3 chemistry cluster and could be compared at the cluster level; Ni at x = 0.3 (6 at%) is a
  cluster of its own and matches none of the pre-registered compounds.
- **Pre-registration B.1: LaRhO3.** Values exist. The undoped member (x = 0.0) is plotted in Fig. 2 (resistivity and S), in Fig. 4 (again, with a different number of markers) and in Fig. 5 (one value at 300 K per property);
  inside the window that is its 300 K end only.
- **Extraction.** Vector, so the proposal is to extract it as for Sebastia-Luna (`na12_extract_shibasaki.py`, `pdf_vector.py`, axis calibration on the printed ticks, series by the legend's symbol colours, SHA256 of the PDF, clean tree).
- **Decisions needed before it is written:** (1) undoped member only, or also the x <= 0.2 members at the cluster level (the pre-registration text says both: the README row "undoped member" and the selection rule "every composition of
  the paper that matches a compound, dopants included"); (2) rows from Fig. 5 (stated 300 K), with the end markers of Figs. 2 and 4 as a reported cross-check, or the end markers themselves as rows; (3) whether the x = 0 series of Figs. 2
  and 4 is one sample.

### Rajasekaran et al. 2020 (La-doped BaSnO3)

- **Figures.** Figs. 1 to 9 are X-ray, microscopy, XPS, band-structure and schematic images (no thermoelectric data). Fig. 10: (a) electrical conductivity, (b) Seebeck coefficient and (c) power factor against temperature, 300 to 600 K,
  for BSOLa5 and BSOLa10. Table 2: sample name, oxygen-vacancy fraction, crystal phase, n, mobility, sigma, S and PF (one entry per sample; its temperature is not in the table header and has to be read from the text; the PF entry of
  BSOLa5 looks like the last marker of Fig. 10c, so Table 2 probably holds the highest-temperature values, to be confirmed).
- **Form.** Fig. 10 is raster: one CMYK JPEG (Adobe, inverted) of 1657 x 1245 pixels. The raw JPEG shows a black background and the red series disappears in a naive conversion, so a bitmap must come from the PDF page or from a corrected
  conversion; a corrected PNG (SHA256 `fe404f61e4c6a23e74039c12bb28b21cbc256be8deaf1ae1b5e0037de134f9ee`) is in `C:\Users\choha\na12_digitising\rajasekaran2020\`. Each panel has two series of six markers joined by a line (BSOLa5 black squares, BSOLa10 red circles) at about 323, 373,
  423, 473, 523 and 573 K (50 K steps). Legibility: all markers are visible; in (a) and (c) the two series touch at about 423 to 473 K, but each marker has its own centre. Fig. 10(c) is derived (S^2 sigma), a check only.
- **Window.** All six temperatures of each series are inside 300 <= T < 800 K (bins 300, 350, 400, 450, 500, 550). The first marker (about 323 K) is within about 2 K of the 325 K bin edge, so it is an edge row under amendment 6.
- **Properties.** sigma and S; no kappa, no zT. Table 2 holds no sigma or S for undoped BSO.
- **Pre-registration B.1: BaSnO3.** La-doped BaSnO3 values exist (BSOLa5 and BSOLa10, at the cluster level: La is below 5 at% in both); undoped BaSnO3 has no sigma or S in Fig. 10 or Table 2 (its row is empty), so it contributes no
  values. The electronic supplementary information is announced but is not in the file.
- **Extraction.** Raster: by hand in WebPlotDigitizer, one panel at a time, Fig. 10(a) first, then (b); (c) is not read.
