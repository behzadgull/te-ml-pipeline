# Source PDFs for the number audit

Please download the PDFs below into this folder, named `<key>.pdf` (the key is the BibTeX key in `paper/refs.bib`); the two papers that appear in both lists, Flahaut 2006 and Ohta 2005, are one file each, `flahaut2006.pdf` and `ohta2005.pdf`. The `pdf_file` column of `reports/literature_value_verification.csv` gives the file of each audit row. **The PDFs are not committed**: `.gitignore` excludes everything in this folder except this README (the repository is public and these are publisher files). They are needed because
the tool used for the audit could not read these publishers; each is listed with the numbers that must be checked against it
(`reports/literature_value_verification.csv`, rows L01 to L25 (L24 and L25 need no PDF)). The text will be rephrased without the number, or the number removed, for any value
the paper does not support.

| File name | DOI | Numbers to check |
|---|---|---|
| `forman2016estimating.pdf` | 10.1016/j.rser.2015.12.192 | 72% of primary energy lost |
| `biswas2012performance.pdf` | 10.1038/nature11439 | PbTe zT about 2 |
| `poudel2008thermoelectric.pdf` | 10.1126/science.1156446 | Bi2Te3 zT 1.0 to 1.4 |
| `bell2008cooling.pdf` | 10.1126/science.1158899 | same |
| `lee2017ultralow.pdf` | 10.1073/pnas.1711744114 | halide perovskite thermal conductivity below 0.5 W/mK |
| `xie2020all.pdf` | 10.1021/jacs.0c03427 | same |
| `stoumpos2013semiconducting.pdf` | 10.1021/ic401215x | same; Sn2+ oxidation |
| `flahaut2006.pdf` | 10.1109/ict.2006.331291 | CaMnO3: zT 0.2 at 1000 K, S about -350 uV/K, zT below 0.1 |
| `androulakis2004coo.pdf` | 10.1063/1.1647686 | published letter: zT 0.18 (read in the preprint); S 200 to 600 uV/K |
| `ohta2005.pdf` | 10.1063/1.1847723 | SrTiO3: S about -700 uV/K at 300 K, kappa 10 to 12 W/mK, zT below 0.01 |
| `parse2024predicting.pdf` | 10.1002/adts.202400308 | 18,126 instances; 2,761 compounds; R2 0.815; 5-fold CV |
| `jia2024dealing.pdf` | 10.1007/s40843-023-2777-2 | R2 0.89 to 0.90; 92,000 entries |
| `sun2025rationally.pdf` | 10.1002/aelm.202500210 | R2 0.95 (training) and 0.90 (test) |
| `barua2025thermoelectric.pdf` | 10.1021/acsami.4c19149 | about 160,000 points; R2 0.67 to 0.80 on three test sets |
| `wang2025highperformance.pdf` | 10.1016/j.matdes.2024.113552 | rows, cross-validation, R2 0.97 (page or table) |
| `elavunkel2025unlocking.pdf` | 10.1021/acsaem.5c02223 | R2 0.92 to 0.99; split |
| `na2022public.pdf` | 10.1038/s41524-022-00897-2 | 5,205 observations; 880 materials |
| `jain2013commentary.pdf` | 10.1063/1.4812323 | number of compounds |
| `katsura2019datadriven.pdf` | 10.1080/14686996.2019.1603885 | digitization uncertainty 2 to 5% |
| `snyder2008complex.pdf` | 10.1038/nmat2090 | optimum carrier concentration; zT bound |
| `zhao2014ultralow.pdf` | 10.1038/nature13184 | zT about 2.6 |
| `leys2013detecting.pdf` | 10.1016/j.jesp.2013.03.013 | recommended multiplier; the 1.4826 constant |
| `rowe2018thermoelectrics.pdf` (book pages) | 10.1201/9781420038903 | Seebeck 1821 |
| `goldsmid2010introduction.pdf` (book pages) | 10.1007/978-3-642-00716-3 | same |
| `borlido2019large.pdf` | see refs.bib | PBE band gap underestimation (qualitative; check the claim) |

Already read from open versions (no download needed): Ma & Poon (arXiv), Ward et al. (arXiv), Bartel et al. (Europe PMC), Androulakis et al. (arXiv preprint).

## NA12 literature validation (pre-registration B, `docs/decisions.md`, 2026-10-05)

Papers selected before any out-of-fold prediction for these compounds was looked at. Every DOI below was matched to the title, first author and journal by Crossref on 2026-10-05.
Values are read from the PDFs only (page and figure or table) into `docs/na12_literature_values.csv`; the comparison (`scripts/na12_literature_comparison.py`) is run after that file is
filled and committed. Name each file `<first author><year>.pdf`.

| File name | DOI | Compound(s) to read | Notes |
|---|---|---|---|
| `sebastialuna2022.pdf` | 10.1021/acsaem.2c01936 | CsSnI3 (vacuum-deposited films) | S, sigma, kappa as given; open access (PMC9400028) |
| `singh2016.pdf` | 10.1080/14786435.2016.1263404 | LaCoO3 | open version arXiv:1606.01539 |
| `shibasaki2009.pdf` | 10.1007/s11664-009-0666-x | LaRhO3 (undoped member of the B-site series) | related open preprint arXiv:0712.1626 (LaRh1-xNixO3); read the undoped member only if the paper tabulates it |
| `rajasekaran2020.pdf` | 10.1039/d0ce00702a | La-doped BaSnO3 (undoped BaSnO3 if reported) | the BaSnO3 literature found is for La-doped samples; compositions as written in the paper |
| `ohta2005.pdf` | 10.1063/1.1847723 | La- and Nb-doped SrTiO3 single crystals | S, sigma (and kappa if given) against temperature; the same single file as in the first table (audit row L07); download it once |
| `ohtaki1995.pdf` | 10.1006/jssc.1995.1384 | (Ca0.9M0.1)MnO3, M = Y, La, Ce, Sm, In, Sn, Sb, Pb, Bi | doped CaMnO3; all listed M are in the candidate set, see the rule below |
| `flahaut2006.pdf` | 10.1109/ict.2006.331291 | Yb-substituted CaMnO3 | the same single file as in the first table (audit row L05); download it once |

Selection rule for the values (set now, before any prediction was looked at): every value of S, sigma (or resistivity), kappa and zT that a paper tabulates or states in the text at a temperature
from 300 K up to but not including 800 K, for every composition of the paper that matches one of the compounds above (dopants included); where a property is only plotted, it is digitised at
300, 400, 500, 600 and 700 K (the 100 K grid of the screening) wherever the plotted curve covers that temperature, and each such value is marked `digitised` with the tool and the axis
calibration. A value at exactly 800 K is recorded but not compared (the cleaning pipeline's 25 K bins end at 800 K, exclusive).
