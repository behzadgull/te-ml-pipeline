# Supplement

## Supplementary Note [[snote:background]]: Extended background (supplement to Section 1)

### Thermoelectric fundamentals

The Seebeck effect, named after Thomas Johann Seebeck, is the creation of an electric voltage by a temperature difference across a junction of two dissimilar conductors. The magnitude of this voltage per unit temperature difference is the Seebeck coefficient (S, *µ*V K^−1^). A thermoelectric generator directly converts a temperature gradient into electrical power, without moving parts, using an array of p-type and n-type semiconductor legs connected electrically in series and thermally in parallel (Supplementary Figure [[sfig:thermo]]) [@rowe2018thermoelectrics]. The reverse phenomenon, the Peltier effect, uses an electric current to transfer heat across a junction, enabling solid-state cooling, and the Thomson effect is the heating or cooling of a current-carrying conductor in a temperature gradient. Together, these three effects form the basis of thermoelectric energy conversion [@goldsmid2010introduction].

![Supplementary Figure [[sfig:thermo]]](../figures/fig1_thermocouple.png){width=16cm}

**Supplementary Figure [[sfig:thermo]]:** Schematic (drawn by the authors, not to scale) of a thermoelectric p--n couple. Heat flows from the hot to the cold side; the majority carriers of each leg (holes in the p-type leg, electrons in the n-type leg) diffuse from the hot to the cold end, which gives rise to an open-circuit Seebeck voltage *V*~oc~. The two legs are joined in series by electrodes and drive a current *I* (conventional direction) through the load *R*~L~.

In Eq. (1), the numerator S^2^*σ* is the power factor (PF), a measure of electrical power output, whereas *κ* comprises an electronic part (*κ*~e~) controlled by charge carriers and a lattice part (*κ*~l~) controlled by phonons. High zT requires S^2^*σ* to be maximized and *κ* minimized simultaneously, which is inherently difficult because these quantities are interrelated through the carrier concentration (Supplementary Figure [[sfig:carrier]]) [@snyder2008complex]. As carrier concentration increases, *σ* increases while S decreases, because the average entropy per carrier decreases. In addition, the Wiedemann--Franz law links *κ*~e~ to *σ*:

$\kappa_{e} = L\sigma T$ (S1)

where L is the Lorenz number. The only transport parameter that can be reduced relatively independently is *κ*~l~, which has driven research toward materials with complex crystal structures, heavy atoms and large unit cells that scatter phonons efficiently.

![Supplementary Figure [[sfig:carrier]]](../figures/fig2_carrier_concentration.png){width=16cm}

**Supplementary Figure [[sfig:carrier]]:** Schematic (drawn by the authors, arbitrary units) of the dependence of the Seebeck coefficient S, the electrical conductivity *σ*, the thermal conductivity *κ* (electronic *κ*~e~ and lattice *κ*~l~) and the figure of merit zT on the carrier concentration *n* (logarithmic axis), drawn from a simple model of the dependence described by Snyder and Toberer [@snyder2008complex] (S falls with *n*, *σ* is proportional to *n*, *κ*~e~ follows *σ* by the Wiedemann--Franz law, *κ*~l~ is constant), not from data. The optimum range, <!--c:snyder2008_n_lo-->10^19^<!--/c--> to <!--c:snyder2008_n_hi-->10^21^<!--/c--> cm^−3^, is taken from that paper (page <!--c:snyder2008_page-->105<!--/c-->); the boundaries between insulator, semiconductor and metal are schematic.

Several strategies approach the PGEC ideal [@slack1995materials]. Band convergence aligns multiple band extrema to enhance the density-of-states effective mass and hence S without reducing mobility [@pei2011convergence]. Nanostructuring introduces grain boundaries that scatter long-wavelength phonons more effectively than charge carriers, reducing *κ*~l~ while preserving *σ* [@biswas2012performance]. Doping or alloying introduces mass and strain fluctuations that scatter phonons over a range of frequencies. Interest has also grown in materials with intrinsic bond anharmonicity: lone-pair electrons (SnSe, BiCuSeO), rattling cations in oversized cages (skutterudites, clathrates) and soft lattice modes (halide perovskites) produce strong anharmonic phonon--phonon interactions and intrinsically low *κ*~l~ without complex processing [@zhao2014ultralow].

### Perovskite thermoelectric materials

In ABX~3~ perovskites, A is a larger cation (e.g. Sr, Ca, Ba, Cs), B a smaller cation (e.g. Ti, Mn, Co, Sn) and X an anion (O, Cl, Br, I, S, Se, Te). The ideal cubic structure (space group *Pm*3̄*m*) consists of corner-sharing BX~6~ octahedra with the A cation in the cuboctahedral void (Fig. [[fig:perov]]). Structural stability is conventionally assessed by the Goldschmidt tolerance factor t and the octahedral factor *µ* [@goldschmidt1926die]:

$t = \frac{r_{A} + r_{X}}{\sqrt{2}\left( r_{B} + r_{X} \right)},\quad\quad µ = \frac{r_{B}}{r_{X}}$ (S2)

where r~A~, r~B~ and r~X~ are the ionic radii of the A-, B- and X-site ions. The tolerance factor is an imperfect classifier of perovskite formation: Bartel et al. [@bartel2019tolerance] proposed a revised tolerance factor *τ* that accounts for oxidation state and classifies both oxides and halides more accurately than t. While tolerance factors provide useful first-pass estimates, the Materials Project already provides DFT-computed formation energies and energies above the convex hull, which are more reliable stability indicators; this study therefore uses Materials Project stability data directly rather than geometric criteria.

*Oxide perovskites.* Oxide perovskites have been studied as thermoelectrics owing to their chemical stability at high temperature and the abundance of their constituent elements. Heavily La- or Nb-doped SrTiO~3~ single crystals are n-type, with Seebeck coefficients of <!--c:ohta2005_S_lo-->−420<!--/c--> to <!--c:ohta2005_S_hi-->−150<!--/c--> *µ*V K^−1^ and a thermal conductivity of <!--c:ohta2005_k_lo-->9.1<!--/c--> to <!--c:ohta2005_k_hi-->12<!--/c--> W m^−1^ K^−1^ at room temperature, similar to that of undoped SrTiO~3~ and falling as *T*^−1^ (phonon conduction); their zT is <!--c:ohta2005_zT_lo-->0.02<!--/c--> to <!--c:ohta2005_zT_hi-->0.08<!--/c--> at room temperature and reaches <!--c:ohta2005_zT_max-->0.27<!--/c--> at <!--c:ohta2005_zT_T-->1073<!--/c--> K [@ohta2005temperature]. CaMnO~3~ has a Seebeck coefficient of about <!--c:flahaut2006_S-->−350<!--/c--> *µ*V K^−1^ but its resistivity is too high; substituting Yb for part of the Ca (Ca~0.95~Yb~0.05~MnO~3~) raised zT at <!--c:flahaut2006_T-->1000<!--/c--> K by a factor of <!--c:flahaut2006_factor-->4<!--/c--> to <!--c:flahaut2006_zT-->0.2<!--/c--> [@flahaut2006effect]. LaCoO~3~ and its Sr-doped derivatives are p-type, with large Seebeck coefficients but modest zT [@androulakis2004coo].

*Halide perovskites.* Following their success in photovoltaics, halide perovskites have entered the thermoelectric field. CsSnI~3~ and related all-inorganic halides exhibit lattice thermal conductivity below <!--c:halide_kappa_ub-->0.5<!--/c--> W m^−1^ K^−1^ arising from soft, anharmonic vibrations and cation rattling in the halide cage [@lee2017ultralow; @xie2020all], but experimental validation of high zT has been limited by the instability of tin-based compounds under ambient conditions [@stoumpos2013semiconducting]. In hybrid organic--inorganic perovskites such as MASnI~3~, the rotational freedom of the organic cation further complicates transport and is difficult to encode in compositional feature vectors.

*Chalcogenides.* Chalcogenide perovskites (e.g. BaZrS~3~, BaZrSe~3~) and related ternary chalcogenides are attractive lead-free targets because they combine earth-abundant elements with heavy anions and low thermal conductivity [@sopiha2021chalcogenide].

### Computational methods and materials databases

The conventional computational route to thermoelectric properties begins with a DFT band structure from codes such as VASP [@kresse1996efficient] or Quantum ESPRESSO [@giannozzi2009quantum]. Electronic transport coefficients (S, *σ*/*τ*, *κ*~e~/*τ*) are then obtained by solving the BTE within the constant relaxation-time approximation, commonly with BoltzTraP [@madsen2006boltztrap]; because *τ* is assumed constant, absolute *σ* and *κ*~e~ require additional experimental input or separate calculations. Lattice thermal conductivity requires interatomic force constants from supercell calculations and solution of the phonon BTE, e.g. with ShengBTE [@li2014shengbte]. A complete characterization (S, *σ*, *κ*~e~, *κ*~l~ at several temperatures) of one compound can take days to weeks, motivating ML surrogate models.

*Starrydata2* is the largest publicly available database of experimental thermoelectric measurements, assembled by systematic digitization of published figures; each data point is linked to a paper DOI and sample ID [@katsura2019datadriven; @katsura2025starrydata]. Known limitations include digitization uncertainty (a precision of <!--c:katsura2025_prec-->0.3<!--/c-->% of the graph width for semi-automatic extraction, with occasional errors above <!--c:katsura2025_err-->10<!--/c-->% of the original value for points near zero on linear axes [@katsura2025starrydata]), inter-laboratory variability for the same nominal composition and occasional unlabeled computational data. The *Materials Project* provides DFT-computed crystal structures, formation energies, energies above the convex hull and band gaps for inorganic compounds via a REST API [@jain2013commentary], but transport coefficients are not available for most entries. The ESTM (Experimentally Synthesized Thermoelectric Materials) database curated at the Korea Research Institute of Chemical Technology contains <!--c:na2022_obs-->5,205<!--/c--> experimental observations for <!--c:na2022_materials-->880<!--/c--> unique materials [@na2022public]. ESTM was curated independently of Starrydata2, making it a valuable external validation resource. JARVIS, maintained by NIST, contains BoltzTraP-computed transport properties for thousands of compounds [@choudhary2020joint]; these differ systematically from experiment because of the constant relaxation-time approximation and ideal-crystal assumptions.

### Machine learning for thermoelectric property prediction

ML approaches are divided by input representation. Structure-based approaches use atomic coordinates, lattice vectors and site symmetry, e.g. crystal graph neural networks (CGCNN) [@xie2018crystal]. Composition-based methods use only the chemical formula, from which statistical summaries of elemental properties are computed; they are more broadly applicable because they do not require the crystal structure, which is often unknown for hypothetical materials. Tree ensembles (XGBoost, LightGBM, gradient-boosted decision trees) dominate composition-based thermoelectric prediction because they handle tabular features, are robust to missing values and are compatible with SHAP; deep neural networks and transformer-based models (TabPFN) have been used with mixed results. Table [[tbl:lit]] summarizes recent studies.

Parse et al. [@parse2024predicting] trained XGBoost on <!--c:parse2024_rows-->18,126<!--/c--> cleaned Starrydata2 instances covering <!--c:parse2024_compounds-->2,761<!--/c--> compounds, split compound by compound into <!--c:parse2024_train_pct-->80<!--/c-->% training and <!--c:parse2024_test_pct-->20<!--/c-->% test data (R^2^ = <!--c:parse2024_r2-->0.815<!--/c--> on the test set); a <!--c:parse2024_folds-->5<!--/c-->-fold cross-validation over the whole dataset was added. Jia et al. [@jia2024dealing] introduced composition-based cross-validation, excluding individual compositions rather than random rows, and obtained R^2^ of <!--c:jia2024_r2_cv-->0.89<!--/c--> under <!--c:jia2024_folds-->10<!--/c-->-fold composition-based CV and <!--c:jia2024_r2_test-->0.90<!--/c--> on a test set, for zT on <!--c:jia2024_rows-->92,291<!--/c--> data points from <!--c:jia2024_compositions-->7,295<!--/c--> compositions; this is stricter than standard K-fold but less strict than system-level grouping, because members of the same chemical family (e.g. different doping levels of Bi~2~Te~3~) can still appear in both sets. Sun et al. [@sun2025rationally] combined MAGPIE and CBFV features, selected with Pearson correlation and LassoCV, in deep neural networks (R^2^ = <!--c:sun2025_r2_train-->0.95<!--/c--> on the training set and <!--c:sun2025_r2_test-->0.90<!--/c--> on the test set), the dual-featurizer approach used here. Ma and Poon [@ma2025reexamining] added pair-interaction descriptors (Miedema mixing enthalpy) and dopant properties, reporting modest improvements. Barua et al. [@barua2025thermoelectric] trained XGBoost on ≈<!--c:barua2025_rows-->160,000<!--/c--> data points and reported R^2^ between <!--c:barua2025_r2_lo-->0.67<!--/c--> and <!--c:barua2025_r2_hi-->0.80<!--/c--> on three test sets. Stacking ensembles on small, domain-restricted datasets [@wang2025highperformance; @elavunkel2025unlocking] reported R^2^ of <!--c:elavunkel2025_r2_zT-->0.92<!--/c-->--<!--c:elavunkel2025_r2_S-->0.99<!--/c-->. Wang et al. used <!--c:wang2025_folds-->10<!--/c-->-fold cross-validation over the whole dataset, a row-level protocol of the kind whose inflation relative to chemistry-aware splitting this study quantifies; the split of Elavunkel and Padhan is not stated in its abstract and was not checked.

MAGPIE [@ward2016general] computes six statistics (minimum, maximum, range, mean, average deviation, mode) of <!--c:ward2016_props-->22<!--/c--> elemental properties, yielding <!--v:n_magpie-->132<!--/v--> features per composition. CBFV extends this with the Oliynyk elemental property set [@oliynyk2016highthroughput], including polarizability, Gordy and Mulliken electronegativities, Miracle radius, metallic valence and heat of atomization. Physics-informed descriptors such as mass variance (phonon scattering by mass disorder) [@tamura1983isotope], tolerance factors [@goldschmidt1926die; @bartel2019tolerance] and Miedema mixing enthalpy [@ma2025reexamining] have also been explored.

### Validation strategies and data leakage

Random train--test splitting and K-fold cross-validation assume independent and identically distributed (i.i.d.) samples; when this holds, K-fold CV provides a nearly unbiased estimate of generalization error [@hastie2009elements]. Data leakage from improper validation is now recognized as a widespread cause of over-optimistic results across ML-based science [@kapoor2023leakage]. Thermoelectric datasets violate the i.i.d. assumption because a material such as Bi~0.5~Sb~1.5~Te~3~ may appear at <!--v:bins-->21<!--/v--> temperatures (<!--v:t_min-->300<!--/v-->, <!--v:t_second-->325<!--/v-->, . . . , <!--v:t_max-->800<!--/v--> K); under random splitting the test prediction becomes interpolation along a known temperature curve rather than extrapolation to new chemistry. Roberts et al. [@roberts2017cross] showed that grouped or blocked CV, which withholds entire groups of correlated observations, is needed for unbiased estimates with structured data. In materials informatics, Meredig et al. [@meredig2018machine] introduced leave-one-cluster-out CV, Xiong et al. [@xiong2020evaluating] showed that standard K-fold overstates the ability to extrapolate, and Li et al. [@li2023critical] demonstrated systematic degradation of materials ML models under distribution shift. chemistry-cluster CV assigns all measurements of one group exclusively to training or test. The grouping can be defined at the level of compositions (as in [@jia2024dealing]), element sets (e.g. Bi--Sb--Te), chemistry clusters (Section 2.4) or broader families; stricter grouping gives more conservative but more realistic estimates for genuinely new chemistry.

### Model interpretability

SHAP (SHapley Additive exPlanations) attributes each prediction to features using Shapley values from cooperative game theory, and TreeExplainer computes exact values in polynomial time for tree ensembles [@lundberg2017unified]. SHAP provides consistent global (mean \|SHAP\|) and local explanations, unlike gain-based importance. LIME fits local linear surrogates around individual predictions [@ribeiro2016why] but does not provide global patterns. In thermoelectrics, interpretability is a tool for scientific validation: if melting temperature dominates predictions of *κ*, this is consistent with known physics (bond strength governs phonon velocity), whereas a dominant but physically irrelevant feature would indicate a spurious correlation.

## Supplementary Note [[snote:cleaning]]: The eleven cleaning steps in detail (supplement to Section 2.1)

*Data preparation (steps 1--5).* Step 1 extracted the four target properties, converted resistivity to conductivity and applied physical-range bounds: <!--v:b_S-->−1000<!--/v--> ≤ S ≤ <!--v:b_S_hi-->1000<!--/v--> *µ*V K^−1^, <!--v:b_sigma_lo-->10<!--/v--> ≤ *σ* ≤ <!--v:b_sigma_hi-->10^7^<!--/v--> S m^−1^, <!--v:b_kappa_lo-->0.05<!--/v--> ≤ *κ* ≤ <!--v:b_kappa_hi-->25<!--/v--> W m^−1^ K^−1^ and <!--v:b_zT_lo-->0<!--/v--> ≤ zT ≤ <!--v:b_zT_hi-->4<!--/v-->, giving <!--v:n_step1-->1,996,047<!--/v--> observations. The Seebeck bound is an empirical cap that excludes gross digitization errors rather than imposing a tight physical limit; the *σ* and *κ* limits correspond approximately to the semiconductor--insulator boundary and to a permissive value below amorphous (Cahill--Pohl) limits [@cahill1992lower], respectively (the Cahill--Pohl limit is material-specific, so the *κ* bound is not itself that minimum); and the zT bound of <!--d:clean_zT_hi-->4<!--/d--> is a choice of this study, set above the zT of <!--c:zhao2014_zT-->2.6<!--/c--> ± <!--c:zhao2014_zT_err-->0.3<!--/c--> reported at <!--c:zhao2014_T-->923<!--/c--> K for SnSe single crystals along the b axis [@zhao2014ultralow]; it is not a physical limit. Step 2 merged curve data with sample metadata. Step 3 restricted temperatures to <!--d:clean_t_min-->300<!--/d-->--<!--d:clean_t_max-->800<!--/d--> K and binned them into <!--d:clean_bin_width-->25<!--/d--> K intervals so that slightly different reported temperatures (e.g. <!--v:ex_temps-->300, 310 and 320<!--/v--> K) fall in the same bin, leaving <!--v:n_step3-->1,096,324<!--/v--> observations. Step 4 pivoted the data from long format (one row per measurement) to wide format (one row per sample per temperature bin), giving <!--v:n_step4-->398,763<!--/v--> rows, and Step 5 removed formulas that could not be parsed by the pymatgen Composition class [@ong2013python] (<!--v:rem_step5-->5,382<!--/v--> rows, <!--v:remp_step5-->1.3<!--/v-->%).

*Quality filtering (steps 6--8).* Step 6 compared the reported zT with the value recomputed from Eq. (1) for all rows containing the four properties and removed rows whose relative discrepancy exceeded <!--d:clean_zt_tol_pct-->50<!--/d-->%:

$\frac{|{zT}_{rep} - S^{2}\sigma T/\kappa|}{{zT}_{rep}} > 0.5$ (S3)

This removed <!--v:rem_step6-->4,712<!--/v--> rows (<!--v:remp_step6-->1.2<!--/v-->%); it changed the mean zT from <!--v:na5_zt6_before-->0.383<!--/v--> to <!--v:na5_zt6_after-->0.387<!--/v-->. This is conservative (some valid individual property measurements may be lost), but guarantees internal consistency of the retained rows. Step 7 removed DFT-computed entries identified by searching paper metadata for keywords such as "DFT", "first principles", "ab initio" and "VASP", because such values differ systematically from experiment owing to the constant relaxation-time assumption; only <!--v:rem_step7-->986<!--/v--> rows (<!--v:remp_step7-->0.3<!--/v-->%) were flagged, confirming that Starrydata2 is predominantly experimental. Step 8, the largest single step, removed <!--v:rem_step8-->78,685<!--/v--> rows (<!--v:remp_step8-->20.3<!--/v-->%): for each (formula, temperature) group with multiple measurements, the coefficient of variation (CV) was computed and groups exceeding property-specific thresholds were removed (<!--d:clean_cv_S-->0.5<!--/d--> for S, *κ* and zT; <!--d:clean_cv_sigma-->0.8<!--/d--> for *σ*, set higher a priori because conductivity varies more between samples); these are starting values fixed before any model was trained, not values tuned on model performance.

*Statistical filtering (steps 9--11).* Step 9 applied a median absolute deviation (MAD) filter, removing values with

$|x - median(x)| > 3.5\, MAD(x)$ (S4)

a robust alternative to a rule based on the mean and standard deviation [@leys2013detecting], where x is S, log~10~ *σ* or log~10~ *κ*, and the median and the MAD, MAD(x) = median(|x − median(x)|), are taken over all rows that have a value of that property. The MAD is unscaled (no <!--v:mad_const-->1.4826<!--/v--> consistency factor), so the multiplier of <!--d:clean_mad_mult-->3.5<!--/d--> MADs corresponds to about <!--v:mad_sigma-->2.4<!--/v--> standard deviations for normally distributed data; it is a choice of this study (see below), not the multiplier recommended in the cited work. The MAD filter was applied to S, *σ* and *κ* but not to zT, a derived quantity whose plausibility is better judged against the physical bounds of Step 1; it removed <!--v:rem_step9-->19,361<!--/v--> rows (<!--v:remp_step9-->6.3<!--/v-->%). Rows with zT \> <!--v:zt_hi_thr-->3.0<!--/v--> lie above the zT of <!--c:zhao2014_zT-->2.6<!--/c--> ± <!--c:zhao2014_zT_err-->0.3<!--/c--> reported at <!--c:zhao2014_T-->923<!--/c--> K for SnSe single crystals along the b axis [@zhao2014ultralow]; <!--v:na4_zt3-->15<!--/v--> of the <!--v:n_zT-->129,419<!--/v--> zT rows (<!--v:na4_zt3_pct-->0.012<!--/v-->%) are affected; they are retained as statistically immaterial. Step 10 removed formulas with fewer than <!--d:clean_min_temps-->3<!--/d--> distinct temperatures, for which consistency cannot be verified (<!--v:rem_step10-->4,650<!--/v--> rows). Step 11 applied a rolling-median smoothness filter (window = <!--d:clean_smooth_window-->3<!--/d-->) to each temperature curve of each sample and property: a point is a spike if it deviates from the centred rolling median by more than <!--d:clean_mad_mult-->3.5<!--/d--> times the median of those absolute deviations along that curve (the same unscaled multiplier as Step 9, applied to the local deviations), setting <!--v:na5_spikes-->25,166<!--/v--> spike values to NaN and removing rows in which all properties became NaN (<!--v:rem_step11-->4,323<!--/v--> rows, <!--v:remp_step11-->1.5<!--/v-->%).

## Supplementary Note [[snote:protocol]]: Reading the literature values: rules of the comparison (supplement to Section 2.6)

**Reading the literature.** Values are read from the PDFs only, with page and figure or table, and marked as table, text or digitised. A value read from a figure carries the tool (vector extraction from the PDF where the figure is
vector, WebPlotDigitizer, version <!--c:wpd_version-->5<!--/c-->, clicks on the scan otherwise [@rohatgiwebplotdigitizer]) and the axis calibration; no uncertainty is attached to a digitised marker, and an uncertainty is recorded only where the paper prints one. Discrete
markers are read at their own measurement temperature and assigned to the <!--d:clean_bin_width-->25<!--/d--> K floor bins of the cleaning pipeline (a bin is labelled by its lower edge, so the lowest bin is labelled <!--v:t_min-->300<!--/v--> K); curves, and only curves, are read on a <!--d:na12_grid_step_K-->100<!--/d--> K grid (<!--v:t_min-->300<!--/v--> to
<!--d:na12_grid_max_K-->700<!--/d--> K). Values below <!--v:t_min-->300<!--/v--> K (room temperature, <!--d:na12_room_temperature_K-->298.15<!--/d--> K) or at or above <!--v:t_max-->800<!--/v--> K are recorded and not compared. Derived curves are not read (Wiedemann-Franz electronic conductivity, fitted lines, DFT and Boltzmann-transport
results). Units are converted to those of the pipeline (S: µV K^-1^, *σ*: S m^-1^, *κ*: W m^-1^ K^-1^; *zT* as stated, or *Z* multiplied by the temperature). Where S, *σ* (or resistivity), *κ* and *zT* are all read at one
temperature, *zT* is recomputed from the other three and the ratio is reported against the printed value; where the temperatures differ (*κ* of the SrTiO~3~ crystals) an interpolated, labelled approximate check is
made. These checks are reported, never used to exclude or change a value. A value takes the temperature the paper states for it (Shibasaki et al., Fig. 5: <!--v:t_min-->300<!--/v--> K), not the drawn position of its marker. Where the clicked value axis of a digitising project
did not match the printed ticks (Rajasekaran et al., Fig. 10), the value is read at the click's pixel row on the printed ticks and the clicked values are not used; the abstract's printed value is then a check. A table value with no stated temperature
(Rajasekaran et al., Table 2) is not read.

**Marker temperatures in figures with several series.** In a figure whose series were measured at common temperatures, the temperature of a marker is the consensus of its column: the mean of the marker-centre
temperatures of the series, read on a straight line through the printed ticks of the axis (the calibration uncertainty *u* is the largest residual of that line); where several figures report one measurement run, a column has
one temperature, the mean of the figures' consensus weighted by $1/u^{2}$. The marker centre is the centroid of the enclosed white interior of an open symbol or of the opened body of a filled one; a symmetry fit is used
only for symbols with no enclosed interior (hatched, dotted, plus). Where the series do not share temperatures (Ohtaki et al., Figs. 4 and 6), every marker has its own temperature. For an axis of $1000/T$ (Ohtaki et al.,
Fig. 1), $T = 1000/x$, $u(T) = T^{2}/1000\,u(x)$, and *x* is interpolated between the bottom and the top tick lines because the scan is tilted; for Rajasekaran et al., Fig. 10, the small tilt of the frame is corrected
from the frame's own edges.

**Edge sensitivity.** A value whose temperature lies within <!--d:na12_edge_factor-->2<!--/d--> *u* of a bin edge is compared in its floor bin (the headline) and also in the neighbouring bin; both are reported, together with the overall summary
under the neighbouring-bin assignment and the values whose result changes.

**Legibility.** Only markers whose centre can be placed are read. The enclosed interior of an open symbol, or the opened body of a filled one, is searched in two passes (a window centred on the click, then on what the first pass found) so that a
marker is not clipped when the click is off-centre. A marker overlapped by another symbol is recorded as not legible, with its series and column, in a list derived from the digitised files and not typed; the count is reported with the results. A click whose
centre the estimator cannot place is skipped. Three image-derived gates apply, with the same thresholds, to every digitised scanned figure (Ohta et al., Figs. 1, 2, 3 and 4; Ohtaki et al., Figs. 1, 4, 6 and 7; Rajasekaran et al., Fig. 10): a marker is removed if
its enclosed interior is smaller than <!--d:na12_gate_area-->0.8<!--/d--> of the median of its series (interior partly covered), if a left-right symmetry fit computed on its opened body (which a joining line does not bias) and its interior centre are more than
<!--d:na12_gate_sym_px-->10<!--/d--> px apart, or if its image temperature is more than <!--d:na12_gate_click_K-->15<!--/d--> K from the temperature of its click. The columns of a figure for the not-legible list are derived from a reference series (and a
<!--d:na12_column_gap_K-->20<!--/d--> K gap) where the series share columns, and from the clustering of all markers where they do not.

**Paper-specific rules.** For Shibasaki et al., only undoped LaRhO~3~ is compared (the pre-registered compound is the undoped member); the Ni- and Mg-substituted members are recorded and not compared. The rows come from Fig. 5, which states <!--v:t_min-->300<!--/v--> K; the end
markers of Figs. 2 and 4 are a cross-check. Where a Fig. 5 value lies within <!--d:na12_unreadable_pct-->1<!--/d-->% of its axis range it cannot be read, and the same sample's value at that temperature is taken from the figure that resolves it: this applies to the conductivity of undoped LaRhO~3~,
taken from the end markers of the unsubstituted series of Figs. 2 and 4. For Rajasekaran et al., the rows come from Fig. 10, panels (a) and (b) (BaSnO~3~ with La at the Ba site; the undoped sample has no conductivity or Seebeck coefficient in the paper), each row compared at the
cluster level.

## Supplementary Note [[snote:ladder]]: The five validation methods (supplement to Section 3.1.1)

Supplementary Table [[stbl:ladder]] and Supplementary Figure [[sfig:ladder]] compare the five validation methods. Random split, 5-fold and 10-fold CV give practically identical R^2^ (maximum difference <!--v:spread_max-->0.002<!--/v--> per target), showing that all three leak the same temperature-series information and that increasing the number of folds does not address the leakage.

<!-- BEGIN TABLE 6 -->
**Supplementary Table [[stbl:ladder]]:** R^2^ obtained with five validation methods (same model, frozen hyperparameters and 397 features). Random 80/20 pools 20 independent draws; 5-fold and 10-fold are single partitions; composition and chemistry cluster are the mean ± SD across 5 repeats. ∆R^2^ is random 80/20 minus chemistry cluster.

| **Target** | **Random 80/20** | **5-fold** | **10-fold** | **Composition** | **Chemistry cluster** | **∆R^2^** |
|---------|----------|----------|----------|-------------|-------------|----------|
| S | 0.958 | 0.958 | 0.960 | 0.832 ± 0.004 | 0.753 ± 0.005 | +0.205 |
| *σ* | 0.915 | 0.915 | 0.917 | 0.776 ± 0.002 | 0.702 ± 0.002 | +0.213 |
| *κ* | 0.943 | 0.944 | 0.946 | 0.857 ± 0.001 | 0.809 ± 0.002 | +0.134 |
| zT | 0.918 | 0.918 | 0.919 | 0.818 ± 0.001 | 0.746 ± 0.005 | +0.172 |
<!-- END TABLE 6 -->

![Supplementary Figure [[sfig:ladder]]](../figures/paper_a/fig2_validation_ladder.png){width=16cm}

**Supplementary Figure [[sfig:ladder]]:** Pooled R^2^ across the five validation methods. Chemistry-cluster CV is the lowest for every target; ∆ is random 80/20 minus chemistry cluster. Error bars are the standard deviations defined in Supplementary Table [[stbl:ladder]].

## Supplementary Note [[snote:algo]]: Weights of the stacking ensemble (supplement to Section 3.1.2)

The fitted weights are not interpreted: the three models' predictions are near-collinear (the correlation of any pair within any fold is at least <!--v:a2_w_corr_min-->0.976<!--/v-->), the weights range from <!--v:a2_w_min-->0.01<!--/v--> to <!--v:a2_w_max-->0.70<!--/v--> over the <!--v:a2_nfits-->100<!--/v--> fits, and the largest spread of one model's weight across the <!--v:a2_nfolds-->25<!--/v--> folds of a target is <!--v:a2_w_range_min-->0.35<!--/v--> to <!--v:a2_w_range_max-->0.60<!--/v--> (depending on the target).

## Supplementary Note [[snote:classifier]]: The carrier-type classifier in detail (supplement to Section 3.1.3)

For the p class precision is <!--v:cl_prec_p-->0.955<!--/v--> (<!--v:cl_prec_p_lo-->0.940<!--/v-->--<!--v:cl_prec_p_hi-->0.966<!--/v-->) and recall <!--v:cl_rec_p-->0.951<!--/v--> (<!--v:cl_rec_p_lo-->0.938<!--/v-->--<!--v:cl_rec_p_hi-->0.962<!--/v-->), for the n class <!--v:cl_prec_n-->0.940<!--/v--> (<!--v:cl_prec_n_lo-->0.930<!--/v-->--<!--v:cl_prec_n_hi-->0.948<!--/v-->) and <!--v:cl_rec_n-->0.944<!--/v--> (<!--v:cl_rec_n_lo-->0.934<!--/v-->--<!--v:cl_rec_n_hi-->0.954<!--/v-->), and the area under the ROC curve is <!--v:cl_auc-->0.987<!--/v--> (<!--v:cl_auc_lo-->0.985<!--/v-->--<!--v:cl_auc_hi-->0.990<!--/v-->, first repeat only). Summed over the five repeats the confusion matrix is <!--v:cl_tn-->388,028<!--/v--> n-type rows correct and <!--v:cl_fp-->22,832<!--/v--> predicted p; <!--v:cl_fn-->24,978<!--/v--> p-type rows predicted n and <!--v:cl_tp-->489,442<!--/v--> correct. Accuracy is lower for the <!--v:cl_single_n-->59<!--/v--> rows of single-row clusters (<!--v:cl_single_acc-->0.78<!--/v-->) and for clusters of two to nine rows (<!--v:cl_small_acc-->0.92<!--/v-->) than for larger ones.

## Supplementary Note [[snote:featsel]]: Feature selection: protocol, convergence, device and a post hoc check (supplement to Section 3.1.4)

Inside every outer training fold (<!--v:n_repeats-->5<!--/v--> repeats of <!--v:n_folds-->5<!--/v--> folds, the folds of Section 3.1.1) three steps were applied to the training rows only: a Pearson filter that drops one of each pair of features with |*r*| above <!--v:na13_pearson-->0.95<!--/v-->, a LassoCV (<!--v:na13_alphas-->20<!--/v--> penalties, inner chemistry-cluster folds) that keeps the features with non-zero coefficients, and a ranking of those by mutual information with the target on a subsample of <!--v:na13_mi_rows-->20,000<!--/v--> rows, from which the top *k* of the earlier work were kept. The target's frozen XGBoost was then fitted twice on the same training rows, on the selected features and on all <!--v:n_feat-->397<!--/v-->, and scored on the same test rows, on one machine and one device. The earlier work names the three steps and the number of features it kept per target, but not the settings of the steps. The number kept, *k*, is the earlier work's, used to reproduce what it compared and not as a recommended choice. The remaining settings were fixed before any result of the comparison was seen and none was tuned: the Pearson threshold, a common near-duplicate cut; the penalty grid and the three inner chemistry-cluster folds of the LassoCV, with scikit-learn's defaults for the range of the grid and for the tolerance; and the size of the mutual-information subsample, chosen for run time. The one setting changed in the course of this work is the Lasso's iteration limit (below). Because the earlier work did not state the settings, the comparison tests the procedure as instantiated here; other settings could change the numbers.

*Convergence.* At the first setting tried, <!--v:na13_conv_lo-->2000<!--/v--> iterations, every one of the <!--v:na13_conv_folds-->20<!--/v--> folds diagnosed (the first repeat of each target) raised a convergence warning (<!--v:na13_conv_warn_2000-->258<!--/v--> warnings); at <!--v:na13_maxiter-->20,000<!--/v--> iterations none did, and that setting is used throughout. The selected sets nevertheless depend on the setting: the Jaccard similarity between the sets chosen at the two settings has a median of <!--v:na13_conv_jac_med-->0.93<!--/v--> and a minimum of <!--v:na13_conv_jac_min-->0.56<!--/v-->, and the fold R^2^ changed by more than <!--v:na13_conv_d_thr-->0.005<!--/v--> in <!--v:na13_conv_d_n-->6<!--/v--> of the <!--v:na13_conv_folds-->20<!--/v--> folds (mean absolute change <!--v:na13_conv_d_mean-->0.003<!--/v-->), so a selection made at the lower setting depends on the number of iterations. The change in R^2^ is not interpreted on its own: a per-fold change of <!--v:na13_conv_d_thr-->0.005<!--/v--> is comparable to the variation caused by running the same fit with the same selected set on a different machine, <!--v:na13_platform_delta-->0.0062<!--/v--> in the one fold compared (the first fold of the first repeat for *κ*, a Windows PC and a Linux server, both on the CPU). Of the <!--v:na13_units-->100<!--/v--> selection fits at the adopted setting, <!--v:na13_warn_n-->1<!--/v--> raised a convergence warning (<!--v:na13_warn_target-->S<!--/v-->, repeat <!--v:na13_warn_repeat-->2<!--/v-->, fold <!--v:na13_warn_fold-->0<!--/v-->, in an inner path of the penalty search; its final refit converged).

*Device.* The XGBoost fits were run with the CUDA device requested (a Tesla V100S is listed in the run record). The scripts record the device requested, not the device the booster used; the paired comparison does not depend on it, because both fits of a fold ran on the same machine and device. The all-<!--v:n_feat-->397<!--/v--> fits reproduce the committed chemistry-cluster values of Section 3.1.1, which were fitted on a different machine, to within <!--v:na13_maxdiff-->1.2 × 10^−16^<!--/v--> in every fold.

*Post hoc check (exploratory; added after the result above was seen, and it does not change it).* The feature `temperature_bin` is one of the <!--v:n_feat-->397<!--/v--> features and was selected in none of the <!--v:na13_units-->100<!--/v--> folds. Refitting the frozen model on the selected features plus `temperature_bin`, on the same folds and device, gives R^2^ of <!--v:na13_S_temp-->0.720<!--/v--> (S), <!--v:na13_sigma_temp-->0.687<!--/v--> (*σ*), <!--v:na13_kappa_temp-->0.795<!--/v--> (*κ*) and <!--v:na13_zT_temp-->0.736<!--/v--> (zT), which recovers <!--v:na13_S_temp_share-->34<!--/v-->%, <!--v:na13_sigma_temp_share-->61<!--/v-->%, <!--v:na13_kappa_temp_share-->70<!--/v-->% and <!--v:na13_zT_temp_share-->97<!--/v-->% of the gap between the selected set and all <!--v:n_feat-->397<!--/v--> (pooled per-repeat means). The combined set still stays below all <!--v:n_feat-->397<!--/v--> in <!--v:na13_S_temp_below-->25<!--/v-->, <!--v:na13_sigma_temp_below-->24<!--/v-->, <!--v:na13_kappa_temp_below-->24<!--/v--> and <!--v:na13_zT_temp_below-->20<!--/v--> of the <!--v:na13_nfolds-->25<!--/v--> folds. These figures are descriptive: no threshold, interval or test was set for them, and the grouped comparison above remains the result.

<!-- BEGIN TABLE 7b -->
**Supplementary Table [[stbl:featsel]]:** Chemistry-cluster CV R^2^ (mean ± SD over 5 repeats of 5 grouped folds) of each target's frozen XGBoost on all 397 features and on the features kept by the selection of the earlier work, applied inside every outer training fold (Pearson filter at |r| = 0.95, a LassoCV over 20 penalties, then the top k by mutual information on a 20,000-row subsample; k = 25, 44, 39, 32 for S, *σ*, *κ*, zT). Both fits of a fold use the same rows, machine and device. The paired difference is the comparison fixed before the run (difference of the pooled R^2^, chemistry-cluster bootstrap interval in brackets). The last three rows are post hoc and exploratory: the same selected features plus temperature_bin, added after the result above was seen; no interval or test is attached to them.

| **Quantity** | **S** | ***σ*** | ***κ*** | **zT** |
|--------------------|------------|------------|------------|------------|
| All 397 features | 0.753 ± 0.005 | 0.702 ± 0.002 | 0.809 ± 0.002 | 0.746 ± 0.005 |
| Selected features (k imposed) | 0.702 ± 0.008 | 0.664 ± 0.003 | 0.762 ± 0.003 | 0.451 ± 0.006 |
| Selected minus all 397 (paired, fixed before the run) | −0.050 [−0.060, −0.042] | −0.038 [−0.045, −0.032] | −0.047 [−0.057, −0.038] | −0.295 [−0.328, −0.262] |
| Features kept by the Lasso, mean (range over 25 folds) | 182 (129–206) | 156 (119–184) | 184 (176–195) | 161 (138–183) |
| k (earlier work) | 25 | 44 | 39 | 32 |
| Post hoc: selected + temperature_bin | 0.720 ± 0.010 | 0.687 ± 0.003 | 0.795 ± 0.003 | 0.736 ± 0.005 |
| Post hoc: (selected + temperature_bin) minus all 397 | −0.033 | −0.015 | −0.014 | −0.009 |
| Post hoc: share of the selected-to-all-397 gap recovered (%) | 34 | 61 | 70 | 97 |
| Post hoc: folds of 25 with selected + temperature_bin below all 397 | 25 | 24 | 24 | 20 |
<!-- END TABLE 7b -->

## Supplementary Note [[snote:shap]]: Feature importance in detail (supplement to Section 3.1.5)

SHAP values are in the units of each target (*µ*V K^−1^ for S; log~10~ units for *σ* and *κ*). Feature labels read "Elemental <property>, <statistic> (MAGPIE or CBFV)": the property is one of the elements, the statistic is taken over the elements of the formula (mean = atomic-fraction-weighted mean, mean abs. deviation = the weighted mean absolute deviation about that mean, range, minimum and maximum over the elements, mode = the value of the most abundant element). For S the top feature, for instance, is the spread of the elemental thermal conductivities of the formula, not the thermal conductivity of the sample.

![Supplementary Figure [[sfig:beeswarm]]](../figures/fig11_shap_beeswarm.png){width=16cm}

**Supplementary Figure [[sfig:beeswarm]]:** SHAP beeswarm plots for (a) S, (b) *σ*, (c) *κ* and (d) zT: one point per test row (<!--v:bs_rows-->5,000<!--/v--> per model, first repeat of the chemistry-cluster folds), coloured by the percentile rank of the row's value of the feature; the features are the ones with the largest mean \|SHAP\| among these rows, so their order can differ from Fig. [[fig:shapg]], which averages all folds of all repeats. Every feature other than temperature is a statistic of an ELEMENTAL property of the elements in the formula, not a measured property of the material (for S, the top feature is the spread of the elemental thermal conductivities, not the sample's thermal conductivity).

The two featurization schemes contribute differently to the targets (Supplementary Figure [[sfig:shares]]). In the zT model, temperature accounts for <!--v:shap_share_temperature-->20<!--/v-->% of the mean \|SHAP\|, CBFV features for <!--v:shap_share_cbfv-->53<!--/v-->% and MAGPIE features for <!--v:shap_share_magpie-->28<!--/v-->% (shares over all <!--v:n_feat-->397<!--/v--> features, chemistry-cluster folds). In the other models temperature accounts for <!--v:shap_share_S_temperature-->2<!--/v-->% (S), <!--v:shap_share_sigma_temperature-->4<!--/v-->% (*σ*) and <!--v:shap_share_kappa_temperature-->5<!--/v-->% (*κ*), CBFV features for <!--v:shap_share_S_cbfv-->69<!--/v-->%, <!--v:shap_share_sigma_cbfv-->68<!--/v-->% and <!--v:shap_share_kappa_cbfv-->60<!--/v-->%, and MAGPIE features for <!--v:shap_share_S_magpie-->29<!--/v-->%, <!--v:shap_share_sigma_magpie-->28<!--/v-->% and <!--v:shap_share_kappa_magpie-->35<!--/v-->%. CBFV features carry the largest share of every model (<!--v:shap_cbfv_min-->53<!--/v--> to <!--v:shap_cbfv_max-->69<!--/v-->%), but they are also the more numerous (<!--v:shap_n_cbfv-->264<!--/v--> against <!--v:shap_n_magpie-->132<!--/v--> MAGPIE features): per feature, the mean share of a CBFV feature is <!--v:shap_ratio_min-->0.85<!--/v--> to <!--v:shap_ratio_max-->1.22<!--/v--> times that of a MAGPIE feature, so neither scheme is more informative per feature. Both carry a substantial share of every model; this attribution does not show whether either could be dropped.

![Supplementary Figure [[sfig:shares]]](../figures/fig12_shap_shares.png){width=8.5cm}

**Supplementary Figure [[sfig:shares]]:** Relative contribution of MAGPIE, CBFV and temperature features (share of mean \|SHAP\| over all features) to each model.

## Supplementary Note [[snote:dvd]]: Direct versus component-wise zT prediction in detail (supplement to Section 3.1.6)

Direct zT prediction was compared with the component-wise derivation S^2^*σ*T/*κ* under chemistry-cluster CV (Supplementary Table [[stbl:dvd]]). Direct prediction is more accurate (R^2^ = <!--v:dvd_direct-->0.727<!--/v--> vs <!--v:dvd_derived-->0.540<!--/v-->), with each model using its own target's frozen hyperparameters. The derived pathway suffers from error propagation; to first order, the relative error of the derived zT is

$\frac{\delta zT}{zT} \approx \sqrt{\left( 2\frac{\delta S}{S} \right)^{2} + \left( \frac{\delta\sigma}{\sigma} \right)^{2} + \left( \frac{\delta\kappa}{\kappa} \right)^{2}}$ (S5)

so the quadratic dependence on S doubles its relative error, and the weakest component model (*σ*, R^2^ = <!--v:dvd_sigma-->0.692<!--/v-->, against <!--v:dvd_S-->0.820<!--/v--> for S and <!--v:dvd_kappa-->0.821<!--/v--> for *κ*) adds a large further term. Back-transforming the log-space predictions of *σ* and *κ* with a smearing correction changes the derived R^2^ by <!--v:dvd_duan-->−0.005<!--/v-->, too little to alter the comparison. The individual component models remain useful diagnostically: a material predicted to have high S but low *σ*, for example, may benefit from carrier-concentration optimization through doping. The comparison was repeated under random row-level validation (shuffled 5-fold, five repeats, the same <!--v:dvd_rows-->56,088<!--/v--> rows and the same frozen hyperparameters; Supplementary Table [[stbl:dvd]]). Direct prediction is again more accurate (R^2^ = <!--v:na7_direct-->0.917<!--/v--> vs <!--v:na7_derived-->0.880<!--/v-->), so the ordering does not depend on grouped validation. The size of the advantage does: the gap between the two pathways is <!--v:na7_gap-->0.036<!--/v--> under random validation and <!--v:na7_grouped_gap-->0.186<!--/v--> under chemistry-cluster validation (differences of unrounded R^2^), a factor of <!--v:na7_ratio-->5.1<!--/v-->. Random validation therefore understates how much the derived route loses when it has to generalize to unseen chemistry.

<!-- BEGIN TABLE 10 -->
**Supplementary Table [[stbl:dvd]]:** Direct versus component-wise zT prediction on the 56,088 rows with all four properties present (4,139 chemistry clusters); each model uses its own target's frozen hyperparameters. Chemistry-cluster CV is 5 repeats x 5 folds of grouped folds; random CV is 5 repeats x 5 folds of shuffled row-level folds; both pool the out-of-fold predictions (n = 280,440).

| **Pathway** | **R^2^ (chemistry-cluster)** | **MAE (chemistry-cluster)** | **R^2^ (random)** | **MAE (random)** |
|------------------------|----------|---------|----------|---------|
| Direct (features → zT) | 0.727 | 0.132 | 0.917 | 0.072 |
| Derived (S^2^*σ*T/*κ*) | 0.540 | 0.166 | 0.880 | 0.083 |
<!-- END TABLE 10 -->

## Supplementary Note [[snote:na12]]: The literature comparison per paper (supplement to Section 3.4)

<!-- BEGIN TABLE 12 -->
**Supplementary Table [[stbl:na12]]:** Comparison of out-of-fold predictions with published measurements, per paper. n: values compared; inside: values whose absolute relative error is within the relative uncertainty of the measurement (S 0.06, *σ* 0.08, *κ* 0.11, zT 0.19), with the fraction; median abs. rel. error: median absolute relative error (linear units); log~10~ ratio: median absolute and signed median of log~10~(prediction) − log~10~(literature) for *σ* and *κ*. Exact level: the same composition at the same temperature bin; cluster level: the mean of the held-out predictions of the sample's chemistry cluster at that bin (a family average). Part A: the independent papers; Part B: the papers whose DOI is among the DOIs of the training data (training sources), not pooled with part A.

*Part A: independent papers.*

| **Paper** | **Level** | **Property** | **n** | **Inside (fraction)** | **Median abs. rel. error** | **Median abs. log~10~ ratio** | **Signed median log~10~ ratio** |
|---|---|---|---|---|---|---|---|
| Ohtaki et al. [@ohtaki1995] | exact | S | 10 | 3 (0.300) | 0.086 | – | – |
| Ohtaki et al. [@ohtaki1995] | exact | *σ* | 22 | 0 (0.000) | 0.416 | 0.206 | −0.185 |
| Ohtaki et al. [@ohtaki1995] | exact | *κ* | 2 | 0 (0.000) | 0.362 | 0.195 | −0.195 |
| Ohtaki et al. [@ohtaki1995] | exact | zT | 3 | 0 (0.000) | 1.139 | – | – |
| Sebastia-Luna et al. [@sebastialuna2022] | exact | S | 3 | 0 (0.000) | 0.449 | – | – |
| Sebastia-Luna et al. [@sebastialuna2022] | exact | *σ* | 3 | 0 (0.000) | 4.896 | 0.768 | +0.768 |
| Sebastia-Luna et al. [@sebastialuna2022] | exact | *κ* | 3 | 0 (0.000) | 0.160 | 0.062 | +0.062 |
| Sebastia-Luna et al. [@sebastialuna2022] | exact | zT | 3 | 0 (0.000) | 5.194 | – | – |
| Singh and Pandey [@singh2016] | exact | S | 3 | 1 (0.333) | 0.145 | – | – |
| Ohtaki et al. [@ohtaki1995] | cluster | S | 40 | 0 (0.000) | 0.596 | – | – |
| Ohtaki et al. [@ohtaki1995] | cluster | *σ* | 66 | 7 (0.106) | 6.105 | 0.851 | +0.851 |
| Ohtaki et al. [@ohtaki1995] | cluster | *κ* | 5 | 1 (0.200) | 0.248 | 0.124 | −0.124 |
| Ohtaki et al. [@ohtaki1995] | cluster | zT | 5 | 0 (0.000) | 0.917 | – | – |
| Sebastia-Luna et al. [@sebastialuna2022] | cluster | S | 3 | 0 (0.000) | 0.504 | – | – |
| Sebastia-Luna et al. [@sebastialuna2022] | cluster | *σ* | 3 | 0 (0.000) | 5.230 | 0.793 | +0.793 |
| Sebastia-Luna et al. [@sebastialuna2022] | cluster | *κ* | 3 | 0 (0.000) | 0.137 | 0.054 | +0.054 |
| Sebastia-Luna et al. [@sebastialuna2022] | cluster | zT | 3 | 0 (0.000) | 4.485 | – | – |
| Singh and Pandey [@singh2016] | cluster | S | 13 | 0 (0.000) | 0.354 | – | – |

*Overall, independent papers: papers weigh equally (the mean over papers of the fraction inside the band; the median over papers of each paper's median).*

| **Level** | **Property** | **Papers** | **Mean fraction inside** | **Median of median abs. rel. error** | **Median of median abs. log~10~ ratio** | **Median of signed median log~10~ ratio** |
|---|---|---|---|---|---|---|
| exact | S | 3 | 0.211 | 0.145 | – | – |
| exact | *σ* | 2 | 0.000 | 2.656 | 0.487 | +0.291 |
| exact | *κ* | 2 | 0.000 | 0.261 | 0.128 | −0.067 |
| exact | zT | 2 | 0.000 | 3.166 | – | – |
| cluster | S | 3 | 0.000 | 0.504 | – | – |
| cluster | *σ* | 2 | 0.053 | 5.668 | 0.822 | +0.822 |
| cluster | *κ* | 2 | 0.100 | 0.193 | 0.089 | −0.035 |
| cluster | zT | 2 | 0.000 | 2.701 | – | – |

*Part B: training-source papers (not pooled with part A).*

| **Paper** | **Level** | **Property** | **n** | **Inside (fraction)** | **Median abs. rel. error** | **Median abs. log~10~ ratio** | **Signed median log~10~ ratio** |
|---|---|---|---|---|---|---|---|
| Flahaut et al. [@flahaut2006effect] | exact | *σ* | 4 | 0 (0.000) | 0.550 | 0.348 | −0.348 |
| Shibasaki et al. [@shibasaki2009] | exact | S | 1 | 0 (0.000) | 0.601 | – | – |
| Flahaut et al. [@flahaut2006effect] | cluster | S | 6 | 0 (0.000) | 0.846 | – | – |
| Flahaut et al. [@flahaut2006effect] | cluster | *σ* | 13 | 1 (0.077) | 0.331 | 0.175 | −0.175 |
| Flahaut et al. [@flahaut2006effect] | cluster | *κ* | 15 | 2 (0.133) | 0.271 | 0.104 | +0.104 |
| Flahaut et al. [@flahaut2006effect] | cluster | zT | 12 | 4 (0.333) | 0.629 | – | – |
| Ohta et al. [@ohta2005temperature] | cluster | S | 35 | 0 (0.000) | 0.544 | – | – |
| Ohta et al. [@ohta2005temperature] | cluster | *σ* | 34 | 0 (0.000) | 0.874 | 0.388 | +0.269 |
| Ohta et al. [@ohta2005temperature] | cluster | *κ* | 16 | 10 (0.625) | 0.091 | 0.039 | −0.020 |
| Ohta et al. [@ohta2005temperature] | cluster | zT | 19 | 2 (0.105) | 0.473 | – | – |
| Rajasekaran et al. [@rajasekaran2020] | cluster | S | 5 | 0 (0.000) | 0.433 | – | – |
| Rajasekaran et al. [@rajasekaran2020] | cluster | *σ* | 2 | 0 (0.000) | 16.319 | 1.173 | +1.173 |
| Shibasaki et al. [@shibasaki2009] | cluster | S | 1 | 0 (0.000) | 0.601 | – | – |
<!-- END TABLE 12 -->

<!-- BEGIN TABLE 13 -->
**Supplementary Table [[stbl:edge]]:** Edge sensitivity: the overall summary of the independent papers (papers weigh equally) with every value that lies within twice the calibration uncertainty of a temperature-bin edge evaluated in its floor bin (headline) and in the neighbouring bin.

| **Level** | **Property** | **Fraction inside, headline** | **Fraction inside, neighbouring bin** | **Median abs. rel. error, headline** | **Median abs. rel. error, neighbouring bin** | **Median abs. log~10~ ratio, headline** | **Median abs. log~10~ ratio, neighbouring bin** |
|---|---|---|---|---|---|---|---|
| exact | S | 0.211 | 0.111 | 0.145 | 0.313 | – | – |
| exact | *σ* | 0.000 | 0.000 | 2.656 | 2.659 | 0.487 | 0.488 |
| exact | *κ* | 0.000 | 0.000 | 0.261 | 0.265 | 0.128 | 0.131 |
| exact | zT | 0.000 | 0.000 | 3.166 | 3.098 | – | – |
| cluster | S | 0.000 | 0.000 | 0.504 | 0.504 | – | – |
| cluster | *σ* | 0.053 | 0.045 | 5.668 | 5.700 | 0.822 | 0.824 |
| cluster | *κ* | 0.100 | 0.125 | 0.193 | 0.189 | 0.089 | 0.088 |
| cluster | zT | 0.000 | 0.000 | 2.701 | 2.701 | – | – |
<!-- END TABLE 13 -->

## Supplementary Note [[snote:nomenclature]]: Nomenclature and abbreviations

The symbols and abbreviations used in the manuscript are listed in Supplementary Table [[stbl:nomenclature]].

**Supplementary Table [[stbl:nomenclature]]:** Symbols and abbreviations used in the manuscript.

| **Symbol / abbreviation** | **Meaning** |
|---------------------|---------------------------------------------------|
| AI | Artificial intelligence |
| ML | Machine learning |
| DL | Deep learning |
| TE | Thermoelectric |
| DFT | Density functional theory |
| BTE | Boltzmann transport equation |
| PGEC | Phonon-glass electron-crystal |
| XGBoost | Extreme gradient boosting |
| LightGBM | Light gradient boosting machine |
| GBDT | Gradient-boosted decision trees |
| RF | Random forest |
| DNN | Deep neural network |
| CGCNN | Crystal graph convolutional neural network |
| TabPFN | Tabular prior-data fitted network |
| CV | Cross-validation (also coefficient of variation in Step 8) |
| MAGPIE | Materials-Agnostic Platform for Informatics and Exploration |
| CBFV | Composition-based feature vector |
| SHAP | SHapley Additive exPlanations |
| LIME | Local interpretable model-agnostic explanations |
| MAE | Mean absolute error |
| MAD | Median absolute deviation |
| MP | Materials Project |
| ESTM | Experimentally Synthesized Thermoelectric Materials (database) |
| JARVIS | Joint Automated Repository for Various Integrated Simulations |
| VASP | Vienna Ab initio Simulation Package |
| KRICT | Korea Research Institute of Chemical Technology |
| API | Application programming interface |
| REST | Representational state transfer |
| zT | Dimensionless figure of merit |
| S | Seebeck coefficient |
| *σ* | Electrical conductivity |
| *κ*, *κ*~e~, *κ*~l~ | Total, electronic and lattice thermal conductivity |
| PF | Power factor (S^2^*σ*) |
| L | Lorenz number |
| R^2^ | Coefficient of determination |
| T | Absolute temperature |
| E~hull~ | Energy above the convex hull |
| E~g~ | Band gap |
| ABX~3~ | General perovskite stoichiometry |
