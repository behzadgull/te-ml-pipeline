\# Validation Inflation, Descriptor Saturation, and the Label-Noise Ceiling in Thermoelectric Property Prediction



**Muhammad Behzad Gull**
TODO: Affiliation



## Abstract



\## Abstract



Machine learning models predicting thermoelectric transport properties from composition are routinely reported with cross-validated R² above 0.9. We show that this figure is substantially inflated by the validation protocol. On a pinned snapshot of the Starrydata2 database, with hyperparameters held fixed throughout, moving from ungrouped to chemistry-cluster grouped cross-validation lowers R² by 0.134 to 0.213 across the Seebeck coefficient, electrical conductivity, thermal conductivity and figure of merit. k-fold cross-validation provides no protection: five-fold and ten-fold results are indistinguishable from repeated random 80/20 holdout. A feature-attribution control shows that the model relies on the same descriptors under both protocols, locating the inflation in test-set composition rather than in learned structure.



We then ask what limits honest performance. Combining an inter-laboratory measurement round-robin with a direct comparison of two independent digitizations of the same published figures gives a label-noise ceiling of 0.96 to 0.99, against which grouped performance of 0.70 to 0.81 leaves 0.15 to 0.29 of headroom. A descriptor ablation shows this gap is not one of descriptor count: tripling the feature set from 133 to 397 closes under 4.2% of the headroom on every property, and two descriptor families of different construction are nearly interchangeable. Predicting the figure of merit directly outperforms reconstructing it from separately predicted components by 0.186 in R². Transfer to two independent databases degrades further, though 13 to 17 percent of external rows fall outside the training data's property range, and restricting to in-support rows recovers a substantial share of the apparent loss.



The results indicate that composition-based thermoelectric models are further from their achievable limit than published figures suggest, and that the remaining distance is a property of the representation rather than of the number of descriptors or the size of the dataset.



# Introduction



\## 1. Introduction



Thermoelectric materials convert heat to electricity, and their performance is summarised by the dimensionless figure of merit zT = S²σT/κ, combining the Seebeck coefficient, electrical conductivity and thermal conductivity. Because the three transport properties are coupled and their measurement is slow, machine learning models that predict them from composition alone have become a standard tool for narrowing experimental search spaces, increasingly trained on large databases digitized from the published literature [@katsura2025starrydata; @na2022public; @parse2024predicting; @barua2025thermoelectric].



Such models are routinely reported with R² between 0.90 and 0.98 [@athar2026beyond]; one recent study reports 0.970 under ten-fold cross-validation over 5,226 temperature-resolved rows drawn from 1,022 materials [@wang2025highperformance]. The figure is arrived at honestly in the sense that the reported protocol is followed, but the protocol itself is usually a random split or k-fold cross-validation over rows, and rows in thermoelectric databases are not independent. A single material contributes a measurement at every temperature on its curve, and a doped series contributes one row per composition per temperature. Under random splitting, the same specimen appears on both sides of the split at different temperatures, and near duplicates of it appear under different dopant labels (Figure 1). The model is asked to interpolate within a curve it has already seen.



![Figure 1](figures/fig1_leakage_schematic.png)

*Figure 1. PLACEHOLDER -- file pending. Schematic of within-curve and near-duplicate leakage: a single material's temperature curve contributes one row per bin, and a doped series shares its host lattice across dopant labels, so rows that appear independent under random row-level splitting are not.*



That this inflates reported accuracy is not in itself new. Leave-cluster-out validation was proposed for materials property prediction to separate extrapolation from interpolation [@meredig2018machine], and redundancy between training and test sets has been shown to overstate performance on materials benchmarks [@li2024mdhit]. In thermoelectrics, the study that introduced the ESTM dataset reported R² above 0.9 for composition-based models, while zT prediction for material groups absent from training reached R² of 0.13 before transfer learning [@na2022public]. On a 3,879-row subset of the same dataset, holding out all temperature records of a composition lowered R² for all three transport properties, and holding out whole material families gave negative values [@ho2026physicsinspired]. Representative splitting has been proposed as part of more reliable thermoelectric workflows [@athar2026robust], and some studies on Starrydata2 already hold out whole compositions [@jia2024dealing].



What these studies leave open is the size of the effect, protocol by protocol, with its variance across independent partitions measured; the thermoelectric comparisons above each rest on a single draw over a few thousand rows. Nor is it known what would remain once the inflation is removed: whether composition-based descriptors are close to their limit or far from it, and whether that limit is set by the descriptors, the data volume or the noise in the labels themselves. Measurement uncertainty in thermoelectric transport properties is known from inter-laboratory round-robins [@alleno2015round; @wang2015international], literature-derived databases carry inconsistencies of their own [@ryu2025highquality; @athar2025tackling], and label noise sets an expected upper bound on the R² any regression model can reach [@li2021performance]. To our knowledge these have not been combined into a ceiling for composition-based thermoelectric models.



This paper addresses both questions on a single pinned snapshot of Starrydata2 [@katsura2025starrydata], using one model family [@chen2016xgboost] and one frozen set of hyperparameters per property, so that every difference we report is attributable to the condition under test rather than to a change in the model.



We make four contributions. First, we quantify validation inflation across five protocols and four properties, with the grouped protocols repeated over independent partitions. Grouping by chemistry cluster lowers R² by 0.134 to 0.213 relative to ungrouped validation; grouping by exact composition removes only 58 to 65% of that inflation; and k-fold cross-validation provides no protection. Second, a feature-attribution control shows that the inflation reflects test-set composition rather than a change in what the model learns: the same function is applied to an easier test set. Third, we construct a label-noise ceiling from two independently measured components, an inter-laboratory round-robin and a direct comparison of two independent digitizations of the same figures. Honest performance sits 0.15 to 0.29 below it, and an ablation shows that the gap is not a descriptor-count problem: tripling the number of descriptors closes at most 4.2% of it. Fourth, we show that the figure of merit is better predicted directly than reconstructed from separately predicted components, by 0.186 in R² under grouped validation.



We then test transfer to two independent databases, finding a further degradation beyond the validation gap, of which a substantial share is extrapolation outside the training data's property range rather than failure to generalise.



# Methods



\## 2. Data and Methods



\### 2.1 Dataset



Training data comes from Starrydata2 [@katsura2025starrydata], a community-curated database of thermoelectric property measurements digitized from figures in the primary literature. The snapshot used throughout is a single frozen pull of 9,494 publications, 55,261 samples and 156,101 digitized property curves, taken on 22 August 2026 and identified by SHA-256 hash. Starrydata2 regenerates its public export daily, so an unpinned pull is not reproducible; every result reported here derives from this one snapshot, which is archived separately from the live source.



\### 2.2 Cleaning pipeline



Digitized curves are converted to model-ready rows through eleven stages (Figure 2). Each curve's digitized points are first expanded into individual temperature-property observations and screened against physical bounds, giving 1,996,047 observations. Resistivity is inverted to conductivity and merged with directly reported conductivity. Observations outside 300 to 800 K are discarded and the remainder binned at 25 K, leaving 1,096,324. Pivoting to one row per material and temperature bin, averaging where multiple digitized points fall in the same bin, gives 398,763 rows.



![Figure 2](figures/cleaning_funnel.png)

*Figure 2. Row count through the eleven-stage cleaning pipeline, from 1,996,047 property observations after range filtering to 280,664 rows in the final cleaned dataset.*



Six filters then act on those rows. Formulas that cannot be parsed are dropped (98.7% retained). Rows reporting all four properties are checked against S²σT/κ and discarded if the relative discrepancy exceeds 50% (98.8%). Publications identified as computational rather than experimental are excluded entirely (99.7%). Where several sources report the same composition at the same temperature, the group is discarded if any property's coefficient of variation exceeds a per-property threshold (79.7%). A median-absolute-deviation filter at 3.5σ removes outliers (93.7%). Materials measured at fewer than three distinct temperature bins are removed (98.4%), and a rolling-median smoothness filter removes remaining spikes within each measurement series (98.5%).



The cleaned dataset contains 280,664 rows. After featurization it provides 185,064 rows with a Seebeck coefficient, 182,755 with electrical conductivity, 121,110 with thermal conductivity and 129,419 with a figure of merit; the four sets overlap and are not a partition.



\### 2.3 Features



Each composition is described by 132 MAGPIE elemental-property attributes [@ward2016general], computed with matminer [@ward2018matminer], and 264 CBFV features [@murdock2020domain] built on the Oliynyk element-property set [@oliynyk2016highthroughput], both computed once per unique formula and joined onto every row of that composition. With the temperature bin this gives 397 features. Formulas that fail either featurizer are excluded and logged rather than silently dropped.



\### 2.4 Targets



The Seebeck coefficient and figure of merit are modelled in linear space; electrical and thermal conductivity span several orders of magnitude and are modelled in log10. Reported R² is always in the space the target was trained in.



\### 2.5 Validation protocols



Five protocols are compared. Twenty independent random 80/20 holdout draws, pooled, together with five-fold and ten-fold cross-validation, impose no constraint on what appears in both training and test. Composition grouping requires that no exact composition be split across folds. Chemistry-cluster grouping applies a stricter rule (Figure 3). It follows the leave-cluster-out principle for separating extrapolation from interpolation [@meredig2018machine]. Elements present below 5 atomic percent are treated as dopants and removed before forming the cluster identifier, and remaining amounts within 5% of an integer are snapped to that integer before reduction. A doped material therefore groups with its parent, and so does one whose host stoichiometry is reported with measurement-level imprecision. (PbTe)₀.₉₇(SrTe)₀.₀₂(Na₂Te)₀.₀₁ groups with PbTe, and Pb₀.₉₇Te groups with PbTe, while Bi₂Te₂.₇Se₀.₃, where selenium occupies 6.0 atomic percent and is therefore an alloy component rather than a dopant, does not group with Bi₂Te₃. The 5 atomic percent threshold follows the convention that separates principal elements from minor additions in multicomponent alloys [@yeh2004nanostructured], also used to distinguish doping from alloying in thermoelectric compositions [@ma2025reexamining].



![Figure 3](figures/fig3_grouping_rule_schematic.png)

*Figure 3. PLACEHOLDER -- file pending. The chemistry-cluster grouping rule: elements below 5 atomic percent are treated as dopants and collapsed into the host lattice, and host amounts within 5% of an integer are snapped to that integer before reduction, so Pb0.97Te and (PbTe)0.97(SrTe)0.02(Na2Te)0.01 both group with PbTe while Bi2Te2.7Se0.3 (Se at 6.0 at%) remains distinct from Bi2Te3.*



Random holdout draws and k-fold partitions are generated with scikit-learn [@pedregosa2011scikit]. Grouped protocols are run as five repeats of five-fold cross-validation, with a custom fold assignment in place of scikit-learn's deterministic GroupKFold. Because the group-size distribution is heavy-tailed, the assignment of the largest clusters to folds is randomised between repeats rather than fixed, so that the across-repeat standard deviation reflects which chemistries are held out and not only residual model variance. Reported values are the mean and standard deviation of per-repeat pooled R². The random 80/20 protocol pools twenty independent holdout draws into one value; five-fold and ten-fold cross-validation are each a single partition and likewise report one pooled value.



\### 2.6 Model and hyperparameters



All results use gradient-boosted trees (XGBoost [@chen2016xgboost]). Hyperparameters were tuned once per target on all rows, by an Optuna search of 20 trials scored with three-fold cross-validation grouped by the original chemistry-cluster identifier, then frozen and reused unchanged across every protocol, feature set and experiment reported here. Differences between conditions therefore reflect the condition rather than the tuning. In particular, the descriptor ablation of Section 3.4 applies full-feature hyperparameters to reduced feature sets, and the direct-versus-derived comparison of Section 3.6 applies each target's own frozen set to its model on a subset, with a sensitivity run applying the figure-of-merit set to all four models; retuning on the subset in either case would confound the effect of interest with a tuning change.



\### 2.7 Significance testing



Where two protocols differ by an amount comparable to repeat-to-repeat variation, we apply the Nadeau-Bengio corrected paired t-test [@nadeau2003inference], which accounts for the dependence between overlapping training sets that inflates the naive paired test. We report the mean paired difference as the effect size and use the test only to establish that the direction is not attributable to chance.



\### 2.8 External validation



For external testing, models are refit on 100% of the training data with frozen hyperparameters and applied once. Smearing factors [@duan1983smearing] correcting the log-space back-transformation are computed from training-side out-of-fold residuals and held fixed; external labels are never used for calibration. Each external database is partitioned by source publication and by chemistry cluster, and the resulting strata are scored separately and never pooled. Because external datasets extend beyond the property ranges the training data covers, results are reported both over the full external set and restricted to rows falling within training's per-property range in S, σ and κ simultaneously, with the excluded fraction stated.



\### 2.9 Reproducibility



The dataset snapshot, model checkpoints, per-row predictions, frozen hyperparameters and analysis scripts are version-controlled and archived, and software environments for every run are recorded. As a check on pipeline determinism, one grouped cross-validation run was regenerated three weeks after its original execution, on a different software environment, and reproduced every held-out prediction bit-for-bit.



# Results



\## 3. Results



\### 3.1 Grouped validation lowers reported accuracy by 0.13 to 0.21



Table 1 gives pooled out-of-fold R² for four thermoelectric properties under five validation protocols (Figure 4), with hyperparameters tuned once per target and held fixed across every rung, so that differences between rungs reflect the validation protocol alone.



![Figure 4](figures/fig2_validation_ladder.png)

*Figure 4. Pooled out-of-fold R² by validation protocol and property (Table 1), with error bars showing each rung's own across-repeat, across-fold or across-draw standard deviation and brackets giving the random-80/20-to-chemistry-cluster gap.*



\*\*Table 1.\*\* Pooled out-of-fold R² by validation protocol. Composition and chemistry cluster report mean ± standard deviation across five independent repeats. Random 80/20, 5-fold and 10-fold report mean ± standard deviation across that rung's own folds or draws within a single pass (twenty draws for random 80/20, five or ten folds for k-fold), not across independent repeats. These two kinds of standard deviation are not directly comparable: one measures repeat-to-repeat spread under a newly drawn held-out partition each time, the other measures fold-to-fold or draw-to-draw spread within one partition.



| Target | random 80/20 (20 draws) | 5-fold | 10-fold | composition | chemistry cluster | gap |

|---|---|---|---|---|---|---|

| S | 0.9582 ± 0.0013 | 0.9585 ± 0.0013 | 0.9595 ± 0.0014 | 0.8322 ± 0.0044 | 0.7528 ± 0.0050 | 0.205 |

| σ (log10) | 0.9150 ± 0.0009 | 0.9152 ± 0.0009 | 0.9174 ± 0.0024 | 0.7762 ± 0.0015 | 0.7020 ± 0.0020 | 0.213 |

| κ (log10) | 0.9434 ± 0.0012 | 0.9436 ± 0.0013 | 0.9455 ± 0.0021 | 0.8565 ± 0.0013 | 0.8092 ± 0.0021 | 0.134 |

| zT | 0.9180 ± 0.0014 | 0.9181 ± 0.0029 | 0.9193 ± 0.0032 | 0.8178 ± 0.0009 | 0.7456 ± 0.0045 | 0.172 |



Three ungrouped protocols produce nearly identical estimates. Twenty independent random 80/20 holdout draws, pooled, five-fold and ten-fold cross-validation give 0.9582, 0.9585 and 0.9595 for the Seebeck coefficient; the largest spread across the four properties is 0.0024, for electrical conductivity. Increasing the number of folds does not change the picture. Whatever these protocols measure, they measure it consistently.



Grouping changes the picture substantially. Requiring that no exact composition appear in both training and test folds lowers R² to 0.8322 ± 0.0044 for S. Grouping instead by chemistry cluster, which additionally collapses dopants below 5 at% into their parent lattice, lowers it to 0.7528 ± 0.0050. The corresponding chemistry-cluster values for the other properties are 0.7020 ± 0.0020 for σ, 0.8092 ± 0.0021 for κ and 0.7456 ± 0.0045 for zT.



The gap between the ungrouped and chemistry-cluster rungs ranges from \*\*0.134 for thermal conductivity to 0.213 for electrical conductivity\*\*, with a mean of 0.181 across the four properties. Set against the across-repeat standard deviations of the grouped rungs, which run from 0.0020 to 0.0050, these gaps are 30 to 100 times the repeat-to-repeat spread. No formal test is required to establish that they are real, and none is available in any case: the ungrouped and grouped rungs use different resampling structures that cannot be paired. Their stability is instead attested by the mutual agreement of the three ungrouped protocols noted above. Figure 5 shows the random-versus-chemistry-cluster gap directly, as predicted-versus-actual zT under both protocols.



![Figure 5](figures/zt_parity_random_vs_grouped.png)

*Figure 5. Predicted versus actual zT under random 80/20 (R² = 0.9180, n = 517,680) and chemistry-cluster (R² = 0.7456, n = 647,095) validation, identical axes and colour scale, showing the spread difference directly.*



The narrower comparison between the two grouped rungs does warrant a test. Composition grouping scores higher than chemistry-cluster grouping by 0.047 to 0.079 depending on property. A Nadeau-Bengio corrected paired t-test over five repeats rejects chance for every property (p < 0.001 in all four cases). We report the difference itself rather than the t-statistic as the headline: with five repeats the test has four degrees of freedom, and a large t there is unremarkable, whereas a difference of 0.047 to 0.079 in R² is a substantial change in what the protocol reports. Chemistry-cluster grouping is therefore a measurably stricter anchor than composition grouping, which is why we adopt it as the honest reference throughout.



The practically important observation is that \*\*k-fold cross-validation offers no protection\*\*. Five-fold and ten-fold sit with the random split, not with the grouped rungs, because the leakage they fail to prevent is not fold-count dependent: it arises from the same material appearing, at different temperatures or under different dopant labels, on both sides of the split. A practitioner who moves from repeated random 80/20 holdout to ten-fold cross-validation in the belief that this makes the estimate more honest gains nothing.



Thermal conductivity inflates least, at 0.134. It is also the property that transfers best across databases (Section 3.5), consistent with κ being the most genuinely predictable of the four from composition alone.



\### 3.2 The inflation reflects test-set composition, not a shift in learned structure



Section 3.1 establishes that ungrouped validation reports higher accuracy than grouped validation. It does not establish why. Two mechanisms are consistent with the observation. The model might learn something different when near-duplicate materials are available in training, leaning on features that act as chemistry fingerprints and effectively identifying the test material rather than predicting its behaviour. Alternatively, the model might learn the same function in both cases and simply be evaluated on easier examples.



These are distinguishable through feature attribution. If the first mechanism operates, the relative importance the model assigns to different descriptor families should differ between the two protocols.



We fit the figure-of-merit model under both random and chemistry-cluster splits across all twenty-five folds of five repeats, with identical hyperparameters, and computed exact TreeSHAP attributions [@lundberg2020local] on a fixed 20,000-row subsample of each fold's held-out set. Attributions were normalised to shares of each model's total, since the two models have different prediction variances and raw magnitudes are not comparable. Shares were aggregated to descriptor families, which is robust to the arbitrary splitting of credit among correlated features that individual feature rankings suffer from (Figure 6).



![Figure 6](figures/shap_attribution.png)

*Figure 6. TreeSHAP attribution shares for zT, random versus chemistry-cluster splits, twenty-five folds per protocol: (a) three coarse descriptor families, (b) ten fine semantic groups sorted by mean share. Every pair overlaps within its across-fold standard deviation; the largest group difference (valence electron configuration) is 0.78 pooled fold-SD.*



The two protocols reproduce the expected R² gap: 0.9180 pooled under random splitting against 0.7456 under grouping, a difference of 0.172 that matches the ladder's value of 0.173 for this property.



\*\*Table 2.\*\* Attribution shares by descriptor family, zT, twenty-five folds per protocol.



| Family | random | chemistry cluster | Δ | Δ / pooled fold-SD |

|---|---|---|---|---|

| CBFV | 0.5377 ± 0.0049 | 0.5288 ± 0.0251 | −0.0089 | −0.49 |

| MAGPIE | 0.2654 ± 0.0056 | 0.2759 ± 0.0228 | +0.0105 | +0.63 |

| temperature | 0.1969 ± 0.0014 | 0.1953 ± 0.0058 | −0.0015 | −0.36 |



Their attribution shares are indistinguishable. Disaggregating into ten semantic groups changes nothing: the largest difference between protocols in any group is 0.78 times the pooled fold-to-fold standard deviation, and most fall below 0.6. No group separates.



\*\*The model does not rely on different descriptors when near-duplicates are available to it.\*\* It applies the same learned structure and is graded on an easier test set, because the random-split held-out fold contains materials that also appear, at other temperatures or under other dopant labels, in training. Validation inflation in this setting is a property of how the data is partitioned, not of what the model learns.



Two methodological observations accompany this result, both of which bear on how such comparisons should be conducted.



First, the number of folds matters more than it might appear. An earlier version of this analysis used a single repeat, five folds per protocol. On that evidence three of the ten semantic groups exceeded one fold-standard-deviation, the largest at 1.9, and the null result did not hold. Extending to twenty-five folds per protocol resolved it: every group fell below 0.8, and the three apparent effects disappeared. With five observations the standard deviation is itself poorly estimated, and differences of one to two such units cannot be distinguished from noise. We report this because the underpowered version was not obviously wrong on its face.



Second, the grouped protocol's fold-to-fold variance in attribution shares runs two to sixteen times the random protocol's, and does not narrow as the number of folds increases. This is not an estimation artefact. Under grouped cross-validation the composition of each fold genuinely differs, since which material families are held out varies between folds and repeats, whereas under random splitting every fold resembles every other. The same property that makes grouped R² estimates noisier shows up here in a different measurement.



This is a negative result and we report it as such: it rules out an alternative explanation rather than establishing a new mechanism. One structural caveat applies. The two models are necessarily trained on different row sets, and that difference is the phenomenon under study, so this compares two fitted models rather than ablating one. Averaging across twenty-five folds addresses sampling variation within each protocol but cannot remove that confound, which is inherent to comparing the protocols at all.



\### 3.3 A measured ceiling on achievable performance



Honest grouped performance of 0.70 to 0.81 is only interpretable against some notion of what is achievable. No model can predict labels more accurately than those labels are themselves reproducible, so the ceiling is set by label noise. We estimate it from two independent sources and combine them.



\*\*Measurement noise.\*\* Alleno et al. [@alleno2015round] report relative uncertainties from a round-robin in which the same specimen was measured by multiple laboratories: approximately 6% for the Seebeck coefficient, 8% for electrical resistivity (equivalently, to first order, conductivity), 11% for thermal conductivity and 19% for the figure of merit. The 19% figure for zT is the per-measurement standard uncertainty, the quantity that matches this dataset's row-level noise floor; Alleno et al. separately report a 17% expanded uncertainty on the mean of zT across the round-robin's repeated measurements, a different statistic that is not used here. Converting each to a noise variance and dividing by this dataset's own property variance in the matched space gives the expected upper bound on R² under label noise [@li2021performance], R²max = 1 − σ²noise/σ²total: 0.9974 for S, 0.9968 for σ, 0.9777 for κ and 0.9761 for zT. This estimate rests on a single skutterudite compound and excludes any error introduced by reading values from published figures.



\*\*Digitization noise.\*\* That second component can be measured directly. The training data is digitized from figures in the primary literature, and an independent group [@ryu2025highquality] has digitized an overlapping set of the same publications. Taking samples whose DOI appears in both databases and whose canonical composition matches, and comparing the two sets of labels directly without any model, gives agreement over 300 to 800 K of R² = 0.9639 for S, 0.9840 for σ, 0.9833 for κ, and 0.9807 or 0.9839 for zT depending on whether the digitized or reconstructed target is used. This is measured on 96 of 176 DOI-overlap samples, those whose compositions canonicalise identically on both sides.



Agreement between two noisy measurements understates the ceiling on predicting the underlying quantity, because it absorbs noise twice. Under equal and independent noise, R²max = (1 + R²agree)/2, which bounds the digitization-limited ceiling between the raw agreement and that corrected value.



\*\*Combined.\*\* The two components are independent — the round-robin measures inter-laboratory scatter on physical specimens and explicitly excludes figure-reading error — so their noise fractions add: (1 − R²comb) = (1 − R²meas) + (1 − R²dig). Table 3 gives the result (Figure 7).



![Figure 7](figures/fig5_headroom.png)

*Figure 7. Decomposition of each property's R² = 0 to 1 range into achieved (chemistry-cluster grouped R²), headroom to the combined label-noise ceiling, digitization noise and measurement noise, with the ungrouped random-80/20 R² marked to show how much apparent performance is validation artefact rather than real headroom closed.*



\*\*Table 3.\*\* Combined label-noise ceiling and remaining headroom.



| Target | R²max (measurement) | Digitization ceiling | Combined | Grouped R² | Headroom |

|---|---|---|---|---|---|

| S | 0.9974 | 0.9639–0.9819 | 0.9613–0.9794 | 0.7528 | 0.209–0.227 |

| σ | 0.9968 | 0.9840–0.9920 | 0.9807–0.9888 | 0.7020 | 0.279–0.287 |

| κ | 0.9777 | 0.9833–0.9916 | 0.9609–0.9693 | 0.8092 | 0.152–0.160 |

| zT (declared) | 0.9761 | 0.9807–0.9903 | 0.9568–0.9665 | 0.7456 | 0.211–0.221 |

| zT (recomputed) | 0.9761 | 0.9839–0.9919 | 0.9600–0.9680 | 0.7456 | 0.214–0.222 |



Combined ceilings fall between 0.96 and 0.99, leaving \*\*headroom of 0.15 to 0.29 R² on every property\*\*.



Three qualifications. The digitization term was measured on a subset whose target variance is narrower than the full database, which makes the combined ceiling conservative rather than optimistic. It shifted by up to 0.02 between two snapshots of the same source database taken one week apart, so the floor is quoted to two decimals at most. And both components read the same printed curve: neither captures synthesis-to-synthesis variation or processing differences between nominally identical specimens. The true label-noise floor is therefore larger than what is measured here, and the reported headroom is a lower bound on the gap that remains.



\### 3.4 Tripling the descriptor count closes under five percent of the remaining gap



A natural explanation for the headroom in Section 3.3 is that the descriptor set is too small. The pipeline uses 397 features, combining 132 MAGPIE attributes with 264 CBFV features and a temperature bin, and it would be reasonable to suppose that a richer representation would recover part of that gap.



To test this we refit each target under chemistry-cluster grouped CV using three feature sets: MAGPIE alone (133 features including temperature), CBFV alone (265), and the full combination (397). Hyperparameters were those tuned on the full feature set and were not retuned for the subsets, so that any difference reflects the descriptors rather than the tuning. All other settings, including folds, seeds and repeat structure, were held identical to the ladder (Figure 8).



![Figure 8](figures/descriptor_ablation.png)

*Figure 8. Chemistry-cluster grouped R² at 133 (MAGPIE), 265 (CBFV) and 397 (full) features, one panel per property, against the combined label-noise ceiling band; the full-minus-MAGPIE gain closes 2.2 to 4.2% of the remaining headroom on every property.*



\*\*Table 4.\*\* Descriptor ablation under chemistry-cluster grouped CV.



| Target | MAGPIE (133) | CBFV (265) | Full (397) | Δ full−MAGPIE | % of headroom |

|---|---|---|---|---|---|

| S | 0.7464 ± 0.0039 | 0.7502 ± 0.0029 | 0.7528 ± 0.0050 | +0.0065 | 2.9–3.1 |

| σ | 0.6919 ± 0.0011 | 0.7018 ± 0.0020 | 0.7020 ± 0.0020 | +0.0101 | 3.5–3.6 |

| κ | 0.8028 ± 0.0025 | 0.8048 ± 0.0026 | 0.8092 ± 0.0021 | +0.0064 | 4.0–4.2 |

| zT | 0.7406 ± 0.0024 | 0.7443 ± 0.0042 | 0.7456 ± 0.0045 | +0.0049 | 2.2–2.3 |



Adding 264 CBFV features to the 132 MAGPIE attributes changes R² by 0.0049 to 0.0101 depending on property. These differences are systematic rather than noise; at this sample size the across-repeat standard deviations are between 0.0011 and 0.0050, and most differences exceed twice their standard deviation. But statistical detectability at this sample size is close to free, and the quantity that matters is not whether the improvement is real but whether it is useful.



Measured against the headroom to the combined ceiling, it is not. \*\*Tripling the descriptor count closes between 2.2 and 4.2 percent of the gap that remains to the label-noise limit.\*\* On the current trajectory, closing the remaining headroom by adding composition-derived descriptors would require a representation orders of magnitude larger than anything in use.



The saturation is symmetric. CBFV alone, at 265 features, comes within 0.0002 to 0.0044 of the full set on every property, so MAGPIE contributes almost nothing on top of CBFV just as CBFV contributes almost nothing on top of MAGPIE. Two descriptor families built on different principles converge on nearly the same predictive content, which is what one expects when both are functions of the same underlying variable and that variable has been exhausted.



This also disposes of a natural methodological question. With 397 features and no feature selection, one might ask whether a reduced set chosen by importance would perform comparably or better. The ablation answers it: selection operates within a representation that has already saturated at roughly a third of its size, so it cannot recover headroom that the representation does not contain. We note additionally that gradient-boosted trees are largely insensitive to uninformative features, which cost training time rather than accuracy, and that any selection performed outside the training fold would reintroduce precisely the leakage this paper is concerned with.



The conclusion is that the gap between honest performance and the noise ceiling is not a descriptor-count problem. Section 3.5 identifies what it is instead.



\### 3.5 Transfer to independent databases degrades further, and much of the loss is extrapolation rather than failure to generalise



Grouped cross-validation removes materials from training that appear in test, but both sides still come from one database, digitized by one group under one set of curation decisions. Whether grouped performance survives a change of source is a separate question, and it is the one a practitioner applying a published model to their own measurements actually faces.



We test against two independent databases. ESTM [@na2022public] is a separately compiled thermoelectric dataset. teMatDb [@ryu2025highquality; @ryu2025tematdb] is an independent digitization effort by a different group, drawing on overlapping primary literature but producing its own values from its own reading of the published figures. Neither shares rows with the training data.



Models were refit on the full training set with frozen hyperparameters and applied once to each external set. Smearing factors [@duan1983smearing] correcting the log-space back-transformation were computed from training-side out-of-fold residuals and held fixed; external labels were never used for calibration. Each external set was partitioned by source publication and by chemistry cluster, and the resulting strata were scored separately and never pooled.



\*\*Table 5.\*\* External transfer on ESTM. Full-set and in-support results, the latter restricted to rows within training's per-property range in S, σ and κ simultaneously.



| Stratum | Property | Full R² | n | In-support R² | n | OOD fraction |

|---|---|---|---|---|---|---|

| DOI-disjoint | S | 0.5664 | 3,123 | 0.7404 | 2,709 | 2.4% |

| DOI-disjoint | σ | 0.3988 | 3,123 | 0.6132 | 2,709 | 12.0% |

| DOI-disjoint | κ | 0.6892 | 3,123 | 0.7314 | 2,709 | 2.9% |

| DOI-disjoint | zT direct | 0.6615 | 3,123 | 0.6686 | 2,709 | — |

| DOI-disjoint | zT derived | 0.2064 | 3,123 | 0.3286 | 2,709 | — |

| cluster-disjoint | S | 0.3536 | 1,448 | 0.5115 | 1,196 | 2.3% |

| cluster-disjoint | σ | 0.2746 | 1,448 | 0.4085 | 1,196 | 15.5% |

| cluster-disjoint | κ | 0.6184 | 1,448 | 0.6565 | 1,196 | 4.5% |

| cluster-disjoint | zT direct | 0.4982 | 1,448 | 0.5343 | 1,196 | — |

| cluster-disjoint | zT derived | −0.0959 | 1,448 | −0.0814 | 1,196 | — |



\*\*Transfer degrades substantially on both databases.\*\* On ESTM, for samples whose chemistry cluster is absent from training, R² falls to 0.354 for the Seebeck coefficient, 0.275 for electrical conductivity, 0.618 for thermal conductivity and 0.498 for the figure of merit, against internal grouped values of 0.753, 0.702, 0.809 and 0.746. teMatDb degrades comparably on samples from publications absent from training: 0.699, 0.175, 0.653 and 0.496. A second gap therefore exists beyond the one Section 3.1 measures, of similar or larger magnitude (Figure 9).



![Figure 9](figures/external_transfer.png)

*Figure 9. Internal chemistry-cluster, external full-set and external in-support R² by property: (a) ESTM DOI-disjoint, (b) ESTM cluster-disjoint, (c) teMatDb DOI-disjoint (no in-support split; out-of-support tail 0.12%). Out-of-support fraction annotated above each property group; zT derived is excluded as numerically unstable. teMatDb's 27-cluster chemistry-disjoint stratum is not shown.*



\*\*A large share of that loss is extrapolation, not failure to generalise.\*\* External datasets extend beyond the property ranges the training data covers, and the model is being asked to predict outside its support. Restricting to rows within training's per-property range in all of S, σ and κ simultaneously, 13.3% of ESTM's DOI-disjoint rows and 17.4% of its cluster-disjoint rows fall outside. Electrical conductivity dominates that exclusion: 12.0% and 15.5% of rows sit below training's cleaned conductivity floor of roughly 959 S m⁻¹, against 2 to 5% for the other two properties (Figure 10). Temperature contributes nothing, because the 300–800 K window is enforced on both sides before anything else runs.



![Figure 10](figures/sigma_extrapolation.png)

*Figure 10. Density of log10(electrical conductivity) for the training set and for ESTM's cluster-disjoint rows, with training's cleaned conductivity floor (≈959 S m⁻¹) marked; 15.5% of ESTM cluster-disjoint mass falls below it.*



Within support, performance recovers markedly. For the cluster-disjoint stratum, R² rises from 0.354 to 0.512 for S, from 0.275 to 0.409 for σ, from 0.618 to 0.657 for κ and from 0.498 to 0.534 for zT. The gap to internal grouped performance narrows correspondingly, to 0.241, 0.294, 0.153 and 0.211.



The interpretation matters for what the result means in practice. \*\*A model that fails on out-of-support inputs is not the same as a model that fails to generalise.\*\* Roughly a third of σ's apparent external failure is the model being asked to predict conductivities an order of magnitude below anything it was trained on. The remaining 0.29 is genuine cross-database degradation on inputs the model should in principle handle, and it is that residual, not the headline full-set number, that the second inflation gap should be read as.



\*\*Degradation is not uniform across databases.\*\* ESTM transfers worse on the Seebeck coefficient; teMatDb transfers worse on electrical conductivity. The databases differ in curation, teMatDb applying a self-consistency filter that ESTM does not, and in chemistry breadth. We report both rather than averaging them, and note that characterising this heterogeneity properly would require more external sets than two.



Two constraints on how these numbers should be read. teMatDb's chemistry-disjoint stratum contains 27 clusters across 412 rows, with only four material families represented by three or more samples, and per-property R² there ranges from +0.96 to −49 depending on family. We report that stratum's sizes as an inventory finding and do not quote its R² values, which cannot be estimated from a sample of that size. Separately, zT reconstructed from predicted components is numerically unstable: the frozen-smear correction changes derived-zT by under 0.005 in a direction that is not consistent between runs differing only in hyperparameter choice.



Finally, the chemistry-cluster rule collapses dopants below 5 at% but not alloy stoichiometry, so Bi₂Te₂.₇Se₀.₃, where selenium occupies 6.0 at%, forms a cluster distinct from Bi₂Te₃, while (PbTe)₀.₉₇(SrTe)₀.₀₂(Na₂Te)₀.₀₁ collapses into PbTe. The cluster-disjoint strata therefore test unseen stoichiometry within familiar host lattices rather than genuinely novel chemistry. It is the stricter of the two available tests, not an absolute one.



\### 3.6 Predicting zT directly outperforms reconstructing it from components



The figure of merit can be predicted directly or assembled from separately predicted S, σ and κ via S²σT/κ. We compare both on the subset where all four properties are reported, 56,088 rows across 4,139 chemistry clusters, under the same grouped protocol, each model with its own target's frozen hyperparameters (Figure 11).



![Figure 11](figures/zt_direct_vs_derived.png)

*Figure 11. Predicted versus actual zT, direct prediction (R² = 0.7267) versus reconstruction from S, σ and κ via S²σT/κ (R² = 0.5403), identical axes and colour scale, n = 280,440 pooled across five repeats of the 56,088-row subset.*



Direct prediction gives pooled R² = 0.7267; reconstruction gives 0.5403, a gap of 0.186. Giving all four models the figure-of-merit hyperparameters instead, so that the two pathways differ only in what is predicted, gives a gap of 0.201; the ordering does not depend on the hyperparameter choice. The component models are individually adequate: S at 0.8198, σ at 0.6920 and κ at 0.8209 in their respective spaces. Combining three imperfect predictions through a product of powers compounds their errors, and the residuals of σ and κ are positively correlated at 0.43, so the errors reinforce rather than cancel.



The gap is not an artefact of the log-space back-transformation. Applying Duan's smearing correction [@duan1983smearing] changes reconstructed R² by −0.0052, and by +0.0023 when all four models share the figure-of-merit hyperparameters; the direction is not stable and the magnitude is under 3% of the gap. Tail contributions are similar for both pathways, with the worst 1% of rows carrying 23.0% of squared error for direct prediction and 21.8% for reconstruction, so the difference is a broadly wider residual distribution rather than a small number of catastrophic failures.



The same ordering holds on both external databases and on every stratum. Direct prediction should be preferred where a direct target is available.



# Discussion



\## 4. Discussion



\### 4.1 What this means for reported performance



The practical implication is direct. Thermoelectric property models evaluated by random splitting or k-fold cross-validation report R² values 0.134 to 0.213 higher than the same models evaluated under chemistry-cluster grouping, with the gap averaging 0.181 across the four properties. Increasing the number of folds does not reduce it. A reader encountering a reported R² above 0.9 for a composition-based thermoelectric model has no way to tell, from the number alone, whether it reflects predictive capability or the fact that the same material appears on both sides of the split at different temperatures.



One other study reports a comparable split in thermoelectrics. On ESTM, a separately curated database whose source literature partly overlaps Starrydata2's, holding out compositions lowered R² for the Seebeck coefficient by 0.094 from a single random split [@ho2026physicsinspired]; the corresponding drop here is 0.126. Electrical and thermal conductivity are scored in linear units there and in log10 here, so those drops are not comparable, and neither study's absolute values bear comparison, since data, descriptors and protocols differ.



The remedy is not costly. Grouping by chemistry cluster requires a formula parser and a threshold, and the computational cost is identical to ungrouped cross-validation. What it costs is the reported number.



\### 4.2 Where the remaining gap lives



Honest grouped performance of 0.70 to 0.81 sits 0.15 to 0.29 below a label-noise ceiling built from two independently measured components. Section 3.4 rules out descriptor count as the binding constraint: adding 264 CBFV features to 132 MAGPIE attributes closes at most 4.2% of that headroom, and the two descriptor families are nearly interchangeable, each coming within 0.005 of their union on every property.



What remains is a property of the representation rather than its size. Thermoelectric transport depends on structure, texture, grain size and processing history, none of which a composition-and-temperature feature vector can express. Two specimens of identical stated composition, prepared differently, can differ in electrical conductivity by more than the model's dynamic range. No amount of additional composition-derived descriptors resolves this, because the information is not present in the composition.



The ceiling we report is conservative in a specific way worth stating. Both noise components read measurements from the same published figure: the inter-laboratory round-robin measures apparatus scatter, and the cross-database digitization comparison measures reading error. Neither captures synthesis-to-synthesis variation between nominally identical specimens. The true label-noise floor is therefore higher than the one we measure, and the reported headroom is a lower bound on the gap that remains.



\### 4.3 Extrapolation versus generalisation in cross-database transfer



Section 3.5 finds a second gap beyond the validation-protocol gap, but decomposes it. Between 13 and 17 percent of external rows fall outside the property ranges the training data covers, almost entirely in electrical conductivity, where the external tail reaches an order of magnitude below training's cleaned floor. Restricting to rows within support recovers a substantial share of the apparent loss: σ improves from 0.275 to 0.409 on the cluster-disjoint stratum.



The distinction matters for how such results should be read. A model asked to predict conductivities below anything in its training range is not failing to generalise; it is being used outside its domain of validity. The residual after that restriction, 0.15 to 0.29 depending on property, is the genuine cross-database degradation, and it is that figure rather than the full-set number that should be compared against the internal validation gap.



Transfer is also heterogeneous between databases. ESTM degrades most on the Seebeck coefficient, teMatDb most on electrical conductivity. The two differ in curation and chemistry breadth, and characterising this heterogeneity properly would require more external datasets than the two available here. We report both rather than averaging them.



\### 4.4 Two leakage defects in our own pipeline



During preparation of this manuscript we found two defects in the grouping implementation. The cluster-identifier defect made the honest anchor less strict than its stated definition; the fold-assignment defect broke the independence of the repeats. We report them because the direction of the cluster-identifier error is instructive and because the paper's central claim is precisely that this class of mistake is easy to make and hard to notice.



The first concerned the cluster identifier. After removing sub-threshold dopants, the implementation reduced the remaining composition to its canonical formula, which does not collapse near-integer stoichiometries. Pb₀.₉₇Te therefore received a different cluster identifier from PbTe, and the two could be split across folds. Because stoichiometries reported to two decimal places are common in the literature, this affected 79% of rows: CoSb₃ fragmented into 157 pseudo-clusters, PbTe into 170.



The second concerned fold assignment. Groups were bin-packed by descending size into the fold with the smallest running total, which meant the largest group was always assigned to the first fold, the second largest to the second, and so on, regardless of the per-repeat shuffle. Repeats were therefore not independent partitions for exactly the rows that dominate the result.



To check how much of the change came from fold assignment alone, we reran the chemistry-cluster rung with the corrected fold assignment and the original cluster identifier. Across the four properties the rung moved by −0.0022 to +0.0011, with no consistent sign, and the across-repeat standard deviation changed by less than 0.001. The decrease appeared only once the snapped identifier was introduced. We did not run the reverse control (original fold assignment with the snapped identifier), so an interaction between the two corrections, which would act mainly through the large clusters the snapped identifier creates, is not separated.



Together, the two corrections lowered the chemistry-cluster rung by 0.037 to 0.058 across the four properties and widened the validation gap correspondingly, from a mean of 0.131 to 0.181, both figures computed on the same rows, since the pre-fix chemistry-cluster checkpoint and the post-fix ungrouped rerun both use File A's row set. Every chemistry-cluster number we had previously computed was therefore optimistic, and every gap measured against one was a lower bound.



Neither defect produced an error, a warning, or an implausible result. In both cases the code ran, reported a number, and the number was wrong in the direction that flatters the analysis. That is the same failure mode this paper documents in published validation practice, occurring in a codebase written specifically to avoid it.



\### 4.5 Limitations



Chemistry-cluster grouping is stricter than composition grouping but is not family-level grouping. The 5 at% threshold collapses dopants and near-integer stoichiometric variation but not alloy composition, so Bi₀.₅Sb₁.₅Te₃ and Bi₀.₄Sb₁.₆Te₃ remain distinct clusters. A large fraction of clusters contain a single composition. Cross-family generalisation is therefore not tested here and requires a coarser grouping than we adopt.



The measurement-noise component of the ceiling derives from a round-robin on a single skutterudite compound and is applied to all four properties across all chemistries. This is the best available estimate in the literature but it is not a per-chemistry figure.



The digitization-noise component was measured on 96 of 176 candidate samples, those whose compositions canonicalise identically in both databases. Samples that match are those with simpler formulas, which may also be easier to digitize accurately, so this component may be optimistic. It also shifted by up to 0.02 between two snapshots of the same source database taken one week apart.



Reconstructing zT from separately predicted components is numerically unstable in a way that limits what can be claimed about it. Direct prediction outperforms reconstruction on every stratum of both external databases and by 0.19 internally, and that ordering is robust; the reconstructed values themselves move substantially under small changes to the back-transformation correction, and we report the ordering rather than the magnitudes.



Hyperparameters were selected by grouped cross-validation on the same rows later used for evaluation, so the evaluation is not nested and grouped R² is optimistic by an amount we did not measure. The search was small: one Optuna search per target, 20 trials over eight XGBoost hyperparameters, each scored by three-fold grouped cross-validation. Two biases of known direction run against our conclusions. Selection on the evaluation rows raises grouped R², which understates both the validation gap and the headroom to the label-noise ceiling. Applying hyperparameters tuned on the full 397-feature set to the MAGPIE-only and CBFV-only subsets, without retuning, penalises the subsets, so the 4.2% of headroom closed by the full descriptor set is an upper bound. A third effect has unknown sign: tuning used the cluster identifier as it stood before the snapping correction described in Section 4.4, so its inner folds were less strict than the grouping used for evaluation.



Finally, the external chemistry-disjoint strata are small. teMatDb's contains 27 clusters across 412 rows with four material families represented by three or more samples. We report its composition as an inventory finding and do not estimate per-property performance from it.



# Conclusion

\## 5. Conclusion



Reported accuracy for composition-based thermoelectric property prediction depends heavily on the validation protocol. Moving from ungrouped to chemistry-cluster grouped cross-validation lowers R² by 0.134 to 0.213 across the four transport properties, and increasing the number of folds in an ungrouped protocol does not reduce the gap at all. Feature attributions are indistinguishable between the two protocols, which locates the effect in how the data is partitioned rather than in what the model learns.



Honest grouped performance of 0.70 to 0.81 sits 0.15 to 0.29 below a label-noise ceiling assembled from inter-laboratory measurement scatter and cross-database digitization agreement. That gap is not closed by adding descriptors: tripling the feature count from 133 to 397 recovers under 4.2% of it on every property, and two descriptor families built on different principles are nearly interchangeable. The limit is what a composition-and-temperature representation can express, not how much of it is used.



Transfer to independent databases degrades further, though between a third and a half of the apparent loss is extrapolation beyond the property ranges the training data covers rather than failure to generalise on inputs the model should handle.



For practitioners, the recommendation is specific: report grouped cross-validation, state the grouping rule, and report the fraction of any external evaluation that falls outside the training range. For the field, the implication is that current composition-based models are further from the achievable limit than published figures suggest, and that closing the remaining distance will require representations that encode structure and processing rather than larger sets of composition-derived features.



# 

