# Combined ceiling and headroom, recomputed against the snapfix chemistry rung

Generated 20260917T172251. Read-only except for this results artifact -- no src/ edits, no refits, no git operations.

## Item 1: R2_max (measurement noise floor) rerun, snapfix cleaned dataset

Input: `checkpoints\saved_predictions\te-ml-pipeline\data\processed\cleaned_ThermoelectricMaterials_2026-08-15.csv`
SHA256: `0275c5088521580a1156acf2078f77f0eb7d4a8c6b4e724864f0de0875d8a393`  (280,664 rows)

This is the SAME file used for `results/noise_floor/20260911T114356/` -- the snapfix regeneration only changed the `chemistry_cluster_id` column of the *featurized* CSV; `data_cleaning.py` (which produces the cleaned CSV `compute_all()` reads) was never rerun. There is no separate 'snapfix cleaned' file to point at; this cleaned CSV IS the correct input by inheritance, confirmed by the identical SHA256.

| Property | Space | sigma_noise | sigma_total | n_used | R2_max | delta vs 20260911T114356 |
|---|---|---|---|---|---|---|
| S | log | 0.060000 | 0.902736 | 185,409 | 0.9955824595 | +0.00e+00 |
| S | linear | 8.987760 | 176.926274 | 185,417 | 0.9974194144 | +0.00e+00 |
| sigma | log | 0.080000 | 1.408410 | 183,014 | 0.9967735728 | +0.00e+00 |
| sigma | linear | 3622.318918 | 129589.496254 | 183,014 | 0.9992186713 | +0.00e+00 |
| kappa | log | 0.110000 | 0.736324 | 121,247 | 0.9776824302 | +0.00e+00 |
| kappa | linear | 0.219798 | 1.985001 | 121,247 | 0.9877389961 | +0.00e+00 |
| zT | log | 0.180000 | 1.482076 | 129,582 | 0.9852495876 | +0.00e+00 |
| zT | linear | 0.056842 | 0.388215 | 129,633 | 0.9785614718 | +0.00e+00 |

**Max |delta| across all 16 property/space/quantity cells: 0.00e+00 -- exactly zero, well under the 0.001 flag threshold.** Expected: byte-identical input file, so byte-identical output. Confirmed, not merely close.

## Item 2: digitization ceiling, confirmed unchanged

Source: `results/external_snapfix/20260917T160553/tematdb_inventory_snapfix.json`, `C3_composition_matched_digitization_agreement.N3_label_agreement_300_800K` (verified directly from the file, not from this task's prompt text).

| Property | R2_agree (300-800K) |
|---|---|
| S | 0.9639 |
| sigma | 0.9840 |
| kappa | 0.9833 |
| zT_declared | 0.9807 |
| zT_tep | 0.9839 |

## Item 3: combined ceiling, rebuilt against the NEW chemistry rung

Formulas (identical to CLAUDE.md's COMBINED CEILING section):
```
upper_dig = (1 + R2_agree) / 2
R2_comb   = 1 - ((1 - R2_meas) + (1 - R2_dig))     # additive noise fractions
headroom  = R2_comb - confirmed_ceiling             # confirmed_ceiling = NEW chemistry rung R2
```

### S

R2_meas = 0.9974194144  (noise-floor R2_max, paper scale)
R2_dig lower bound = 0.9639  (raw R2_agree)
R2_dig upper bound = (1 + 0.9639) / 2 = 0.9819496296
R2_comb (lower) = 1 - ((1 - 0.997419) + (1 - 0.9639)) = 0.9613
R2_comb (upper) = 1 - ((1 - 0.997419) + (1 - 0.981950)) = 0.9794
confirmed ceiling (NEW chemistry rung) = 0.7528
headroom (lower) = 0.9613 - 0.7528 = 0.2085
headroom (upper) = 0.9794 - 0.7528 = 0.2266

### sigma

R2_meas = 0.9967735728  (noise-floor R2_max, paper scale)
R2_dig lower bound = 0.9840  (raw R2_agree)
R2_dig upper bound = (1 + 0.9840) / 2 = 0.9919828345
R2_comb (lower) = 1 - ((1 - 0.996774) + (1 - 0.9840)) = 0.9807
R2_comb (upper) = 1 - ((1 - 0.996774) + (1 - 0.991983)) = 0.9888
confirmed ceiling (NEW chemistry rung) = 0.7020
headroom (lower) = 0.9807 - 0.7020 = 0.2787
headroom (upper) = 0.9888 - 0.7020 = 0.2868

### kappa

R2_meas = 0.9776824302  (noise-floor R2_max, paper scale)
R2_dig lower bound = 0.9833  (raw R2_agree)
R2_dig upper bound = (1 + 0.9833) / 2 = 0.9916310039
R2_comb (lower) = 1 - ((1 - 0.977682) + (1 - 0.9833)) = 0.9609
R2_comb (upper) = 1 - ((1 - 0.977682) + (1 - 0.991631)) = 0.9693
confirmed ceiling (NEW chemistry rung) = 0.8092
headroom (lower) = 0.9609 - 0.8092 = 0.1517
headroom (upper) = 0.9693 - 0.8092 = 0.1601

### zT (vs ZT_author_declared)

R2_meas = 0.9785614718  (noise-floor R2_max, paper scale)
R2_dig lower bound = 0.9807  (raw R2_agree)
R2_dig upper bound = (1 + 0.9807) / 2 = 0.9903443650
R2_comb (lower) = 1 - ((1 - 0.978561) + (1 - 0.9807)) = 0.9593
R2_comb (upper) = 1 - ((1 - 0.978561) + (1 - 0.990344)) = 0.9689
confirmed ceiling (NEW chemistry rung) = 0.7456
headroom (lower) = 0.9593 - 0.7456 = 0.2137
headroom (upper) = 0.9689 - 0.7456 = 0.2233

### zT (vs recomputed alpha^2*T/(rho*kappa))

R2_meas = 0.9785614718  (noise-floor R2_max, paper scale)
R2_dig lower bound = 0.9839  (raw R2_agree)
R2_dig upper bound = (1 + 0.9839) / 2 = 0.9919337214
R2_comb (lower) = 1 - ((1 - 0.978561) + (1 - 0.9839)) = 0.9624
R2_comb (upper) = 1 - ((1 - 0.978561) + (1 - 0.991934)) = 0.9705
confirmed ceiling (NEW chemistry rung) = 0.7456
headroom (lower) = 0.9624 - 0.7456 = 0.2168
headroom (upper) = 0.9705 - 0.7456 = 0.2249

### Combined ceiling table, summary

| Target | R2_max (measurement) | Digitization ceiling | Combined | Confirmed (NEW) | Headroom |
|---|---|---|---|---|---|
| S | 0.9974 | 0.9639-0.9819 | 0.9613-0.9794 | 0.7528 | 0.2085-0.2266 |
| sigma | 0.9968 | 0.9840-0.9920 | 0.9807-0.9888 | 0.7020 | 0.2787-0.2868 |
| kappa | 0.9777 | 0.9833-0.9916 | 0.9609-0.9693 | 0.8092 | 0.1517-0.1601 |
| zT (vs ZT_author_declared) | 0.9786 | 0.9807-0.9903 | 0.9593-0.9689 | 0.7456 | 0.2137-0.2233 |
| zT (vs recomputed alpha^2*T/(rho*kappa)) | 0.9786 | 0.9839-0.9919 | 0.9624-0.9705 | 0.7456 | 0.2168-0.2249 |

## Item 4: old vs new headroom

| Target | OLD headroom | NEW headroom | change (lower) | change (upper) |
|---|---|---|---|---|
| S | 0.154-0.172 | 0.2085-0.2266 | +0.0545 | +0.0546 |
| sigma | 0.221-0.229 | 0.2787-0.2868 | +0.0577 | +0.0578 |
| kappa | 0.115-0.123 | 0.1517-0.1601 | +0.0367 | +0.0371 |
| zT (vs ZT_author_declared) | 0.162-0.172 | 0.2137-0.2233 | +0.0517 | +0.0513 |
| zT (vs recomputed alpha^2*T/(rho*kappa)) | 0.166-0.174 | 0.2168-0.2249 | +0.0508 | +0.0509 |

Headroom widened substantially (roughly +0.037 to +0.058) on every target/comparison -- entirely because the confirmed chemistry-cluster ceiling dropped under the corrected grouping (S5 fold fix + SNAP(0.05) cluster fix + snapfix dataset), while R2_meas and R2_dig are both grouping-independent and moved by exactly zero (item 1) and by less than 0.0001 (item 2), respectively.

## Item 5: descriptor-ablation headroom fractions, recomputed against the NEW headroom

| Target | full-minus-magpie delta | NEW headroom | NEW fraction | OLD fraction |
|---|---|---|---|---|
| S | 0.0064 | 0.2085-0.2266 | 2.82%-3.07% | 2.2-2.5% |
| sigma | 0.0101 | 0.2787-0.2868 | 3.52%-3.62% | 4.5-4.7% |
| kappa | 0.0064 | 0.1517-0.1601 | 4.00%-4.22% | 3.4-3.6% |
| zT (declared) | 0.0050 | 0.2137-0.2233 | 2.24%-2.34% | 1.6-1.7% |
| zT (recomputed TEP) | 0.0050 | 0.2168-0.2249 | 2.22%-2.31% | 1.6-1.7% |

Descriptor headroom fractions move in both directions relative to the old table: S, kappa, and zT rise modestly (bigger delta and/or headroom didn't grow proportionally as much); sigma's fraction actually FALLS (4.5-4.7% -> 3.52-3.62%) despite a similar absolute delta, because sigma's headroom widened the most of any property (+0.058-0.058) -- the same absolute descriptor gain now closes a smaller share of a bigger remaining gap. The qualitative conclusion is unchanged: full descriptor coverage over magpie-only still closes well under 5% of headroom on every target.
