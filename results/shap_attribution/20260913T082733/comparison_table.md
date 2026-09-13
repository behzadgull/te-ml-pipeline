# SHAP Attribution Comparison -- zT, random vs chemistry-cluster (repeat 0, all 5 folds)

Generated: 20260913T082733. Dataset: `data/processed/featurized_ThermoelectricMaterials_2026-08-15.csv` (sha256 e97406fa5223...). Device: cuda.

## Outer R^2 per fold and pooled (D2)

| split_strategy | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | pooled |
|---|---|---|---|---|---|---|
| random | 0.9155 | 0.9156 | 0.9187 | 0.9188 | 0.9164 | 0.9170 |
| chemistry | 0.8018 | 0.7791 | 0.7981 | 0.8181 | 0.7911 | 0.7973 |

### Coarse group comparison (D6, D7)

| group | random mean share +/- SD | chemistry mean share +/- SD | delta (chem-rand) | delta / pooled SD |
|---|---|---|---|---|
| cbfv | 0.5385 +/- 0.0052 | 0.5297 +/- 0.0191 | -0.0088 | -0.63 |
| magpie | 0.2647 +/- 0.0062 | 0.2731 +/- 0.0212 | +0.0083 | +0.54 |
| temperature | 0.1968 +/- 0.0016 | 0.1972 +/- 0.0047 | +0.0004 | +0.12 |

### Fine descriptor-semantic group comparison (D6, D7)

| group | random mean share +/- SD | chemistry mean share +/- SD | delta (chem-rand) | delta / pooled SD |
|---|---|---|---|---|
| atomic_radius | 0.1942 +/- 0.0042 | 0.1859 +/- 0.0164 | -0.0083 | -0.69 |
| dft_groundstate | 0.0218 +/- 0.0014 | 0.0289 +/- 0.0146 | +0.0071 | +0.68 |
| electronegativity | 0.0708 +/- 0.0033 | 0.0668 +/- 0.0070 | -0.0040 | -0.74 |
| periodic_position | 0.1116 +/- 0.0024 | 0.1152 +/- 0.0071 | +0.0036 | +0.68 |
| thermodynamic_bulk | 0.1535 +/- 0.0023 | 0.1561 +/- 0.0067 | +0.0026 | +0.52 |
| valence_electron_config | 0.1917 +/- 0.0043 | 0.1906 +/- 0.0193 | -0.0011 | -0.08 |
| atomic_mass | 0.0174 +/- 0.0019 | 0.0166 +/- 0.0023 | -0.0008 | -0.35 |
| temperature | 0.1968 +/- 0.0016 | 0.1972 +/- 0.0047 | +0.0004 | +0.12 |
| metal_class | 0.0231 +/- 0.0006 | 0.0234 +/- 0.0030 | +0.0004 | +0.16 |
| melting_point | 0.0191 +/- 0.0008 | 0.0194 +/- 0.0014 | +0.0002 | +0.21 |

`other` fine-group column count: 0 (expected 0 -- every column should resolve to a named semantic group; see summary.json's unmapped_fine_group_columns if nonzero).