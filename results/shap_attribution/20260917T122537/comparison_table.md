# SHAP Attribution Comparison -- zT, random vs chemistry-cluster (repeat 0, all 5 folds)

Generated: 20260917T122537. Dataset: `data\processed\featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv` (sha256 d9fc1e5d942e...). Device: cpu.

## Outer R^2 per fold and pooled (D2)

| split_strategy | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | pooled |
|---|---|---|---|---|---|---|
| random | 0.9165 | 0.9163 | 0.9190 | 0.9189 | 0.9178 | 0.9177 |
| chemistry | 0.7745 | 0.7613 | 0.6879 | 0.6780 | 0.8032 | 0.7442 |

### Coarse group comparison (D6, D7)

| group | random mean share +/- SD | chemistry mean share +/- SD | delta (chem-rand) | delta / pooled SD |
|---|---|---|---|---|
| magpie | 0.2648 +/- 0.0064 | 0.2800 +/- 0.0207 | +0.0152 | +0.99 |
| cbfv | 0.5387 +/- 0.0066 | 0.5247 +/- 0.0199 | -0.0140 | -0.95 |
| temperature | 0.1964 +/- 0.0016 | 0.1953 +/- 0.0032 | -0.0011 | -0.45 |

### Fine descriptor-semantic group comparison (D6, D7)

| group | random mean share +/- SD | chemistry mean share +/- SD | delta (chem-rand) | delta / pooled SD |
|---|---|---|---|---|
| atomic_radius | 0.1966 +/- 0.0048 | 0.1753 +/- 0.0349 | -0.0213 | -0.85 |
| periodic_position | 0.1092 +/- 0.0037 | 0.1273 +/- 0.0373 | +0.0181 | +0.68 |
| valence_electron_config | 0.1939 +/- 0.0054 | 0.1761 +/- 0.0181 | -0.0178 | -1.33 |
| dft_groundstate | 0.0224 +/- 0.0014 | 0.0330 +/- 0.0284 | +0.0106 | +0.53 |
| electronegativity | 0.0704 +/- 0.0020 | 0.0631 +/- 0.0072 | -0.0073 | -1.38 |
| thermodynamic_bulk | 0.1550 +/- 0.0039 | 0.1616 +/- 0.0138 | +0.0066 | +0.65 |
| metal_class | 0.0217 +/- 0.0024 | 0.0274 +/- 0.0099 | +0.0056 | +0.78 |
| melting_point | 0.0171 +/- 0.0003 | 0.0207 +/- 0.0026 | +0.0035 | +1.88 |
| atomic_mass | 0.0171 +/- 0.0012 | 0.0202 +/- 0.0044 | +0.0031 | +0.95 |
| temperature | 0.1964 +/- 0.0016 | 0.1953 +/- 0.0032 | -0.0011 | -0.45 |

`other` fine-group column count: 0 (expected 0 -- every column should resolve to a named semantic group; see summary.json's unmapped_fine_group_columns if nonzero).