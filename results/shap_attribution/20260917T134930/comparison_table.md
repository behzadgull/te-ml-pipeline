# SHAP Attribution Comparison -- zT, random vs chemistry-cluster (repeat 0, all 5 folds)

Generated: 20260917T134930. Dataset: `data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv` (sha256 d9fc1e5d942e...). Device: cuda.

## Outer R^2 per fold and pooled (D2)

| split_strategy | r0f0 | r0f1 | r0f2 | r0f3 | r0f4 | r1f0 | r1f1 | r1f2 | r1f3 | r1f4 | r2f0 | r2f1 | r2f2 | r2f3 | r2f4 | r3f0 | r3f1 | r3f2 | r3f3 | r3f4 | r4f0 | r4f1 | r4f2 | r4f3 | r4f4 | pooled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 0.9155 | 0.9156 | 0.9187 | 0.9188 | 0.9164 | 0.9194 | 0.9168 | 0.9160 | 0.9189 | 0.9196 | 0.9173 | 0.9177 | 0.9187 | 0.9174 | 0.9204 | 0.9187 | 0.9162 | 0.9177 | 0.9166 | 0.9178 | 0.9195 | 0.9198 | 0.9172 | 0.9197 | 0.9191 | 0.9180 |
| chemistry | 0.7732 | 0.7601 | 0.6861 | 0.6763 | 0.8049 | 0.7559 | 0.7069 | 0.7764 | 0.7144 | 0.7782 | 0.7402 | 0.7187 | 0.6911 | 0.7816 | 0.7498 | 0.7797 | 0.6984 | 0.6788 | 0.7728 | 0.8035 | 0.7311 | 0.6720 | 0.7556 | 0.7909 | 0.7652 | 0.7456 |

### Coarse group comparison (D6, D7)

| group | random mean share +/- SD | chemistry mean share +/- SD | delta (chem-rand) | delta / pooled SD |
|---|---|---|---|---|
| magpie | 0.2654 +/- 0.0056 | 0.2759 +/- 0.0228 | +0.0105 | +0.63 |
| cbfv | 0.5377 +/- 0.0049 | 0.5288 +/- 0.0251 | -0.0089 | -0.49 |
| temperature | 0.1969 +/- 0.0014 | 0.1953 +/- 0.0058 | -0.0015 | -0.36 |

### Fine descriptor-semantic group comparison (D6, D7)

| group | random mean share +/- SD | chemistry mean share +/- SD | delta (chem-rand) | delta / pooled SD |
|---|---|---|---|---|
| atomic_radius | 0.1945 +/- 0.0046 | 0.1802 +/- 0.0277 | -0.0143 | -0.72 |
| periodic_position | 0.1120 +/- 0.0021 | 0.1257 +/- 0.0337 | +0.0137 | +0.57 |
| valence_electron_config | 0.1939 +/- 0.0051 | 0.1814 +/- 0.0219 | -0.0125 | -0.78 |
| dft_groundstate | 0.0206 +/- 0.0022 | 0.0277 +/- 0.0184 | +0.0071 | +0.54 |
| thermodynamic_bulk | 0.1533 +/- 0.0036 | 0.1587 +/- 0.0160 | +0.0053 | +0.46 |
| electronegativity | 0.0700 +/- 0.0026 | 0.0671 +/- 0.0058 | -0.0029 | -0.64 |
| metal_class | 0.0226 +/- 0.0010 | 0.0250 +/- 0.0058 | +0.0024 | +0.57 |
| temperature | 0.1969 +/- 0.0014 | 0.1953 +/- 0.0058 | -0.0015 | -0.36 |
| melting_point | 0.0185 +/- 0.0009 | 0.0199 +/- 0.0025 | +0.0014 | +0.76 |
| atomic_mass | 0.0177 +/- 0.0015 | 0.0189 +/- 0.0042 | +0.0012 | +0.37 |

`other` fine-group column count: 0 (expected 0 -- every column should resolve to a named semantic group; see summary.json's unmapped_fine_group_columns if nonzero).