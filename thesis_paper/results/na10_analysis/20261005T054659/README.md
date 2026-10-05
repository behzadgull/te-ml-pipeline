# NA10 (third run): the JARVIS comparison with the classifier's sign (the screening configuration)

Predictions `results/final_b/20261004T201433/predictions_jarvis.csv` (the regressor's S with the carrier-type classifier's sign; `predicted_sign_source` classifier), otherwise the same
definitions and strata as `20261004T190444` (which used the regressor's sign, `final_a`). Magnitude statistics are identical (|S| is unchanged by the sign override); only sign agreement differs.

| stratum (same-sign entries, reference >= 20 uV/K) | n | regressor sign | classifier sign [cluster-bootstrap interval] |
|---|---|---|---|
| all | 3,885 | 0.504 | 0.520 [0.503, 0.540] |
| metals and semimetals | 3,534 | 0.502 | 0.516 [0.498, 0.532] |
| semiconductors | 351 | 0.521 | 0.567 [0.511, 0.620] |
| metals, cluster seen | 172 | 0.616 | 0.669 [0.590, 0.742] |
| metals, cluster unseen | 3,362 | 0.496 | 0.508 [0.490, 0.527] |

With the classifier the agreement is marginally above chance overall and clearly better only for clusters seen in training; for unseen clusters it stays at chance.
