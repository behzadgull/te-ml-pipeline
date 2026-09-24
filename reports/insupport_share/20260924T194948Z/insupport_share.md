# In-support share of ESTM's external loss

recovered share = (R2 in-support - R2 full) / (internal chemistry-cluster R2 - R2 full).
ood SSE share = fraction of the external squared error from out-of-support rows.

## DOI-disjoint (n = 3123, in-support 2709, out-of-support 13.3%)

| Property | R2 full | R2 in-support | R2 internal | recovered share | OOD share of SSE |
|---|---|---|---|---|---|
| S | 0.5664 | 0.7404 | 0.7528 | 93.3% | 63.8% |
| sigma | 0.3988 | 0.6132 | 0.7020 | 70.7% | 79.1% |
| kappa | 0.6892 | 0.7314 | 0.8092 | 35.2% | 35.5% |
| zT | 0.6615 | 0.6686 | 0.7456 | 8.5% | 12.8% |

## cluster-disjoint (n = 1448, in-support 1196, out-of-support 17.4%)

| Property | R2 full | R2 in-support | R2 internal | recovered share | OOD share of SSE |
|---|---|---|---|---|---|
| S | 0.3536 | 0.5115 | 0.7528 | 39.6% | 60.8% |
| sigma | 0.2746 | 0.4085 | 0.7020 | 31.3% | 78.4% |
| kappa | 0.6184 | 0.6565 | 0.8092 | 20.0% | 38.5% |
| zT | 0.4982 | 0.5343 | 0.7456 | 14.6% | 19.7% |

The DOI-disjoint stratum is shown for completeness; internal chemistry-cluster R2 is not its matched
reference (it tests measurement transfer, not chemistry transfer), so its recovered shares are not
comparable to the cluster-disjoint ones and are not quoted in the paper.
