# NA13 exploratory follow-up: the selected features plus temperature_bin

EXPLORATORY and descriptive (docs/decisions.md, 2026-10-10, post hoc, after the NA13 results were seen). No threshold, test or interval is attached; it does not change the pre-registered paired verdict.

100 units; device cuda; complete True; smoke False.

| Target | R2 committed selected | R2 selected + temperature_bin | R2 committed all 397 | per-fold difference to selected, mean (min to max) | per-fold difference to all 397, mean (min to max) |
|---|---|---|---|---|---|
| S | 0.7025 | 0.7198 | 0.7528 | +0.0175 (-0.0006 to +0.0291) | -0.0332 (-0.0736 to -0.0083) |
| sigma | 0.6638 | 0.6871 | 0.7020 | +0.0236 (+0.0149 to +0.0329) | -0.0151 (-0.0340 to +0.0008) |
| kappa | 0.7622 | 0.7951 | 0.8092 | +0.0336 (+0.0208 to +0.0574) | -0.0142 (-0.0267 to +0.0016) |
| zT | 0.4507 | 0.7365 | 0.7456 | +0.2863 (+0.2488 to +0.3478) | -0.0092 (-0.0297 to +0.0158) |
