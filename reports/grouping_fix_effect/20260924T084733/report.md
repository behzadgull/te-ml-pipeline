# Grouping-fix effect on the chemistry-cluster rung

Generated 20260924T084733 UTC by scripts/grouping_fix_effect.py. All assertions passed.

The split into fold-code effect and remaining change is arithmetic, not causal: the reverse control
(original fold code with the snapped identifier) was not run.

| Target | pre-fix | control (fold code only) | post-fix | total drop | fold-code effect | remaining change (grouping key + interaction) |
|---|---|---|---|---|---|---|
| S | 0.8076 | 0.8064 | 0.7528 | 0.0548 | -0.0012 | -0.0536 |
| sigma | 0.7600 | 0.7579 | 0.7020 | 0.0580 | -0.0022 | -0.0558 |
| kappa | 0.8460 | 0.8441 | 0.8092 | 0.0369 | -0.0019 | -0.0349 |
| zT | 0.7968 | 0.7979 | 0.7456 | 0.0512 | +0.0011 | -0.0524 |

Mean gap (random 80/20 minus chemistry rung): pre-fix 0.1310, post-fix 0.1812.
Total drop range: 0.0369 to 0.0580.
Fold-code effect range: -0.0022 to +0.0011.
Remaining-change range (grouping key + interaction): -0.0558 to -0.0349.


Across-repeat SD of the per-repeat pooled R^2:

| Target | pre-fix | control | post-fix | control minus pre-fix |
|---|---|---|---|---|
| S | 0.0018 | 0.0020 | 0.0050 | +0.0002 |
| sigma | 0.0008 | 0.0012 | 0.0020 | +0.0003 |
| kappa | 0.0011 | 0.0009 | 0.0021 | -0.0002 |
| zT | 0.0030 | 0.0024 | 0.0045 | -0.0006 |
