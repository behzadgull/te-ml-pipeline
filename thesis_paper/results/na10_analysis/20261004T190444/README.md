# NA10 (second run): the JARVIS comparison with a metal / semiconductor split

Same predictions (`results/final_a/20261004T102855/predictions_jarvis.csv`, the regressor's S, no classifier) and the same definitions as `20261004T140442` (see its README for how the predicted
and JARVIS signs were chosen), plus strata by the JARVIS optb88vdw gap: metal or semimetal (gap < 0.05 eV, a stated choice, not tuned) and semiconductor (gap >= 0.05 eV). The numbers of the
earlier run are reproduced; this run adds the gap strata and the counts per class (`analysis.json`).

The sign test covers mostly metals. Of the 13,123 entries whose n and p values have the same sign (the only entries with a JARVIS sign), 12,733 are metals or semimetals and 390 are
semiconductors; of the 3,885 with a reference magnitude of at least 20 uV/K, 3,534 are metals and 351 semiconductors. For a semiconductor JARVIS's n and p values have opposite signs by
construction (9,372 of the 9,762 semiconductors), so the sign is set by the imposed doping and carries no carrier type to compare with.

| stratum | n | Spearman of \|S_pred\| with the JARVIS magnitude | sign agreement (n) |
|---|---|---|---|
| all | 23,218 | 0.227 [0.208, 0.245] | 0.504 (3,885) |
| metals and semimetals | 13,456 | 0.112 [0.092, 0.133] | 0.502 [0.485, 0.519] (3,534) |
| semiconductors | 9,762 | -0.081 [-0.108, -0.053] | 0.521 [0.468, 0.579] (351) |
| metals, cluster seen in training | 627 | 0.307 [0.207, 0.398] | 0.616 [0.534, 0.692] (172) |
| metals, cluster unseen | 12,829 | 0.096 [0.078, 0.115] | 0.496 [0.479, 0.514] (3,362) |
| semiconductors, cluster seen | 523 | -0.032 [-0.264, 0.178] | not analysed (fewer than 30) |
| semiconductors, cluster unseen | 9,239 | -0.065 [-0.092, -0.038] | 0.519 [0.462, 0.577] (335) |

Reading: sign agreement is at chance for metals and for semiconductors; the overall rank correlation (0.23) is carried by the metals and by clusters seen in training, and is negative for
semiconductors. Intervals are chemistry-cluster bootstrap (1,000 draws).
