# Descriptor Ablation Results

Generated: 2026-09-12T18:43:18Z. Read-only, no refits.

Sources: `kaggle_out/descriptor_ablation/{target}_chemistry_{magpie,cbfv}/` (this task's ablation runs) and `checkpoints/saved_predictions/checkpoints/{target}_chemistry/` (the confirmed full-feature ladder checkpoints, already tracked in git) -- all computed by the same pooling code in this task, not read from CLAUDE.md.

## Item 2: run completeness

All 8 ablation run directories present. Each has 25/25 `repeat*_fold*_predictions.npz` and 25/25 `repeat*_fold*.json` (no missing/incomplete runs). `n_features` confirmed exactly 133 (magpie) / 265 (cbfv) for all 8, matching run_config.json.

## Item 5: per-target table

### S (scored in linear space)

| feature_set | n_features | per-repeat pooled R2 (mean +/- SD) | pooled R2 |
|---|---|---|---|
| magpie | 133 | 0.8038 +/- 0.0031 | 0.8038 |
| cbfv | 265 | 0.8060 +/- 0.0025 | 0.8060 |
| full | 397 | 0.8076 +/- 0.0018 | 0.8076 |

Per-repeat values:
- magpie: [0.8013, 0.8031, 0.8043, 0.8015, 0.8089]
- cbfv: [0.8041, 0.8046, 0.8052, 0.8057, 0.8103]
- full: [0.8062, 0.8064, 0.8077, 0.8071, 0.8108]

### sigma (scored in log10 space)

| feature_set | n_features | per-repeat pooled R2 (mean +/- SD) | pooled R2 |
|---|---|---|---|
| magpie | 133 | 0.7497 +/- 0.0010 | 0.7497 |
| cbfv | 265 | 0.7585 +/- 0.0013 | 0.7585 |
| full | 397 | 0.7600 +/- 0.0008 | 0.7600 |

Per-repeat values:
- magpie: [0.7502, 0.7505, 0.7507, 0.7487, 0.7486]
- cbfv: [0.7593, 0.7574, 0.7604, 0.7573, 0.7582]
- full: [0.7608, 0.7599, 0.7608, 0.7598, 0.7588]

### kappa (scored in log10 space)

| feature_set | n_features | per-repeat pooled R2 (mean +/- SD) | pooled R2 |
|---|---|---|---|
| magpie | 133 | 0.8419 +/- 0.0009 | 0.8419 |
| cbfv | 265 | 0.8436 +/- 0.0013 | 0.8436 |
| full | 397 | 0.8460 +/- 0.0011 | 0.8460 |

Per-repeat values:
- magpie: [0.8421, 0.8410, 0.8431, 0.8423, 0.8408]
- cbfv: [0.8441, 0.8428, 0.8447, 0.8446, 0.8417]
- full: [0.8463, 0.8463, 0.8470, 0.8465, 0.8441]

### zT (scored in linear space)

| feature_set | n_features | per-repeat pooled R2 (mean +/- SD) | pooled R2 |
|---|---|---|---|
| magpie | 133 | 0.7940 +/- 0.0016 | 0.7940 |
| cbfv | 265 | 0.7945 +/- 0.0042 | 0.7945 |
| full | 397 | 0.7968 +/- 0.0030 | 0.7968 |

Per-repeat values:
- magpie: [0.7959, 0.7916, 0.7937, 0.7951, 0.7939]
- cbfv: [0.7919, 0.7886, 0.7958, 0.7978, 0.7985]
- full: [0.7973, 0.7915, 0.7985, 0.7984, 0.7983]

## Item 6: deltas (full minus other), in units of full run's across-repeat SD

| Target | full SD | full - magpie | in SD units | >2x SD? | full - cbfv | in SD units | >2x SD? |
|---|---|---|---|---|---|---|---|
| S | 0.0018 | +0.0038 | +2.08 | YES | +0.0017 | +0.91 | no |
| sigma | 0.0008 | +0.0103 | +12.56 | YES | +0.0015 | +1.86 | no |
| kappa | 0.0011 | +0.0042 | +3.78 | YES | +0.0025 | +2.22 | YES |
| zT | 0.0030 | +0.0028 | +0.93 | no | +0.0023 | +0.76 | no |

No conclusions drawn. Report only.