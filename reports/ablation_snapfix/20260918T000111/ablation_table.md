# Snapfix descriptor ablation, tabulated

Generated 20260918T000111. Sources: `results/descriptor_ablation_snapfix/20260917T184650/` (magpie/cbfv), `results/ladder_regen_snapfix/20260917T150000/` (full -- the snapfix chemistry rung, computed here by the SAME code as magpie/cbfv, not read from any prior report).

## Per-run detail

| Target | Feature set | n_features | scale | per-repeat R2 (5 values) | mean +/- SD | pooled R2 (25 folds) | fold-level mean +/- SD |
|---|---|---|---|---|---|---|---|
| S | magpie | 133 | linear | 0.7494, 0.7499, 0.7477, 0.7441, 0.7406 | 0.7464 +/- 0.0039 | 0.7464 | 0.7445 +/- 0.0323 |
| S | cbfv | 265 | linear | 0.7535, 0.7514, 0.7514, 0.7481, 0.7464 | 0.7502 +/- 0.0029 | 0.7502 | 0.7482 +/- 0.0314 |
| S | full | 397 | linear | 0.7581, 0.7559, 0.7549, 0.7464, 0.7488 | 0.7528 +/- 0.0050 | 0.7528 | 0.7510 +/- 0.0325 |
| sigma | magpie | 133 | log10 | 0.6922, 0.6915, 0.6932, 0.6902, 0.6925 | 0.6919 +/- 0.0011 | 0.6919 | 0.6883 +/- 0.0306 |
| sigma | cbfv | 265 | log10 | 0.7022, 0.7036, 0.7033, 0.6985, 0.7013 | 0.7018 +/- 0.0020 | 0.7018 | 0.6983 +/- 0.0298 |
| sigma | full | 397 | log10 | 0.7018, 0.7028, 0.7044, 0.6988, 0.7025 | 0.7020 +/- 0.0020 | 0.7020 | 0.6986 +/- 0.0294 |
| kappa | magpie | 133 | log10 | 0.8048, 0.8061, 0.8011, 0.8011, 0.8009 | 0.8028 +/- 0.0025 | 0.8028 | 0.7993 +/- 0.0184 |
| kappa | cbfv | 265 | log10 | 0.8059, 0.8036, 0.8010, 0.8079, 0.8057 | 0.8048 +/- 0.0026 | 0.8048 | 0.8012 +/- 0.0176 |
| kappa | full | 397 | log10 | 0.8128, 0.8088, 0.8071, 0.8085, 0.8087 | 0.8092 +/- 0.0021 | 0.8092 | 0.8058 +/- 0.0172 |
| zT | magpie | 133 | linear | 0.7378, 0.7428, 0.7383, 0.7418, 0.7425 | 0.7406 +/- 0.0024 | 0.7406 | 0.7373 +/- 0.0448 |
| zT | cbfv | 265 | linear | 0.7443, 0.7458, 0.7375, 0.7487, 0.7453 | 0.7443 +/- 0.0042 | 0.7443 | 0.7410 +/- 0.0418 |
| zT | full | 397 | linear | 0.7433, 0.7481, 0.7392, 0.7509, 0.7463 | 0.7456 +/- 0.0045 | 0.7456 | 0.7425 +/- 0.0420 |

## Summary table: magpie | cbfv | full | delta(full-magpie)

Per-repeat pooled R2, mean +/- across-repeat SD (k=5), matching CLAUDE.md's existing ablation-table convention.

| Target | magpie (133 feat) | cbfv (265 feat) | full (397 feat) | full - magpie |
|---|---|---|---|---|
| S | 0.7464 +/- 0.0039 | 0.7502 +/- 0.0029 | 0.7528 +/- 0.0050 | +0.0065 |
| sigma (log10) | 0.6919 +/- 0.0011 | 0.7018 +/- 0.0020 | 0.7020 +/- 0.0020 | +0.0101 |
| kappa (log10) | 0.8028 +/- 0.0025 | 0.8048 +/- 0.0026 | 0.8092 +/- 0.0021 | +0.0064 |
| zT | 0.7406 +/- 0.0024 | 0.7443 +/- 0.0042 | 0.7456 +/- 0.0045 | +0.0049 |

## Fraction-of-headroom framing (full-minus-magpie / NEW headroom range)

Using the NEW headroom from `results/noise_floor/20260917T172251/noise_floor_inputs.json` (item3_combined_ceiling_new), not the old CLAUDE.md headroom values.

| Target | full-minus-magpie delta | NEW headroom range | fraction of headroom |
|---|---|---|---|
| S | +0.0065 | 0.2085-0.2266 | 2.86%-3.10% |
| sigma | +0.0101 | 0.2787-0.2868 | 3.52%-3.63% |
| kappa | +0.0064 | 0.1517-0.1601 | 3.97%-4.19% |
| zT (declared) | +0.0049 | 0.2137-0.2233 | 2.21%-2.31% |
| zT (recomputed TEP) | +0.0049 | 0.2168-0.2249 | 2.20%-2.28% |
