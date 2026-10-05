# NA2 run-time estimate for the Kaggle CPU sessions (4 cores)

Code: `run_config.json` (clean tree). Measured here with 4 threads on the maintainer's 8-core machine; `timing.json` holds every measurement. No prediction or score was kept.

- **Calibration**: the Kaggle CPU time of tuning trial 12 of the first random-forest session (3 inner fits, 600 trees, S; `results/na2_random_forest_tuning/20261004T100611`) was 3,534 s; the same
  configuration timed here (trees timed at 4 and 12, scaled to 600) gives 2,152 s, so a Kaggle fit takes **1.64 times** the local time (an assumption: the same ratio holds for other fits).
- **Random forest, fixed setting** (500 trees, a third of the features, node size 5, no depth limit), one fit on the first outer training fold, scaled to 500 trees and by 1.64:
  S 1,580 s (26 min), sigma 1,271 s (21 min), kappa 938 s (16 min), zT 1,060 s (18 min); 25 fits per target: S 11.0 h, sigma 8.8 h, kappa 6.5 h, zT 7.4 h (33.7 h in all).
- **LightGBM** (local seconds, 4 threads, S; fit and predict; not calibrated): smallest search-space configuration 5.6 s (inner fold) / 8.8 s (outer fold), middle 22.6 / 26.4 s, largest 90.7 / 102.8 s.
  A tuning trial is three inner fits; the factor of 1.64 is applied in the cell estimates as a guess only.
