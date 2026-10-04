# NA11 screening sensitivity (from the stored candidate table; no model predictions yet)

Code `b95e78dfccb1a4fa16053b2655ba5e41595f504f`, clean tree (`run_config.json`). Input: `results/na11/20261004T135055/abx3_candidates.csv` (E_hull <= 0.10).
Axes: toxic list (main: Tl, Hg, Cd, As, Be excluded; relaxed: only Hg and Cd excluded), E_hull limit (0.10, 0.05 main, 0.025, 0), gap cap (0.6 main, 0.9 eV);
16 variants, listed with their sizes in `counts.json`; `membership.csv` has every compound in every variant. The candidates are not ranked here: no model has
predicted them yet, so the ranking by predicted zT is to be redone per variant once `final_models` has scored the matrix.

Main list 30 compounds. Main-list members surviving every E_hull tightening (0.025 and stable-only): 12 (BaSnO3, CsSnI3, ErCrO3, EuHfO3, EuHfS3, EuZrO3, EuZrS3,
KBiO3, LaCoO3, LaRhO3, LuVO3, SrBiO3). E_hull 0.10 keeps all 30 and adds 14 entries (44); the gap cap 0.9 eV adds 19 (49); allowing Tl, As, Be adds only TlCrO3 and
TaTlO3 (32); E_hull 0.10 with cap 0.9 gives 73.
