# NA11: Materials Project candidates in chemistry clusters the training data already contain

Code: the commit in `run_config.json` (clean tree). Inputs: the 409-row MP candidate file (`mp_perovskite_candidates_featurized_20261004T135055.csv`, gitignored, SHA256 in
`run_config.json`), the snapfix training CSV and the committed sensitivity membership (`results/na11_sensitivity/20261004T135205/membership.csv`).

"Seen" = the entry's `chemistry_cluster_id` occurs among the training rows (any target; also per target, among the rows with a measured value of that target), the same definition as the
NA10 JARVIS analysis. Counts are of compounds (rows); several polymorphs of one formula share a cluster, so the number of unique clusters is smaller (`counts.json`).

| list | compounds | unique clusters | cluster seen | cluster unseen |
|---|---|---|---|---|
| main (E_hull <= 0.05, gap <= 0.6) | 30 | 26 | 9 | 21 |
| E_hull <= 0.10 | 44 | 37 | 9 | 35 |
| gap cap 0.9 | 49 | 40 | 13 | 36 |
| E_hull <= 0.10 and cap 0.9 | 73 | 60 | 15 | 58 |
| all 409 perovskite-type compounds (gap window 0.1 to 3.0) | 409 | 285 | 48 | 361 |

The 14 compounds that E_hull <= 0.10 adds to the main list are all in unseen clusters; of the 19 added by the 0.9 eV cap, 4 are seen. The seen compounds of the main list are BaSnO3,
CaMnO3, CsSnI3 (two entries), LaCoO3, LaNiO3, LaRhO3 and YCoO3 (two entries). `candidate_lists.csv` has the lists and seen flags of every compound; BaZrSe3 (mp-998350, E_hull 0.0775,
PBE gap 0.20 eV, not stable) is in the E_hull <= 0.10 lists only, in no list at E_hull <= 0.05, and its cluster is unseen.
