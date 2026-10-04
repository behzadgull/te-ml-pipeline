# NA11: second Materials Project query (E_hull <= 0.10 eV/atom), filter, structural test, featurisation (no predictions)

Repeats `../20261004T100423` with the E_hull limit of the query raised from 0.05 to 0.10 eV/atom, so that the screening sensitivity to that limit can be reported.
The main criteria are unchanged (E_hull <= 0.05, gap 0.1 to 3.0 eV, final cap 0.6 eV, Tl/Hg/Cd/As/Be excluded) and the main counts and the main candidate
set reproduce the first run exactly (315 perovskite-type, 32 below the cap, 30 in the ranked list; the same 315 material IDs and bit-identical features).

- **Code**: query from commit `a9045c73dfa2067deac7ff9350070dea7a8d6f31`, clean tree (`query_config.json`); filter from the same commit, clean tree (`run_config.json`).
- **Query**: Materials Project API (`mp-api` 0.46.5, database version 2026.04.13), 2026-10-04 13:42:20 UTC; raw file `data/external/mp/mp_query_20261004T134220.json`
  (gitignored), SHA256 `2bd8bcf523a567bda7085cb93fc98694fe54336536af3393d63ababcc392a30a`, 128,704,134 bytes; 101,161 materials in query A, 18,092 three-element
  compounds with structures in query B. The API key is recorded nowhere.
- **Candidate matrix for the models** (gitignored): `data/external/mp/mp_perovskite_candidates_featurized_20261004T135055.csv`, 409 rows (every perovskite-type
  compound in the main gap window with E_hull <= 0.10; the 315 rows with `ehull` <= 0.05 are the main set), 397 model features plus metadata, `temperature_bin` 600
  (replaced by the 300 to 800 K grid at prediction time), SHA256 `6386c09781667cc86a52d23dee1ee6f86ea899c39cea4d22f4ca10e143aec95c`, 1,178,742 bytes.

## Counts (`counts.json`)

| Step | Main (E_hull <= 0.05) | E_hull <= 0.10 (S3) |
|---|---|---|
| f0 E_hull limit | 75,508 | 101,161 |
| f1 band gap 0.1 to 3.0 eV | 26,355 | 37,821 |
| f2 lead-free, no radioactive elements | 24,723 | 35,916 |
| f3 ABX3 | 806 (30 anti-perovskites) | 1,025 (46) |
| f4 perovskite-type by connectivity | 315 | 409 |
| f5 gap <= 0.6 eV | 32 | 52 |
| f6 no Tl, Hg, Cd, As, Be (ranked list) | 30 | 44 |
| f6 with the cap at 0.9 eV as well (S4) | 49 | 73 |

Raising E_hull to 0.10 adds 94 perovskite-type compounds and 14 to the ranked list. Sensitivity rows S1 (gap window upper bound x1.5) and S2 (cap x1.5) are in
`counts.json` and are unchanged from the first run. Files: `abx3_candidates.csv` (every lead- and radioactive-free ABX3 compound with E_hull <= 0.10 in the widened
window, with `in_main_window` and `in_main_ehull`), `ranked_list_f6.csv` (the 30 main compounds), `ranked_list_f6_ehull0.10.csv` (the 44 with E_hull <= 0.10).
