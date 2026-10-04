# NA11: Materials Project candidates (query, filters, structural test, featurisation; no predictions)

- **Code**: commit `5e9487be1ad69a4a4538ff75f664306641b70b41`, tree clean (`run_config.json`: `tree_clean` true). The query (`query_config.json`) ran from
  `f9444249adbbefde3f08d7518ae997d284ffab48` with a clean tree. A first filter run from the same query was discarded because its run config recorded a dirty
  tree (it measured the tree after creating its own output folder); the script was fixed and the run repeated. Both runs gave the same featurised matrix
  (same SHA256).
- **Query**: Materials Project API (`mp-api` 0.46.5, database version 2026.04.13), 2026-10-04 09:56:33 UTC; raw result `data/external/mp/mp_query_20261004T095633.json`
  (gitignored), SHA256 `5f44fcedd3e7dcae0814a4a4fa4afb2590436a873a2a56917fb13a38686439e7`, 100,188,212 bytes. The API key is not recorded anywhere.
- **Candidate matrix for the models** (gitignored): `data/external/mp/mp_perovskite_candidates_featurized_20261004T100423.csv`, 315 rows, 417 columns (397
  model features; `temperature_bin` is set to 600 here and replaced by the 300 to 800 K grid at prediction time), SHA256
  `b345e0f6965567469254864b7900b218d5ae32c1ee0ee471153d72c7bf5e1a6f`, 907,925 bytes.

## Thresholds as used (thesis section 3.6, claims C135 to C142)

E_hull <= 0.05 eV/atom; band gap 0.1 to 3.0 eV; lead and radioactive/unstable elements removed (Tc, Pm, Po, At, Rn, Fr, Ra, Ac, Th, Pa, U, Np, Pu and the heavier
actinides); ABX3 stoichiometry; perovskite-type by the connectivity test (`scripts/perovskite_test.py`: six-fold B coordination, X bridging exactly two B sites, no
shared edges or faces, three-dimensional connectivity), anti-perovskites excluded; final band-gap cap 0.6 eV; Tl, Hg, Cd, As and Be excluded from the ranked list.
These are the thesis's stated values; none was changed.

## Counts after every filter (`counts.json`)

| Step | Count |
|---|---|
| f0 E_hull <= 0.05 eV/atom | 75,508 |
| f1 band gap 0.1 to 3.0 eV | 26,355 |
| f2 lead-free and free of radioactive elements | 24,723 (840 removed for Pb, 815 for radioactive elements) |
| f3 ABX3 stoichiometry | 806 (30 anti-perovskites) |
| f4 perovskite-type by connectivity | 315 |
| f5 band gap <= 0.6 eV | 32 |
| f6 free of Tl, Hg, Cd, As, Be (ranked list, `ranked_list_f6.csv`) | 30 |

**Space group versus connectivity** (among the 806 ABX3 compounds): the old rule (Pm-3m, Pnma, R-3c, I4/mcm, Imma, P4/mbm, Cmcm) keeps 281; connectivity keeps
315. They agree on 197; 118 connectivity-type compounds lie outside the space-group list; 62 compounds in the list are not perovskite-type (and not
anti-perovskites). The space-group rule therefore both misses and admits compounds.

## Band-gap sensitivity (main criteria unchanged)

Materials Project band gaps are PBE values and underestimate experimental gaps.
- S1, window upper bound x1.5 (0.1 to 4.5 eV): f1 35,811; f2 33,861; f3 1,151 (32 anti-perovskites); f4 447 (132 more than the main window). f5 and f6 are
  unchanged (32, 30) by construction, because the final cap of 0.6 eV lies inside both windows.
- S2, final cap x1.5 (0.9 eV): f5 59 and the ranked list 49 (19 more than the main list of 30).

## Files

`counts.json`, `run_config.json` (inputs with SHA256, code commit, tree_clean, script SHA256, perovskite-test parameters), `query_config.json` (copy of the query's
own record), `abx3_candidates.csv` (every lead- and radioactive-free ABX3 compound in the widened window, with MP ID, formula, E_hull, gap, space group, the
structural verdict and `in_main_window`), `ranked_list_f6.csv` (the 30 compounds of the main ranked list). The thesis's own counts (20,586; 19,144; 550; 204; 346;
16) came from an undated query and an older database release and are not comparable one to one; they are not used here.
