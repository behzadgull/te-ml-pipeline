# Grouping Key statistics (CLAUDE.md lines 303-348), remeasured under SNAP(0.05)

Generated 20260918T000232. Source: `data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv`
(280,173 rows). Read-only, no refits.

## 1. Total chemistry_cluster_id groups, whole pull

| | OLD (2026-08-15 pull, pre-fix) | NEW (snapfix, fixed) |
|---|---|---|
| Total groups | 12,454 | 8,908 |

## 2. parent_system vs chemistry_cluster_id, unique formulas

| | OLD | NEW |
|---|---|---|
| Unique composition strings | 17,207 | 17,977 |
| parent_system groups | 3,921 | 3,909 |
| chemistry_cluster_id groups (over these formulas) | 12,024 | 8,908 |
| chemistry_cluster_id groups spanning >1 parent_system | 1,399 of 12,024 (11.6%) | 1,285 of 8,908 (14.4%) |

Note: the unique-formula count itself differs (17,207 old vs 17,977 new) because these come from different
underlying pulls/regenerations, not just the grouping fix -- both counts are reported for context, not as a
controlled comparison of formula count alone.

## 3. Named-cluster stats: CoSb3, Ca3Co4O9

| Cluster | OLD samples | OLD rows | OLD % of fold | OLD parent_systems spanned | NEW samples | NEW rows | NEW parent_systems spanned |
|---|---|---|---|---|---|---|---|
| CoSb3 | 780 | 9,898 | 14.5% | 69 | 959 | 13,185 | 139 |
| Ca3Co4O9 | 548 | 5,027 | 7.4% | -- | 281 | 3,239 | 52 |

CoSb3 GREW under the fix (780->959 samples, +23%): consistent with the fix's mechanism -- doped variants whose
host amounts are near-integer (e.g. Co0.98Sb2.97) previously produced their own fractional cluster identity
under the old reduced_formula-without-snapping code, and now round cleanly to Co1Sb3 = "CoSb3", merging into
the same cluster that used to only catch already-exact-integer formulas.

Ca3Co4O9 SHRANK under the fix (548->281 samples, -49%): the opposite direction from CoSb3. Not investigated
further here (out of this job's scope) -- flagged as a real, non-obvious finding, not assumed to follow the
same direction as CoSb3 just because both are named clusters in the original note.

## 4. sklearn GroupKFold(n_splits=5) fold row-count balance

| | OLD | NEW |
|---|---|---|
| Fold row count range | 68,165-68,167 | 56,034-56,035 |
| Fold size std (% of mean) | <0.01% | 0.0010% |

**Confounded comparison, flagged explicitly**: the OLD figure's implied total row count (5 x ~68,166 =~340,830)
does not match the snapfix CSV's 280,173 rows. 340,831 is the exact row count CLAUDE.md's own Data Cleaning
Pipeline funnel records as the step-11 "Before fix" total (i.e. before the 2026-08-17 step8/9 multi-source-
consistency fix), so the OLD GroupKFold measurement was very likely taken on that now-superseded (differently
sized) cleaned dataset -- not the same total N as the snapfix CSV. This fold-count comparison is confounded by
TWO independent changes (dataset row count AND the SNAP grouping fix), not attributable to the grouping fix
alone.

CoSb3/Ca3Co4O9's own new fold share: CoSb3 (fold 0, 56,035 rows) = 23.5% of its fold (up from 14.5%, larger both
in absolute row terms and as a fold share); Ca3Co4O9 (fold 4, 56,034 rows) = 5.8% of its fold (down from 7.4%,
consistent with its shrinking membership).

## Which CLAUDE.md statements need their numbers replaced

Every specific figure quoted in lines 303-348 needs replacing:
- Line 304-305: "12,454 measured on the 2026-08-15 pull" -> 8,908 under the fix (different source dataset too,
  see caveat above).
- Lines 313-320: "3,921 parent_system groups vs. 12,024 chemistry_cluster_id groups... 1,399 of 12,024 (11.6%)...
  CoSb3... 780 samples... splits into 69 different parent_system groups" -> all four figures superseded (3,909 /
  8,908 / 1,285 of 8,908 (14.4%) / CoSb3 959 samples, 139 parent_system groups).
- Lines 345-348: "CoSb3... 780 samples / 9,898 rows... makes up 14.5%... Ca3Co4O9 (548 samples / 5,027 rows)
  makes up 7.4%" -> CoSb3 959 samples / 13,185 rows / 23.5%; Ca3Co4O9 281 samples / 3,239 rows / 5.8%.
- Lines 340-341: "fold ROW COUNTS come out essentially balanced (68,165-68,167 rows per fold...)" -> qualitatively
  still true (still balanced to <0.01%), but the exact range is stale and was measured on a different-sized
  dataset in addition to the different grouping; replace with 56,034-56,035 and note the dataset-size caveat if
  citing the old figure for comparison.

The QUALITATIVE arguments in this section (repeated grouped CV is needed because of group-size imbalance, not
scarcity; parent_system and chemistry_cluster_id cross-cut rather than nest) still hold under the new numbers --
only the specific figures are stale, not the reasoning built on them.
