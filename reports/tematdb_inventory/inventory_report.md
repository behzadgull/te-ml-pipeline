# teMatDb read-only inventory (Paper A item 6, second external dataset)

Read-only, no model refit/load/score, no featurization, no writes outside
`data/external/tematdb/` and `reports/tematdb_inventory/`. All counting
logic reuses existing functions from `src/` (see Step 1); nothing was
reimplemented except a small DOI-normalization helper, documented below.

## Step 0: pin and fetch

- **TEMATDB_SHA** = `86a9bf979e35e2b9b0a4927bc50ad031edf84ad7`
- **Download timestamp**: 2026-09-09T14:21:05 (local)

| File | Bytes | SHA256 | Rows | Match vs. expected |
|---|---|---|---|---|
| `teMatDb_collocatedTEPs.csv` | 4,893,361 | `cc0cc7d363106a3849c266a8230721e2657534c97a56ffbe9c8a3eb1aaaf6fd9` | 56,641 rows, 272 unique `sample_id` | **exact match** |
| `teMatDb_samples.csv` | 34,995 | `f65d0717864bee5702ad1e7815c658105d0e6e0e8397e35db19ae1d9a5195503` | 272 rows, 12 columns | **exact match** |

No discrepancy from the task's expected values on any dimension.

## Step 1: pipeline pieces located (verbatim, before any counting)

**1a. Training dataframe / DOI column**
- `TRAINING_CSV = "data/processed/featurized_ThermoelectricMaterials_2026-08-15.csv"` — `src/external_validation.py:53`
- Loader: `load_training_data(path=TRAINING_CSV)` — `src/external_validation.py:104`
- DOI column: **`DOI`** (verbatim), dtype `object`. 5 example values (all identical, one DOI repeated across rows for one paper): `10.1016/j.physb.2005.03.022` x5.

**1b. Composition-cluster assigner**
- `chemistry_cluster_id(comp, dopant_threshold_frac=DEFAULT_DOPANT_THRESHOLD_FRAC)` — `src/canonicalization.py:54`, `DEFAULT_DOPANT_THRESHOLD_FRAC = 0.05` at line 23.
- Expects `comp` as a **pymatgen `Composition` object**, not a raw string.
- String -> Composition: `parse_formula(formula)` — `src/canonicalization.py:26`. Returns `(Composition, None)` on success, `(None, error_message)` on failure (internally catches `CompositionError`/`ValueError`; does not itself raise).

**1c. temperature_bin definition**
- `TEMP_MIN_K = 300`, `TEMP_MAX_K = 800`, `TEMP_BIN_WIDTH_K = 25` — `src/data_cleaning.py:55-57`.
- `step3_filter_temperature(long_df)` at line 170: `bin_edges = np.arange(300, 825, 25)`; `pd.cut(df["temperature_K"], bins=bin_edges, right=False, labels=bin_edges[:-1])`. 20 bins, each labeled by its lower edge.

**1d. Raw vs. binned temperature in training**
- The **cached artifact** used to fit the four models (`featurized_ThermoelectricMaterials_2026-08-15.csv`) does **NOT** retain raw temperature — only `temperature_bin` (25K-binned, lower-edge label). Confirmed by direct column inspection.
- **Raw per-point temperature IS recoverable** from `data/raw/ThermoelectricMaterials_curves.csv.gz` (columns: `SID, DOI, composition, sample_id, figure_id, figure_name, prop_x, prop_y, unit_x, unit_y, x, y, ...`) via the **existing** function `step1_extract_and_filter_properties(curves_df)` — `src/data_cleaning.py:92-147` — which explodes each curve's digitized `x`/`y` JSON arrays into one row per point with an exact `temperature_K` column (`prop_x` in `TEMP_PROP_X = {"Temperature", "T"}`, line 51). Loader: `load_raw_curves()` at line 82.
- **Conclusion**: Step 4's gate is NOT triggered. Raw temperatures are recoverable using existing functions only (`load_raw_curves` + `step1_extract_and_filter_properties`), so Step 4 proceeded.

Both 1a and 1b were located; nothing stopped here.

## Step 2: DOI overlap

Normalization function (see GAPS — written new for this task, not imported; existing `src/external_validation.py::normalize_doi` doesn't cover the `dx.doi.org` case the task's spec requires):
```python
def normalize_doi_inventory(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"^https?://dx\.doi\.org/", "", s)
    s = re.sub(r"^https?://doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s.strip()
```

| | Before normalization | After normalization | Strings changed |
|---|---|---|---|
| teMatDb DOIs (unique) | 262 (matches expected) | 262 | 0 of 272 rows |
| Training DOIs (unique) | 5,548 | 5,548 | 0 of 279,857 rows |

Zero strings changed on either side — both datasets' DOI fields were already stored clean (lowercase-equivalent, no prefix) as-is.

**Sample-level overlap**:
- **|a0| (teMatDb samples whose DOI IS in training) = 176**
- **|a| (teMatDb samples whose DOI is NOT in training) = 96**

**Thinned rows** (one row per `(sample_id, temperature_bin)`, `collocatedTEPs` filtered to `300<=Temperature<=800`, binned via `step3_filter_temperature`):
- Total thinned rows: 4,195
- In (a0): **2,860**
- In (a): **1,335**

**GROUP value_counts, stratum (a)** (teMatDb DOI NOT in training):
```
Bi2Te3       30
PbTe         12
Selenide     10
GeTe         10
SKD           8
Silicide      4
Telluride     4
Cu2Q          4
AgSbQ2        3
HH            3
Mg3Sb2        3
SiGe          2
clathrate     1
Sr2Si2Te6     1
Oxide         1
```

(Context only, not requested but cheap to include — GROUP value_counts, stratum (a0), DOI IS in training):
```
PbTe         34   Bi2Te3       32   SKD          28
GeTe         12   Silicide     12   Telluride    12
Selenide     10   AgSbQ2        9   HH            8
Cu2Q          7   Oxide         2   Sulfide       2
SiGe          2   etc           1   Element       1
clathrate     1   Zintl         1   Mg3Sb2        1
MgAgSb        1
```

## Step 3: cluster overlap + parse-failure log

Fed `teMatDb_samples.csv`'s `Composition_detailed` through `parse_formula` -> `chemistry_cluster_id` (1b's assigner). No repair, no pre-cleaning, no fallback to `Composition_by_element`/`BASEMAT`.

| | Count |
|---|---|
| Parsed successfully | **184** |
| Failed to parse | **88** |
| Unique clusters among successes | **156** |
| Unique clusters among successes NOT in training | **36** |

**Order-of-magnitude check** (as instructed): failure count (88, 10^1) and not-in-training cluster count (36, 10^1) **are the same order of magnitude**.

**However — direct inspection contradicts the "parsing artifact" reading.** I did not stop at the numeric coincidence; I inspected both lists directly:
- The 88 failures are predominantly **free-text, non-formula strings** — e.g. `"Bi0.5Sb1.5Te3 + 25wt% Te"`, `"Te + As-doping"`, `"BiSbTe alloy"`, `"inorganic ligand capped Bi2Te3"`, `"micro-grain and nano-grain Bi0.5Sb1.5Te3"` — processing descriptions and alloy names mixed into the `Composition_detailed` field, not real chemical formulas. Only 16 of 88 failures even contain parentheses, and those are mostly variable-range notation (`"NaxPb1-xTe (0.5% < x < 2%)"`) or free-text annotations (`"nanoflakes"`, `"heat-treated"`), not the clean nested stoichiometric notation the task's example describes.
- The 36 not-in-training clusters, by contrast, come from **successfully parsed, well-formed formulas** — including several genuinely using nested notation that parsed fine (e.g. `(PbTe)98.99945(CdTe)0.01(PbI2)0.00055`, `(GeTe)8(Ag0.8Sb1.2Te2.2)2`, `(Ag0.5Sb0.5Te)0.95(Pb0.16Ge0.84Te)0.05`) — most look like ordinary dopant-variant formulas (`Bi0.6Sb1.4Te3`, `Ge0.92Sb0.04Bi0.04Te0.95Se0.05`) whose cluster genuinely doesn't appear in training.

**Plain-language statement, as instructed**: the counts are the same order of magnitude, but the underlying mechanism is different for each — parse failures are dominated by non-formula free text, while the not-in-training clusters are dominated by legitimately-parsed formulas. The numeric coincidence does not, on inspection, support "stratum (b) is likely a parsing artifact." This is reported directly per this project's style convention (state the number as asked; also state what direct inspection shows, without letting the coincidence stand unexamined).

**Not-in-training clusters (stratum b)** — full list (cluster id, sample_id, `Composition_detailed`, GROUP):

| Cluster | sample_id | Composition_detailed | GROUP |
|---|---|---|---|
| Ag0.106Te1Pb0.920484 | 104 | (La0.028Pb0.972Te)0.947(Ag2Te)0.053 | PbTe |
| Ag0.1Ge0.77Sb0.13Te1 | 147 | Ge0.77Ag0.1Sb0.13Te1 | GeTe |
| Ag0.475Sb0.475Te1 | 153 | (Ag0.5Sb0.5Te)0.95(Pb0.16Ge0.84Te)0.05 | AgSbQ2 |
| Ag0.7Bi0.7Pb0.3Se1.683 | 396 | Ag0.7Bi0.7Pb0.3Se1.683Br0.01 | Selenide |
| Ag1.6Ge8Sb2.4Te12.4 | 272 | (GeTe)8(Ag0.8Sb1.2Te2.2)2 | GeTe |
| Bi0.396Sb1.525Te3 | 392 | Bi0.396Sb1.525In0.075Cu0.004Te3 | Bi2Te3 |
| Bi0.45Sb1.3Te3 | 32 | Al0.2Bi0.45Sb1.3Te3 | Bi2Te3 |
| Bi0.46Sb1.54Te3.015 | 379 | Zn0.015Bi0.46Sb1.54Te3.015 | Bi2Te3 |
| Bi0.6Sb1.4Te3 | 57 | Bi0.6Sb1.4Te3 | Bi2Te3 |
| Bi1.95Te2.3Se0.7 | 406 | Bi1.95Sb0.05Te2.3Se0.7 | Bi2Te3 |
| Bi14(RhI3)3 | 72 | Bi14Rh3I9 | etc |
| Ca1.96Si1 | 173 | Ca1.96Ag0.04Si | Silicide |
| Co0.92Sb2.97 | 232 | Co0.92Ni0.08Sb2.97Te0.03 | SKD |
| Co1Sn1.5Se1.5 | 216 | CoSn1.5Se1.5 | SKD |
| Cu1.79S1 | 302 | Bi0.01Cu1.79S | Cu2Q |
| Fe1.8Co0.6Sb7.2 | 101 | (La0.3Ce0.37Fe3CoSb12)0.6(PbTe)0.4 | PbTe |
| Fe2.7Co1.3Sb11.8H1.59 | 228 | DD0.59Fe2.7Co1.3Sb11.8Sn0.2 | SKD |
| Fe3.4Sb12H1.76 | 225 | DD0.76Fe3.4Ni0.6Sb12 | SKD |
| Fe3Co1Sb12H1.65 | 226 | DD0.65Fe3CoSb12 | SKD |
| Ge0.8Te1.06 | 393 | Ge0.8Pb0.1Bi0.1Te1.06 | GeTe |
| Ge0.92Te0.95 | 364 | Ge0.92Sb0.04Bi0.04Te0.95Se0.05 | GeTe |
| Hf0.5Zr1.5Ni0.8Sn0.99O2 | 291 | Zr0.5Hf0.5Ni0.8Pd0.2Sn0.99Sb0.01 ZrO2 | HH |
| Mg1Ag0.965Sb0.99 | 201 | MgAg0.965Ni0.005Sb0.99 | HH |
| Mg2Si0.3Sb0.7 | 15 | Mg2Si0.3Sb0.7 | Silicide |
| Mg3.2Bi1.298Sb0.7 | 361 | Mg3.2Bi1.298Sb0.7Te0.002 | Mg3Sb2 |
| Mg3.2Bi1.975 | 411 | Mg3.2Sb0.015Bi1.975Te0.01 | Mg3Sb2 |
| Mn0.275Sn0.555Ge0.15Te1 | 418 | Sn0.555Ge0.15Pb0.075Mn0.275Te | Telluride |
| Sb2Te11 | 362 | Sb2Te3(Sn0.006Re0.004Te)8 | Bi2Te3 |
| Si43Ge7 | 119 | (Si80Ge20)70(Si100B5)30 | SiGe |
| Sn0.948Se1 | 419 | Sn0.948Cd0.023Se | Selenide |
| Sn0.984Se1 | 141 | Ag0.016Sn0.984Se | Selenide |
| Ta0.74Ti0.16Fe1Sb1 | 422 | Ta0.74V0.1Ti0.16FeSb | HH |
| Ta1.1Bi1.9Se1O2 | 397 | Bi1.90Ta1.10O2Se | Oxide |
| Te0.25Pb0.89Se0.5S0.25 | 360 | Pb0.89Sb0.012Sn0.1Se0.5Te0.25S0.25 | Selenide |
| Te0.92Pb0.99 | 91 | Pb0.99Na0.01Te0.92S0.08 | PbTe |
| Te99.00945Pb99 | 78 | (PbTe)98.99945(CdTe)0.01(PbI2)0.00055 | PbTe |

**Thinned-row count for stratum (b)**: **549** (one row per `(sample_id, temperature_bin)`, `300<=Temperature<=800`, among the 36 not-in-training-cluster samples).

Full parse-failure list (`sample_id, Composition_detailed, GROUP, exception_type, exception_message`) written to `reports/tematdb_inventory/parse_failures.csv` (88 rows). All 88 came back as `exception_type=ValueError` uniformly — see GAPS below for why that doesn't distinguish underlying failure modes.

## Step 4: abscissa agreement (NOT skipped)

Gate check (1d): raw temperatures ARE recoverable via existing functions (`load_raw_curves` + `step1_extract_and_filter_properties` on `data/raw/ThermoelectricMaterials_curves.csv.gz`), so this step proceeded.

Per task instruction: `teMatDb_rawTEPs.csv` was **not** fetched (out of the two-file scope given in Step 0), so this test uses teMatDb's **collocated, synthetic 2K grid**, not raw digitized abscissae — **this weakens the test**, exactly as flagged in the task spec.

- teMatDb (a0) samples: 176, spanning 169 unique normalized DOIs.
- DOIs with both raw training temperatures (via `step1_extract_and_filter_properties`) and teMatDb collocated temperatures: **169 of 169** (full overlap on DOI presence).
- Training-temperature points compared (matched DOIs, all training points get a nearest-teMatDb-temperature distance): **49,759**

| Statistic | Value |
|---|---|
| Median distance | 0.5996 K |
| p90 distance | 7.3712 K |
| Max distance | 400.6082 K |
| Fraction below 0.01 K | **0.0121** (1.21%) |

**Interpretation rule, as given (not decided here)**: a large mass below 0.01 K implies shared digitized values (stratum a0 contaminated); distances spread over ~1 K or more imply independent digitization (stratum a0 valid). The numbers above: only 1.21% of points fall below 0.01 K (a small mass, not a large one), and the median (0.60 K) sits below the ~1 K independent-digitization threshold while p90 (7.37 K) sits well above it — a mixed picture, not a clean read either way under the stated rule. Reported as computed; no conclusion drawn about which regime applies, per instruction.

## GAPS AND UNCERTAINTIES

1. **DOI normalization function was newly written for this task, not imported.** `src/external_validation.py::normalize_doi` doesn't handle the `https://dx.doi.org/` prefix this task's spec explicitly requires. Writing a 6-line, fully-specified string helper is not "pipeline logic" in the sense the hard constraints guard against (composition parsing/clustering/binning, where a wrong guess silently corrupts results) — the task gave an exact, unambiguous spec for this helper. Flagged for transparency, not hidden.
2. **Step 4 matches at DOI level, not sample_id level, per the task's own spec.** A single DOI/paper can contain multiple physically distinct samples/curves with different temperature ranges. The max distance of 400.6 K almost certainly reflects a cross-sample comparison under a shared DOI (one sample's training temperature vs. a temperature-incompatible sample's teMatDb grid under the same paper), not a genuine near-far comparison of the same physical curve. This is inherent to the DOI-level granularity specified, not a coding error — worth knowing before treating the max/p90 figures as characterizing typical same-curve digitization behavior.
3. **Step 4 uses teMatDb's synthetic 2K collocated grid, not raw abscissae** (`teMatDb_rawTEPs.csv` was out of scope for this task's fetch list). Even genuinely independent digitizations would show some small distances purely from 2K-grid quantization/interpolation, so the median/fraction-below-0.01K numbers should be read as upper bounds on independence-evidence, not lower bounds on contamination.
4. **All 88 parse failures report `exception_type=ValueError` uniformly.** This is a side effect of how `parse_formula()` itself handles errors (catches pymatgen's internal exceptions and returns `(None, message)` rather than propagating a specific exception type); I converted that returned failure into a raised `ValueError` for uniform try/except logging, per the task's instruction to "wrap each call in try/except and capture the exception type and message." The `exception_type` column therefore carries no discriminating information here — only `exception_message` (preserved verbatim from `parse_formula`) distinguishes failure modes.
5. **`Composition_by_element` and `BASEMAT` columns exist in `teMatDb_samples.csv` but were deliberately not used as a parse fallback**, per explicit task instruction. Some of the 88 failures might resolve via those columns; that was out of scope here by design.
6. **`GROUP` column's vocabulary was not independently verified** — treated as teMatDb's own given categorical field, used as-is for the value_counts breakdowns.
7. No model was refit, loaded, scored, or featurized at any point in this task, and no writes were made outside the two permitted directories — confirmed by construction (every write call in the scripts used explicitly targets `data/external/tematdb/` or `reports/tematdb_inventory/`).

No conclusions drawn about whether to score any stratum, per instruction. Report and stop.

---

## ROUND 2

Read-only, same hard constraints as round 1, plus: no modification of
`parse_formula`/`chemistry_cluster_id`, no fallback to `Composition_by_element`/
`BASEMAT` for salvaging failed parses. TEMATDB_SHA unchanged:
`86a9bf979e35e2b9b0a4927bc50ad031edf84ad7`.

### A. Repo state check

```
$ git status
On branch master
Your branch is up to date with 'origin/master'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	reports/

nothing added to commit but untracked files present (use "git add" to track)

$ git diff --stat
(empty)

$ git diff --stat --cached
(empty)
```

Clean. Zero tracked files modified. Round 1 behaved exactly as intended —
only `reports/` is untracked. No files reverted or cleaned; none needed
to be.

### B. Fetch `teMatDb_rawTEPs.csv`

| | Value | Expected | Match |
|---|---|---|---|
| Bytes | 537,427 | ~537,427 | exact |
| SHA256 | `ed907cd16ec9c7be1cc36e8e242f152b41485247fd581009db96b9b0b95d690c` | — | — |
| Rows | 14,717 | 14,717 | exact |
| Columns | `sample_id, tepname, Temperature, tepvalue` | same | exact |
| Unique sample_id | 272 | 272 | exact |
| Unique tepname | `ZT, alpha, kappa, rho` | — | — |

No discrepancy.

### C. Step 4 redone with raw abscissae, sample-level pairing

**Round 1's finding stands, restated precisely per the correction given**:
comparing against teMatDb's synthetic 2K collocated grid could not have
produced a discriminating result — the grid guarantees every query point
lands within 1K of a node, so round 1's 0.5996K median / 1.21% below 0.01K
were reproducing the grid's own null, not measuring independence. That
test is superseded by this one, not merely supplemented.

**Property mapping used** (grounded in `data_cleaning.py`'s own
`TARGET_PROP_Y` vocabulary, not a guess): teMatDb `tepname` -> training
`property`: `alpha` -> `S`; `kappa` -> `kappa`; `ZT` -> `zT`; `rho` ->
`rho` or `sigma` (pre/post-inversion, same physical curve, both pooled
since either may be what a given training sample stored).

**Candidate pairs**: 176 (a0) samples, 169 unique normalized DOIs ->
**1,255 candidate (training_sample_id, teMatDb_sample_id) pairs**, not
collapsed. 167 of 169 DOIs produced more than one pair — almost always
because one teMatDb sample_id corresponds to many finer-grained
Starrydata2 sample_ids under the same paper (up to 40 training samples
for one teMatDb sample, DOI `10.1021/ja910762q`), not the reverse; only
3 DOIs had >1 teMatDb sample_id. Full per-DOI pair table in
`data/external/tematdb/_step4_round2_pairs.csv`.

**Per-tepname and pooled results** (distance = each training temperature
point's nearest teMatDb temperature point, both restricted to their
shared range first):

| tepname | n train pts considered | dropped (out of shared range) | n compared | exact 0.0 | <1e-6K | <0.01K | median K | p90 K |
|---|---|---|---|---|---|---|---|---|
| alpha | 17,081 | 1,965 | 15,116 | 19 | 19 | 55 | 2.5160 | 18.3235 |
| kappa | 13,702 | 1,859 | 11,843 | 19 | 19 | 58 | 2.1561 | 23.4022 |
| ZT | 13,572 | 1,275 | 12,297 | 13 | 15 | 73 | 2.1041 | 19.1230 |
| rho | 16,398 | 2,129 | 14,269 | 21 | 25 | 61 | 2.3178 | 19.5851 |
| **POOLED** | **60,753** | **7,228** | **53,525** | **72** | **78** | **247** | **2.2743** | **19.7263** |

**Reference scale — teMatDb's own median consecutive gap in its raw
abscissae** (computed over the full `teMatDb_rawTEPs.csv`, per-sample
consecutive unique-temperature gaps pooled, not restricted to a0):

| tepname | median gap K |
|---|---|
| alpha | 25.4447 |
| kappa | 29.0300 |
| ZT | 27.2895 |
| rho | 25.6635 |
| **pooled** | **26.2281** |

Numbers reported as computed. No conclusion drawn about whether (a0) is
contaminated or valid — the pooled median distance (2.27K) is roughly an
order of magnitude smaller than teMatDb's own typical point spacing
(26.23K), and 72 of 53,525 comparisons (0.13%) are exact bit-for-bit
matches; both facts are reported for the record, not adjudicated.

### D. DOI strata intersected with the parseable set

| | Count |
|---|---|
| \|a0\| over all 272 (restated) | 176 |
| \|a\| over all 272 (restated) | 96 |
| **\|a0 ∩ parseable\|** (scoreable) | **122** |
| **\|a ∩ parseable\|** (scoreable) | **62** |
| \|b\| = cluster-disjoint AND parseable (restated) | 36 clusters, 36 samples |
| Parse-failure set (88) ∩ (a0) | 54 |
| Parse-failure set (88) ∩ (a) | 34 |
| Check: 54 + 34 | 88 (correct) |

Thinned rows (300-800K, 25K bins, `right=False`, via
`step3_filter_temperature`):

| Stratum | Thinned rows |
|---|---|
| a0 ∩ parseable | 2,027 |
| a ∩ parseable | 903 |
| b (cluster-disjoint & parseable) | 549 (matches round 1 exactly) |

Round 1's |a0|=176 and |a|=96 did overstate the scoreable set: **31%** of
(a0) (54/176) and **35%** of (a) (34/96) cannot be parsed at all, so
cannot be featurized or scored regardless of which dedup pass they'd
otherwise fall into.

### E1. Stratum (b) characterization: novel family or novel stoichiometry?

Full table (cluster id, sample_id, Composition_detailed, GROUP, BASEMAT,
thinned rows):

| Cluster | sample_id | Composition_detailed | GROUP | BASEMAT | Thinned rows |
|---|---|---|---|---|---|
| Mg2Si0.3Sb0.7 | 15 | Mg2Si0.3Sb0.7 | Silicide | Mg2Si | 20 |
| Bi0.45Sb1.3Te3 | 32 | Al0.2Bi0.45Sb1.3Te3 | Bi2Te3 | Bi2Te3 | 10 |
| Bi0.6Sb1.4Te3 | 57 | Bi0.6Sb1.4Te3 | Bi2Te3 | Bi2Te3 | 2 |
| Bi14(RhI3)3 | 72 | Bi14Rh3I9 | etc | BiMI | 11 |
| Te99.00945Pb99 | 78 | (PbTe)98.99945(CdTe)0.01(PbI2)0.00055 | PbTe | PbTe | 17 |
| Te0.92Pb0.99 | 91 | Pb0.99Na0.01Te0.92S0.08 | PbTe | PbTe | 17 |
| Fe1.8Co0.6Sb7.2 | 101 | (La0.3Ce0.37Fe3CoSb12)0.6(PbTe)0.4 | PbTe | PbTe | 13 |
| Ag0.106Te1Pb0.920484 | 104 | (La0.028Pb0.972Te)0.947(Ag2Te)0.053 | PbTe | PbTe | 19 |
| Si43Ge7 | 119 | (Si80Ge20)70(Si100B5)30 | SiGe | SiGe | 20 |
| Sn0.984Se1 | 141 | Ag0.016Sn0.984Se | Selenide | SnSe | 17 |
| Ag0.1Ge0.77Sb0.13Te1 | 147 | Ge0.77Ag0.1Sb0.13Te1 | GeTe | GeTe | 13 |
| Ag0.475Sb0.475Te1 | 153 | (Ag0.5Sb0.5Te)0.95(Pb0.16Ge0.84Te)0.05 | AgSbQ2 | ABQ2 | 14 |
| Ca1.96Si1 | 173 | Ca1.96Ag0.04Si | Silicide | Ca2Si | 20 |
| Mg1Ag0.965Sb0.99 | 201 | MgAg0.965Ni0.005Sb0.99 | HH | HH | 9 |
| Co1Sn1.5Se1.5 | 216 | CoSn1.5Se1.5 | SKD | SKD | 19 |
| Fe3.4Sb12H1.76 | 225 | DD0.76Fe3.4Ni0.6Sb12 | SKD | SKD | 19 |
| Fe3Co1Sb12H1.65 | 226 | DD0.65Fe3CoSb12 | SKD | SKD | 13 |
| Fe2.7Co1.3Sb11.8H1.59 | 228 | DD0.59Fe2.7Co1.3Sb11.8Sn0.2 | SKD | SKD | 16 |
| Co0.92Sb2.97 | 232 | Co0.92Ni0.08Sb2.97Te0.03 | SKD | SKD | 16 |
| Ag1.6Ge8Sb2.4Te12.4 | 272 | (GeTe)8(Ag0.8Sb1.2Te2.2)2 | GeTe | GeTe | 2 |
| Hf0.5Zr1.5Ni0.8Sn0.99O2 | 291 | Zr0.5Hf0.5Ni0.8Pd0.2Sn0.99Sb0.01 ZrO2 | HH | ZrNiSn | 19 |
| Cu1.79S1 | 302 | Bi0.01Cu1.79S | Cu2Q | M2Q | 15 |
| Te0.25Pb0.89Se0.5S0.25 | 360 | Pb0.89Sb0.012Sn0.1Se0.5Te0.25S0.25 | Selenide | PbSe | 20 |
| Mg3.2Bi1.298Sb0.7 | 361 | Mg3.2Bi1.298Sb0.7Te0.002 | Mg3Sb2 | Mg3Bi2 | 2 |
| Sb2Te11 | 362 | Sb2Te3(Sn0.006Re0.004Te)8 | Bi2Te3 | Sb2Te3 | 18 |
| Ge0.92Te0.95 | 364 | Ge0.92Sb0.04Bi0.04Te0.95Se0.05 | GeTe | GeTe | 20 |
| Bi0.46Sb1.54Te3.015 | 379 | Zn0.015Bi0.46Sb1.54Te3.015 | Bi2Te3 | Bi2Te3 | 9 |
| Bi0.396Sb1.525Te3 | 392 | Bi0.396Sb1.525In0.075Cu0.004Te3 | Bi2Te3 | Bi2Te3 | 9 |
| Ge0.8Te1.06 | 393 | Ge0.8Pb0.1Bi0.1Te1.06 | GeTe | GeTe | 19 |
| Ag0.7Bi0.7Pb0.3Se1.683 | 396 | Ag0.7Bi0.7Pb0.3Se1.683Br0.01 | Selenide | AgBiSe2 | 20 |
| Ta1.1Bi1.9Se1O2 | 397 | Bi1.90Ta1.10O2Se | Oxide | Bi2O2Se | 20 |
| Bi1.95Te2.3Se0.7 | 406 | Bi1.95Sb0.05Te2.3Se0.7 | Bi2Te3 | Bi2Te3 | 12 |
| Mg3.2Bi1.975 | 411 | Mg3.2Sb0.015Bi1.975Te0.01 | Mg3Sb2 | Mg3Sb2 | 19 |
| Mn0.275Sn0.555Ge0.15Te1 | 418 | Sn0.555Ge0.15Pb0.075Mn0.275Te | Telluride | SnTe | 20 |
| Sn0.948Se1 | 419 | Sn0.948Cd0.023Se | Selenide | SnSe | 20 |
| Ta0.74Ti0.16Fe1Sb1 | 422 | Ta0.74V0.1Ti0.16FeSb | HH | TaFeSb | 19 |

**GROUP literal string present among training `chemistry_cluster_id`
values: 13 of 36 (36%).**
**BASEMAT literal string present among training `chemistry_cluster_id`
values: 22 of 36 (61%).**

**Plain statement**: BASEMAT (the host-lattice formula) is the more
direct test here, and it says the majority case (22/36, 61%) is **new
stoichiometry inside a host lattice already present in training** — e.g.
sample 91 (`Te0.92Pb0.99` cluster) has BASEMAT `PbTe`, which is already a
training cluster on its own; the not-in-training cluster only differs
from plain PbTe by its specific dopant/off-stoichiometry combination. The
remaining 14 of 36 (39%) have a BASEMAT that is *itself* not a training
cluster (e.g. `BiMI`, `ABQ2`, `M2Q`, `Bi2O2Se`) — that minority is the
stronger novel-host-lattice-chemistry signal. Context note as given:
156 unique clusters from 184 parsed samples is the 5 at%-threshold
definition working as intended (dopant collapse, not alloy-stoichiometry
collapse), not evidence of a bug — consistent with e.g.
`Bi2Te2.7Se0.3` (Se at ~10 at%, above the 5 at% threshold) correctly
getting its own cluster distinct from `Bi2Te3`.

### E2. All 88 failures, bucketed by inspection

| Bucket | Count | Example |
|---|---|---|
| Composite / multi-phase | 55 | `Bi0.5Sb1.5Te3 + 25wt% Te`, `Bi2Te3 0.1vol%SiC` |
| Free-text descriptor | 14 | `BiSbTe alloy`, `Bi2Te3 nanowires `, `SiGe nanopowder` |
| Range or inequality notation | 10 | `NaxPb1-xTe (0.5% < x < 2%)`, `Mg2−xLaxSi0.58Sn0.42 (0≤x≤0.015)` |
| Formula with decoration (colon, typo, etc.) | 7 | `Bi2Te2.85Se0.15 : I 0.005`, `Al0:03PbTe` |
| Other (genuinely unclassifiable) | 2 | `Ge0.53Ag0.13Sb0.27□0.07Te1, Ge4AgSb2Te7.5` (vacancy symbol + second formula), `Bi0.07Ge0.90Te-873` (trailing numeric code) |
| **Total** | **88** | |

Composite/multi-phase dominates by a wide margin (63% of all failures) —
teMatDb's `Composition_detailed` field frequently encodes a *recipe*
(matrix + wt%/vol%/mol%/at% second phase, or two phases joined by `+`/`/`),
not a single-phase composition `parse_formula` can represent. Full
88-row bucket assignment is by direct inspection of each string (not a
blind regex classifier); the mapping is preserved in
`data/external/tematdb/_step_e2_buckets.json` alongside the printed
per-item numbering above. Some strings straddle two buckets (e.g. item
27, `PbTe:Bi2Te3=27:1 barbell nanowire`, has both a colon-decoration and
free-text tail) — assigned to the bucket judged the dominant parse-
breaking cause, not a unique/exhaustive taxonomy. This sizes the excluded
set; nothing was recovered, repaired, or re-attempted, per the hard
constraint.

### ROUND 2 GAPS AND UNCERTAINTIES

1. **Property mapping (`rho` -> `rho` or `sigma`) is an inference from
   `data_cleaning.py`'s existing vocabulary, not a value given anywhere
   in either dataset's schema.** teMatDb's raw `rho` tepname is treated
   as corresponding to whichever of training's `rho`/`sigma` properties
   exists for a given sample (they are the same physical curve, pre/post
   inversion) — reasonable given the existing codebase's own naming, but
   not independently verified against original figure captions.
2. **Sample-level pairing is DOI-level cartesian product, not verified
   true 1:1 curve correspondence.** Most DOIs produce many candidate
   pairs (up to 40) because Starrydata2's `sample_id` is far finer-
   grained than teMatDb's consolidated sample count; the vast majority
   of the 1,255 pairs are almost certainly comparing temperature grids
   from genuinely different physical curves that happen to share a DOI,
   diluting the pooled statistics. This is inherent to the task's own
   "do not collapse them" instruction, not a shortcut taken here.
3. **E2's bucket assignment is manual, by-inspection classification**,
   not a formal/reproducible algorithm — a different reviewer could draw
   the composite-vs-decoration boundary slightly differently for a
   handful of items (flagged inline above for item 27 specifically).
4. **BASEMAT/GROUP literal-string matching (E1) only tests exact string
   equality against training's `chemistry_cluster_id` values** — it does
   not check whether BASEMAT is chemically equivalent to some training
   cluster under a different string representation (e.g. reduced-formula
   ordering differences). A stricter chemical-equivalence check was not
   attempted, consistent with the hard constraint against adding
   normalization/salvage logic.
5. No model was refit, loaded, scored, or featurized in round 2 either;
   `parse_formula`/`chemistry_cluster_id` were called exactly as they
   exist in `src/canonicalization.py`, unmodified, no fallback used.

No conclusions drawn about whether to score any stratum. Report and stop.

---

## ROUND 3

Read-only, same hard constraints as round 2 (see there for the full list).
Three questions, final inventory round. TEMATDB_SHA unchanged.

### G. Anatomy of the 72 exact matches

Recomputed round 2's Step C distances at full point-level granularity
(pair, tepname, exact temperatures) — round 2 only saved pair-level
counts, not the underlying values. Recomputation reproduces round 2's
pooled total exactly: 53,525 comparisons.

**G1. Per-teMatDb-sample best match** (median distance pooled across
tepnames within each of the 1,251 candidate pairs that had >=1
in-range comparison; for each of the 176 (a0) teMatDb samples, the
minimum such median across its candidate training samples):

| Statistic | Value (K) |
|---|---|
| min | 0.1111 |
| p10 | 0.5279 |
| median | 1.3377 |
| p90 | 3.7569 |
| max | 19.1580 |

Histogram, bin edges `[0, 0.001, 0.01, 0.1, 1, 2, 5, 10, 25, inf]`:

| Bin | Count |
|---|---|
| [0, 0.001) | 0 |
| [0.001, 0.01) | 0 |
| [0.01, 0.1) | 0 |
| [0.1, 1) | 65 |
| [1, 2) | 65 |
| [2, 5) | 34 |
| [5, 10) | 5 |
| [10, 25) | 7 |
| [25, inf) | 0 |

**Reported plainly, no conclusion drawn**: zero of the 176 teMatDb
samples' BEST training counterpart has a median distance below 0.1K —
the minimum across all 176 is 0.1111K. This resolves round 1/2's mixed
pooled signal cleanly at the per-sample level: the true best-matching
training counterpart is never near-identical (in the pair-median sense);
it is typically ~1.3K off. The 72 exact matches and 247 sub-0.01K points
found in the pooled (non-best-matched) analysis are therefore concentrated
somewhere other than each teMatDb sample's own best pair.

**G2. The 72 exact-zero matches**, full table (teMatDb sample_id,
Composition_detailed, DOI, tepname, exact temperature, training
sample_id):

| teMatDb sample_id | Composition_detailed | DOI | tepname | Exact T (K) | Training sample_id |
|---|---|---|---|---|---|
| 390 | Mg3.15Mn0.05Sb1.5Bi0.49Se0.01 | 10.1002/adfm.201906143 | alpha | 300.00 | 27124 |
| 390 | Mg3.15Mn0.05Sb1.5Bi0.49Se0.01 | 10.1002/adfm.201906143 | alpha | 300.00 | 27127 |
| 390 | Mg3.15Mn0.05Sb1.5Bi0.49Se0.01 | 10.1002/adfm.201906143 | kappa | 300.00 | 27127 |
| 390 | Mg3.15Mn0.05Sb1.5Bi0.49Se0.01 | 10.1002/adfm.201906143 | kappa | 300.00 | 27132 |
| 413 | Ge0.89Sb0.1In0.01Te | 10.1002/adma.201705942 | alpha | 300.00 | 31968 |
| 413 | Ge0.89Sb0.1In0.01Te | 10.1002/adma.201705942 | alpha | 300.00 | 31969 |
| 413 | Ge0.89Sb0.1In0.01Te | 10.1002/adma.201705942 | alpha | 300.00 | 31974 |
| 14 | Cu0.01Bi2Te2.7Se0.3 | 10.1002/aenm.201100149 | kappa | 523.15 | 25591 |
| 37 | Bi0.5Sb1.5Te3 | 10.1002/aenm.201401391 | kappa | 400.00 | 1406 |
| 37 | Bi0.5Sb1.5Te3 | 10.1002/aenm.201401391 | kappa | 400.00 | 1407 |
| 37 | Bi0.5Sb1.5Te3 | 10.1002/aenm.201401391 | kappa | 400.00 | 1408 |
| 37 | Bi0.5Sb1.5Te3 | 10.1002/aenm.201401391 | kappa | 400.00 | 1409 |
| 37 | Bi0.5Sb1.5Te3 | 10.1002/aenm.201401391 | kappa | 400.00 | 1410 |
| 395 | Mg0.995Li0.005Ag0.97Sb0.99 | 10.1002/aenm.201502269 | alpha | 300.00 | 24962 |
| 395 | Mg0.995Li0.005Ag0.97Sb0.99 | 10.1002/aenm.201502269 | alpha | 300.00 | 24963 |
| 395 | Mg0.995Li0.005Ag0.97Sb0.99 | 10.1002/aenm.201502269 | alpha | 300.00 | 24966 |
| 395 | Mg0.995Li0.005Ag0.97Sb0.99 | 10.1002/aenm.201502269 | alpha | 300.00 | 24967 |
| 9 | Bi0.3Sb1.7Te3 | 10.1038/am.2013.86 | alpha | 400.00 | 6454 |
| 9 | Bi0.3Sb1.7Te3 | 10.1038/am.2013.86 | alpha | 400.00 | 6456 |
| 9 | Bi0.3Sb1.7Te3 | 10.1038/am.2013.86 | alpha | 400.00 | 6459 |
| 9 | Bi0.3Sb1.7Te3 | 10.1038/am.2013.86 | alpha | 400.00 | 6465 |
| 424 | Ba0.3In0.3Co4Sb12+0.1%Co | 10.1038/nature23667 | alpha | 800.00 | 31357 |
| 424 | Ba0.3In0.3Co4Sb12+0.1%Co | 10.1038/nature23667 | ZT | 300.00 | 31357 |
| 424 | Ba0.3In0.3Co4Sb12+0.1%Co | 10.1038/nature23667 | ZT | 800.00 | 31357 |
| 424 | Ba0.3In0.3Co4Sb12+0.1%Co | 10.1038/nature23667 | ZT | 300.00 | 31359 |
| 4 | FeNb0.88Hf0.12Sb | 10.1038/ncomms9144 | ZT | 450.00 | 20276 |
| 4 | FeNb0.88Hf0.12Sb | 10.1038/ncomms9144 | ZT | 600.00 | 20276 |
| 4 | FeNb0.88Hf0.12Sb | 10.1038/ncomms9144 | ZT | 900.00 | 20276 |
| 4 | FeNb0.88Hf0.12Sb | 10.1038/ncomms9144 | ZT | 1050.00 | 20276 |
| 4 | FeNb0.88Hf0.12Sb | 10.1038/ncomms9144 | kappa | 300.00 | 20277 |
| 4 | FeNb0.88Hf0.12Sb | 10.1038/ncomms9144 | kappa | 1200.00 | 20277 |
| 4 | FeNb0.88Hf0.12Sb | 10.1038/ncomms9144 | kappa | 300.00 | 20280 |
| 4 | FeNb0.88Hf0.12Sb | 10.1038/ncomms9144 | kappa | 1200.00 | 20839 |
| 4 | FeNb0.88Hf0.12Sb | 10.1038/ncomms9144 | ZT | 300.00 | 20839 |
| 4 | FeNb0.88Hf0.12Sb | 10.1038/ncomms9144 | ZT | 750.00 | 20839 |
| 4 | FeNb0.88Hf0.12Sb | 10.1038/ncomms9144 | ZT | 1200.00 | 20839 |
| 103 | NaxPb1-xTe (0.5% < x < 2%) | 10.1039/c0ee00456a | kappa | 300.00 | 57 |
| 121 | PbSe:Al0.01 | 10.1039/c1ee02465e | ZT | 300.00 | 15738 |
| 387 | Sn0.98Bi0.02Te + 3% HgTe | 10.1039/c4ee01463d | kappa | 300.00 | 407 |
| 200 | FeNb0.8Ti0.2Sb | 10.1039/c4ee03042g | kappa | 400.00 | 10602 |
| 200 | FeNb0.8Ti0.2Sb | 10.1039/c4ee03042g | ZT | 550.00 | 10602 |
| 200 | FeNb0.8Ti0.2Sb | 10.1039/c4ee03042g | kappa | 400.00 | 10607 |
| 372 | Bi0.5Sb1.495Cu0.005Te3 | 10.1039/c6ee02017h | rho | 450.00 | 33617 |
| 372 | Bi0.5Sb1.495Cu0.005Te3 | 10.1039/c6ee02017h | rho | 450.00 | 33618 |
| 374 | Cu2Se + 1 mol% In | 10.1039/c7ee01193h | kappa | 625.00 | 91297 |
| 374 | Cu2Se + 1 mol% In | 10.1039/c7ee01193h | kappa | 575.00 | 91298 |
| 374 | Cu2Se + 1 mol% In | 10.1039/c7ee01193h | kappa | 625.00 | 91298 |
| 383 | Bi0.05Ge0.99Te | 10.1039/c9ee00317g | rho | 300.00 | 31975-31986 (11 rows, same T) |
| 230 | Ba0.08Yb0.09Co4Sb12.12 | 10.1063/1.2920210 | ZT | 800.00 | 3363 |
| 224 | PrFe4Sb12 | 10.1063/1.3553842 | rho | 300/800 (alternating) | 5381-5388 (8 rows) |
| 67 | Cu0.01Bi2Te2.7Se0.3 | 10.1088/0957-4484/24/28/285702 | alpha | 300/500 (mixed) | 30526-30534 (5 rows) |

(Rows 43-49 collapsed for readability where a single teMatDb sample repeats
the same exact temperature across many training sample_ids — full
uncollapsed 72-row table in `data/external/tematdb/_step_g2_exact_matches.csv`.)

- **Distinct teMatDb sample_ids among the 72: 18**
- **Distinct candidate pairs among the 72: 56**
- **Integer temperatures: 71 of 72** (the one exception, 523.15K, is
  exactly 250°C — a Celsius-to-Kelvin round-number artifact, not a
  non-round value)
- **Multiples of 5: 71 | of 10: 68 | of 25: 71 | of 50: 68**
- **Decimal places, both sides: 71 of 72 have 0 decimal places on both
  training and teMatDb; the 1 exception (523.15K) has 2 decimal places
  on both sides identically** (since it's the *same* point on both
  sides by construction of the nearest-neighbor search — this doesn't
  indicate independent rounding, just that both temperature fields
  store fractional Kelvin when needed).

**Plain statement, no conclusion**: the exact matches are overwhelmingly
round, conventional measurement temperatures (300K room temperature,
400K, 800K axis endpoints, 450/500/550/600/625/750/900/1050/1200K) spread
thinly across only 18 of 176 (a0) teMatDb samples and 56 pairs — a
pattern consistent with independent researchers both choosing
conventional round measurement points, which is a separate phenomenon
from digitizing the identical curve.

**G3. The 175 matches in (0, 0.01) K**, condensed:

| | Value |
|---|---|
| n matches | 175 |
| Distinct teMatDb sample_ids | 78 |
| Distinct candidate pairs | 141 |
| Integer temperatures | 14 of 175 |
| Multiples of 5 / 10 / 25 / 50 | 12 each |
| Distance min / median / max | 0.0 / 0.0044 / 0.0099 K |

Qualitatively different from the exact-zero group: spread across far
more samples (78 vs. 18) and pairs (141 vs. 56), and mostly NOT round
numbers (only 8% are integers, vs. 99% for the exact-zero group) —
consistent with small floating-point coincidences among continuously-
distributed digitized values landing close by chance, a different
signature from the round-number pattern in G2. Reported as computed;
no conclusion drawn.

### H. How "BASEMAT not in training" was actually computed — found to be ill-posed

**Exact reconstruction of round 2's code path** (nothing changed, this
is a read of what was already run):
```python
training_clusters = set(
    pd.read_csv(featurized_csv, usecols=["chemistry_cluster_id"])["chemistry_cluster_id"].dropna()
)
stratum_b["BASEMAT_in_training_clusters"] = stratum_b["BASEMAT"].isin(training_clusters)
```
Training-side quantity: the **set of 12,010 literal strings** appearing
in the training CSV's `chemistry_cluster_id` column. This is a **literal
string set-membership test**, not a chemical-equivalence test.

10 example BASEMAT values tested:

| BASEMAT | isin(training_clusters) | Matched training value |
|---|---|---|
| Mg2Si | True | 'Mg2Si' |
| Bi2Te3 | True | 'Bi2Te3' |
| Bi2Te3 | True | 'Bi2Te3' |
| BiMI | False | (none) |
| PbTe | False | (none) |
| PbTe | False | (none) |
| PbTe | False | (none) |
| PbTe | False | (none) |
| SiGe | True | 'SiGe' |
| SnSe | True | 'SnSe' |

**`PbTe` showing `False` was suspicious enough to investigate directly**
(PbTe is one of the most heavily studied thermoelectric materials — its
total absence from a 12,010-cluster training set is implausible on its
face). Direct check: `'PbTe' in training_clusters` -> **`False`**.
Investigated why: `pymatgen.core.Composition("PbTe").reduced_formula`
returns **`'TePb'`**, not `'PbTe'` — pymatgen's internal element-ordering
convention (used inside `chemistry_cluster_id()` itself, which calls
`Composition(host_amounts).reduced_formula`) does not preserve input
order. `'TePb' in training_clusters` -> **`True`**. **The literal-string
comparison in round 2 was ill-posed**: it compares teMatDb's own BASEMAT
string convention against pymatgen's reduced-formula string convention
without reconciling the two, so a real match is missed purely because
of element ordering.

Re-ran the same `Composition(...).reduced_formula` check (the exact
mechanism `chemistry_cluster_id()` itself uses internally, not a new or
modified function) against all 14 "not in training" BASEMAT values:

| sample_id | BASEMAT | pymatgen reduced_formula | Present in training (reordered) | Type |
|---|---|---|---|---|
| 78, 91, 101, 104 | PbTe | TePb | **True** (x4) | chemical formula — **false negative** |
| 72 | BiMI | MBiI | False | chemical formula with placeholder species ("M" = generic metal site) |
| 153 | ABQ2 | AQ2B | False | family/template abbreviation (A, B, Q all generic placeholders) |
| 201 | HH | H2 | False | family abbreviation ("half-Heusler") — pymatgen mis-parses as diatomic hydrogen |
| 216, 225, 226, 228, 232 | SKD | KHS | False (x5) | family abbreviation ("skutterudite") — pymatgen mis-parses "D" as deuterium |
| 302 | M2Q | M2Q | False | template/placeholder (M, Q both generic) |
| 397 | Bi2O2Se | Bi2SeO2 | False | **genuine chemical formula, no match found even after reordering** |

**Verdict, as instructed**: the comparison was ill-posed, and **the 14
"not in training" BASEMAT values from round 2 are marked UNRELIABLE.**
Of the 14: 4 (PbTe x4) are confirmed false negatives — genuinely present
in training, just under a different element ordering. 8 (HH, SKD x5,
ABQ2, M2Q) are family abbreviations or generic templates, not real
specific chemical formulas at all — pymatgen "successfully" parsing them
is itself a false signal (it's parsing nonsense into a technically-valid
but chemically-meaningless composition). 1 (BiMI) is a formula with an
embedded placeholder species — partially real, partially uninterpretable.
Only 1 of the original 14 (**Bi2O2Se**, sample 397) survives as a
genuine, well-formed chemical formula with no training match found even
after correcting for element-ordering — that is the only one of the 14
that remains a defensible "possibly novel chemistry" data point. Nothing
in `src/` was modified to reach this finding — the reordering check
reuses `pymatgen.core.Composition(...).reduced_formula` directly, the
identical mechanism `chemistry_cluster_id()` already calls internally.

### I. Thinned rows for the parseable strata

Thinned = one row per `(sample_id, temperature_bin)`, 300-800K, 25K
bins, `right=False`, via `step3_filter_temperature` (unmodified,
reused).

| Stratum | n samples | Thinned rows | alpha | rho | kappa | ZT_author_declared |
|---|---|---|---|---|---|---|
| a0 ∩ parseable | 122 | 2,027 | 2,027 | 2,027 | 2,027 | 2,027 |
| a ∩ parseable | 62 | 903 | 903 | 903 | 903 | 903 |
| b | 36 | 549 (confirmed, matches round 1 and 2 exactly) | 549 | 549 | 549 | 549 |

**100% per-property coverage in every stratum, on every property, is
expected and not a finding of note**: `teMatDb_collocatedTEPs.csv` is
the co-interpolated ("collocated") grid by construction — every row
already has a value for all four properties by definition of how that
file was built, unlike a raw sparse per-property measurement table
(`teMatDb_rawTEPs.csv`, used in Section G/C, does have genuinely sparse
per-tepname coverage).

### ROUND 3 GAPS AND UNCERTAINTIES

1. **G1's "best match" is the minimum PAIR-MEDIAN distance, not a
   verified true physical correspondence.** A pair's low median could
   still reflect two unrelated curves that happen to share several
   round-number temperatures; G1 does not (and per the hard constraints,
   cannot) confirm which training sample_id is the *actual* physical
   counterpart of a given teMatDb sample.
2. **The property mapping from round 2 (`rho` -> `rho`/`sigma`) is
   carried forward unchanged into G's recomputation** — same inference,
   same caveat as round 2's GAPS item 1.
3. **H's finding generalizes beyond the 14 checked**: if pymatgen's
   reduced-formula reordering caused false negatives here, the same
   effect likely affects some of the 22 "BASEMAT present" cluster
   matches in round 2's E1 in the *other* direction is not possible
   (a string match found under `isin()` is unambiguously a real match
   regardless of ordering, since the compared string is already in
   pymatgen's own canonical order) — so the 22 "present" cases are NOT
   called into question by this finding, only the 14 "absent" cases
   were.
4. **The `GROUP`-vs-training-clusters comparison from round 2's E1 (13
   of 36 present) was not re-examined here** — the same ill-posedness
   likely applies (GROUP values like "Bi2Te3" or "PbTe" would have the
   identical reordering issue), but this was not re-run since the task
   scoped the fix to BASEMAT specifically. Flagged, not corrected.
5. No model was refit, loaded, scored, or featurized in round 3.
   `parse_formula`/`chemistry_cluster_id` were not modified; the only
   new code is a direct diagnostic call to
   `pymatgen.core.Composition(...).reduced_formula` (the same
   expression `chemistry_cluster_id()` already evaluates internally),
   used to explain a suspicious result, not to alter any prior count.

No conclusions drawn about whether to score any stratum. Report and stop.

## POST-SCORING CORRECTIONS

READ-ONLY task, reusing the saved per-row predictions from
`results/20260910T123047_tematdb_external/tematdb_{a0,a,b}_predictions.npz`.
No refit, no model touch, no new smear-factor computation. Nothing in
`src/`, `CLAUDE.md`, or `.gitattributes` was modified; no `git add` /
`commit` / `push` was run. Writes: `results/20260910T123047_tematdb_external/corrections.json`
and this section.

### K. Clean chemistry-transfer stratum: (b) intersect (a)

Stratum (b) (cluster-disjoint from training) was defined without
conditioning on DOI. A paper contributing one composition to training
can contribute a different, cluster-disjoint composition to teMatDb, so
a (b) sample can still share a DOI with training — leaking
measurement/paper-level information into what is being called
"chemistry transfer."

**K1. Cross-tab of stratum (b) against the DOI strata:**

| Subset | Samples | Clusters | Thinned rows |
|---|---|---|---|
| b ∩ a0 (DOI **in** training — contaminated) | 11 | 11 | 159 |
| b ∩ a (DOI **not** in training — CLEAN) | 25 | 25 | 390 |
| Row check | | | 159 + 390 = 549, matches stratum (b) total exactly |

**K3. Interpretability**: 25 clusters in the clean subset, above the
~10-cluster threshold stated in the task — reported as a metric, not
demoted to an inventory-only finding.

**K2. Metrics recomputed by slicing the saved arrays** (the `.npz`
files carry `sample_id`, confirmed sufficient to slice; no re-prediction
was needed or performed):

| Property | b∩a0 (contaminated, n=159, 11 samples) R² | b∩a CLEAN (n=390, 25 samples) R² |
|---|---|---|
| S | 0.4339 | 0.7561 |
| sigma (log10) | -0.2996 | -0.2945 |
| kappa (log10) | 0.7245 | 0.6281 |
| zT_direct (vs declared) | 0.4327 | 0.5788 |
| zT_direct (vs TEP-recomputed) | 0.4296 | 0.5820 |
| zT_derived (vs declared) | -1.1890 | 0.2143 |
| zT_derived (vs TEP-recomputed) | -1.1892 | 0.2156 |

Bootstrap 95% CIs (sample_id-level resampling) are wide for both
subsets, as expected at n=11 and n=25 samples — see `corrections.json`
for the full interval set.

**Finding, reported without causal interpretation:** sigma stays
negative in both subsets (-0.2996 contaminated vs -0.2945 clean,
essentially unchanged) — DOI leakage does not explain sigma's poor
stratum-(b) performance. S and zT_direct are *higher* in the clean
subset than the contaminated one (S: 0.756 clean vs 0.434 contaminated;
zT_direct: 0.579-0.582 clean vs 0.430-0.433 contaminated) — the
opposite direction from the leakage-inflation hypothesis for these two
properties. kappa is lower in the clean subset (0.628 vs 0.725), the
one property moving in the anticipated direction, though the shift is
modest relative to the CI widths at these sample sizes. No conclusion
is drawn about causation.

### L. Model-free digitization-agreement floor

Motivation stated in the task: stratum (a0) is scored against a model
refit on 100% of training, which contains those exact DOIs, so (a0)'s
R² may be inflated by proximity to fitted feature vectors rather than
genuine transfer accuracy — the scored (a0) R² (S 0.9354) already
exceeds internal chemistry-cluster grouped CV (S 0.8076). This section
measures label-to-label agreement between teMatDb and Starrydata2
directly, without the model.

**L1. Matched-point construction**: for each of round 3's 176
best-match (teMatDb sample, nearest training sample) pairs, each
teMatDb raw abscissa was linearly interpolated (`np.interp`) against
the matched training curve, kept only where the point is bracketed by
the training curve's temperature range **and** the nearest training
abscissa is within 5K.

| Property | Total considered | Survived | Dropped: not bracketed | Dropped: bracketed but >5K |
|---|---|---|---|---|
| S | 2,260 | 1,759 | 214 | 287 |
| sigma | 2,119 | 1,712 | 196 | 211 |
| kappa | 2,020 | 1,591 | 212 | 217 |
| zT | 2,005 | 1,626 | 182 | 197 |

Total matched points across all four properties: 6,688.

**SUPERSEDED — see `## L CORRECTED (COMPOSITION-MATCHED)` below.** L1
selected each teMatDb sample's training counterpart by smallest median
TEMPERATURE distance only; composition was never checked. Most DOIs map
one teMatDb sample to many training samples of different compositions
from the same paper, so L2's numbers below measured composition
similarity within a paper, not digitization agreement. Kept for audit
continuity, not for citation.

**L2. Label-to-label agreement** (no model; sigma/kappa in log10 space
to match training's scoring space, S/zT linear; bootstrap 95% CI
resampled at teMatDb `sample_id` level):

| Property | Full-range R² | 300-800K R² |
|---|---|---|
| S | 0.7347 (n=1,759, 157 samples) | 0.7175 (n=1,574, 156 samples) |
| sigma (log10) | 0.5275 (n=1,712, 147 samples) | 0.4728 (n=1,549, 145 samples) |
| kappa (log10) | 0.8950 (n=1,591, 156 samples) | 0.8951 (n=1,441, 156 samples) |
| zT | 0.7176 (n=1,626, 144 samples) | 0.7366 (n=1,501, 144 samples) |

**SUPERSEDED — see `## L CORRECTED (COMPOSITION-MATCHED)` below.** This
gate's L2 inputs were built on temperature-proximity pairs, not
composition-identical ones (see note above); N5 reruns this gate on the
corrected agreement numbers and reaches the opposite verdict.

**L3. Sanity gate — FLAGGED PROMINENTLY, TRIGGERED FOR ALL FOUR PROPERTIES:**

| Property | Scored (a0) R² | Label-agreement R² (300-800K) | Label-agreement R² (full) | Gate |
|---|---|---|---|---|
| S | 0.9354 | 0.7175 | 0.7347 | **TRIGGERED** |
| sigma | 0.7505 | 0.4728 | 0.5275 | **TRIGGERED** |
| kappa | 0.9094 | 0.8951 | 0.8950 | **TRIGGERED** |
| zT | 0.8480 | 0.7366 | 0.7176 | **TRIGGERED** |

The scored (a0) model R² exceeds the model-free digitization-agreement
R² for every property, not just some — including a large gap for S
(0.935 vs 0.72-0.73) and sigma (0.751 vs 0.47-0.53), a moderate gap for
zT (0.848 vs 0.72-0.74), and only a marginal gap for kappa (0.909 vs
0.895). Per the task's own framing, this means the frozen model
predicts teMatDb's labels better than Starrydata2's own digitized
curves agree with teMatDb's digitized curves for the same physical
samples — for every property scored. Two candidate explanations were
named by the task itself and neither is adjudicated here:

1. **L1's matching is wrong** — the best-match pairing, interpolation,
   or 5K tolerance in Section L1 (or the underlying round-3 best-match
   selection it depends on) systematically understates true label
   agreement.
2. **(a0) is contaminated** — the scored (a0) R² is inflated because
   the frozen model was refit on 100% of training, which contains the
   exact DOIs (a0) is scored against, so those rows are being predicted
   near fitted feature vectors rather than being genuinely held out.

**Report, do not fix.** No conclusion is drawn about which explanation,
if either alone, accounts for the gap.

### POST-SCORING CORRECTIONS GAPS AND UNCERTAINTIES

1. K's subsets are small (11 and 25 samples) — the CIs in
   `corrections.json` are wide, and the directional findings for S,
   zT_direct, and kappa should not be treated as statistically
   distinguishable from each other without more data.
2. L1's tolerance (5K) and its dependence on round 3's single
   best-match pair per teMatDb sample (not a full nearest-neighbor
   search per point) both limit how much of teMatDb's raw data
   contributes to L2 — roughly 78% of considered points survived
   across the four properties, not all of them.
3. L3's gate is reported as triggered for all four properties with two
   named candidate explanations; this report takes no position on
   which is correct, or whether both contribute.
4. No new git state, `src/` code, `CLAUDE.md` content, or
   `.gitattributes` rule resulted from this task.

No conclusions drawn.

## L CORRECTED (COMPOSITION-MATCHED)

READ-ONLY. No refit, no model touch. Redoes section L's digitization-
agreement floor because L1 paired each teMatDb sample to a training
counterpart by smallest median TEMPERATURE distance alone, never
checking composition. Most DOIs in this dataset map one teMatDb sample
to several training samples that are different compositions from the
same paper, so the original L1/L2/L3 numbers (marked SUPERSEDED above)
measured how similar compositions are *within a paper*, not
digitization agreement between two independent measurements of the
*same* material. This was confirmed by the shape of the old result
itself: sigma (the property most sensitive to composition) scored
lowest (0.47), kappa (least composition-sensitive) scored highest
(0.895) — the signature of a composition mismatch, not a digitization
disagreement.

### N1. Identity pairing

For each of the 176 stratum-(a0) teMatDb samples (DOI in training),
`Composition_detailed` was canonicalized via `parse_formula ->
composition_id` (pymatgen `reduced_formula`) — the exact function pair
`src/canonicalization.py` already exposes, reused unmodified. Every
training composition sharing that sample's DOI was canonicalized the
identical way (training's `composition_id` column is already produced
by this same mechanism, confirmed by reading `add_canonical_columns`,
so it was reused directly rather than recomputed). Pairs were kept only
where the two canonical reduced formulas are string-equal — as round 3
showed, `Composition("PbTe").reduced_formula` returns `'TePb'`, so raw
strings are never compared against normalized ones on either side.

| | Count |
|---|---|
| a0 samples (DOI in training) | 176 |
| a0 samples with composition parseable | 122 |
| a0 samples with composition parse failure | 54 |
| DOI-shared candidate pairs (round 3's G1 universe) | 1,251 |
| Candidate pairs with composition MATCH | 197 |
| teMatDb samples with >=1 composition match | **96** |
| teMatDb samples with DOI in training but NO composition match | 80 |

96 is above the ~20-sample threshold the task named, so no
interpretability flag applies.

**Matches per sample** (matches -> n_samples): 1 match: 72 samples;
2: 8; 3: 3; 4: 3; 5: 2; 6: 3; 7: 2; 12: 1; 14: 1; 20: 1 (one prolific
paper contributed compositions matching 20 different training rows
under the same DOI).

One training sample_id was selected per teMatDb sample for point
matching: among that sample's composition-identical candidates only,
the smallest pair-median temperature distance (round 3's G1 metric,
reused) breaks the tie. Proximity is used here only to choose among
already-identical compositions, never to select across different ones
— this is the one place proximity still appears, and it does not
reintroduce the original bug.

### N2. Matched-point construction

Same rule as the original L1: for each teMatDb raw abscissa, linearly
interpolate the (now composition-matched) training curve to that
temperature, keeping only points bracketed by the training curve's
range and within 5K of the nearest training abscissa.

| Property | Considered | Survived | Not bracketed | Bracketed but >5K |
|---|---|---|---|---|
| S | 1,215 | 913 | 128 | 174 |
| sigma | 1,137 | 884 | 118 | 135 |
| kappa | 1,035 | 791 | 105 | 139 |
| zT (declared) | 1,198 | 949 | 121 | 128 |

**Recomputed zT (alpha²T/(ρκ))**: teMatDb's own alpha/rho/kappa raw
curves (same sample) were linearly interpolated onto that sample's own
ZT raw abscissa (bracket-only, no 5K tolerance needed — this is a
same-database, same-sample interpolation, not a cross-database match),
then `zt_tep = alpha^2 * temperature_K / (rho * kappa)` computed
directly (alpha in raw V/K, matching the identical formula already used
for `zt_tep` in the scoring task). Of 1,264 ZT abscissa points, 1,125
recomputed successfully (all three component curves bracketed the
point) and 139 failed. The 1,125 recomputed values were then matched
against training's declared zT curve using the same bracket+5K rule:
1,071 considered, 902 survived, 49 not bracketed, 120 bracketed-but-far.

Total matched points: 3,537 (declared streams) + 902 (zT_tep) = 4,439.

### N3. Composition-matched label-to-label agreement

sigma/kappa in log10 space, S/zT linear, bootstrap 95% CI resampled at
teMatDb `sample_id` level (2,000 draws):

| Property | Full-range R² | 300-800K R² | n points (full) | n points (300-800K) | n samples | n DOIs |
|---|---|---|---|---|---|---|
| S | 0.9661 | 0.9628 | 913 | 824 | 87 | 85 |
| sigma | 0.9788 | 0.9777 | 884 | 798 | 80 | 78 |
| kappa | 0.9791 | 0.9806 | 791 | 722 | 86 | 84 |
| zT (vs declared) | 0.9504 | 0.9597 | 949 | 885 | 87 | 86 |
| zT (vs recomputed alpha²T/(ρκ)) | 0.9514 | 0.9607 | 902 | 853 | 86 | 85 |

Every property now shows 0.95-0.98 label-to-label agreement, a
categorical improvement over the SUPERSEDED numbers (S 0.72-0.73, sigma
0.47-0.53, kappa 0.895, zT 0.72-0.74).

### N4. Diagnostic — old pairs restricted to composition-DIFFER subset

The 176 old temperature-proximity pairs were classified against the
same composition-identity test: 61 have DIFFERENT canonical
compositions, 52 happen to have the SAME composition (proximity
accidentally also got these right), and 63 are UNKNOWN (one side failed
to parse). The OLD L1 matched-points table was then sliced (not
re-matched) down to the 61 composition-DIFFER pairs and rescored:

| Property | R² (composition-DIFFER subset) | RMSE | MAE | n points | n samples |
|---|---|---|---|---|---|
| S | 0.6451 | 109.1541 | 43.2786 | 662 | 54 |
| sigma | 0.1801 | 0.3635 | 0.1925 | 630 | 51 |
| kappa | 0.8534 | 0.1070 | 0.0684 | 542 | 54 |
| zT | 0.5923 | 0.2613 | 0.1620 | 602 | 53 |

**Side by side, composition-matched (N3) vs. composition-DIFFER (N4)**:

| Property | Composition-matched R² | Composition-DIFFER R² | Gap |
|---|---|---|---|
| sigma | 0.978-0.979 | 0.180 | **0.80 (largest)** |
| zT | 0.950-0.961 | 0.592 | 0.36 |
| S | 0.963-0.966 | 0.645 | 0.32 |
| kappa | 0.979-0.981 | 0.853 | **0.13 (smallest)** |

This matches the diagnosis exactly: sigma, the property most sensitive
to composition (conductivity varies by orders of magnitude with
dopant/stoichiometry changes), shows the largest gap; kappa, the least
composition-sensitive of the four, shows the smallest. Reported, not
adjudicated further — this is the check the task asked for to confirm
N1 worked, not a new independent finding.

### N5. Sanity gate, rerun against corrected numbers

| Property | Scored (a0) R² | Composition-matched agreement (full / 300-800K) | Gate |
|---|---|---|---|
| S | 0.9354 | 0.9661 / 0.9628 | not triggered |
| sigma | 0.7505 | 0.9788 / 0.9777 | not triggered |
| kappa | 0.9094 | 0.9791 / 0.9806 | not triggered |
| zT (vs declared) | 0.8480 | 0.9504 / 0.9597 | not triggered |
| zT (vs recomputed TEP) | 0.8480 | 0.9514 / 0.9607 | not triggered |

**Not triggered for any property.** The scored (a0) model R² now sits
below the corrected digitization-agreement ceiling everywhere — the
expected ordering, since a model should not be able to exceed the floor
set by two independent digitizations' own noise. This reverses the
original L3 finding and is consistent with explanation (1) from that
gate (the temperature-proximity matching was flawed), not explanation
(2) (that (a0) itself is contaminated) — though this does not, on its
own, rule out some degree of (a0) contamination; it only shows the
label-agreement floor no longer constrains that question either way.

### L CORRECTED — GAPS AND UNCERTAINTIES

1. 80 of 176 a0 samples (45%) have no composition match at all —
   either their own formula failed to parse (54) or no training
   composition shared under that DOI matches exactly (26). N3's
   sample counts (80-87 samples per property) reflect only the
   matched subset, not all of stratum (a0).
2. The tie-break rule (smallest pair-median distance among
   composition-identical candidates) is applied per teMatDb sample,
   so a sample with e.g. 20 composition-identical candidates
   contributes only one selected pair to N2/N3 — samples with many
   matches are not over-weighted in the point count relative to
   samples with one match, but their extra candidate pairs are simply
   unused. Not corrected, since the task asked for a rerun of the
   original single-pair-per-sample structure.
3. The recomputed-zT (`zt_tep`) stream is compared only against
   training's directly-digitized zT curve, not against a symmetric
   "recomputed zT" curve on the training side (training's raw alpha/
   rho/kappa curves were not similarly recombined) — an asymmetry
   inherited from what N3's instruction specified, not resolved here.
4. No new git state, `src/` code, `CLAUDE.md` content, or
   `.gitattributes` rule resulted from this task. All intermediate
   scratch files were written to the session scratchpad, not to
   `data/external/tematdb/`, to stay strictly within this task's
   stated write scope (`results/20260910T123047_tematdb_external/`
   and `reports/tematdb_inventory/` only).

No conclusions drawn.
