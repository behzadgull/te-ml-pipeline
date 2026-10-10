"""
Fill every number-bearing part of thesis_paper/paper/paper.md from committed artifacts, so that no value in the manuscript is typed by hand.

Two mechanisms, rewritten in place by this script:
  1. Inline values: an HTML-comment opening marker "v:key", the text, and a closing marker "/v" (see INLINE below). The text between the
     markers is replaced by the value computed for `key` (thesis_values.values() for Paper A numbers, plus the results of the new
     analyses once they exist). Pandoc drops the markers from the docx.
  2. Table blocks: BEGIN TABLE n / END TABLE n comment lines hold a generated table, caption included.
Anything still to be computed is written as [[PENDING: NAx]] (NAx as in reports/claim_inventory_summary.md); the build script refuses
to build the final docx while any remain, and prints them with `--list-pending`.

Usage (from the repository root):
    python thesis_paper/scripts/make_thesis_values.py                 # rewrite paper.md
    python thesis_paper/scripts/make_thesis_values.py --check         # exit 1 if paper.md is out of date or has an unknown key
    python thesis_paper/scripts/make_thesis_values.py --list-pending  # every [[PENDING: ...]] with its line
"""

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_a_values as pav  # noqa: E402
import thesis_values as tv  # noqa: E402
import audit_numbers as an  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "thesis_paper" / "paper" / "paper.md"
T4 = pav.TARGETS
INLINE = re.compile(r"<!--v:(\w+)-->(.*?)<!--/v-->")
DESIGN_MARK = re.compile(r"<!--d:([\w.\-]+)-->(.*?)<!--/d-->")
PENDING = re.compile(r"\[\[PENDING: [^\]]+\]\]")
MINUS = "−"


def _pend(na):
    return f"[[PENDING: {na}]]"


def extra_values(v):
    """Results of the new analyses (NA4, NA5, NA7, NA8, NA9); the analyses still to run keep their PENDING markers in the text."""
    return tv.new_analysis_values()


def table2(v):
    """Final dataset statistics (NA4), per property. Every cell is a key of the value dictionary (thesis_values.table_values)."""
    labels = {"S": "S (*µ*V K^−1^)", "sigma": "*σ* (S m^−1^)", "kappa": "*κ* (W m^−1^ K^−1^)", "zT": "zT"}
    lines = [f"**Table 2:** Final dataset statistics: the {v['n_featurized']} featurised rows, per property (coverage is the share of those rows with a value; "
             "SD is the sample standard deviation).", "",
             "| **Property** | **Rows** | **Coverage** | **Mean** | **Median** | **SD** | **Range** |",
             "|---------------|---------|-----------|-----------|-----------|-----------|----------------|"]
    for t in T4:
        lines.append(f"| {labels[t]} | {v[f't2_{t}_n']} | {v[f't2_{t}_cov']} | {v[f't2_{t}_mean']} | {v[f't2_{t}_median']} | {v[f't2_{t}_sd']} | "
                     f"{v[f't2_{t}_min']} to {v[f't2_{t}_max']} |")
    return "\n".join(lines)


def table4(v):
    """XGBoost hyperparameters: the four frozen sets and the search space. Every cell is a key of the value dictionary."""
    head = ("**Table 4:** XGBoost hyperparameters: the four frozen sets (each tuned once on all rows of its target, by a "
            f"{v['optuna_trials']}-trial Optuna search scored with {v['inner_folds']}-fold chemistry-cluster cross-validation) and the search space.")
    lines = [head, "", "| **Parameter** | **S** | ***σ*** | ***κ*** | **zT** | **Search range** |",
             "|---------------------|-----------|-----------|-----------|-----------|------------------|"]
    for k in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight", "reg_lambda", "reg_alpha"):
        lines.append(f"| {k} | {v[f't4_S_{k}']} | {v[f't4_sigma_{k}']} | {v[f't4_kappa_{k}']} | {v[f't4_zT_{k}']} | {v[f't4_range_{k}']} |")
    return "\n".join(lines)


def table5(v):
    """Chemistry-cluster results per target."""
    labels = {"S": "S", "sigma": "log~10~*σ*", "kappa": "log~10~*κ*", "zT": "zT"}
    head = (f"**Table 5:** Chemistry-cluster grouped cross-validation results ({v['repeats']} repeats of {v['outer_folds']}-fold CV; R^2^ is the mean "
            "± standard deviation across repeats of the per-repeat pooled value, and MAE and RMSE are the means of the per-repeat pooled values; the "
            "fold-level SD is over all folds). MAE and RMSE are in *µ*V K^−1^ for S and in log~10~ units for *σ* and *κ*.")
    lines = [head, "", "| **Target** | **R^2^** | **Fold-level SD** | **MAE** | **RMSE** | **Rows** | **Features** |",
             "|------------|-------------|----------|-------|-------|----------|----------|"]
    for t in T4:
        lines.append(f"| {labels[t]} | {v[f'chem_{t}']} ± {v[f'chem_sd_{t}']} | {v[f'foldsd_{t}']} | {v.get(f'mae_{t}', _pend('NA8'))} | "
                     f"{v.get(f'rmse_{t}', _pend('NA8'))} | {v[f'n_{t}']} | {v['n_feat']} |")
    return "\n".join(lines)


def table6(v):
    """The five-way validation ladder."""
    head = (f"**Table 6:** R^2^ obtained with five validation methods (same model, frozen hyperparameters and {v['n_feat']} features). "
            f"Random 80/20 pools {v['rand_draws']} independent draws; 5-fold and 10-fold are single partitions; composition and chemistry cluster are the "
            f"mean ± SD across {v['repeats']} repeats. ∆R^2^ is random 80/20 minus chemistry cluster.")
    lines = [head, "", "| **Target** | **Random 80/20** | **5-fold** | **10-fold** | **Composition** | **Chemistry cluster** | **∆R^2^** |",
             "|---------|----------|----------|----------|-------------|-------------|----------|"]
    names = {"S": "S", "sigma": "*σ*", "kappa": "*κ*", "zT": "zT"}
    for t in T4:
        lines.append(f"| {names[t]} | {v[f'rand_{t}']} | {v[f'k5_{t}']} | {v[f'k10_{t}']} | {v[f'comp_{t}']} ± {v[f'comp_sd_{t}']} | "
                     f"{v[f'chem_{t}']} ± {v[f'chem_sd_{t}']} | {v[f'gap_{t}']} |")
    return "\n".join(lines)


def table8(v):
    """ESTM external validation."""
    head = (f"**Table 8:** ESTM external validation (R^2^). The {v['estm_scope']} ESTM rows within {v['t_min']}--{v['t_max']} K are scored in two strata that are never pooled: "
            f"rows sharing no source DOI with the training data (a, n = {v['estm_a_n']}) and rows whose chemistry cluster is absent from training "
            f"(b, n = {v['estm_b_n']}). The in-support column restricts (b) to rows inside the training range of S, *σ*, *κ* and temperature "
            f"(n = {v['estm_b_insup_n']}). The drop is stratum (b) minus the internal chemistry-cluster value.")
    lines = [head, "", "| **Target** | **Internal (chemistry cluster)** | **ESTM (a) DOI-disjoint** | **ESTM (b) cluster-disjoint** | **(b), in-support** | **Drop (b)** |",
             "|---------|-------------|-------------|-------------|-----------|---------|"]
    names = {"S": "S", "sigma": "*σ*", "kappa": "*κ*", "zT": "zT (direct)"}
    for t in T4:
        lines.append(f"| {names[t]} | {v[f'chem_{t}']} | {v[f'estm_a_{t}']} | {v[f'estm_b_{t}']} | {v[f'estm_b_insup_{t}']} | {v[f'estm_b_drop_{t}']} |")
    return "\n".join(lines)


def table8b(v):
    """teMatDb scoring with the saved final models: R2 per target and stratum with the sample-level bootstrap interval."""
    head = (f"**Table 8b:** teMatDb scoring with the same final models (R^2^ with the {v['tm_ci_level']}% interval of a bootstrap over whole samples, {v['tm_nboot']} draws, in brackets). "
            "Strata are never pooled: a0, the sample's source DOI is in the training data (not an external test); a, the DOI is not in the training data; "
            "b, the chemistry cluster is absent from training. zT (direct) is scored against the declared zT; *σ* and *κ* in log~10~ units. "
            "An ordering is stated only where the difference exceeds the fit-to-fit difference of the ESTM scoring (Section 4.2) and the intervals do not overlap; this is a descriptive rule, not a test.")
    lines = [head, "", "| **Stratum** | **Rows** | **Samples** | **S** | ***σ*** | ***κ*** | **zT (direct)** |", "|-------|-----|-----|---------|---------|---------|---------|"]
    names = {"a0": "a0, DOI in training", "a": "a, DOI not in training", "b": "b, cluster absent from training"}
    for s in ("a0", "a", "b"):
        cells = [f"{v[f'tm_{s}_{t}']} [{v[f'tm_{s}_{t}_lo']}, {v[f'tm_{s}_{t}_hi']}]" for t in T4]
        lines.append(f"| {names[s]} | {v[f'tm_{s}_rows']} | {v[f'tm_{s}_samples']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def table10(v):
    """Direct versus derived zT, under chemistry-cluster CV (Paper A run) and random row-level CV (NA7)."""
    lines = [f"**Table 10:** Direct versus component-wise zT prediction on the {v['dvd_rows']} rows with all four properties present "
             f"({v['dvd_clusters']} chemistry clusters); each model uses its own target's frozen hyperparameters. Chemistry-cluster CV is {v['dvd_repeats']} repeats x {v['dvd_folds']} folds "
             f"of grouped folds; random CV is {v['na7_repeats']} repeats x {v['na7_folds']} folds of shuffled row-level folds; both pool the out-of-fold predictions "
             f"(n = {v.get('na7_n', _pend('NA7'))}).", "",
             "| **Pathway** | **R^2^ (chemistry-cluster)** | **MAE (chemistry-cluster)** | **R^2^ (random)** | **MAE (random)** |",
             "|------------------------|----------|---------|----------|---------|",
             f"| Direct (features → zT) | {v['dvd_direct']} | {v.get('mae_dvd_direct', _pend('NA8'))} | {v.get('na7_direct', _pend('NA7'))} | {v.get('na7_mae_direct', _pend('NA7'))} |",
             f"| Derived (S^2^*σ*T/*κ*) | {v['dvd_derived']} | {v.get('mae_dvd_derived', _pend('NA8'))} | {v.get('na7_derived', _pend('NA7'))} | {v.get('na7_mae_derived', _pend('NA7'))} |"]
    return "\n".join(lines)


def table9(v):
    """Top five features per target by mean |SHAP| (NA3, chemistry-cluster folds), in the order S, sigma, kappa, zT; with the interval across the folds and the number of folds in which the feature is in the top three."""
    lines = [f"**Table 9:** The five features with the largest mean \\|SHAP\\| per target (mean over {v['shap_folds']} chemistry-cluster test folds, each on {v['shap_rows']} random test rows), each with the {v['iv_lo']}th to {v['iv_hi']}th percentile "
             f"of the per-fold mean across the {v['shap_folds']} folds and the number of folds in which it is among the three most important features. The folds are five repeats of five grouped folds, so they are not independent. "
             "Only orderings the folds support are interpreted: the intervals of ranks two to five overlap in every model, so those ranks are listed by their mean without a ranking claim. "
             "Units of the SHAP values are those of the target (*µ*V K^−1^ for S; log~10~ units for *σ* and *κ*). Every feature is a statistic of an ELEMENTAL property of the elements in the formula (or the temperature), "
             "not a measured property of the material; the statistic is the atomic-fraction-weighted mean, the weighted mean absolute deviation about it (mean abs. deviation), the range, the minimum or the maximum over the elements, "
             "or the mode (the value of the most abundant element), from the MAGPIE or CBFV element tables. The labels come from docs/feature_labels.csv.", "",
             "| **Rank** | **S** | ***σ*** | ***κ*** | **zT** |", "|----|-------------|-------------|-------------|-------------|"]
    for i in range(1, 6):
        lines.append(f"| {v[f't9_rank_{i}']} | {v[f't9_S_{i}']} | {v[f't9_sigma_{i}']} | {v[f't9_kappa_{i}']} | {v[f't9_zT_{i}']} |")
    return "\n".join(lines)


def table11(v):
    """The ranked list of the main-list candidates, in two groups (chemistry cluster seen or not seen in the training data)."""
    head = (f"**Table 11:** Ranked list of the {v['mp_f6']} lead-free, narrow-gap, perovskite-type candidates of the main list: hypotheses to test, not findings. Within each group "
            f"(chemistry cluster present in the training data, or absent) the candidates are ordered by the maximum over {v['t_min']}--{v['t_max']} K of the predicted zT, reached at the temperature "
            "T; the other columns are the predictions at T (S with the sign of the carrier-type classifier). E~hull~ in eV atom^−1^ (on hull: no energy above the convex hull), E~g~ the PBE gap in eV, "
            "*σ* in S m^−1^, *κ* in W m^−1^ K^−1^. † the shortlist of the pre-registered rule (docs/decisions.md). ‡ the maximum lies above the melting point of the compound; see Table 11b. Studied before: a paper whose title or abstract reports a thermoelectric measurement "
            "or calculation of the compound was found by a literature check made independently of the predictions (no evidence = none found, not none existing). The features are composition-only, so polymorphs of one compound (separate Materials Project entries) share their predictions.")
    lines = [head, "", "| **Cluster** | **#** | **Compound (MP id)** | **E~hull~** | **Stability** | **E~g~** | **T (K)** | **S (*µ*V K^−1^)** | ***σ*** | ***κ*** | **zT** | **Studied before** |",
             "|------|---|-----------|------|--------|-----|-----|------|--------|-----|-----|---------|"]
    for k in range(1, 31):
        lines.append(f"| {v[f't11_{k}_group']} | {v[f't11_{k}_rank']} | {v[f't11_{k}_name']} | {v[f't11_{k}_ehull']} | {v[f't11_{k}_stab']} | {v[f't11_{k}_gap']} | {v[f't11_{k}_T']} | "
                     f"{v[f't11_{k}_S']} | {v[f't11_{k}_sigma']} | {v[f't11_{k}_kappa']} | {v[f't11_{k}_zT']} | {v[f't11_{k}_lit']} |")
    return "\n".join(lines)


def table11b(v):
    """The shortlist: thermal limit, the maximum within the limit and the secondary view at 600 K."""
    lines = [f"**Table 11b:** The {v['sl_n']} shortlist compounds with their thermal limits (sources in the text) and the predictions within them: the highest grid temperature reported, the maximum predicted zT up to it (at the temperature T), "
             f"and the secondary view, the predicted zT at {v['lim_secondary_T']} K with the rank at {v['lim_secondary_T']} K next to the primary rank (maximum over the whole grid). Model outputs, not findings.", "",
             f"| **Compound** | **Thermal limit** | **Highest T reported (K)** | **Maximum zT within the limit** | **T (K)** | **zT at {v['lim_secondary_T']} K** | **Rank, primary** | **Rank, {v['lim_secondary_T']} K** |",
             "|-------------|--------------------|----------|---------|-----|-------|-----|-----|"]
    for k in range(1, 5):
        lines.append(f"| {v[f'lim_{k}_name']} | {v[f'lim_{k}_text']} | {v[f'lim_{k}_range']} | {v[f'lim_{k}_zt']} | {v[f'lim_{k}_zt_T']} | {v[f'lim_{k}_zt600']} | {v[f'lim_{k}_rank1']} | {v[f'lim_{k}_rank2']} |")
    return "\n".join(lines)


def table7(v):
    """Architecture comparison: R2 under chemistry-cluster CV (mean and SD over 5 repeats), and the paired difference from XGBoost with its cluster-bootstrap interval."""
    names = {"xgb": "XGBoost (tuned once, frozen)", "lgbm": "LightGBM (tuned, same protocol)", "rf": "Random forest (one fixed setting, not tuned)"}
    lines = [f"**Table 7:** Chemistry-cluster CV R^2^ (mean ± SD over {v['n_repeats']} repeats of {v['n_folds']} grouped folds) of XGBoost, LightGBM, a random forest, the mean of the three and their stack on the same folds, with each model's difference from XGBoost "
             f"(same folds, paired) and the chemistry-cluster bootstrap interval of that difference in brackets. XGBoost and LightGBM are tuned with the same {v['optuna_trials']}-trial search; the random forest uses one fixed, "
             "literature-based setting and is not tuned (Section 3.4), which favours XGBoost. The mean of the three models has no fitted weights. The stack is a non-negative ridge meta-learner over the three models whose weights are fitted on inner out-of-fold predictions inside each "
             "outer training fold, never on the outer test rows (Section 4.1.3). The last three rows are the paired comparisons fixed before the stack was run (difference of the pooled R^2^ on the same folds, "
             "chemistry-cluster bootstrap interval in brackets); the best single model is chosen on the same folds, which favours the single model, and the last row is given to four decimals because it is "
             "below the third.", "",
             "| **Model** | **S** | ***σ*** | ***κ*** | **zT** |", "|--------------------|------------|------------|------------|------------|"]
    for k, nm in names.items():
        cells = []
        for t in T4:
            if f"a2_{k}_{t}" not in v:
                cells.append(_pend("NA2: random forest for S"))
            elif k == "xgb":
                cells.append(f"{v[f'a2_{k}_{t}']} ± {v[f'a2_{k}_{t}_sd']}")
            else:
                cells.append(f"{v[f'a2_{k}_{t}']} ± {v[f'a2_{k}_{t}_sd']} ({v[f'a2_{k}_{t}_d']} [{v[f'a2_{k}_{t}_lo']}, {v[f'a2_{k}_{t}_hi']}])")
        lines.append(f"| {nm} | " + " | ".join(cells) + " |")
    lines.append("| Mean of the three (no fitted weights) | " + " | ".join(f"{v[f'a2_m3_{t}']} ± {v[f'a2_m3_{t}_sd']} ({v[f'a2_m3_{t}_d']} [{v[f'a2_m3_{t}_lo']}, {v[f'a2_m3_{t}_hi']}])" for t in T4) + " |")
    lines.append("| Stacking (XGBoost + LightGBM + random forest, ridge meta-learner, nested) | " + " | ".join(f"{v[f'a2_stk_{t}']} ± {v[f'a2_stk_{t}_sd']}" for t in T4) + " |")
    lines.append("| Stack minus XGBoost | " + " | ".join(f"{v[f'a2_stk_{t}_d']} [{v[f'a2_stk_{t}_lo']}, {v[f'a2_stk_{t}_hi']}]" for t in T4) + " |")
    lines.append("| Stack minus the best single model (S: random forest; the other targets: XGBoost) | " + " | ".join(
        f"{v[f'a2_stkbest_{t}_d']} [{v[f'a2_stkbest_{t}_lo']}, {v[f'a2_stkbest_{t}_hi']}]" for t in T4) + " |")
    lines.append("| Stack minus the mean of the three | " + " | ".join(f"{v[f'a2_stkmean_{t}_d']} [{v[f'a2_stkmean_{t}_lo']}, {v[f'a2_stkmean_{t}_hi']}]" for t in T4) + " |")
    return "\n".join(lines)


def table7b(v):
    """Explicit feature selection (NA13): the frozen XGBoost on all 397 features and on the features the thesis's selection keeps, paired on the same folds; and the post hoc refit with temperature_bin added."""
    lines = [f"**Table 7b:** Chemistry-cluster CV R^2^ (mean ± SD over {v['n_repeats']} repeats of {v['n_folds']} grouped folds) of each target's frozen XGBoost on all {v['n_feat']} features and on the features kept by the selection of the "
             f"earlier work, applied inside every outer training fold (Pearson filter at |r| = {v['na13_pearson']}, a LassoCV over {v['na13_alphas']} penalties, then the top k by mutual information on a {v['na13_mi_rows']}-row subsample; "
             f"k = {v['na13_S_k']}, {v['na13_sigma_k']}, {v['na13_kappa_k']}, {v['na13_zT_k']} for S, *σ*, *κ*, zT). Both fits of a fold use the same rows, machine and device. The paired difference is the comparison fixed before the run "
             "(difference of the pooled R^2^, chemistry-cluster bootstrap interval in brackets). The last three rows are post hoc and exploratory: the same selected features plus temperature_bin, added after the result above was seen; "
             "no interval or test is attached to them.", "",
             "| **Quantity** | **S** | ***σ*** | ***κ*** | **zT** |", "|--------------------|------------|------------|------------|------------|"]
    row = lambda name, f: lines.append(f"| {name} | " + " | ".join(f(t) for t in T4) + " |")  # noqa: E731
    row(f"All {v['n_feat']} features", lambda t: f"{v[f'na13_{t}_all']} ± {v[f'na13_{t}_all_sd']}")
    row("Selected features (k imposed)", lambda t: f"{v[f'na13_{t}_sel']} ± {v[f'na13_{t}_sel_sd']}")
    row(f"Selected minus all {v['n_feat']} (paired, fixed before the run)", lambda t: f"{v[f'na13_{t}_d']} [{v[f'na13_{t}_lo']}, {v[f'na13_{t}_hi']}]")
    row(f"Features kept by the Lasso, mean (range over {v['na13_nfolds']} folds)", lambda t: f"{v[f'na13_{t}_nl']} ({v[f'na13_{t}_nl_min']}–{v[f'na13_{t}_nl_max']})")
    row("k (earlier work)", lambda t: f"{v[f'na13_{t}_k']}")
    row("Post hoc: selected + temperature_bin", lambda t: f"{v[f'na13_{t}_temp']} ± {v[f'na13_{t}_temp_sd']}")
    row(f"Post hoc: (selected + temperature_bin) minus all {v['n_feat']}", lambda t: f"{v[f'na13_{t}_temp_d']}")
    row(f"Post hoc: share of the selected-to-all-{v['n_feat']} gap recovered (%)", lambda t: f"{v[f'na13_{t}_temp_share']}")
    row(f"Post hoc: folds of {v['na13_nfolds']} with selected + temperature_bin below all {v['n_feat']}", lambda t: f"{v[f'na13_{t}_temp_below']}")
    return "\n".join(lines)


LIT_ROWS = [  # (study, model, data source, rows, reported metric, validation protocol); values as the studies report them, N/R = not reported
    ("Parse et al. [@parse2024predicting]", "XGBoost", "Starrydata2", "18.1K", "R^2^ (zT) 0.815", "5-fold CV"),
    ("Jia et al. [@jia2024dealing]", "GBDT", "Starrydata2", "92K", "R^2^ (zT) 0.89--0.90", "Composition-level CV"),
    ("Ma & Poon [@ma2025reexamining]", "LightGBM", "Compiled TE", "<!--c:ma2025_rows-->14.1<!--/c-->K",
     "R^2^ (zT) <!--c:ma2025_zT_r2-->0.86<!--/c-->; R^2^ (\\|S\\|) <!--c:ma2025_S_r2-->0.8<!--/c-->", "Not stated"),
    ("Sun et al. [@sun2025rationally]", "DNN", "Starrydata2", "N/R", "R^2^ (zT) 0.90 (test)", "Train/test split"),
    ("Barua et al. [@barua2025thermoelectric]", "XGBoost", "Starrydata2", "∼160K", "R^2^ (zT) 0.67--0.80", "Three external test sets"),
    ("Wang et al. [@wang2025highperformance]", "Stacking", "Mixed", "5.2K", "R^2^ (zT) 0.97", "10-fold CV"),
    ("Elavunkel & Padhan [@elavunkel2025unlocking]", "Stacking", "Half-Heusler", "small", "R^2^ (S) 0.99; R^2^ (zT) 0.92", "Random split"),
]


def table1(v):
    """Recent ML studies: the metric and the validation protocol each reports (no value of this work, no ranking)."""
    lines = ["**Table 1:** Recent ML studies of thermoelectric property prediction: the metric and the validation protocol each reports, as the authors report them. "
             "The studies differ in dataset, target, preprocessing and protocol, so the values are not comparable with one another or with the R^2^ of this work. "
             "N/R = not reported.", "",
             "| **Study** | **Model** | **Source** | **Rows** | **Reported metric** | **Validation protocol** |",
             "|--------------|----------|-----------|-----------|-------------------|------------|"]
    lines += [f"| {a} | {b} | {c} | {d} | {e} | {f} |" for (a, b, c, d, e, f) in LIT_ROWS]
    return "\n".join(lines)


BLOCKS = {"TABLE 1": table1, "TABLE 2": table2, "TABLE 4": table4, "TABLE 5": table5, "TABLE 6": table6, "TABLE 7": table7, "TABLE 7b": table7b, "TABLE 8": table8, "TABLE 8b": table8b, "TABLE 9": table9, "TABLE 10": table10, "TABLE 11": table11, "TABLE 11b": table11b}


def render(text):
    """Return `text` with every inline value and block recomputed. Raises KeyError on an unknown key or missing block."""
    v = tv.values()
    v.update(extra_values(v))

    def sub(m):
        if m.group(1) not in v:
            raise KeyError(f"unknown inline value key {m.group(1)!r}")
        return f"<!--v:{m.group(1)}-->{v[m.group(1)]}<!--/v-->"

    text = INLINE.sub(sub, text)
    reg = an.load_registry()

    def sub_design(m):
        row = reg.get(m.group(1))
        if row is None or row["class"] != "DESIGN":
            raise KeyError(f"design marker {m.group(1)!r} has no DESIGN row in docs/number_registry.csv")
        return f"<!--d:{m.group(1)}-->{tv.config_value(row['source_or_config'])}<!--/d-->"

    text = DESIGN_MARK.sub(sub_design, text)
    for name, fn in BLOCKS.items():
        pat = re.compile(rf"(<!-- BEGIN {re.escape(name)} -->\n)(?:.*?\n)?(<!-- END {re.escape(name)} -->)", re.S)
        if not pat.search(text):
            raise KeyError(f"block {name!r} not found in paper.md")
        body = fn(v) + "\n"
        text = pat.sub(lambda m, body=body: m.group(1) + body + m.group(2), text)
    return text


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--list-pending", action="store_true")
    ap.add_argument("--paper", default=str(PAPER))
    args = ap.parse_args()
    path = Path(args.paper)
    old = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    if args.list_pending:
        for i, line in enumerate(old.split("\n"), 1):
            for m in PENDING.finditer(line):
                print(f"line {i}: {m.group(0)}")
        return 0
    new = render(old)
    if args.check:
        if new != old:
            print("paper.md is out of date: run python thesis_paper/scripts/make_thesis_values.py")
            return 1
        print(f"paper.md values and blocks are current ({len(INLINE.findall(new))} inline values, {len(PENDING.findall(new))} pending)")
        return 0
    path.write_bytes(new.encode("utf-8"))
    print(f"wrote {path} ({len(INLINE.findall(new))} inline values, {len(BLOCKS)} blocks, {len(PENDING.findall(new))} pending)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
