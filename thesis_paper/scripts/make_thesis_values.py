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
    head = (f"**Table 8:** ESTM external validation (R^2^). The {v['estm_scope']} ESTM rows within 300--800 K are scored in two strata that are never pooled: "
            f"rows sharing no source DOI with the training data (a, n = {v['estm_a_n']}) and rows whose chemistry cluster is absent from training "
            f"(b, n = {v['estm_b_n']}). The in-support column restricts (b) to rows inside the training range of S, *σ*, *κ* and temperature "
            f"(n = {v['estm_b_insup_n']}). The drop is stratum (b) minus the internal chemistry-cluster value.")
    lines = [head, "", "| **Target** | **Internal (chemistry cluster)** | **ESTM (a) DOI-disjoint** | **ESTM (b) cluster-disjoint** | **(b), in-support** | **Drop (b)** |",
             "|---------|-------------|-------------|-------------|-----------|---------|"]
    names = {"S": "S", "sigma": "*σ*", "kappa": "*κ*", "zT": "zT (direct)"}
    for t in T4:
        lines.append(f"| {names[t]} | {v[f'chem_{t}']} | {v[f'estm_a_{t}']} | {v[f'estm_b_{t}']} | {v[f'estm_b_insup_{t}']} | {v[f'estm_b_drop_{t}']} |")
    return "\n".join(lines)


def table10(v):
    """Direct versus derived zT, under chemistry-cluster CV (Paper A run) and random row-level CV (NA7)."""
    lines = [f"**Table 10:** Direct versus component-wise zT prediction on the {v['dvd_rows']} rows with all four properties present "
             f"({v['dvd_clusters']} chemistry clusters); each model uses its own target's frozen hyperparameters. Chemistry-cluster CV is 5 repeats x 5 folds "
             f"of grouped folds; random CV is 5 repeats x 5 folds of shuffled row-level folds; both pool the out-of-fold predictions "
             f"(n = {v.get('na7_n', _pend('NA7'))}).", "",
             "| **Pathway** | **R^2^ (chemistry-cluster)** | **MAE (chemistry-cluster)** | **R^2^ (random)** | **MAE (random)** |",
             "|------------------------|----------|---------|----------|---------|",
             f"| Direct (features → zT) | {v['dvd_direct']} | {v.get('mae_dvd_direct', _pend('NA8'))} | {v.get('na7_direct', _pend('NA7'))} | {v.get('na7_mae_direct', _pend('NA7'))} |",
             f"| Derived (S^2^*σ*T/*κ*) | {v['dvd_derived']} | {v.get('mae_dvd_derived', _pend('NA8'))} | {v.get('na7_derived', _pend('NA7'))} | {v.get('na7_mae_derived', _pend('NA7'))} |"]
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


BLOCKS = {"TABLE 1": table1, "TABLE 2": table2, "TABLE 4": table4, "TABLE 5": table5, "TABLE 6": table6, "TABLE 8": table8, "TABLE 10": table10}


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
