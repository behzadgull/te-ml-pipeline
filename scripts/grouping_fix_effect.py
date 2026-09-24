"""
Effect of the 2026-09-16 grouping fixes on the chemistry-cluster rung,
decomposed into its two components. Reproduces every number quoted in
paper.md section 4.4 ("lowered the chemistry-cluster rung by 0.037 to
0.058 ... from a mean of 0.131 to 0.181") from tracked inputs only.

Three chemistry-cluster conditions per target, all on File A's rows:
  pre-fix     old chemistry_cluster_id (unsnapped) + old fold code.
              Recomputed here from the tracked per-fold predictions in
              checkpoints/saved_predictions/checkpoints/<t>_chemistry/
              (.npz are Git LFS objects: run `git lfs pull` first).
  control     old chemistry_cluster_id + corrected fold code (S5).
              Read from reports/regen_postfix/20260916T161906/
              regen_metrics.json (the run's own .npz are archived
              off-machine, see results/regen_postfix/20260916T161906/
              README.md). Tests the fold-code correction on the original
              cluster identifiers.
  post-fix    SNAP(0.05) chemistry_cluster_id + corrected fold code.
              Read from reports/regen_snapfix/20260917T150000/
              ladder_metrics.json. The final, published rung.

Decomposition (additive by construction):
  total drop = pre - post = (pre - control) + (control - post)
             =  fold-code effect  +  remaining change (grouping key + interaction).
The split is arithmetic, not causal: the reverse control (original fold code
with the snapped identifier) was not run, so the remaining change also
contains any interaction between the two corrections.

Gap = random 80/20 pooled R^2 (reports/ungrouped_snapfix/20260922T093243/
metrics.json, same File A rows) minus the chemistry rung.

Every printed value the paper or CLAUDE.md quotes is asserted at its
published precision; the script raises if any input drifts. Writes
reports/grouping_fix_effect/<timestamp>/{report.md,metrics.json}.
Run from the repo root.
"""

import datetime
import json
import os

import numpy as np
from sklearn.metrics import r2_score

TARGETS = ["S", "sigma", "kappa", "zT"]
N_REPEATS = 5
N_FOLDS = 5

PREFIX_DIR = "checkpoints/saved_predictions/checkpoints/{t}_chemistry"
CONTROL_JSON = "reports/regen_postfix/20260916T161906/regen_metrics.json"
POSTFIX_JSON = "reports/regen_snapfix/20260917T150000/ladder_metrics.json"
RANDOM_JSON = "reports/ungrouped_snapfix/20260922T093243/metrics.json"

# Published values this script must reproduce (4 decimals). Effects and
# drops are differences of UNROUNDED rung values, rounded once at the end.
# Differencing the already-rounded 4-decimal rung values (as the CLAUDE.md
# tables invite) shifts two cells by 0.0001: the kappa total drop
# (0.036855 -> 0.0369, not 0.0368) and the sigma fold-code effect
# (-0.002160 -> -0.0022, not -0.0021); likewise the mean pre-fix gap is
# 0.131011 -> 0.1310, not 0.1311, and the mean post-fix gap is 0.181236 ->
# 0.1812 (matching reports/ungrouped_snapfix/.../table1.md), not 0.1813 (both
# quoted values are means of the four rounded gaps). The
# paper's 3-decimal figures ("0.037 to 0.058", "0.131") are unaffected.
EXPECT_PREFIX = {"S": 0.8076, "sigma": 0.7600, "kappa": 0.8460, "zT": 0.7968}
EXPECT_POSTFIX = {"S": 0.7528, "sigma": 0.7020, "kappa": 0.8092, "zT": 0.7456}
EXPECT_CONTROL = {"S": 0.8064, "sigma": 0.7579, "kappa": 0.8441, "zT": 0.7979}
EXPECT_DROP = {"S": 0.0548, "sigma": 0.0580, "kappa": 0.0369, "zT": 0.0512}
EXPECT_MEAN_GAP_PREFIX = 0.1310
EXPECT_MEAN_GAP_POSTFIX = 0.1812
EXPECT_SD_PREFIX = {"S": 0.0018, "sigma": 0.0008, "kappa": 0.0011, "zT": 0.0030}
EXPECT_SD_CONTROL = {"S": 0.0020, "sigma": 0.0012, "kappa": 0.0009, "zT": 0.0024}
EXPECT_SD_POSTFIX = {"S": 0.0050, "sigma": 0.0020, "kappa": 0.0021, "zT": 0.0045}
EXPECT_FOLD_CODE_DELTA = {"S": -0.0012, "sigma": -0.0022, "kappa": -0.0019, "zT": 0.0011}


def prefix_rung(target):
    """Per-repeat pooled R^2 (mean, SD) of the pre-fix chemistry rung, from saved predictions."""
    per_repeat = []
    for r in range(N_REPEATS):
        y_true, y_pred = [], []
        for f in range(N_FOLDS):
            path = os.path.join(PREFIX_DIR.format(t=target), f"repeat{r}_fold{f}_predictions.npz")
            with np.load(path) as z:
                y_true.append(z["y_true"])
                y_pred.append(z["y_pred"])
        per_repeat.append(r2_score(np.concatenate(y_true), np.concatenate(y_pred)))
    return float(np.mean(per_repeat)), float(np.std(per_repeat, ddof=1))


def check(name, got, want, ndigits=4):
    """Raise unless `got` rounds to `want` at ndigits decimals."""
    if round(got, ndigits) != round(want, ndigits):
        raise AssertionError(f"{name}: computed {got:.6f} rounds to {round(got, ndigits)}, expected {want}")


def main():
    with open(CONTROL_JSON, encoding="utf-8") as f:
        control_json = json.load(f)
    with open(POSTFIX_JSON, encoding="utf-8") as f:
        postfix_json = json.load(f)
    with open(RANDOM_JSON, encoding="utf-8") as f:
        random_json = json.load(f)

    rows = {}
    for t in TARGETS:
        pre_mean, pre_sd = prefix_rung(t)
        control = control_json["table_a_ladder"][t]["chemistry_new_mean"]
        post = postfix_json["runs"][f"{t}_chemistry_full"]["per_repeat_r2_mean"]
        rand = random_json["per_run"][t]["random"]["pooled_r2"]
        rows[t] = {
            "prefix": pre_mean,
            "prefix_sd": pre_sd,
            "control_sd": control_json["table_b_ablation"][t]["full_new_sd"],
            "postfix_sd": postfix_json["runs"][f"{t}_chemistry_full"]["per_repeat_r2_std"],
            "control": control,
            "postfix": post,
            "random": rand,
            "total_drop": pre_mean - post,
            "fold_code_effect": control - pre_mean,
            "remaining_change": post - control,
            "gap_prefix": rand - pre_mean,
            "gap_postfix": rand - post,
        }
        check(f"{t} pre-fix rung", pre_mean, EXPECT_PREFIX[t])
        check(f"{t} control rung", control, EXPECT_CONTROL[t])
        check(f"{t} post-fix rung", post, EXPECT_POSTFIX[t])
        check(f"{t} total drop", rows[t]["total_drop"], EXPECT_DROP[t])
        check(f"{t} fold-code effect", rows[t]["fold_code_effect"], EXPECT_FOLD_CODE_DELTA[t])
        assert abs(rows[t]["total_drop"] + rows[t]["fold_code_effect"] + rows[t]["remaining_change"]) < 1e-12

    mean_gap_pre = float(np.mean([rows[t]["gap_prefix"] for t in TARGETS]))
    mean_gap_post = float(np.mean([rows[t]["gap_postfix"] for t in TARGETS]))
    check("mean gap pre-fix", mean_gap_pre, EXPECT_MEAN_GAP_PREFIX)
    check("mean gap post-fix", mean_gap_post, EXPECT_MEAN_GAP_POSTFIX)
    for t in TARGETS:
        check(f"{t} pre-fix SD", rows[t]["prefix_sd"], EXPECT_SD_PREFIX[t])
        check(f"{t} control SD", rows[t]["control_sd"], EXPECT_SD_CONTROL[t])
        check(f"{t} post-fix SD", rows[t]["postfix_sd"], EXPECT_SD_POSTFIX[t])
    drops = [rows[t]["total_drop"] for t in TARGETS]
    check("smallest drop", min(drops), 0.037, 3)
    check("largest drop", max(drops), 0.058, 3)

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = os.path.join("reports", "grouping_fix_effect", ts)
    os.makedirs(out_dir, exist_ok=True)

    lines = [
        "# Grouping-fix effect on the chemistry-cluster rung",
        "",
        f"Generated {ts} UTC by scripts/grouping_fix_effect.py. All assertions passed.",
        "",
        "The split into fold-code effect and remaining change is arithmetic, not causal: the reverse control",
        "(original fold code with the snapped identifier) was not run.",
        "",
        "| Target | pre-fix | control (fold code only) | post-fix | total drop | fold-code effect | remaining change (grouping key + interaction) |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in TARGETS:
        r = rows[t]
        lines.append(
            f"| {t} | {r['prefix']:.4f} | {r['control']:.4f} | {r['postfix']:.4f} | "
            f"{r['total_drop']:.4f} | {r['fold_code_effect']:+.4f} | {r['remaining_change']:+.4f} |"
        )
    lines += [
        "",
        f"Mean gap (random 80/20 minus chemistry rung): pre-fix {mean_gap_pre:.4f}, post-fix {mean_gap_post:.4f}.",
        f"Total drop range: {min(drops):.4f} to {max(drops):.4f}.",
        f"Fold-code effect range: {min(rows[t]['fold_code_effect'] for t in TARGETS):+.4f} to "
        f"{max(rows[t]['fold_code_effect'] for t in TARGETS):+.4f}.",
        f"Remaining-change range (grouping key + interaction): {min(rows[t]['remaining_change'] for t in TARGETS):+.4f} to "
        f"{max(rows[t]['remaining_change'] for t in TARGETS):+.4f}.",
        "",
        "",
        "Across-repeat SD of the per-repeat pooled R^2:",
        "",
        "| Target | pre-fix | control | post-fix | control minus pre-fix |",
        "|---|---|---|---|---|",
    ]
    for t in TARGETS:
        r = rows[t]
        lines.append(
            f"| {t} | {r['prefix_sd']:.4f} | {r['control_sd']:.4f} | {r['postfix_sd']:.4f} | "
            f"{r['control_sd'] - r['prefix_sd']:+.4f} |"
        )
    lines.append("")
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(
            {"generated_at_utc": ts, "per_target": rows,
             "mean_gap_prefix": mean_gap_pre, "mean_gap_postfix": mean_gap_post,
             "inputs": {"prefix_dir": PREFIX_DIR, "control": CONTROL_JSON,
                        "postfix": POSTFIX_JSON, "random": RANDOM_JSON}},
            f, indent=2)
    print("\n".join(lines))
    print(f"Wrote {out_dir}/report.md and metrics.json")


if __name__ == "__main__":
    main()
