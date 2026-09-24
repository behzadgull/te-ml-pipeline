"""
How much of ESTM's external loss is extrapolation beyond the training data's property range?

For each property and each ESTM stratum (DOI-disjoint, cluster-disjoint) this reports two measures,
computed from the saved per-row predictions of the snapfix external validation
(results/external_snapfix/20260917T160553/estm_predictions_pass{a,b}.npz):

  recovered_share: (R2 on in-support rows - R2 on all rows) / (internal chemistry-cluster R2 - R2 on all
      rows). The fraction of the drop from internal grouped performance that disappears when the scoring
      set is restricted to rows inside the training data's per-property range. In-support rows are a
      different subset from the full set, so this is a descriptive ratio, not an exact decomposition of
      the loss. This is the measure quoted in the paper (Section 3.5 and the Conclusion): 15 to 40% on the
      cluster-disjoint stratum.
  ood_sse_share: the fraction of the external squared error, in the property's scoring space, that comes
      from out-of-support rows.

"In-support" means inside the per-property min and max of the snapfix training set (300-800 K rows) in S,
sigma and kappa simultaneously (bounds reconstructed 2026-09-19, CLAUDE.md, External Validation). The
script asserts the row counts that reconstruction is validated against (2,709 and 1,196 in-support rows)
and asserts the cluster-disjoint recovered shares quoted in the paper (S 40, sigma 31, kappa 20, zT 15%)
against its own output before writing anything.

Output: reports/insupport_share/<UTC timestamp>/{insupport_share.json, insupport_share.md, run_config.json}.
run_config.json records the SHA256 of every input, the code commit, the SHA256 of this script and the paths
that were uncommitted when it ran (a commit hash alone does not identify code with uncommitted edits).

Usage (repository root):  python scripts/insupport_share.py
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

EXTERNAL_DIR = Path("results/external_snapfix/20260917T160553")
LADDER_METRICS_PATH = Path("reports/regen_snapfix/20260917T150000/ladder_metrics.json")
REPORT_ROOT = Path("reports/insupport_share")

# Per-property min and max of the snapfix training set, 300-800 K rows (same values as
# scripts/make_figures.py's ESTM_TRAINING_BOUNDS).
TRAINING_BOUNDS = {
    "S": (-461.0258, 562.5),
    "sigma": (958.7831, 1656678.2772),
    "kappa": (0.2830364, 13.7753),
}
EXPECTED_IN_SUPPORT = {"a": 2709, "b": 1196}
STRATA = {"a": "DOI-disjoint", "b": "cluster-disjoint"}
# Shares (percent, rounded) quoted in paper.md for the cluster-disjoint stratum.
PAPER_CLUSTER_DISJOINT_PERCENT = {"S": 40, "sigma": 31, "kappa": 20, "zT": 15}

# property -> (true column, prediction column) in the saved predictions, in the property's scoring space
COLUMNS = {
    "S": ("S_true", "S_pred"),
    "sigma": ("sigma_log_true", "sigma_log_pred"),
    "kappa": ("kappa_log_true", "kappa_log_pred"),
    "zT": ("zT_true", "zT_direct_pred"),
}
LADDER_KEY = {"S": "S", "sigma": "sigma", "kappa": "kappa", "zT": "zT"}


def sha256(path):
    """SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    """Run a git command and return its stdout, stripped."""
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def r2(y, yhat):
    """Coefficient of determination of yhat against y."""
    return 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def in_support_mask(z):
    """Rows inside the training per-property range in S, sigma and kappa simultaneously."""
    mask = np.ones(len(z["S_true"]), dtype=bool)
    for prop, (lo, hi) in TRAINING_BOUNDS.items():
        col = z[{"S": "S_true", "sigma": "sigma_true", "kappa": "kappa_true"}[prop]]
        mask &= (col >= lo) & (col <= hi)
    return mask


def compute():
    """Return {stratum letter: {...}} with per-property shares, from the saved predictions."""
    with open(LADDER_METRICS_PATH, encoding="utf-8") as f:
        ladder = json.load(f)
    internal = {p: ladder["runs"][f"{LADDER_KEY[p]}_chemistry_full"]["per_repeat_r2_mean"] for p in COLUMNS}
    out = {}
    for letter, name in STRATA.items():
        z = np.load(EXTERNAL_DIR / f"estm_predictions_pass{letter}.npz", allow_pickle=True)
        ins = in_support_mask(z)
        assert int(ins.sum()) == EXPECTED_IN_SUPPORT[letter], (letter, int(ins.sum()), EXPECTED_IN_SUPPORT[letter])
        props = {}
        for prop, (tcol, pcol) in COLUMNS.items():
            y, yhat = np.asarray(z[tcol], float), np.asarray(z[pcol], float)
            e2 = (y - yhat) ** 2
            full, sup = r2(y, yhat), r2(y[ins], yhat[ins])
            props[prop] = dict(
                r2_full=full, r2_in_support=sup, r2_internal_chemistry=internal[prop],
                drop_from_internal=internal[prop] - full,
                recovered_share=(sup - full) / (internal[prop] - full),
                ood_sse_share=float(e2[~ins].sum() / e2.sum()),
            )
        out[letter] = dict(stratum=name, n=int(len(ins)), n_in_support=int(ins.sum()),
                           ood_row_fraction=float((~ins).mean()), properties=props)
    return out


def main():
    results = compute()
    # assert the shares quoted in the paper against this output
    for prop, quoted in PAPER_CLUSTER_DISJOINT_PERCENT.items():
        got = 100 * results["b"]["properties"][prop]["recovered_share"]
        assert round(got) == quoted, (prop, got, quoted)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPORT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=False)
    inputs = [EXTERNAL_DIR / f"estm_predictions_pass{k}.npz" for k in STRATA] + [LADDER_METRICS_PATH]
    dirty = [line for line in git("status", "--porcelain").splitlines() if line.strip()]
    config = dict(
        timestamp_utc=ts,
        code_commit=git("rev-parse", "HEAD"),
        tree_clean=not dirty,
        uncommitted_paths=[line[3:] for line in dirty],
        script=dict(path="scripts/insupport_share.py", sha256=sha256(__file__)),
        inputs={str(p).replace("\\", "/"): dict(sha256=sha256(p), bytes=p.stat().st_size) for p in inputs},
        training_bounds=TRAINING_BOUNDS,
        asserted_in_support_rows=EXPECTED_IN_SUPPORT,
        asserted_cluster_disjoint_percent=PAPER_CLUSTER_DISJOINT_PERCENT,
    )
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out_dir / "insupport_share.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = ["# In-support share of ESTM's external loss", "",
             "recovered share = (R2 in-support - R2 full) / (internal chemistry-cluster R2 - R2 full).",
             "ood SSE share = fraction of the external squared error from out-of-support rows.", ""]
    for letter, r in results.items():
        lines += [f"## {r['stratum']} (n = {r['n']}, in-support {r['n_in_support']}, "
                  f"out-of-support {100 * r['ood_row_fraction']:.1f}%)", "",
                  "| Property | R2 full | R2 in-support | R2 internal | recovered share | OOD share of SSE |",
                  "|---|---|---|---|---|---|"]
        for prop, v in r["properties"].items():
            lines.append(f"| {prop} | {v['r2_full']:.4f} | {v['r2_in_support']:.4f} | {v['r2_internal_chemistry']:.4f} | "
                         f"{100 * v['recovered_share']:.1f}% | {100 * v['ood_sse_share']:.1f}% |")
        lines.append("")
    lines += ["The DOI-disjoint stratum is shown for completeness; internal chemistry-cluster R2 is not its matched",
              "reference (it tests measurement transfer, not chemistry transfer), so its recovered shares are not",
              "comparable to the cluster-disjoint ones and are not quoted in the paper."]
    (out_dir / "insupport_share.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir}")
    print("\n".join(lines[4:]))


if __name__ == "__main__":
    sys.exit(main())
