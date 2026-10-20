"""
Screening view by predicted S and kappa (ADDED AFTER NA12, 2026-10-20; not pre-registered). Reads only the committed ranked lists of the screening
(results/na11_ranked/20261005T053426/): nothing is refitted and no new prediction is made. The pre-registered shortlist and its rule are untouched.

For the 30 compounds of the main list (values at the temperature of the predicted maximum of zT) it writes, to results/screening_s_kappa/<UTC>/:
  table11c.csv   each compound with its rank by predicted zT, by |S| (large is first), by kappa (low is first) and by the rank sum of the last two
  summary.json   rank correlations (Spearman), top-ten overlaps, seen-cluster counts, the temperature of the maxima, duplicated predictions
  fig_s_kappa.png  |S| against kappa, seen and unseen clusters marked, zT rank written beside each point
  run_config.json  inputs with SHA256, code commit, tree state

Usage (repository root):  python thesis_paper/scripts/screening_s_kappa_view.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_record as rr  # noqa: E402

RANKED = "thesis_paper/results/na11_ranked/20261005T053426"
TOP_N = 10  # size of the top lists compared (a design choice of this view, fixed before it was run)


def main():
    """Compute the view and write the run folder."""
    main30 = pd.read_csv(REPO / RANKED / "ranked_main_30.csv")
    all409 = pd.read_csv(REPO / RANKED / "ranked_all_409.csv")
    assert len(main30) == 30 and main30["in_main"].all() and len(all409) == 409
    d = main30.copy()
    d["abs_S"] = d["S_at_max"].abs()
    d["rank_zT"] = d["zT_max"].rank(ascending=False, method="min").astype(int)
    d["rank_absS"] = d["abs_S"].rank(ascending=False, method="min").astype(int)
    d["rank_kappa"] = d["kappa_at_max"].rank(ascending=True, method="min").astype(int)
    d["rank_sum"] = d["rank_absS"] + d["rank_kappa"]
    d = d.sort_values(["rank_sum", "rank_zT"]).reset_index(drop=True)
    d["rank_S_kappa"] = np.arange(1, len(d) + 1)

    def rho(a, b):
        return float(spearmanr(a, b).correlation)

    top = lambda col, n=TOP_N: set(d.nsmallest(n, col, keep="all").index)  # noqa: E731
    zt_top, s_top, k_top, sk_top = top("rank_zT"), top("rank_absS"), top("rank_kappa"), top("rank_S_kappa")
    seen = d["cluster_seen_any"].astype(bool)
    dup = d.duplicated(subset=["zT_max", "S_at_max", "kappa_at_max"], keep=False)
    summary = {
        "note": "added after NA12, not pre-registered; reads the committed ranked lists only",
        "n_main": int(len(d)), "top_n": TOP_N,
        "spearman_zT_absS": rho(d["zT_max"], d["abs_S"]),
        "spearman_zT_lowkappa": rho(d["zT_max"], -d["kappa_at_max"]),
        "spearman_absS_lowkappa": rho(d["abs_S"], -d["kappa_at_max"]),
        "top_zT_and_top_absS": len(zt_top & s_top), "top_zT_and_top_lowkappa": len(zt_top & k_top), "top_absS_and_top_lowkappa": len(s_top & k_top),
        "seen_in_top_zT": int(seen[list(zt_top)].sum()), "seen_in_top_S_kappa": int(seen[list(sk_top)].sum()),
        "n_seen_main": int(seen.sum()), "n_unseen_main": int((~seen).sum()),
        "main_temperatures_K": sorted(int(x) for x in d["T_at_zT_max"].unique()),
        "n_main_at_800K": int((d["T_at_zT_max"] == 800).sum()), "n_all_at_800K": int((all409["T_at_zT_max"] == 800).sum()), "n_all": int(len(all409)),
        "n_sign_overridden_main": int(d["sign_overridden_at_max"].astype(bool).sum()),
        "n_rows_in_duplicate_prediction_groups": int(dup.sum()), "n_duplicate_prediction_groups": int(d[dup].groupby(["zT_max", "S_at_max", "kappa_at_max"]).ngroups),
        "duplicate_formulas": sorted(d[dup]["formula"].unique().tolist()),
    }
    out = rr.new_run_dir("screening_s_kappa")
    cols = ["formula", "material_id", "cluster_seen_any", "zT_max", "S_at_max", "kappa_at_max", "rank_zT", "rank_absS", "rank_kappa", "rank_sum", "rank_S_kappa", "sign_overridden_at_max", "shortlist"]
    d[cols].to_csv(out / "table11c.csv", index=False)
    rr.write_json(out / "summary.json", summary)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for flag, lab, mk in ((True, "chemistry cluster in training data", "o"), (False, "cluster not in training data", "s")):
        sel = d[d["cluster_seen_any"].astype(bool) == flag]
        ax.scatter(sel["kappa_at_max"], sel["abs_S"], marker=mk, s=34, label=lab, edgecolor="k", linewidth=0.5, alpha=0.9)
    for _, r in d.iterrows():
        ax.annotate(str(r["rank_zT"]), (r["kappa_at_max"], r["abs_S"]), textcoords="offset points", xytext=(3, 3), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("predicted κ at the zT maximum (W m$^{-1}$ K$^{-1}$)")
    ax.set_ylabel("predicted |S| at the zT maximum (µV K$^{-1}$)")
    ax.set_title("Main list: |S| against κ (numbers: rank by predicted zT)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_s_kappa.png", dpi=300)
    prov = rr.provenance({"ranked_main_30": f"{RANKED}/ranked_main_30.csv", "ranked_all_409": f"{RANKED}/ranked_all_409.csv"}, __file__)
    rr.write_json(out / "run_config.json", prov)
    print("wrote", out, "tree_clean", prov["tree_clean"])
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
