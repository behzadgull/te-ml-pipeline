"""
Screening view by predicted S and kappa (ADDED AFTER NA12, 2026-10-20; not pre-registered). Reads only committed files: the ranked lists of the screening
(results/na11_ranked/20261005T053426/), the per-temperature predictions behind them and the thermal limits of Table 11b (docs/shortlist_thermal_limits.csv). Nothing is refitted and no new
prediction is made. The pre-registered shortlist and its rule are untouched.

Temperature (corrected 2026-10-20 after the first run, which had used 800 K for all 30): for each of the 30 compounds of the main list, S and kappa are taken at the grid temperature of its
maximum predicted zT within its physical limit, exactly as Table 11b does (na11_shortlist_limits.py): only grid temperatures below an unconditional limit of docs/shortlist_thermal_limits.csv are
used (CsSnI3: melting, 724.15 K, so 700 K; applied to every Materials Project entry of that formula, as Table 11 marks them); a conditional limit (LaCoO3 in a reducing atmosphere) is flagged, not
applied; compounds not in that file have no limit applied (limits were sought only for the shortlist). The zT used for the correlations and the top-10 lists is the predicted zT at that same temperature.
The zT rank SHOWN in Table 11c is the rank of Table 11 (within the cluster group, from the pre-registered primary maximum); the script asserts that it equals the order of the ranked list.

It writes, to results/screening_s_kappa/<UTC>/:
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
PRED = "thesis_paper/results/final_b/20261004T201433/predictions_mp.csv"
LIMITS = "thesis_paper/docs/shortlist_thermal_limits.csv"
GRID = (300, 400, 500, 600, 700, 800)
TOP_N = 10  # size of the top lists compared (a design choice of this view, fixed before it was run)


def main():
    """Compute the view and write the run folder."""
    main30 = pd.read_csv(REPO / RANKED / "ranked_main_30.csv")
    all409 = pd.read_csv(REPO / RANKED / "ranked_all_409.csv")
    assert len(main30) == 30 and main30["in_main"].all() and len(all409) == 409
    pred = pd.read_csv(REPO / PRED)
    lim = pd.read_csv(REPO / LIMITS)
    d = main30.copy()
    # Table 11's rank: within the cluster group, from the pre-registered primary maximum (the order of the ranked list); asserted, not recomputed
    for _, g in d.groupby("cluster_seen_any"):
        assert list(g.sort_values("zT_max", ascending=False, kind="stable")["rank_in_group_main_30"]) == sorted(g["rank_in_group_main_30"]), "Table 11 order differs from the primary maximum"
    rows = []
    for _, r in d.iterrows():
        g = pred[pred["material_id"] == r["material_id"]].sort_values("T_K")
        assert list(g["T_K"]) == list(GRID)
        l = lim[lim["formula"] == r["formula"]]
        ul = float(l["unconditional_limit_K"].iloc[0]) if len(l) and pd.notna(l["unconditional_limit_K"].iloc[0]) else None
        gk = g[g["T_K"] < ul] if ul is not None else g
        i = gk["zT_direct"].to_numpy().argmax()
        b = gk.iloc[i]
        rows.append({"T_view": int(b["T_K"]), "zT_view": float(b["zT_direct"]), "S_view": float(b["S"]), "kappa_view": float(b["kappa"]), "limited": bool(ul is not None and ul <= 800), "sign_overridden_view": bool(b["sign_overridden"])})
        if int(b["T_K"]) == int(r["T_at_zT_max"]):
            assert abs(b["zT_direct"] - r["zT_max"]) < 1e-9 and abs(b["S"] - r["S_at_max"]) < 1e-9 and abs(b["kappa"] - r["kappa_at_max"]) < 1e-9, "view disagrees with the ranked list at the same temperature"
    d = pd.concat([d.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    d["abs_S"] = d["S_view"].abs()
    d["rank_zT"] = d["zT_view"].rank(ascending=False, method="min").astype(int)
    d["rank_absS"] = d["abs_S"].rank(ascending=False, method="min").astype(int)
    d["rank_kappa"] = d["kappa_view"].rank(ascending=True, method="min").astype(int)
    d["rank_sum"] = d["rank_absS"] + d["rank_kappa"]
    d = d.sort_values(["rank_sum", "rank_zT"]).reset_index(drop=True)
    d["rank_S_kappa"] = np.arange(1, len(d) + 1)

    def rho(a, b):
        return float(spearmanr(a, b).correlation)

    top = lambda col, n=TOP_N: set(d.nsmallest(n, col, keep="all").index)  # noqa: E731
    zt_top, s_top, k_top, sk_top = top("rank_zT"), top("rank_absS"), top("rank_kappa"), top("rank_S_kappa")
    seen = d["cluster_seen_any"].astype(bool)
    dup = d.duplicated(subset=["zT_view", "S_view", "kappa_view"], keep=False)
    summary = {
        "note": "added after NA12, not pre-registered; reads the committed ranked lists only",
        "n_main": int(len(d)), "top_n": TOP_N,
        "spearman_zT_absS": rho(d["zT_view"], d["abs_S"]),
        "spearman_zT_lowkappa": rho(d["zT_view"], -d["kappa_view"]),
        "spearman_absS_lowkappa": rho(d["abs_S"], -d["kappa_view"]),
        "top_zT_and_top_absS": len(zt_top & s_top), "top_zT_and_top_lowkappa": len(zt_top & k_top), "top_absS_and_top_lowkappa": len(s_top & k_top),
        "seen_in_top_zT": int(seen[list(zt_top)].sum()), "seen_in_top_S_kappa": int(seen[list(sk_top)].sum()),
        "n_seen_main": int(seen.sum()), "n_unseen_main": int((~seen).sum()),
        "main_temperatures_K": sorted(int(x) for x in d["T_view"].unique()),
        "n_main_at_800K": int((d["T_view"] == 800).sum()), "n_main_limited": int(d["limited"].sum()), "limited_formulas": sorted(d[d["limited"]]["formula"].unique().tolist()),
        "n_main_changed_by_limit": int((d["T_view"] != d["T_at_zT_max"]).sum()), "n_all_at_800K": int((all409["T_at_zT_max"] == 800).sum()), "n_all": int(len(all409)),
        "n_sign_overridden_main": int(d["sign_overridden_view"].sum()),
        "n_rows_in_duplicate_prediction_groups": int(dup.sum()), "n_duplicate_prediction_groups": int(d[dup].groupby(["zT_view", "S_view", "kappa_view"]).ngroups),
        "duplicate_formulas": sorted(d[dup]["formula"].unique().tolist()),
    }
    prov = rr.provenance({"ranked_main_30": f"{RANKED}/ranked_main_30.csv", "ranked_all_409": f"{RANKED}/ranked_all_409.csv", "predictions_mp": PRED, "limits": LIMITS}, __file__)
    out = rr.new_run_dir("screening_s_kappa")  # the provenance is taken first: the new folder would otherwise make the tree dirty
    cols = ["formula", "material_id", "cluster_seen_any", "rank_in_group_main_30", "T_view", "zT_view", "S_view", "kappa_view", "limited", "rank_zT", "rank_absS", "rank_kappa", "rank_sum", "rank_S_kappa", "sign_overridden_view", "shortlist"]
    d[cols].to_csv(out / "table11c.csv", index=False)
    rr.write_json(out / "summary.json", summary)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for flag, lab, mk in ((True, "chemistry cluster in training data", "o"), (False, "cluster not in training data", "s")):
        sel = d[d["cluster_seen_any"].astype(bool) == flag]
        ax.scatter(sel["kappa_view"], sel["abs_S"], marker=mk, s=34, label=lab, edgecolor="k", linewidth=0.5, alpha=0.9)
    for _, r in d.iterrows():
        ax.annotate(str(r["rank_in_group_main_30"]), (r["kappa_view"], r["abs_S"]), textcoords="offset points", xytext=(3, 3), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("predicted κ (W m$^{-1}$ K$^{-1}$)")
    ax.set_ylabel("predicted |S| (µV K$^{-1}$)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_s_kappa.png", dpi=300)
    rr.write_json(out / "run_config.json", prov)
    print("wrote", out, "tree_clean", prov["tree_clean"])
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
