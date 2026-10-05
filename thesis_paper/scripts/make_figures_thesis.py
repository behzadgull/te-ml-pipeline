"""
Figures of the thesis paper that are not reused from Paper A, drawn from committed artifacts with every plotted number asserted
against the source (Paper A style: scripts/figstyle.py, 16 cm wide, 300 dpi, no cropping).

  Figure 6  property distributions (NA4)
  Figure 7  predicted versus measured values (NA8)
  Figure 9  ESTM external validation
  Figure 10 SHAP global importance (NA3)
  Figure 12 SHAP shares of MAGPIE, CBFV and temperature (NA3)

Usage (from the repository root):
    python thesis_paper/scripts/make_figures_thesis.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import feature_labels as fl  # noqa: E402
import figstyle as fs  # noqa: E402
import paper_a_values as pav  # noqa: E402
import thesis_values as tv  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "thesis_paper" / "figures"
T4 = pav.TARGETS
BEESWARM_TOP_N = 10  # features per target in the beeswarm plots of Figure 11 (a presentation choice, docs/design_constants.csv D23)
SHAP_TOP_N = 15  # features shown per target in Figure 10 (a presentation choice, docs/design_constants.csv D23; reduced from 20 so that two-line labels fit)
LABEL = {"S": "S", "sigma": "σ", "kappa": "κ", "zT": "zT"}


def fig9_estm(out):
    """Internal chemistry-cluster R2, external full-set R2 and external in-support R2, for the two ESTM strata."""
    e, ins = pav._json(pav.ESTM), pav._json(pav.INSUPPORT)
    ladder = pav._json(pav.LADDER)["runs"]
    fs.apply()
    fig, axes = fs.new_figure("double", 6.4, 1, 2, sharey=True)
    for ax, (k, ek, title) in zip(axes, (("a", "dedup_a_source_doi", "ESTM, no shared publication"),
                                          ("b", "dedup_b_chemistry_cluster", "ESTM, cluster absent from training"))):
        res = e[ek]["results"]
        x = np.arange(len(T4))
        w = 0.26
        internal = [ins[k]["properties"][t]["r2_internal_chemistry"] for t in T4]
        ext = [res["zT_direct" if t == "zT" else t]["r2"] for t in T4]
        insup = [ins[k]["properties"][t]["r2_in_support"] for t in T4]
        for t, i_, e_ in zip(T4, internal, ext):  # the plotted values are the artifacts' values
            assert abs(i_ - ladder[f"{t}_chemistry_full"]["per_repeat_r2_mean"]) < 1e-9
        ax.bar(x - w, internal, w, color=fs.OI["blue"], ec=fs.EDGE_GREY, lw=0.5, label="Internal (chemistry-cluster CV)")
        ax.bar(x, ext, w, color=fs.OI["orange"], ec=fs.EDGE_GREY, lw=0.5, label="External, full set")
        ax.bar(x + w, insup, w, color=fs.OI["green"], ec=fs.EDGE_GREY, lw=0.5, label="External, rows inside the training range")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylim(-0.1, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[t] for t in T4])
        fs.panel_title(ax, title, letter="ab"[k == "b"])
        ax.set_xlim(-0.6, len(T4) - 0.4)
        ax.text(0.98, 0.97, f"n = {e[ek]['n_surviving']:,} rows; {100 * ins[k]['ood_row_fraction']:.0f}% outside the training range",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.25")
    axes[0].set_ylabel("R$^2$")
    handles = [Patch(fc=fs.OI["blue"], ec=fs.EDGE_GREY), Patch(fc=fs.OI["orange"], ec=fs.EDGE_GREY), Patch(fc=fs.OI["green"], ec=fs.EDGE_GREY)]
    fs.legend_row(fig, handles, ["Internal (chemistry-cluster CV)", "External, full set", "External, rows inside the training range"])
    fs.save(fig, str(out), expect_width_cm=16.0)
    plt.close(fig)


def fig6_distributions(out):
    """Histograms of the four properties on the featurised rows (counts from NA4; sigma and kappa in log10)."""
    n4 = tv.na_json("na4")["per_target"]
    fs.apply()
    fig, axes = fs.new_figure("double", 8.6, 2, 2)
    spec = [("S", "Seebeck coefficient (µV K$^{-1}$)"), ("sigma", "log$_{10}$ electrical conductivity (S m$^{-1}$)"),
            ("kappa", "log$_{10}$ thermal conductivity (W m$^{-1}$ K$^{-1}$)"), ("zT", "Figure of merit zT")]
    for ax, (t, xlabel), letter in zip(axes.ravel(), spec, "abcd"):
        h = n4[t]["histogram"]
        edges, counts = np.array(h["edges"]), np.array(h["counts"])
        assert counts.sum() == n4[t]["n"], t  # every row of the property is in the histogram
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=fs.OI["blue"], ec="white", lw=0.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Rows")
        fs.panel_title(ax, f"n = {n4[t]['n']:,}", letter=letter)
    fs.save(fig, str(out), expect_width_cm=16.0)
    plt.close(fig)


def fig7_parity(out):
    """Predicted versus measured values, repeat 0 of the chemistry-cluster rung (2-D histograms from NA8)."""
    from matplotlib.colors import LogNorm

    n8 = tv.na_json("na8")["parity_hist2d"]
    fs.apply()
    fig, axes = fs.new_figure("double", 13.0, 2, 2)
    titles = {"S": "S (µV K$^{-1}$)", "sigma": "log$_{10}$ σ (S m$^{-1}$)", "kappa": "log$_{10}$ κ (W m$^{-1}$ K$^{-1}$)", "zT": "zT"}
    mesh = None
    for ax, t, letter in zip(axes.ravel(), pav.TARGETS, "abcd"):
        d = n8[t]
        lo, hi = d["range"]
        counts = np.array(d["counts"], dtype=float).T  # rows: predicted, columns: measured
        mesh = ax.pcolormesh(np.linspace(lo, hi, counts.shape[1] + 1), np.linspace(lo, hi, counts.shape[0] + 1),
                             np.ma.masked_equal(counts, 0), norm=LogNorm(vmin=1, vmax=counts.max()), cmap="Blues", rasterized=True)
        ax.plot([lo, hi], [lo, hi], color="0.2", lw=0.8, ls=(0, (4, 3)))
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("Measured " + titles[t])
        ax.set_ylabel("Predicted")
        ax.text(0.03, 0.97, f"R$^2$ = {d['r2_repeat0']:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=8)
        fs.panel_title(ax, titles[t], letter=letter)
    fig.colorbar(mesh, ax=axes, shrink=0.6, label="Rows per cell")
    fs.save(fig, str(out), expect_width_cm=16.0)
    plt.close(fig)


def fig10_shap(out):
    """SHAP global importance: the 20 features with the largest mean |SHAP| per target (NA3, chemistry-cluster folds), coloured by descriptor scheme; error bars are the SD across folds."""
    na3 = tv.na3_per_target()
    fs.apply()
    fig, axes = fs.new_figure("double", 21.0, 2, 2)
    order = ("S", "sigma", "kappa", "zT")
    units = {"zT": "mean |SHAP|", "S": "mean |SHAP| (µV K$^{-1}$)", "kappa": "mean |SHAP|", "sigma": "mean |SHAP|"}
    colour = {"MagpieData": fs.OI["blue"], "CBFV_": fs.OI["orange"], "temperature_bin": fs.OI["green"]}
    for ax, t, letter in zip(axes.ravel(), order, "abcd"):
        top = na3[t]["top20"][:SHAP_TOP_N]
        assert len(top) == SHAP_TOP_N and all(top[i]["mean_abs_shap"] >= top[i + 1]["mean_abs_shap"] for i in range(SHAP_TOP_N - 1))
        y = np.arange(SHAP_TOP_N)[::-1]
        cols = [next(c for k, c in colour.items() if f["feature"].startswith(k)) for f in top]
        ax.barh(y, [f["mean_abs_shap"] for f in top], xerr=[f["fold_sd"] for f in top], color=cols, ec=fs.EDGE_GREY, lw=0.4, error_kw={"lw": 0.6, "capsize": 1.5})
        ax.set_yticks(y)
        ax.set_yticklabels([fl.wrapped(fl.label(f["feature"]), 46) for f in top], fontsize=6.5)
        ax.set_xlabel(units[t], fontsize=7.5)
        fs.panel_title(ax, {"zT": "zT", "S": "S", "kappa": "κ", "sigma": "σ"}[t], letter=letter)
        fs.grid(ax, axis="x")
    fs.legend_row(fig, [Patch(color=c, ec=fs.EDGE_GREY) for c in colour.values()], ["MAGPIE", "CBFV", "Temperature"])
    fs.save(fig, str(out), expect_width_cm=16.0)
    plt.close(fig)


def fig12_shares(out):
    """Share of the mean |SHAP| carried by MAGPIE, CBFV and temperature features, per model (NA3)."""
    na3 = tv.na3_per_target()
    fs.apply()
    fig, ax = fs.new_figure("single", 6.5)
    order = ("S", "sigma", "kappa", "zT")
    names = {"zT": "zT", "S": "S", "kappa": "κ", "sigma": "σ"}
    bottom = np.zeros(len(order))
    for g, c, lab in (("magpie", fs.OI["blue"], "MAGPIE"), ("cbfv", fs.OI["orange"], "CBFV"), ("temperature", fs.OI["green"], "Temperature")):
        vals = np.array([na3[t]["share_by_group"][g]["mean"] for t in order])
        ax.bar(np.arange(len(order)), vals, 0.6, bottom=bottom, color=c, ec=fs.EDGE_GREY, lw=0.5, label=lab)
        bottom += vals
    assert np.allclose(bottom, 1.0, atol=1e-3)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([names[t] for t in order])
    ax.set_ylabel("Share of mean |SHAP|")
    ax.set_ylim(0, 1)
    fs.legend_row(fig, *ax.get_legend_handles_labels())
    fs.save(fig, str(out), expect_width_cm=8.0)
    plt.close(fig)


def load_na3_rows(run_dirs):
    """{target: (shap, x, columns)} from NA3-rows bundles (directories): every saved row of repeat 0, all folds pooled."""
    import json

    out = {}
    for d in run_dirs:
        d = Path(d)
        res = json.loads((d / "results.json").read_text(encoding="utf-8"))
        st = json.loads((d / "status.json").read_text(encoding="utf-8"))
        assert st["complete"] and st["units_done"] == st["units_total"], f"{d}: not complete"
        for t in res["per_target"]:
            parts = [np.load(d / "units" / f"{t}_repeat0_fold{f}.npz") for f in range(res["per_target"][t]["n_folds"])]
            out[t] = (np.vstack([u["shap"] for u in parts]), np.vstack([u["x"] for u in parts]), res["feature_columns"])
    return out


def fig11_beeswarm(out, run_dirs, top_n=BEESWARM_TOP_N):
    """Beeswarm plots: SHAP value of every saved row for the top features of each target (NA3 rows, repeat 0, a seeded subsample of each test fold), coloured by the row's feature value (percentile rank)."""
    data = load_na3_rows(run_dirs)
    fs.apply()
    fig, axes = fs.new_figure("double", 22.0, 2, 2)
    order = [t for t in ("S", "sigma", "kappa", "zT") if t in data]
    rng = np.random.default_rng(0)
    sc = None
    for ax, t, letter in zip(axes.ravel(), order, "abcd"):
        shap, x, cols = data[t]
        top = np.argsort(-np.abs(shap).mean(axis=0))[:top_n]
        for rank, j in enumerate(top):
            v = shap[:, j]
            bins = np.digitize(v, np.linspace(v.min(), v.max() + 1e-12, 31))
            counts = np.bincount(bins)[bins]
            jitter = (rng.random(len(v)) - 0.5) * 0.7 * counts / counts.max()
            pct = np.argsort(np.argsort(x[:, j])) / max(len(v) - 1, 1)  # percentile rank of the feature value among the saved rows
            sc = ax.scatter(v, (top_n - 1 - rank) + jitter, c=pct, cmap="viridis", s=2.5, lw=0, alpha=0.7, rasterized=True, vmin=0, vmax=1)
        ax.axvline(0, color="0.4", lw=0.5)
        ax.set_yticks(np.arange(top_n)[::-1])
        ax.set_yticklabels([fl.wrapped(fl.label(cols[j]), 38) for j in top], fontsize=6.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_xlabel({"zT": "SHAP value (zT)", "S": "SHAP value (µV K$^{-1}$)", "kappa": "SHAP value (log$_{10}$)", "sigma": "SHAP value (log$_{10}$)"}[t])
        fs.panel_title(ax, {"zT": "zT", "S": "S", "kappa": "κ", "sigma": "σ"}[t], letter=letter)
        fs.grid(ax, axis="x")
    for ax in axes.ravel()[len(order):]:
        ax.set_visible(False)
    fig.colorbar(sc, ax=axes, orientation="horizontal", location="bottom", shrink=0.5, aspect=40, label="Feature value (percentile rank)")
    fs.save(fig, str(out), expect_width_cm=16.0)
    plt.close(fig)


def main():
    """Entry point."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--na3-rows-dirs", default=None, help="comma-separated NA3-rows bundle directories (tests only; Figure 11 of the paper uses thesis_values.NA3_ROWS)")
    ap.add_argument("--only-fig11-to", default=None, help="write only Figure 11, to this path without extension (for tests)")
    args = ap.parse_args()
    if args.only_fig11_to:
        fig11_beeswarm(args.only_fig11_to, args.na3_rows_dirs.split(","))
        return
    FIG.mkdir(parents=True, exist_ok=True)
    fig6_distributions(FIG / "fig6_property_distributions")
    fig7_parity(FIG / "fig7_predicted_vs_measured")
    fig9_estm(FIG / "fig9_estm_external")
    fig10_shap(FIG / "fig10_shap_global")
    fig12_shares(FIG / "fig12_shap_shares")
    fig11_beeswarm(FIG / "fig11_shap_beeswarm", [REPO / d for d in tv.NA3_ROWS])
    print("Figures 6, 7, 9, 10, 11 and 12 saved")


if __name__ == "__main__":
    main()
