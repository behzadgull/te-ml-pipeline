"""
Figures of the thesis paper that are not reused from Paper A, drawn from committed artifacts with every plotted number asserted
against the source (Paper A style: scripts/figstyle.py, 16 cm wide, 300 dpi, no cropping).

  Figure 6  property distributions (NA4)
  Figure 7  predicted versus measured values (NA8)
  Figure 9  ESTM external validation

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

import figstyle as fs  # noqa: E402
import paper_a_values as pav  # noqa: E402
import thesis_values as tv  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "thesis_paper" / "figures"
T4 = pav.TARGETS
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


def main():
    """Entry point."""
    FIG.mkdir(parents=True, exist_ok=True)
    fig6_distributions(FIG / "fig6_property_distributions")
    fig7_parity(FIG / "fig7_predicted_vs_measured")
    fig9_estm(FIG / "fig9_estm_external")
    print("Figures 6, 7 and 9 saved")


if __name__ == "__main__":
    main()
