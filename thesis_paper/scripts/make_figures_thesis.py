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
    """SHAP global importance: the features with the largest mean |SHAP| per target (NA3, chemistry-cluster folds), coloured by descriptor scheme; error bars are the 2.5th to 97.5th percentile of the per-fold mean over the 25 folds."""
    na3 = tv.na3_per_target()
    fs.apply()
    fig, axes = fs.new_figure("double", 21.0, 2, 2)
    order = ("S", "sigma", "kappa", "zT")
    units = {"zT": "mean |SHAP| (zT)", "S": "mean |SHAP| (µV K$^{-1}$)", "kappa": "mean |SHAP| (log$_{10}$)", "sigma": "mean |SHAP| (log$_{10}$)"}
    colour = {"MagpieData": fs.OI["blue"], "CBFV_": fs.OI["orange"], "temperature_bin": fs.OI["green"]}
    for ax, t, letter in zip(axes.ravel(), order, "abcd"):
        top = na3[t]["top20"][:SHAP_TOP_N]
        assert len(top) == SHAP_TOP_N and all(top[i]["mean_abs_shap"] >= top[i + 1]["mean_abs_shap"] for i in range(SHAP_TOP_N - 1))
        y = np.arange(SHAP_TOP_N)[::-1]
        cols = [next(c for k, c in colour.items() if f["feature"].startswith(k)) for f in top]
        st_ = tv.shap_fold_stats(na3[t])
        idx = [na3[t]["feature_columns"].index(f["feature"]) for f in top]
        mean_ = np.array([f["mean_abs_shap"] for f in top])
        lo_, hi_ = st_["lo"][idx], st_["hi"][idx]
        assert (lo_ >= 0).all() and (lo_ <= mean_ + 1e-9).all() and (mean_ <= hi_ + 1e-9).all()  # a non-negative interval that contains the mean
        ax.barh(y, mean_, xerr=[mean_ - lo_, hi_ - mean_], color=cols, ec=fs.EDGE_GREY, lw=0.4, error_kw={"lw": 0.6, "capsize": 1.5})
        ax.set_xlim(left=0)
        ax.set_yticks(y)
        ax.set_yticklabels([fl.wrapped(fl.label(f["feature"]), 46) for f in top], fontsize=6.5)
        ax.set_xlabel(units[t].replace(" (", "\n("), fontsize=7.5)
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


def fig4_workflow(out):
    """
    Study workflow (Figure 4): data pipeline on top, four validation and model analyses, the shared model band, then interpretation, transfer and screening, and the outputs. Drawn at 16 cm width in
    figstyle (text 8 pt or larger, every text block checked to lie inside its box); every count and every section number is read from a committed artifact or from paper.md and asserted.
    """
    import json
    import re
    from datetime import datetime

    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    v = tv.values()
    v.update(tv.new_analysis_values())
    paper = (REPO / "thesis_paper" / "paper" / "paper.md").read_text(encoding="utf-8")
    heads = {m.group(1): m.group(2).strip() for m in re.finditer(r"^#{2,3} (\d(?:\.\d+)*) (.+)$", paper, flags=re.M)}

    def sec(title):
        found = [n for n, h in heads.items() if h == title]
        assert len(found) == 1, (title, found)
        return found[0]

    s_ladder, s_methods, s_algo = sec("Chemistry-cluster cross-validation"), sec("Validation method comparison"), sec("Algorithm comparison")
    s_nested, s_dvd, s_shap, s_ext = sec("Model development"), sec("Direct versus component-wise zT prediction"), sec("Feature importance and explainability"), sec("External validation")
    s_screen = [n for n, h in heads.items() if h == "Virtual screening" and n.startswith("4.")]
    assert len(s_screen) == 1
    s_screen = s_screen[0]

    funnel = pav._json(pav.FUNNEL)
    assert len(funnel["funnel"]) == 11 and funnel["verification_gate"]["match"] is True
    papers, curves, cleaned = funnel["raw_input_row_counts"]["papers"], funnel["raw_input_row_counts"]["curves"], funnel["funnel"][-1]["rows"]
    assert f"{cleaned:,}" == v["n_clean"]
    meta = pav._json(pav.RAWMETA)
    assert meta["files"]["papers"]["counted_row_count"] == papers and meta["files"]["curves"]["counted_row_count"] == curves
    pulled = datetime.fromisoformat(meta["extraction_timestamp_utc"])
    assert meta["upstream_db_snapshot"].startswith(pulled.strftime("%Y-%m-%d"))
    snap = f"{pulled.day} {pulled:%b %Y}"
    ladder = pav._json(pav.LADDER)["runs"]
    rows = {t: ladder[f"{t}_chemistry_full"]["n_rows_header"] for t in pav.TARGETS}
    for t in pav.TARGETS:
        assert f"{rows[t]:,}" == v[f"n_{t}"]
    assert int(v["shap_n_magpie"]) + int(v["shap_n_cbfv"]) + 1 == int(v["n_feat"]) == 397
    # asserted against the stated sources of the later analyses
    assert v["optuna_trials"] == "20" and v["shap_rows"] == "20,000" and v["shap_folds"] == "25" and v["estm_scope"] == "4,539" and v["jv_both"] == "23,218"
    assert v["mp_f0"] == "75,508" and v["mp_f4"] == "315" and v["mp_f6"] == "30" and v["nv_main_seen"] == "9" and v["nv_main_unseen"] == "21"

    W = 16.0
    top_w, top_h, gap = 3.4, 2.45, 0.8
    a_h, m_h, e_h, o_h, g = 3.75, 1.15, 3.3, 1.15, 0.6
    H = 0.1 + o_h + g + e_h + g + m_h + g + a_h + 0.85 + top_h + 0.2
    fsz = 8.0
    fig = plt.figure(figsize=(W * fs.CM, H * fs.CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-0.1, W + 0.1)
    ax.set_ylim(0, H)
    ax.axis("off")
    checks = []

    def box(x, y, w, h, head, body, face="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.12", fc=face, ec="0.15", lw=0.9))
        t1 = ax.text(x + w / 2, y + h - 0.13, head, ha="center", va="top", fontsize=fsz, fontweight="bold", color="0.1", linespacing=1.25)
        fig.canvas.draw()
        hb = ax.transData.inverted().transform((0, t1.get_window_extent(fig.canvas.get_renderer()).y0))[1]
        t2 = ax.text(x + w / 2, hb - 0.2, body, ha="center", va="top", fontsize=fsz, color="0.1", linespacing=1.25)
        checks.extend([(t1, (x, y, x + w, y + h)), (t2, (x, y, x + w, y + h))])

    def arrow(p0, p1, dashed=False):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, lw=0.9, color="0.25", shrinkA=0, shrinkB=0, ls=(0, (3, 2)) if dashed else "-"))

    # data pipeline
    top_y = H - 0.2 - top_h
    xs = [i * (top_w + gap) for i in range(4)]
    box(xs[0], top_y, top_w, top_h, "Starrydata2\nsnapshot", f"{snap}\n{papers:,} papers\n{curves:,} curves")
    box(xs[1], top_y, top_w, top_h, "Cleaning", f"11 steps\n{cleaned:,} rows")
    box(xs[2], top_y, top_w, top_h, "Featurisation", f"{v['shap_n_magpie']} MAGPIE\n+ {v['shap_n_cbfv']} CBFV\n+ temperature\n= {v['n_feat']} features")
    box(xs[3], top_y, top_w, top_h, "Per-target data", "\n".join(f"{n}  {rows[t]:,}" for n, t in (("S", "S"), ("σ", "sigma"), ("κ", "kappa"), ("zT", "zT"))))
    for i in range(3):
        arrow((xs[i] + top_w + 0.03, top_y + top_h / 2), (xs[i + 1] - 0.03, top_y + top_h / 2))

    # analyses A to D
    a_gap = 0.3
    a_w = (W - 3 * a_gap) / 4
    a_y = 0.1 + o_h + g + e_h + g + m_h + g
    A = [
        (f"(A) §{s_ladder}, §{s_methods}\nValidation ladder\nand ceiling", f"five splits, grouped\nCV {v['n_repeats']} × {v['n_folds']};\nchemistry-cluster\nceilings R²:\nS {v['chem_S']}, σ {v['chem_sigma']}\nκ {v['chem_kappa']}, zT {v['chem_zT']}"),
        (f"(B) §{s_algo}\nModel\ncomparison", "XGBoost, LightGBM,\nrandom forest,\nstacking; paired,\nsame folds"),
        (f"(C) §{s_nested}\nNested CV", f"{v['optuna_trials']} trials inside\neach outer fold;\noptimism ≤ {v['n1_max']} R²"),
        (f"(D) §{s_dvd}\nDirect vs\nderived zT", f"{v['dvd_rows']} rows\ndirect R² {v['dvd_direct']}\nderived R² {v['dvd_derived']}"),
    ]
    cx = []
    for i, (hd, bd) in enumerate(A):
        x0 = i * (a_w + a_gap)
        cx.append(x0 + a_w / 2)
        box(x0, a_y, a_w, a_h, hd, bd, face="0.95")
    bus_y = top_y - 0.42
    x_src = cx[-1]
    assert xs[3] < x_src < xs[3] + top_w
    ax.plot([x_src, x_src], [top_y, bus_y], color="0.25", lw=0.9)
    ax.plot([cx[0], x_src], [bus_y, bus_y], color="0.25", lw=0.9)
    for xc in cx:
        arrow((xc, bus_y), (xc, a_y + a_h + 0.03))

    # model band
    m_y = a_y - g - m_h
    ax.add_patch(FancyBboxPatch((0, m_y), W, m_h, boxstyle="round,pad=0.0,rounding_size=0.12", fc="0.85", ec="0.15", lw=0.9))
    t = ax.text(W / 2, m_y + m_h / 2 + 0.22, "XGBoost, frozen per-target hyperparameters; carrier-type classifier", ha="center", va="center", fontsize=fsz, fontweight="bold", color="0.1")
    t2 = ax.text(W / 2, m_y + m_h / 2 - 0.26, f"classifier accuracy {v['cl_acc']} under chemistry-cluster CV", ha="center", va="center", fontsize=fsz, color="0.1", style="italic")
    checks.extend([(t, (0, m_y, W, m_y + m_h)), (t2, (0, m_y, W, m_y + m_h))])
    for xc in cx:
        arrow((xc, a_y), (xc, m_y + m_h + 0.03), dashed=True)

    # E to G
    e_gap = 0.3
    e_w = (W - 2 * e_gap) / 3
    e_y = m_y - g - e_h
    E = [
        (f"(E) §{s_shap}\nSHAP attribution", f"{v['shap_folds']} folds, {v['shap_rows']} rows\neach; features are\nstatistics of elemental\nproperties, not measured\nproperties"),
        (f"(F) §{s_ext}\nExternal transfer", f"ESTM: {v['estm_scope']} rows,\nDOI-disjoint and\ncluster-disjoint strata\nJARVIS: {v['jv_both']}\nDFT entries (description)"),
        (f"(G) §{s_screen}\nScreening (hypotheses)", f"Materials Project:\n{v['mp_f0']} → {v['mp_f4']} perovskite-type\n(connectivity test)\n→ {v['mp_f6']} candidates:\n{v['nv_main_seen']} seen, {v['nv_main_unseen']} unseen clusters"),
    ]
    ex = []
    for i, (hd, bd) in enumerate(E):
        x0 = i * (e_w + e_gap)
        ex.append(x0 + e_w / 2)
        box(x0, e_y, e_w, e_h, hd, bd, face="0.95")
        arrow((x0 + e_w / 2, m_y), (x0 + e_w / 2, e_y + e_h + 0.03))

    # outputs
    ax.add_patch(FancyBboxPatch((0, 0.1), W, o_h, boxstyle="round,pad=0.0,rounding_size=0.12", fc="0.85", ec="0.15", lw=0.9))
    t = ax.text(W / 2, 0.1 + o_h / 2 + 0.22, "Outputs", ha="center", va="center", fontsize=fsz, fontweight="bold", color="0.1")
    t2 = ax.text(W / 2, 0.1 + o_h / 2 - 0.26, "composition-only ceilings; ranked list of candidates as hypotheses, not findings", ha="center", va="center", fontsize=fsz, color="0.1", style="italic")
    checks.extend([(t, (0, 0.1, W, 0.1 + o_h)), (t2, (0, 0.1, W, 0.1 + o_h))])
    for xc in ex:
        arrow((xc, e_y), (xc, 0.1 + o_h + 0.03))

    fs.check_text_fits(fig, ax, checks)
    fs.save(fig, str(out), expect_width_cm=16.0)
    plt.close(fig)


def main():
    """Entry point."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--na3-rows-dirs", default=None, help="comma-separated NA3-rows bundle directories (tests only; Figure 11 of the paper uses thesis_values.NA3_ROWS)")
    ap.add_argument("--only-fig4-to", default=None, help="write only Figure 4, to this path without extension (for tests)")
    ap.add_argument("--only-fig11-to", default=None, help="write only Figure 11, to this path without extension (for tests)")
    args = ap.parse_args()
    if args.only_fig4_to:
        fig4_workflow(args.only_fig4_to)
        return
    if args.only_fig11_to:
        fig11_beeswarm(args.only_fig11_to, args.na3_rows_dirs.split(","))
        return
    FIG.mkdir(parents=True, exist_ok=True)
    fig6_distributions(FIG / "fig6_property_distributions")
    fig7_parity(FIG / "fig7_predicted_vs_measured")
    fig4_workflow(FIG / "fig4_workflow")
    fig9_estm(FIG / "fig9_estm_external")
    fig10_shap(FIG / "fig10_shap_global")
    fig12_shares(FIG / "fig12_shap_shares")
    fig11_beeswarm(FIG / "fig11_shap_beeswarm", [REPO / d for d in tv.NA3_ROWS])
    print("Figures 4, 6, 7, 9, 10, 11 and 12 saved")


if __name__ == "__main__":
    main()
