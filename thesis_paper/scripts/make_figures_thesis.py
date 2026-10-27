"""
Figures of the thesis paper that are not reused from Paper A, drawn from committed artifacts with every plotted number asserted
against the source (the one shared style, scripts/figstyle.py: Helvetica or Arial, bold axis and tick labels, 16 cm or 8.5 cm wide, 300 dpi, no cropping).

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
sys.path.append(str(HERE.parents[1] / "scripts"))  # the shared style, scripts/figstyle.py (appended: it must not shadow the modules of this folder)
import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402
import matplotlib.transforms  # noqa: E402

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
                transform=ax.transAxes, ha="right", va="top", fontsize=fs.TEXT_PT, color="0.25")
    axes[0].set_ylabel("R$^2$")
    handles = [Patch(fc=fs.OI["blue"], ec=fs.EDGE_GREY), Patch(fc=fs.OI["orange"], ec=fs.EDGE_GREY), Patch(fc=fs.OI["green"], ec=fs.EDGE_GREY)]
    fs.legend_row(fig, handles, ["Internal (chemistry-cluster CV)", "External, full set", "External, rows inside the training range"])
    fs.save(fig, str(out), expect_width_cm=fs.width_cm("double"))
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
    fs.save(fig, str(out), expect_width_cm=fs.width_cm("double"))
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
        ax.text(0.03, 0.97, f"R$^2$ = {d['r2_repeat0']:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=fs.TEXT_PT)
        fs.panel_title(ax, titles[t], letter=letter)
    fig.colorbar(mesh, ax=axes, shrink=0.6, label="Rows per cell")
    fs.save(fig, str(out), expect_width_cm=fs.width_cm("double"))
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
        ax.set_yticklabels([fl.wrapped(fl.label(f["feature"]), 46) for f in top], fontsize=fs.MIN_PT, linespacing=1.0)  # long feature names: the smallest size of the style
        ax.set_xlabel(units[t].replace(" (", "\n("), loc="right")
        fs.panel_title(ax, {"zT": "zT", "S": "S", "kappa": "κ", "sigma": "σ"}[t], letter=letter)
        fs.grid(ax, axis="x")
    fs.legend_row(fig, [Patch(color=c, ec=fs.EDGE_GREY) for c in colour.values()], ["MAGPIE", "CBFV", "Temperature"])
    fs.save(fig, str(out), expect_width_cm=fs.width_cm("double"))
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
    fs.save(fig, str(out), expect_width_cm=fs.width_cm("single"))
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
        ax.set_yticklabels([fl.wrapped(fl.label(cols[j]), 30) for j in top], fontsize=fs.MIN_PT, linespacing=1.0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))  # bold 9 pt tick labels: three ticks keep them apart
        ax.set_xlabel({"zT": "SHAP value (zT)", "S": "SHAP value (µV K$^{-1}$)", "kappa": "SHAP value (log$_{10}$)", "sigma": "SHAP value (log$_{10}$)"}[t], loc="right")
        fs.panel_title(ax, {"zT": "zT", "S": "S", "kappa": "κ", "sigma": "σ"}[t], letter=letter)
        fs.grid(ax, axis="x")
    for ax in axes.ravel()[len(order):]:
        ax.set_visible(False)
    fig.colorbar(sc, ax=axes, orientation="horizontal", location="bottom", shrink=0.5, aspect=40, label="Feature value (percentile rank)")
    fs.save(fig, str(out), expect_width_cm=fs.width_cm("double"))
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

    s_ladder, s_methods, s_algo = sec("Chemistry-cluster cross-validation and validation inflation"), sec("Validation design"), sec("Algorithm comparison")
    s_nested, s_dvd, s_shap, s_ext = sec("Models"), sec("Direct versus component-wise zT prediction"), sec("Feature importance"), sec("External validation")
    s_screen = [n for n, h in heads.items() if h == "Screening results" and n.startswith("3.")]
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

    W = fs.width_cm("double")
    S = W / 16.0  # the layout below was drawn for 16 cm: widths scale with S, heights with K (the text is 9 pt instead of 8 pt)
    K = fs.TEXT_PT / 8.0 * 1.05
    top_w, top_h, gap = 3.4 * S, 2.45 * K, 0.8 * S
    a_h, m_h, e_h, o_h, g = 4.25 * K, 0.9 * K, 3.9 * K, 1.15 * K, 0.6 * K
    H = 0.1 + o_h + g + e_h + g + m_h + g + a_h + 0.85 + top_h + 0.2
    fsz = fs.TEXT_PT
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
    a_gap = 0.3 * S
    a_w = (W - 3 * a_gap) / 4
    a_y = 0.1 + o_h + g + e_h + g + m_h + g
    A = [
        (f"(A) §{s_ladder}, §{s_methods}\nValidation ladder\nand noise ceiling", f"five splits, grouped\nCV {v['n_repeats']} × {v['n_folds']};\nchemistry-cluster R²:\nS {v['chem_S']}, σ {v['chem_sigma']}\nκ {v['chem_kappa']}, zT {v['chem_zT']}\nlabel-noise ceiling\n{v['nc_all_lo']}–{v['nc_all_hi']}"),
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
    t = ax.text(W / 2, m_y + m_h / 2 + 0.22, "XGBoost, frozen per-target hyperparameters (all analyses; alternatives compared in B)", ha="center", va="center", fontsize=fsz, fontweight="bold", color="0.1")
    t.set_position((W / 2, m_y + m_h / 2))
    checks.append((t, (0, m_y, W, m_y + m_h)))
    for xc in cx:
        arrow((xc, a_y), (xc, m_y + m_h + 0.03), dashed=True)

    # E to G
    e_gap = 0.3 * S
    e_w = (W - 2 * e_gap) / 3
    e_y = m_y - g - e_h
    E = [
        (f"(E) §{s_shap}\nSHAP attribution", f"{v['shap_folds']} folds, {v['shap_rows']} rows\neach; features are\nstatistics of elemental\nproperties, not measured\nproperties"),
        (f"(F) §{s_ext}\nExternal transfer", f"ESTM: {v['estm_scope']} rows,\nDOI-disjoint and\ncluster-disjoint strata\nJARVIS: {v['jv_both']}\nDFT entries (description;\nsign by carrier-type\nclassifier)"),
        (f"(G) §{s_screen}\nScreening (hypotheses)", f"Materials Project:\n{v['mp_f0']} → {v['mp_f4']} perovskite-type\n(connectivity test)\n→ {v['mp_f6']} candidates:\n{v['nv_main_seen']} seen, {v['nv_main_unseen']} unseen clusters;\ncarrier-type classifier,\naccuracy {v['cl_acc']}"),
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
    t2 = ax.text(W / 2, 0.1 + o_h / 2 - 0.26, "grouped performance and its label-noise ceiling; ranked candidates as hypotheses, not findings", ha="center", va="center", fontsize=fsz, color="0.1", style="italic")
    checks.extend([(t, (0, 0.1, W, 0.1 + o_h)), (t2, (0, 0.1, W, 0.1 + o_h))])
    for xc in ex:
        arrow((xc, e_y), (xc, 0.1 + o_h + 0.03))

    fs.check_text_fits(fig, ax, checks)
    fs.save(fig, str(out), expect_width_cm=fs.width_cm("double"))
    plt.close(fig)


def fig13_s_kappa(out):
    """Predicted |S| against predicted kappa at the zT maximum for the 30 main-list candidates, seen and unseen chemistry clusters marked, the position # of Table 11 (rank by predicted zT within the cluster group) written beside each point
    (added after NA12, not pre-registered). Every value is read from the table11c.csv of the committed run of scripts/screening_s_kappa_view.py."""
    d = tv.screening_sk_table()
    assert len(d) == 30
    fs.apply()
    fig, ax = fs.new_figure("double", 8.0)
    for seen, mk, col in ((True, "o", fs.OI["blue"]), (False, "s", fs.OI["orange"])):
        sel = d[d["cluster_seen_any"].astype(bool) == seen]
        ax.scatter(sel["kappa_view"], sel["S_view"].abs(), marker=mk, s=34, color=col, edgecolor=fs.EDGE_GREY, linewidth=0.5, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.55, 8.0)
    ax.set_ylim(-15, 450)
    ax.set_xticks([0.6, 0.8, 1, 2, 3, 4, 6])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Predicted κ (W m$^{-1}$ K$^{-1}$)")
    ax.set_ylabel("Predicted |S| (µV K$^{-1}$)")
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    placed = []  # window extents of the labels and markers already on the figure
    pts = [(r["kappa_view"], abs(r["S_view"])) for _, r in d.iterrows()]
    half = 3.8 * fig.dpi / 72  # half the marker size (the marker is about 5.8 pt across), in pixels at the figure dpi
    for x_, y_ in pts:
        px, py = ax.transData.transform((x_, y_))
        placed.append(matplotlib.transforms.Bbox.from_extents(px - half, py - half, px + half, py + half))
    for _, g in d.groupby(["kappa_view", "S_view", "cluster_seen_any"]):  # identical predictions are drawn once and carry every # of Table 11; labels are moved off each other
        r = g.iloc[0]
        for off in ((5, 3), (5, -11), (-15, 3), (-15, -11), (5, 12), (-15, 12), (5, -19), (-15, -19), (-25, -4), (14, -4)):
            t = ax.annotate(", ".join(str(int(x)) for x in sorted(g["rank_in_group_main_30"])), (r["kappa_view"], abs(r["S_view"])), textcoords="offset points", xytext=off, fontsize=fs.MIN_PT, color="0.2")
            bb = t.get_window_extent(rend)
            if not any(bb.overlaps(q) for q in placed):
                placed.append(bb)
                break
            t.remove()
        else:
            raise AssertionError("no free place for a rank label")
    fs.grid(ax, "both")
    handles = [matplotlib.lines.Line2D([], [], marker=m_, ls="", color=c_, mec=fs.EDGE_GREY, mew=0.5, ms=6) for m_, c_ in (("o", fs.OI["blue"]), ("s", fs.OI["orange"]))]
    fs.legend_row(fig, handles, ["Chemistry cluster in the training data", "Chemistry cluster not in the training data"])
    fs.save(fig, str(out), expect_width_cm=fs.width_cm("double"))
    plt.close(fig)


def _assert_texts_apart(fig, texts):
    """No two of the given text artists may overlap (window extents), so that a schematic has no colliding labels."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    bbs = [(t.get_text(), t.get_window_extent(rend)) for t in texts if t.get_text().strip()]
    for i in range(len(bbs)):
        for j in range(i + 1, len(bbs)):
            assert not bbs[i][1].overlaps(bbs[j][1]), f"labels overlap: {bbs[i][0]!r} and {bbs[j][0]!r}"


def fig1_thermocouple(out):
    """Schematic of a thermoelectric p-n couple (Figure 1): hot and cold side, p- and n-type legs, electrodes, load, heat flow, carrier diffusion and conventional current. Drawn in code; no data."""
    from matplotlib.patches import Circle, Rectangle

    fs.apply()
    W, H = 16.0, 10.6
    fig, ax = fs.new_figure("double", H)
    ax.set_xlim(0, W)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    RED_L, BLUE_L, GREY = "#F6D5CC", "#D2E7F5", "#B8BCC2"
    texts, boxes = [], []

    def rect(x0, y0, x1, y1, fc, ec):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fc, ec=ec, lw=1.0, zorder=2))

    def label(x, y, s, box=None, **kw):
        kw.setdefault("ha", "center")
        kw.setdefault("va", "center")
        kw.setdefault("fontsize", fs.TEXT_PT)
        t = ax.text(x, y, s, zorder=6, **kw)
        texts.append(t)
        if box:
            boxes.append((t, box))
        return t

    def arrow(p0, p1, color="black", lw=1.6, ls="-"):
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=11, shrinkA=0, shrinkB=0), zorder=5)

    # plates, electrodes, legs
    rect(3.9, 8.35, 10.9, 9.0, RED_L, fs.OI["vermillion"])
    label(7.4, 8.675, "Hot side, $T_h$", box=(3.9, 8.35, 10.9, 9.0), color=fs.OI["vermillion"], fontweight="bold")
    rect(4.6, 7.85, 10.3, 8.35, GREY, "0.35")
    label(7.45, 8.1, "hot electrode", box=(4.6, 7.85, 10.3, 8.35))
    rect(4.6, 3.35, 6.6, 7.85, fs.OI["vermillion"], "0.25")
    rect(8.3, 3.35, 10.3, 7.85, fs.OI["blue"], "0.25")
    label(5.6, 7.3, "p-type", box=(4.6, 3.35, 6.6, 7.85), color="white", fontweight="bold")
    label(9.3, 7.3, "n-type", box=(8.3, 3.35, 10.3, 7.85), color="white", fontweight="bold")
    rect(4.6, 2.85, 6.6, 3.35, GREY, "0.35")
    rect(8.3, 2.85, 10.3, 3.35, GREY, "0.35")
    label(5.6, 3.1, "electrode", box=(4.6, 2.85, 6.6, 3.35))
    label(9.3, 3.1, "electrode", box=(8.3, 2.85, 10.3, 3.35))
    rect(3.9, 2.15, 10.9, 2.85, BLUE_L, fs.OI["blue"])
    label(7.4, 2.5, "Cold side, $T_c$", box=(3.9, 2.15, 10.9, 2.85), color=fs.OI["blue"], fontweight="bold")

    # heat flow
    for x in np.linspace(4.9, 9.9, 4):
        arrow((x, 9.75), (x, 9.05), color=fs.OI["vermillion"])
        arrow((x, 2.1), (x, 1.5), color=fs.OI["blue"])
    label(11.15, 9.4, "Heat in", ha="left", color=fs.OI["vermillion"], fontweight="bold")
    label(7.4, 1.2, "Heat out", color=fs.OI["blue"], fontweight="bold")

    # carriers (hot to cold diffusion) in the legs
    for cx, sym, pts in ((5.6, "+", ((5.0, 6.4), (6.2, 5.6), (5.0, 4.8))), (9.3, "−", ((8.7, 6.4), (9.9, 5.6), (8.7, 4.8)))):
        arrow((cx, 6.85), (cx, 4.15), color="white", lw=1.2, ls=(0, (3, 2)))
        for (x, y) in pts:
            ax.add_patch(Circle((x, y), 0.26, fc="none", ec="white", lw=1.3, zorder=4))
            label(x, y - 0.02, sym, color="white", fontweight="bold")
    label(5.6, 3.7, "holes", box=(4.6, 3.35, 6.6, 7.85), color="white", fontweight="bold")
    label(9.3, 3.7, "electrons", box=(8.3, 3.35, 10.3, 7.85), color="white", fontweight="bold")

    # external circuit with the load, and the conventional current
    wire = dict(color="black", lw=1.6, solid_capstyle="butt", zorder=3)
    ax.plot([4.6, 3.4, 3.4, 6.2], [3.1, 3.1, 0.5, 0.5], **wire)
    ax.plot([8.6, 11.4, 11.4, 10.3], [0.5, 0.5, 3.1, 3.1], **wire)
    zx = np.linspace(6.2, 8.6, 13)
    zy = 0.5 + np.array([0] + [0.22 * (-1) ** k for k in range(11)] + [0])
    ax.plot(zx, zy, **wire)
    label(7.4, 0.0, "Load $R_L$", va="top")
    arrow((3.55, 0.5), (4.75, 0.5))
    label(4.15, 0.85, "$I$")

    # temperature axis
    ax.annotate("", xy=(1.0, 9.0), xytext=(1.0, 2.5), arrowprops=dict(arrowstyle="-|>", color="0.2", lw=1.3, mutation_scale=10, shrinkA=0, shrinkB=0))
    label(1.35, 8.8, "$T_h$", ha="left")
    label(1.35, 2.5, "$T_c$", ha="left")
    label(1.35, 5.65, "$\\Delta T = T_h - T_c$", ha="left")
    label(0.45, 5.75, "temperature", rotation=90, color="0.2")

    # voltage
    label(11.9, 6.15, "Open-circuit voltage", ha="left")
    label(11.9, 5.55, "$V_{oc} = (S_p - S_n)\\,\\Delta T$", ha="left")

    fs.check_text_fits(fig, ax, boxes)
    _assert_texts_apart(fig, texts)
    fs.save(fig, str(out), expect_width_cm=fs.width_cm("double"))
    plt.close(fig)


def fig2_carrier_concentration(out):
    """Schematic dependence of S, sigma, kappa (electronic and lattice) and zT on the carrier concentration (Figure 2), in arbitrary units. The curves follow the qualitative model S ~ ln(1 + (n_S/n)^(2/3))
    (logarithmic at low n, degenerate n^(-2/3) at high n), sigma ~ n, kappa_e ~ sigma (Wiedemann-Franz) and a constant kappa_l; zT ~ S^2 sigma / kappa. The shaded optimum range is read from the registry rows of
    Snyder and Toberer 2008 (snyder2008_n_lo, snyder2008_n_hi); the zT maximum of the model must fall inside it (asserted)."""
    import audit_numbers as an

    reg = an.load_registry()
    lo, hi = (float(reg[k]["value"].split()[0]) for k in ("snyder2008_n_lo", "snyder2008_n_hi"))
    assert (lo, hi) == (1e19, 1e21), (lo, hi)
    fs.apply()
    u = np.linspace(17.0, 22.0, 801)
    n = 10.0 ** u
    n_s, n_k = 1.0e20, 1.0e20  # schematic scales: the carrier concentration where S bends, and where kappa_e equals kappa_l
    S = np.log1p((n_s / n) ** (2 / 3))
    S = S / S.max()
    sig = n / n.max()
    kl = np.ones_like(n)
    ke = n / n_k
    kt = ke + kl
    zt = S ** 2 * sig / kt
    zt = zt / zt.max()
    u_pk = u[zt.argmax()]
    assert np.log10(lo) <= u_pk <= np.log10(hi), u_pk
    fig, axes = fs.new_figure("double", 13.0, 4, 1, sharex=True)
    axes = list(axes)
    x_ins, x_met = 18.0, 21.0  # region boundaries in log10 n (schematic)
    panels = [("S (a.u.)", False), ("$\\sigma$ (a.u., log)", True), ("$\\kappa$ (a.u., log)", True), ("zT (a.u.)", False)]
    for ax, (yl, logy) in zip(axes, panels):
        ax.axvspan(17.0, x_ins, color="0.5", alpha=0.10, lw=0, zorder=0)
        ax.axvspan(x_met, 22.0, color="0.5", alpha=0.10, lw=0, zorder=0)
        ax.axvspan(np.log10(lo), np.log10(hi), color=fs.OI["orange"], alpha=0.22, lw=0, zorder=0)
        ax.set_ylabel(yl)
        ax.set_xlim(17.0, 22.0)
        if logy:
            ax.set_yscale("log")
        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())  # arbitrary units: no y tick labels
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        fs.grid(ax, "x")
    axes[0].plot(u, S, color=fs.OI["vermillion"], lw=2.0)
    axes[0].set_ylim(0, 1.1)
    axes[1].plot(u, sig, color=fs.OI["blue"], lw=2.0)
    axes[1].set_ylim(1e-5, 3.0)
    axes[2].plot(u, kt, color="black", lw=2.0, label="$\\kappa = \\kappa_e + \\kappa_l$")
    axes[2].plot(u, ke, color=fs.OI["sky"], lw=1.6, ls="--", label="$\\kappa_e$")
    axes[2].plot(u, kl, color=fs.OI["green"], lw=1.6, ls="--", label="$\\kappa_l$")
    axes[2].set_ylim(3e-3, 4e4)
    axes[2].legend(loc="upper left", ncol=1, handlelength=2.2)
    axes[3].plot(u, zt, color=fs.OI["purple"], lw=2.0)
    axes[3].plot([u_pk], [1.0], "o", color=fs.OI["purple"], ms=5, mec=fs.EDGE_GREY, mew=0.6, zorder=5)
    axes[3].set_ylim(0, 1.18)
    ticks = [17, 18, 19, 20, 21, 22]
    axes[3].set_xticks(ticks)
    axes[3].set_xticklabels([f"$10^{{{t}}}$" for t in ticks])
    axes[3].set_xlabel("Carrier concentration $n$ (cm$^{-3}$)")
    top = axes[0]
    texts = []
    for xc, s in ((17.5, "Insulator"), ((x_ins + x_met) / 2, "Semiconductor"), (21.5, "Metal")):
        texts.append(top.annotate(s, xy=(xc, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=fs.TEXT_PT, color="0.25", fontweight="bold"))
    texts.append(top.annotate(f"Optimum zT: $10^{{{int(np.log10(lo))}}}$ to $10^{{{int(np.log10(hi))}}}$ cm$^{{-3}}$", xy=(np.log10(np.sqrt(lo * hi)), 1.0), xycoords=("data", "axes fraction"),
                              xytext=(0, 17), textcoords="offset points", ha="center", va="bottom", fontsize=fs.TEXT_PT, color=fs.OI["vermillion"], fontweight="bold"))
    _assert_texts_apart(fig, texts)
    fs.save(fig, str(out), expect_width_cm=fs.width_cm("double"))
    plt.close(fig)


def fig3_perovskite(out):
    """The ideal cubic ABX3 perovskite (Figure 3), drawn from fractional coordinates in an oblique projection: A at the corners (0 or 1, 0 or 1, 0 or 1), B at the body centre (1/2, 1/2, 1/2),
    X at the six face centres, with the BX6 octahedron (the 12 edges between the six X atoms). No data."""
    from itertools import combinations, product

    from matplotlib.patches import Polygon

    fs.apply()
    W, H = 8.5, 8.9
    fig, ax = fs.new_figure("single", H)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    side, dx, dy = 3.4, 1.0, 0.85  # the cube edge in cm, and the projection of the depth axis
    ox, oy = 2.4, 3.3

    def proj(p):
        x, y, z = p
        return (ox + side * (x + dx / side * z), oy + side * (y + dy / side * z))

    A = [p for p in product((0, 1), repeat=3)]
    B = (0.5, 0.5, 0.5)
    X = [(0.5, 0.5, 0), (0.5, 0.5, 1), (0.5, 0, 0.5), (0.5, 1, 0.5), (0, 0.5, 0.5), (1, 0.5, 0.5)]
    opp = lambda a, b: all(abs(i + j - 1) < 1e-9 for i, j in zip(a, b))  # noqa: E731  (the two X on opposite faces)
    oct_edges = [(a, b) for a, b in combinations(X, 2) if not opp(a, b)]
    assert len(oct_edges) == 12
    # octahedron faces (each triple of mutually adjacent X), drawn first
    for tri in combinations(X, 3):
        if all(not opp(a, b) for a, b in combinations(tri, 2)):
            ax.add_patch(Polygon([proj(p) for p in tri], closed=True, fc=fs.OI["sky"], ec="none", alpha=0.10, zorder=1))
    # cube edges: the front face solid, the others thin and dashed (the cube is drawn transparent)
    for a, b in combinations(A, 2):
        if sum(abs(i - j) for i, j in zip(a, b)) == 1:
            (x0, y0), (x1, y1) = proj(a), proj(b)
            front = a[2] == 0 and b[2] == 0
            ax.plot([x0, x1], [y0, y1], color="black" if front else "0.45", lw=1.4 if front else 0.9, ls="-" if front else (0, (4, 3)), zorder=2)
    for a, b in oct_edges:
        (x0, y0), (x1, y1) = proj(a), proj(b)
        ax.plot([x0, x1], [y0, y1], color=fs.OI["blue"], lw=0.9, alpha=0.9, zorder=3)
    # atoms, back to front
    atoms = [(p, "A") for p in A] + [(B, "B")] + [(p, "X") for p in X]
    style = {"A": (fs.OI["green"], 230), "B": (fs.OI["blue"], 150), "X": (fs.OI["vermillion"], 110)}
    for p, kind in sorted(atoms, key=lambda t: -t[0][2]):
        c, s = style[kind]
        ax.scatter(*proj(p), s=s, color=c, edgecolor=fs.EDGE_GREY, linewidth=0.6, zorder=4 + (1 - p[2]))
    texts = []

    def lab(xy, text, **kw):
        kw.setdefault("ha", "left")
        t = ax.text(*xy, text, fontsize=fs.TEXT_PT, va="center", fontweight="bold", zorder=7, **kw)
        texts.append(t)
        return t

    def leader(p, q):
        ax.annotate("", xy=p, xytext=q, arrowprops=dict(arrowstyle="-", color="0.15", lw=0.8, shrinkA=0, shrinkB=3), zorder=6)

    leader(proj((0, 1, 0)), (0.55, 6.9))
    lab((0.25, 7.1), "A")
    leader(proj(B), (6.8, 6.2))
    lab((6.85, 6.2), "B")
    leader(proj((0, 0.5, 0.5)), (1.45, 5.0))
    lab((1.0, 5.0), "X")
    leader(proj((0.75, 0.75, 0.5)), (5.6, 8.1))
    lab((5.6, 8.4), "BX$_6$ octahedron", ha="center")
    # lattice parameter: a dimension line below the front face, with extension lines
    ya = oy - 0.55
    for xe in (ox, ox + side):
        ax.plot([xe, xe], [oy - 0.25, ya - 0.12], color="black", lw=0.8, zorder=6)
    ax.annotate("", xy=(ox + side, ya), xytext=(ox, ya), arrowprops=dict(arrowstyle="<|-|>", color="black", lw=1.0, shrinkA=0, shrinkB=0, mutation_scale=9), zorder=6)
    lab((ox + side / 2, ya - 0.4), "$a$", ha="center")
    # key
    key_y = [1.75, 1.15, 0.55]
    key = [("A", "A site (e.g. Sr, Ca, Ba, Cs)"), ("B", "B site (e.g. Ti, Mn, Co, Sn)"), ("X", "X site (O, Cl, Br, I, S, Se, Te)")]
    for y, (kind, s) in zip(key_y, key):
        c, _ = style[kind]
        ax.scatter(0.55, y, s=90, color=c, edgecolor=fs.EDGE_GREY, linewidth=0.6, zorder=4)
        t = ax.text(0.95, y, s, fontsize=fs.TEXT_PT, ha="left", va="center", zorder=7)
        texts.append(t)
    fig.canvas.draw()
    _assert_texts_apart(fig, texts)
    fs.save(fig, str(out), expect_width_cm=fs.width_cm("single"))
    plt.close(fig)


def check_paper_widths():
    """Every figure of paper.md and of the Supplement that carries a width in cm must be included at its true width (the width of its PNG at the style's dpi), so that Word does not rescale it.
    Figure numbers come from the labels (scripts/resolve_labels.py)."""
    import re

    from PIL import Image

    import resolve_labels as rl

    main_text = (REPO / "thesis_paper" / "paper" / "paper.md").read_text(encoding="utf-8")
    supp_text = rl.load_supplement()
    numbers = rl.scan(main_text, supp_text)[0]
    found = [("Figure " + str(numbers["fig"][lb]), rel, w) for lb, rel, w in re.findall(r"!\[Figure \[\[fig:(\w+)\]\]\]\(\.\./figures/([^)]+)\)\{width=([\d.]+)cm\}", main_text)]
    found += [("Supplementary Figure S" + str(numbers["sfig"][lb]), rel, w) for lb, rel, w in re.findall(r"!\[Supplementary Figure \[\[sfig:(\w+)\]\]\]\(\.\./figures/([^)]+)\)\{width=([\d.]+)cm\}", supp_text)]
    drawn = len(re.findall(r"!\[(?:Supplementary )?Figure \[\[(?:fig|sfig):\w+\]\]\]\(\.\./figures/(?!source_media)", main_text + supp_text))
    assert len(found) == 13 and drawn >= 13, f"expected the thirteen drawn figures with a width in cm, found {len(found)} of {drawn}"
    for num, rel, w in found:
        with Image.open(REPO / "thesis_paper" / "figures" / rel) as im:
            cm = im.size[0] / fs.DPI * 2.54
        assert abs(cm - float(w)) < 0.05, f"{num}: the text says {w} cm, the file is {cm:.2f} cm wide"
    return {n: float(w) for n, _, w in found}


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
    fig1_thermocouple(FIG / "fig1_thermocouple")
    fig2_carrier_concentration(FIG / "fig2_carrier_concentration")
    fig3_perovskite(FIG / "fig3_perovskite_structure")
    fig13_s_kappa(FIG / "fig13_s_kappa")
    widths = check_paper_widths()
    print("Figures 1 to 4, 6, 7, 9, 10, 11, 12 and 13 saved; widths in paper.md match the files:", widths)


if __name__ == "__main__":
    main()
