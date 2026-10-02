"""
Figures 1 to 4 of the Paper B preview manuscript (paper_b/paper/paper.md), in the Paper A style
(paper_b/scripts/figstyle.py): 16 cm wide, 300 dpi, saved without cropping, Okabe-Ito colours, no hatching,
text at 8 pt or larger.

  Figure 1  study design: what each condition trains on, for one held-out family F
  Figure 2  rows per family and property, qualifying marked, never-held-out categories apart
  Figure 3  qualification map: families by property, grouped by super-family
  Figure 4  error-decomposition schematic on synthetic data (labelled illustrative)

No model results exist and none are read. Every count comes from paper_b_facts.load_facts(), which
checks the committed artifacts against each other first; every value drawn is asserted equal to its source
before it is drawn. Figures 1 and 4 are schematics: they contain no counts, and every proportion they use is
asserted against the design (five folds; size matching) or computed from synthetic data and checked.

Run from the repository root:  python paper_b/scripts/make_figures_b.py
Writes paper_b/figures/<name>.png and .pdf.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402
import paper_b_facts as pf  # noqa: E402

FIGURES_DIR = pf.PB / "figures"
FIG_STEMS = {1: "fig1_study_design", 2: "fig2_family_coverage", 3: "fig3_qualification_map", 4: "fig4_error_decomposition"}

# one colour per property (Okabe-Ito, from the shared cycle) and one per role in the design schematic
TARGET_COLOURS = {"S": fs.OI["blue"], "sigma": fs.OI["vermillion"], "kappa": fs.OI["green"], "zT": fs.OI["purple"]}
C_TEST, C_REST, C_G, C_OTHER = fs.OI["vermillion"], fs.OI["blue"], fs.OI["purple"], fs.UNGROUPED_GREYS[0]
REMOVED_EDGE = "0.35"
DASH = (0, (3, 2))
QUALIFIES_FILL, FAILS_FILL = "#B9DDF1", "#EDEDED"


def _check_legend_inside(fig, legend):
    fig.canvas.draw()
    bb, fb = legend.get_window_extent(), fig.bbox
    assert bb.x0 >= fb.x0 - 1 and bb.x1 <= fb.x1 + 1 and bb.y0 >= fb.y0 - 1 and bb.y1 <= fb.y1 + 1, "legend leaves the figure"


# ---------------------------------------------------------------------------------------------------------------
# Figure 1: study design
# ---------------------------------------------------------------------------------------------------------------

def make_study_design(out_path, facts):
    """
    One held-out family F as a bar of blocks. Test fold = 1/n_folds of F (n_folds read from the committed splits).
    G has the width of F (the design picks the held-out family closest to F in row count). Dashed white = removed
    from training. Asserted: C1 removes exactly the rest of F, C2 removes slivers totalling exactly that width (so its
    training size equals C1's), C3 removes exactly G, the specialist keeps only the rest of F.
    """
    n_folds = facts["n_folds"]
    assert n_folds == 5, n_folds
    W, label_w = 16.0, 4.5
    x0, B = 4.7, 11.2
    gap_test = 0.25                              # visible gap between F's test fold and every training block
    f_tot = 0.30 * B
    test_w = f_tot / n_folds
    rest_w = f_tot - test_w
    g_w = f_tot
    oth_w = B - f_tot - g_w - gap_test
    assert oth_w > 0 and abs(test_w + gap_test + rest_w + g_w + oth_w - B) < 1e-9
    xt, xr, xg, xo = x0, x0 + test_w + gap_test, x0 + f_tot + gap_test, x0 + f_tot + gap_test + g_w
    assert abs(xo + oth_w - (x0 + B)) < 1e-9

    row_h, gap, head_h, leg_h = 0.9, 0.42, 1.35, 1.75
    n_rows = 5
    H = head_h + n_rows * row_h + (n_rows - 1) * gap + leg_h + 0.15
    fig = plt.figure(figsize=(W * fs.CM, H * fs.CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    checks = []

    def block(x, w, y, kind, present=True):
        colour = {"test": C_TEST, "rest": C_REST, "G": C_G, "oth": C_OTHER}[kind]
        if present:
            ax.add_patch(Rectangle((x, y), w, row_h, fc=colour, ec="white", lw=0.6, zorder=2))
        else:
            ax.add_patch(Rectangle((x, y), w, row_h, fc="white", ec=REMOVED_EDGE, lw=0.8, ls=DASH, zorder=3))

    # header: brackets and labels over the blocks of the first row
    y_head = H - head_h + 0.1
    for (xa, xb, text) in ((xt, xg, "family F"), (xg, xo, "family G"), (xo, x0 + B, "all other families")):
        ax.plot([xa + 0.05, xa + 0.05, xb - 0.05, xb - 0.05], [y_head - 0.12, y_head, y_head, y_head - 0.12], color="0.3", lw=0.7)
        t = ax.text((xa + xb) / 2, y_head + 0.1, text, ha="center", va="bottom", fontsize=8, color="0.1")
        checks.append((t, (xa, y_head, xb, H)))

    ledger = {}                                  # condition -> (training width, removed width) in cm, for the assertions
    labels = [
        ("C0", "C0  pooled"),
        ("C1", "C1  leave-one-\nfamily-out"),
        ("C2", "C2  size-matched\nrandom removal"),
        ("C3", "C3  structured\nremoval (family G)"),
        ("SP", "Specialist\n(F's training folds)"),
    ]
    for i, (key, label) in enumerate(labels):
        y = H - head_h - (i + 1) * row_h - i * gap
        t = ax.text(0.15, y + row_h / 2, label, ha="left", va="center", fontsize=8, fontweight="bold", color="0.1", linespacing=1.2)
        checks.append((t, (0, y, label_w, y + row_h)))
        block(xt, test_w, y, "test")
        rest_in = key in ("C0", "C2", "C3", "SP")
        g_in = key in ("C0", "C1", "C2")
        oth_in = key != "SP"
        block(xr, rest_w, y, "rest", rest_in)
        block(xg, g_w, y, "G", g_in)
        block(xo, oth_w, y, "oth", oth_in)
        train = (rest_w if rest_in else 0) + (g_w if g_in else 0) + (oth_w if oth_in else 0)
        removed = (0 if rest_in else rest_w) + (0 if g_in else g_w) + (0 if oth_in else oth_w)
        if key == "C2":
            # random rows from outside F, drawn as six slivers across G and the other families; total width = the rest of F
            k = 6
            sliver_w = rest_w / k
            region0, region_w = xg, g_w + oth_w
            cell = region_w / k
            jitter = [-0.25, 0.20, -0.10, 0.30, -0.30, 0.15]
            for j in range(k):
                cx = region0 + (j + 0.5) * cell + jitter[j]
                assert region0 < cx - sliver_w / 2 and cx + sliver_w / 2 < region0 + region_w
                ax.add_patch(Rectangle((cx - sliver_w / 2, y), sliver_w, row_h, fc="white", ec=REMOVED_EDGE, lw=0.8, ls=DASH, zorder=3))
            train -= rest_w
            removed += rest_w
            assert abs(k * sliver_w - rest_w) < 1e-9
        ledger[key] = (train, removed)

    # the design's own identities, asserted on the geometry that was drawn
    assert abs(ledger["C1"][1] - rest_w) < 1e-9, "C1 must remove exactly the rest of F"
    assert abs(ledger["C2"][0] - ledger["C1"][0]) < 1e-9, "C2 must have exactly C1's training size"
    assert abs(ledger["C3"][1] - g_w) < 1e-9, "C3 must remove exactly G"
    assert abs(ledger["SP"][0] - rest_w) < 1e-9, "the specialist must keep only the rest of F"
    assert abs(ledger["C0"][0] - (B - gap_test - test_w)) < 1e-9 and ledger["C0"][1] == 0

    handles = [
        Patch(fc=C_TEST, ec="white", label="family F: test fold (predicted, never in training)"), Patch(fc=C_REST, ec="white", label="family F: rest"),
        Patch(fc=C_G, ec="white", label="family G"), Patch(fc=C_OTHER, ec="white", label="other families"),
        Patch(fc="white", ec=REMOVED_EDGE, ls=DASH, lw=0.8, label="removed from training"),
    ]
    leg = ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, handlelength=1.3,
                    handleheight=0.9, columnspacing=1.0, handletextpad=0.5, fontsize=8, borderaxespad=0.1)
    _check_legend_inside(fig, leg)
    fs.check_text_fits(fig, ax, checks)
    fs.save(fig, out_path, expect_width_cm=16.0)
    plt.close(fig)
    return {"n_folds": n_folds, "test_fraction_of_F": test_w / f_tot}


# ---------------------------------------------------------------------------------------------------------------
# Figure 2: family coverage
# ---------------------------------------------------------------------------------------------------------------

def make_family_coverage(out_path, facts):
    summary = facts["summary"].set_index("family")
    fams = [f for f in facts["summary"].sort_values("rows_all", ascending=False).family if f not in pf.NEVER_HELD_OUT]
    apart = [f for f in facts["summary"].sort_values("rows_all", ascending=False).family if f in pf.NEVER_HELD_OUT]
    assert len(fams) == 27 and len(apart) == 2
    min_rows = facts["min_rows"]
    XMIN, XMAX = 100, 100_000
    offsets = np.array([-0.30, -0.10, 0.10, 0.30])
    bar_h = 0.17

    fig, (ax_a, ax_b) = fs.new_figure("double", 22.4, 2, 1, sharex=True, height_ratios=[27, 2.45])
    hollow_count = {t: 0 for t in pf.TARGETS}
    dagger_count = {t: 0 for t in pf.TARGETS}
    min_clusters = facts["min_clusters"]
    for ax, names in ((ax_a, fams), (ax_b, apart)):
        ax.set_xscale("log")
        ax.set_xlim(XMIN, XMAX)
        fs.grid(ax, "x")
        for i, fam in enumerate(names):
            for t, off in zip(pf.TARGETS, offsets):
                rows = int(summary.loc[fam, f"rows_{t}"])
                assert rows > XMIN, (fam, t, rows)
                qual = str(summary.loc[fam, f"qualifies_{t}"]) == "True"
                colour = TARGET_COLOURS[t]
                held_out_applies = fam not in pf.NEVER_HELD_OUT
                hollow = held_out_applies and not qual
                hollow_count[t] += int(hollow)
                ax.barh(i + off, rows - XMIN, left=XMIN, height=bar_h, fc="white" if hollow else colour, ec=colour, lw=0.7, zorder=2)
                if hollow and rows >= min_rows:
                    # enough rows but too few clusters: marked, so the hollow bar that crosses the row line is explained
                    assert int(summary.loc[fam, f"clusters_{t}"]) < min_clusters, (fam, t)
                    x_dag = rows * 1.12
                    assert x_dag > min_rows * 1.05, (fam, t)             # clear of the dashed threshold line
                    ax.text(x_dag, i + off, "†", ha="left", va="center_baseline", fontsize=8, color="0.1", zorder=4)
                    dagger_count[t] += 1
        ax.set_ylim(len(names) - 0.5, -0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([pf.FAMILY_NAMES[f][1] for f in names])
        ax.tick_params(axis="y", length=0)
        ax.axvline(min_rows, color="black", lw=0.9, ls=DASH, zorder=3)
    for t in pf.TARGETS:
        assert hollow_count[t] == 27 - facts["qualifying"][t], (t, hollow_count[t], facts["qualifying"][t])
        assert dagger_count[t] == len(facts["failing"][t]["clusters_only"]), (t, dagger_count[t])
    ax_b.set_xlabel("Rows with a value for the property (log scale)")
    fs.panel_title(ax_a, "Families, held out where they qualify", letter="a")
    fs.panel_title(ax_b, "Never held out: always in the training pool", letter="b")
    ax_a.set_xticks([1e2, 1e3, 1e4, 1e5])
    ax_b.set_xticks([1e2, 1e3, 1e4, 1e5])
    ax_b.set_xticklabels(["100", "1,000", "10,000", "100,000"])
    tgt = [Patch(fc=TARGET_COLOURS[t], ec=TARGET_COLOURS[t], label=pf.TARGET_SYMBOL[t]) for t in pf.TARGETS]
    no_qual = Patch(fc="white", ec="0.3", label="does not qualify")
    dagger = Line2D([], [], color="none", marker=r"$\dagger$", mfc="0.1", mec="0.1", ms=5, ls="none",
                    label=f"{min_rows:,} rows or more, fewer than {min_clusters} clusters")
    line = Line2D([], [], color="black", lw=0.9, ls=DASH, label=f"{min_rows:,} rows")
    # property colours inside panel a, bottom right, where the short bars of the small families leave the axes empty;
    # the three markers in one row below the axes
    leg_t = ax_a.legend(tgt, [h.get_label() for h in tgt], loc="lower right", ncol=4, frameon=False, borderaxespad=0.3)
    leg = fs.legend_row(fig, [no_qual, dagger, line], [h.get_label() for h in (no_qual, dagger, line)], ncol=3)
    fig.canvas.draw()
    lb = leg_t.get_window_extent()
    inv = ax_a.transData.inverted()
    # the property legend must not touch any bar: every bar in the rows it covers ends left of it
    (x_left, _), (_, _) = inv.transform((lb.x0, lb.y0)), inv.transform((lb.x1, lb.y1))
    rows_cov = [i for i in range(len(fams)) if ax_a.transData.transform((100, i))[1] < lb.y1 + 8]
    assert rows_cov, "legend covers no rows"
    for i in rows_cov:
        assert max(int(summary.loc[fams[i], f"rows_{t}"]) for t in pf.TARGETS) < x_left, ("legend overlaps bars of", fams[i])
    _check_legend_inside(fig, leg)
    fs.save(fig, out_path, expect_width_cm=16.0)
    plt.close(fig)
    return {"hollow_bars": hollow_count, "dagger_bars": dagger_count}


# ---------------------------------------------------------------------------------------------------------------
# Figure 3: qualification map
# ---------------------------------------------------------------------------------------------------------------

def make_qualification_map(out_path, facts):
    summary = facts["summary"].set_index("family")
    min_c, min_r = facts["min_clusters"], facts["min_rows"]

    def ordered(members):
        return sorted(members, key=lambda f: -int(summary.loc[f, "rows_all"]))

    lines = []        # ("union"|"member"|"header", key, label)
    for sname in ("iv_vi", "oxides", "zintl"):
        lines.append(("union", sname, f"{pf.SUPER_NAMES[sname][1]} super-family (union)"))
        lines += [("member", f, pf.FAMILY_NAMES[f][1]) for f in ordered(facts["super_members"][sname])]
    lines.append(("header", None, "Other families (standalone)"))
    lines += [("member", f, pf.FAMILY_NAMES[f][1]) for f in ordered(facts["standalone"])]
    assert sum(1 for k, _, _ in lines if k == "member") == 27

    W, label_w = 16.0, 5.6
    col_w = (W - label_w) / 4
    line_h, head_h, leg_h = 0.5, 1.15, 0.95
    H = head_h + len(lines) * line_h + leg_h + 0.1
    fig = plt.figure(figsize=(W * fs.CM, H * fs.CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    checks = []

    ax.text(label_w + 2 * col_w, H - 0.3, "rows (clusters) with a value for the property", ha="center", va="center", fontsize=8, style="italic", color="0.2")
    for j, t in enumerate(pf.TARGETS):
        tx = label_w + (j + 0.5) * col_w
        ax.text(tx, H - 0.85, pf.TARGET_SYMBOL[t], ha="center", va="center", fontsize=9, fontweight="bold", color="0.1")

    qualified_members = {t: 0 for t in pf.TARGETS}
    qualified_standalone = {t: 0 for t in pf.TARGETS}
    qualified_unions = 0
    standalone = set(facts["standalone"])
    for i, (kind, key, label) in enumerate(lines):
        y = H - head_h - (i + 1) * line_h
        indent = 0.15 if kind in ("union", "header") else 0.5
        weight = "bold" if kind in ("union", "header") else "normal"
        t = ax.text(indent, y + line_h / 2, label, ha="left", va="center", fontsize=8, fontweight=weight, color="0.1")
        checks.append((t, (0, y, label_w, y + line_h)))
        if kind == "header":
            continue
        for j, tgt in enumerate(pf.TARGETS):
            x = label_w + j * col_w
            if kind == "union":
                u = facts["supers"][key]
                rows, clusters, qual = u["rows"][tgt], u["clusters"][tgt], u["qualifies"][tgt]
                qualified_unions += int(qual)
            else:
                rows, clusters = int(summary.loc[key, f"rows_{tgt}"]), int(summary.loc[key, f"clusters_{tgt}"])
                qual = str(summary.loc[key, f"qualifies_{tgt}"]) == "True"
                qualified_members[tgt] += int(qual)
                if key in standalone:
                    qualified_standalone[tgt] += int(qual)
            assert qual == (clusters >= min_c and rows >= min_r), (key, tgt)
            ax.add_patch(Rectangle((x + 0.03, y + 0.03), col_w - 0.06, line_h - 0.06, fc=QUALIFIES_FILL if qual else FAILS_FILL,
                                   ec="0.2" if kind == "union" else "none", lw=0.9 if kind == "union" else 0))
            cx = x + col_w / 2
            # the failing count(s) in vermillion, so the reason is visible: rows below the row threshold, clusters below the cluster one
            parts = [(f"{rows:,}", rows < min_r), (" (", False), (f"{clusters:,}", clusters < min_c), (")", False)]
            xs_ = _place_parts(ax, fig, cx, y + line_h / 2, parts, bold=(kind == "union"), checks=checks, box=(x, y, x + col_w, y + line_h))
    assert qualified_members == facts["qualifying"], (qualified_members, facts["qualifying"])
    assert qualified_standalone == facts["standalone_qualifying"], (qualified_standalone, facts["standalone_qualifying"])
    assert qualified_unions == facts["splits_cfg"]["n_super_family_pairs"] == 12

    handles = [Patch(fc=QUALIFIES_FILL, ec="none", label=f"qualifies (at least {min_c} clusters and {min_r:,} rows)"),
               Patch(fc=FAILS_FILL, ec="none", label="does not qualify"),
               Line2D([], [], color=fs.OI["vermillion"], lw=0, marker="s", ms=0, label="below-threshold count")]
    leg_items = [(handles[0], handles[0].get_label()), (handles[1], handles[1].get_label())]
    leg = ax.legend([h for h, _ in leg_items], [l for _, l in leg_items], loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2,
                    frameon=False, handlelength=1.3, handleheight=0.9, columnspacing=1.6, fontsize=8, borderaxespad=0.1)
    _check_legend_inside(fig, leg)
    fs.check_text_fits(fig, ax, checks)
    fs.save(fig, out_path, expect_width_cm=16.0)
    plt.close(fig)
    return {"members_qualifying": qualified_members, "standalone_qualifying": qualified_standalone, "union_cells_qualifying": qualified_unions}


def _place_parts(ax, fig, cx, cy, parts, bold, checks, box):
    """Draw consecutive text parts (each with its own colour) centred on cx; fits are checked per part."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    texts, widths = [], []
    for s, red in parts:
        t = ax.text(0, cy, s, ha="left", va="center", fontsize=8, fontweight="bold" if (bold or red) else "normal",
                    color=fs.OI["vermillion"] if red else "0.1")
        bb = t.get_window_extent(renderer)
        w = inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]
        texts.append(t)
        widths.append(w)
    x = cx - sum(widths) / 2
    for t, w in zip(texts, widths):
        t.set_x(x)
        x += w
        checks.append((t, box))
    return texts


# ---------------------------------------------------------------------------------------------------------------
# Figure 4: error-decomposition schematic (synthetic data)
# ---------------------------------------------------------------------------------------------------------------

def murphy_terms(y, y_hat):
    """The three terms of the draft's decomposition with population standard deviations; they sum to the MSE exactly."""
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    r = np.corrcoef(y, y_hat)[0, 1]
    s_y, s_h = y.std(), y_hat.std()
    offset = (y_hat.mean() - y.mean()) ** 2
    scale = (s_h - r * s_y) ** 2
    lost = (1 - r**2) * s_y**2
    mse = np.mean((y_hat - y) ** 2)
    assert abs(offset + scale + lost - mse) < 1e-9, (offset, scale, lost, mse)
    return {"offset": offset, "scale": scale, "lost": lost, "mse": mse, "r": r}


def make_error_decomposition(out_path):
    rng = np.random.default_rng(7)
    n = 60
    y = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    panels = {
        "a": {"title": "Offset dominates (illustrative)", "y_hat": 1.0 + 0.95 * y + 0.25 * noise},
        "b": {"title": "Signal is lost (illustrative)", "y_hat": 0.25 * y + 0.75 * noise},
    }
    terms = {k: murphy_terms(y, p["y_hat"]) for k, p in panels.items()}
    for k in terms:
        total = terms[k]["mse"]
        terms[k]["shares"] = np.array([terms[k]["offset"], terms[k]["scale"], terms[k]["lost"]]) / total
        assert abs(terms[k]["shares"].sum() - 1) < 1e-9
    assert terms["a"]["shares"][0] > 0.85 and terms["a"]["r"] > 0.9, terms["a"]
    assert terms["b"]["shares"][2] > 0.7 and terms["b"]["shares"][0] < 0.1 and terms["b"]["r"] < 0.4, terms["b"]

    fig, axes = fs.new_figure("double", 8.4, 2, 2, height_ratios=[4.4, 0.9])
    seg_colours = (fs.OI["vermillion"], fs.OI["orange"], fs.OI["blue"])
    lim = (-3.3, 4.3)
    for col, (k, p) in enumerate(panels.items()):
        ax, bar = axes[0, col], axes[1, col]
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_xlabel("true value on the held-out family")
        ax.set_ylabel("predicted value")
        fs.panel_title(ax, p["title"], letter=k)
        ax.plot(lim, lim, color="0.55", lw=0.8, ls="--", zorder=1)
        ax.plot(y, p["y_hat"], ls="none", marker="o", ms=3.4, mfc=fs.OI["blue"], mec="white", mew=0.3, zorder=3)
        my, mh = y.mean(), p["y_hat"].mean()
        ax.plot([my], [mh], marker="P", ms=7, mfc=fs.OI["vermillion"], mec="white", mew=0.6, ls="none", zorder=4)
        if k == "a":
            ax.annotate("", xy=(my, mh), xytext=(my, my), arrowprops=dict(arrowstyle="<->", color=fs.OI["vermillion"], lw=0.9, shrinkA=0, shrinkB=2), zorder=4)
            ax.text(my + 0.2, (my + mh) / 2, "offset", color=fs.OI["vermillion"], fontsize=8, va="center", ha="left")
            ax.text(-3.1, 4.0, "predictions follow the true ranking\nbut sit above it", fontsize=8, va="top", ha="left", color="0.15")
        else:
            ax.text(-3.1, 4.0, "predictions barely follow\nthe true ranking", fontsize=8, va="top", ha="left", color="0.15")
        ax.text(4.1, -3.1, "dashed: predicted = true", fontsize=8, va="bottom", ha="right", color="0.4")

        sh = terms[k]["shares"]
        bar.set_xlim(0, 1)
        bar.set_ylim(0, 1)
        bar.set_yticks([])
        bar.set_xticks([])
        bar.grid(False)
        for spine in ("left", "bottom"):
            bar.spines[spine].set_visible(False)
        left = 0.0
        for share, colour in zip(sh, seg_colours):
            bar.barh(0.5, share, left=left, height=0.8, fc=colour, ec="white", lw=0.6)
            left += share
        bar.set_xlabel(f"offset {sh[0]:.0%}   scale {sh[1]:.0%}   lost signal {sh[2]:.0%}   (r = {terms[k]['r']:.2f})", fontsize=8)
    handles = [Patch(fc=c, label=l) for c, l in zip(seg_colours, ("family offset", "scale error", "lost within-family signal"))]
    handles.append(Line2D([], [], marker="P", ms=6, mfc=fs.OI["vermillion"], mec="white", ls="none", label="family mean (true, predicted)"))
    leg = fs.legend_row(fig, handles, [h.get_label() for h in handles])
    _check_legend_inside(fig, leg)
    fs.save(fig, out_path, expect_width_cm=16.0)
    plt.close(fig)
    return {k: {"shares": [float(x) for x in terms[k]["shares"]], "r": float(terms[k]["r"])} for k in terms}


def main():
    fs.apply()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    facts = pf.load_facts()
    out = {}
    out[1] = make_study_design(FIGURES_DIR / FIG_STEMS[1], facts)
    out[2] = make_family_coverage(FIGURES_DIR / FIG_STEMS[2], facts)
    out[3] = make_qualification_map(FIGURES_DIR / FIG_STEMS[3], facts)
    out[4] = make_error_decomposition(FIGURES_DIR / FIG_STEMS[4])
    for n, stem in FIG_STEMS.items():
        print(f"Figure {n}: saved {stem}.png / .pdf  {out[n]}")
    return out


if __name__ == "__main__":
    main()
