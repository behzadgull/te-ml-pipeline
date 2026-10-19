"""
NA12: electrical conductivity and Seebeck coefficient of LaRhO3 and its Ni- and Mg-substituted members from Shibasaki, Takahashi & Terasaki 2009 (J. Electron. Mater. 38(7):1013-1016,
DOI 10.1007/s11664-009-0666-x), read from the vector figures of the PDF.
Rules: docs/decisions.md, pre-registration B with amendment 1 (stated temperature), amendment 2 (aggregation), amendment 14 (which values are rows) and amendment 15 (a value below 1% of the axis range of Fig. 5 is taken from Figs. 2 and 4). No out-of-fold prediction is read or used here.

Rows (amendment 14): Fig. 5 (PDF page 4), whose caption states 300 K: the conductivity (panel a, 1/rho in ohm^-1 cm^-1 = S/cm) and S (panel b, uV/K) of every marker, Ni series (filled symbols,
x = 0, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3) and Mg series (open symbols, x = 0, 0.03, 0.05, 0.1, 0.2), at T = 300 K as stated. Only undoped LaRhO3 (x = 0, one marker in each series, both with
the same stated sample so that amendment 2 aggregates them) is compared; every doped member is recorded with the token [not-compared: doped member, amendment 14]. The power factor of panel c
is derived and is only a check. Amendment 15: a Fig. 5 value below 1% of its axis range is unreadable; that is the case of the undoped conductivity, which is then taken from the end markers of the x = 0 series
of Figs. 2 and 4 (resistivity in ohm cm, T = 300 K as stated), and the Fig. 5 value stays recorded with a [not-compared] token. Cross-check, reported and not asserted (amendment 14): for each composition shared with Fig. 2 (Ni series, PDF page 2) or Fig. 4 (Mg series, PDF page 3), the marker
of that series nearest 300 K (the text gives 4.2 to 300 K; the last markers are drawn at 297.9 to 306.6 K) against the Fig. 5 value, in the same units (the resistivity axes of Figs. 2 and 4 are in ohm cm, labelled with the symbol of the ohm and no prefix: sigma in S/cm = 1 / rho).

Each marker is a filled path of the content stream (pdf_vector.py); its centre is the centre of its bounding box. The axes are calibrated on the printed major ticks and the tick labels (the
temperature axis of Figs. 2 and 4 on 0, 100, 200, 300 K; the composition axis of Fig. 5 on 0, 0.1, 0.2, 0.3; linear value axes on their labelled ticks; the logarithmic resistivity axes on the decade
exponents); legend symbols are not data. Series are told apart by the fill (Fig. 5: filled = Ni, open = Mg, checked against the legend; Figs. 2 and 4: the colour of the legend symbol of each x).

Validation (asserted): the frame edges of every panel are found among the stroked segments; each axis fit has a maximum residual below 0.02 label steps; every Fig. 5 marker lies within 0.006 of a
composition of its series; the series have 8 (Ni) and 5 (Mg) markers in each of panels a to c; the legend maps filled to Ni and open to Mg; the DOI printed on page 1 is the one of the values.

Usage (repository root, clean tree):
    python thesis_paper/scripts/na12_extract_shibasaki.py                        # writes thesis_paper/results/na12_extraction/<UTC>/
    python thesis_paper/scripts/na12_extract_shibasaki.py --append-values <dir>  # appends the rows of that run to docs/na12_literature_values.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import na12_plot_tools as T  # noqa: E402
import pdf_vector as pv  # noqa: E402
import run_record as rr  # noqa: E402

PDF = "thesis_paper/source/pdfs/shibasaki2009.pdf"
VALUES = REPO / "thesis_paper" / "docs" / "na12_literature_values.csv"
DOI = "10.1007/s11664-009-0666-x"
JOURNAL = "J. Electron. Mater. 38(7):1013-1016 (2009)"
ID_PREFIX = "SH2009_"
COLUMNS = ["lit_id", "compound_label", "composition", "property", "T_K", "value", "unit", "doi", "location", "value_kind", "digitisation_method", "notes"]
T_STATED = 300.0  # the caption of Fig. 5: "at 300 K"
NI_X = (0.0, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
MG_X = (0.0, 0.03, 0.05, 0.1, 0.2)
UNDOPED_LABEL = "LaRhO3 (undoped, x = 0, as stated for Fig. 5)"
NOT_COMPARED = "[not-compared: doped member, amendment 14]"
NOT_COMPARED_UNREADABLE = "[not-compared: unreadable in Fig. 5 (within 1% of the axis range), replaced by Figs. 2 and 4, amendment 15]"
UNREADABLE_FRACTION = 0.01  # amendment 15
UNDOPED_LABEL_FIG24 = "LaRhO3 (undoped, x = 0, as stated for Figs. 2 and 4)"
# frames (x0, y0, x1, y1) of the panels, in PDF points; every edge is asserted to be among the stroked segments
FIG5 = {"a": (74.6, 624.8, 273.1, 709.8), "b": (74.6, 539.8, 273.1, 624.8), "c": (74.6, 454.7, 273.1, 539.8)}
FIG2 = {"a": (341.1, 307.1, 537.0, 447.0), "b": (341.1, 167.2, 537.0, 307.1)}  # resistivity (log), S; Ni series
FIG4 = {"a": (354.1, 568.7, 551.9, 709.9), "b": (354.1, 427.4, 551.9, 568.7)}  # resistivity (log), S; Mg series
PANEL_UNIT = {"Fig. 5a": "S/cm", "Fig. 5b": "uV/K"}


def page_data(pdf, n):
    objs, content, res, _ = pv.page(pdf, n)
    paths, texts = pv.interpret(objs, content, res)
    return paths, texts, T.segments(paths)


def check_frame(segs, f, tol=0.15):
    """The four edges of the frame are long stroked segments."""
    x0, y0, x1, y1 = f
    h = [s for s in segs if abs(s[1] - s[3]) < 1e-6 and abs(s[2] - s[0]) > 60]
    v = [s for s in segs if abs(s[0] - s[2]) < 1e-6 and abs(s[3] - s[1]) > 60]
    assert any(abs(s[1] - y0) < tol and min(s[0], s[2]) <= x0 + tol and max(s[0], s[2]) >= x1 - tol for s in h), ("bottom edge", f)
    assert any(abs(s[1] - y1) < tol and min(s[0], s[2]) <= x0 + tol and max(s[0], s[2]) >= x1 - tol for s in h), ("top edge", f)
    assert any(abs(s[0] - x0) < tol and min(s[1], s[3]) <= y0 + tol and max(s[1], s[3]) >= y1 - tol for s in v), ("left edge", f)
    assert any(abs(s[0] - x1) < tol and min(s[1], s[3]) <= y0 + tol and max(s[1], s[3]) >= y1 - tol for s in v), ("right edge", f)


def match_labels(ticks, labels):
    """Tick positions (pt) matched to labels [(baseline y, value)]: each label's tick is the one nearest to its baseline plus an offset; the offset is the one that gives the best straight line."""
    best = None
    for k in range(-40, 80):
        off = k * 0.1
        tk = [min(ticks, key=lambda y: abs(y - (ly + off))) for ly, _ in labels]
        if len(set(tk)) != len(tk):
            continue
        a, b, r = T.linear_fit(tk, [v for _, v in labels])
        if best is None or r < best[0]:
            best = (r, off, a, b, tk)
    assert best is not None
    return best


def x_calibration(segs, frame, labels_text, values):
    xt = []
    for t in T.bottom_ticks(segs, frame, lo=3.8, hi=5.0):  # a few short strokes near the left end of one frame (Fig. 4b) are not ticks: the major ticks are at least 30 pt apart
        if not xt or t - xt[-1] >= 30.0:
            xt.append(t)
    assert len(xt) == len(values), (len(xt), values)
    a, b, r = T.linear_fit(xt, list(values))
    assert r < 0.02, r
    return {"a": a, "b": b, "residual_label_steps": r, "ticks_pt": xt, "labels": labels_text}


def y_calibration(segs, texts, frame, log=False, expect=None):
    """Linear (labels = values) or logarithmic (the labels are the decade exponents) value axis of a panel; returns value = a * y + b (log10 of the value for a log axis)."""
    x0, y0, x1, y1 = frame
    yt = [y for y in T.left_ticks(segs, frame, lo=3.8, hi=5.0) if y0 - 0.6 <= y <= y1 + 0.6]  # the ticks of this panel only (the segments of the panel above or below end on the same frame edge)
    lab = []
    for t in texts:
        v = T.number(t["text"])
        if v is None or not (x0 - 34 <= t["x"] <= x0 - 1 and y0 - 9 <= t["y"] <= y1 + 9):
            continue
        small = t["size"] < 7.5
        if small == log:  # a log axis uses the small exponent strings, a linear axis the full-size labels
            lab.append((t["y"], v))
    if expect is not None:  # a linear axis with known labels: each expected value is the candidate nearest to the end of the frame it belongs to (0 at the bottom, the largest at the top), so that
        chosen = []         # the labels of the panel below or above (also 0 and the largest value) are not taken
        vmin, vmax = min(expect), max(expect)
        for v in expect:
            cand = [(y, w) for y, w in lab if abs(w - v) < 1e-9]
            assert cand, ("label missing", v, frame)
            ref = y0 if v == vmin else (y1 if v == vmax else (y0 + y1) / 2)
            chosen.append(min(cand, key=lambda c: abs(c[0] - ref)))
        lab = chosen
    lab.sort()
    assert len(lab) >= 4, ("too few value labels", frame, lab)
    r, off, a, b, tk = match_labels(yt, lab)
    assert r < 0.02, ("value axis fit", r, frame)
    return {"a": a, "b": b, "residual_label_steps": r, "label_offset_pt": off, "ticks_pt": tk, "labels": [v for _, v in lab], "log": log}


def centre_markers(paths, frame, min_size=3.5, max_size=6.0):
    """Filled marker paths inside the frame: [(fill, cx, cy)]."""
    x0, y0, x1, y1 = frame
    out = []
    for p in paths:
        if not p["kind"].startswith("f"):
            continue
        pts = [q for sp in p["pts"] for q in sp]
        if not pts:
            continue
        xs = [a for a, b in pts]
        ys = [b for a, b in pts]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        if min_size <= w <= max_size and min_size <= h <= max_size and x0 < cx < x1 and y0 < cy < y1:
            out.append((tuple(round(c, 2) for c in p["color"]), cx, cy))
    return out


def legend_symbols(markers, texts, x_rx=r"^\s*(?:x|A)\s*=\s*(\S+)", dx=(8.0, 15.0), dy=(1.0, 6.5)):
    """The legend entries ('x = 0.03', 'A = Ni') with the fill of their symbol and the position of the symbol: {label: (fill, cx, cy)}."""
    out = {}
    for t in texts:
        m = re.match(x_rx, t["text"])
        if not m:
            continue
        cand = [(f, cx, cy) for f, cx, cy in markers if t["x"] - dx[1] <= cx <= t["x"] - dx[0] and t["y"] + dy[0] <= cy <= t["y"] + dy[1]]
        if cand:
            out[m.group(1)] = min(cand, key=lambda c: abs(c[2] - (t["y"] + 4.0)))
    return out


def fig5(pdf):
    paths, texts, segs = page_data(pdf, 4)
    for f in FIG5.values():
        check_frame(segs, f)
    # composition axis: major ticks 0, 0.1, 0.2, 0.3 (the labels are the run '00.10.20.3' of the lowest panel)
    run = [t for t in texts if t["text"].strip() == "00.10.20.3"]
    assert len(run) == 1
    xcal = x_calibration(segs, FIG5["c"], "0, 0.1, 0.2, 0.3", (0.0, 0.1, 0.2, 0.3))
    allm = []
    for name, f in FIG5.items():
        allm += [(name, *m) for m in centre_markers(paths, f)]
    leg = legend_symbols([m[1:] for m in allm if m[0] == "a"], texts)
    assert set(leg) == {"Ni", "Mg"}, leg
    assert leg["Ni"][0] != (1.0, 1.0, 1.0) and leg["Mg"][0] == (1.0, 1.0, 1.0), "the legend maps a filled symbol to Ni and an open one to Mg"
    leg_pos = {(round(v[1], 1), round(v[2], 1)) for v in leg.values()}
    ycal = {"a": y_calibration(segs, texts, FIG5["a"], expect=[0.0, 10.0, 20.0, 30.0, 40.0]), "b": y_calibration(segs, texts, FIG5["b"], expect=[0.0, 100.0, 200.0, 300.0, 400.0]),
            "c": y_calibration(segs, texts, FIG5["c"], expect=[0.0, 100.0, 200.0, 300.0, 400.0])}
    series = {"Ni": {n: [] for n in "abc"}, "Mg": {n: [] for n in "abc"}}
    for name, fill, cx, cy in allm:
        if (round(cx, 1), round(cy, 1)) in leg_pos:
            continue
        x = xcal["a"] * cx + xcal["b"]
        ser = "Mg" if fill == (1.0, 1.0, 1.0) else "Ni"
        allowed = MG_X if ser == "Mg" else NI_X
        xs = min(allowed, key=lambda v: abs(v - x))
        assert abs(xs - x) <= 0.006, (name, ser, x)
        series[ser][name].append((xs, ycal[name]["a"] * cy + ycal[name]["b"], x, cx, cy))
    for ser, xs in (("Ni", NI_X), ("Mg", MG_X)):
        for name in "abc":
            got = sorted(m[0] for m in series[ser][name])
            assert got == sorted(xs), (ser, name, got)
    for ser in series:
        for name in series[ser]:
            series[ser][name].sort()
    return {"x": xcal, "y": ycal, "legend": {k: list(v) for k, v in leg.items()}, "series": series}


def curve_axes(pdf, page, frames, ser):
    """Figs. 2 and 4: the temperature axis (0, 100, 200, 300 K), the S axis (linear) and the resistivity axis (log); the series of each x by the colour of the legend symbol; markers per series."""
    paths, texts, segs = page_data(pdf, page)
    for f in frames.values():
        check_frame(segs, f)
    runs = [t for t in texts if t["text"].strip() == "0100200300"]
    assert runs, "temperature label run not found"
    xcal = x_calibration(segs, frames["b"], "0, 100, 200, 300", (0.0, 100.0, 200.0, 300.0))
    ycal = {"a": y_calibration(segs, texts, frames["a"], log=True), "b": y_calibration(segs, texts, frames["b"], expect=[0.0, 100.0, 200.0, 300.0, 400.0])}
    markers = {n: centre_markers(paths, f) for n, f in frames.items()}
    leg = legend_symbols([m for ms in markers.values() for m in ms], texts, dx=(8.0, 13.0))  # the legend sits in panel a (Fig. 2) or b (Fig. 4)
    assert sorted(float(k) for k in leg) == sorted(ser), (sorted(leg), ser)
    fill_of = {float(k): v[0] for k, v in leg.items()}
    assert len(set(fill_of.values())) == len(fill_of), "legend symbols of one figure must differ in colour"
    out = {}
    for n in "ab":
        leg_pos = {(round(v[1], 1), round(v[2], 1)) for v in leg.values()}
        for x, fill in fill_of.items():
            ms = [m for m in markers[n] if m[0] == fill and (round(m[1], 1), round(m[2], 1)) not in leg_pos]
            pts = sorted((xcal["a"] * cx + xcal["b"], ycal[n]["a"] * cy + ycal[n]["b"]) for _, cx, cy in ms)
            out.setdefault(x, {})[n] = pts
    return {"x": xcal, "y": ycal, "legend_fill": {str(k): list(v) for k, v in fill_of.items()}, "series": out}


def nearest_300(pts):
    """The marker of a series nearest to 300 K: (T, value)."""
    return min(pts, key=lambda p: abs(p[0] - T_STATED))


def crosscheck(f5, f2, f4):
    """Fig. 2 (Ni) and Fig. 4 (Mg) marker nearest 300 K against the Fig. 5 value of the same composition and series; reported, not asserted (ratio flag outside 0.8 to 1.25)."""
    rec = []
    for ser, fig, axes in (("Ni", "Fig. 2", f2), ("Mg", "Fig. 4", f4)):
        five = {n: {m[0]: m[1] for m in f5["series"][ser][n]} for n in "ab"}
        for x, d in sorted(axes["series"].items()):
            if x not in five["a"]:
                continue
            for panel, five_panel, prop in (("a", "a", "sigma"), ("b", "b", "S")):
                Tm, v = nearest_300(d[panel])
                if prop == "sigma":
                    v = 1.0 / (10.0 ** v)  # log10 rho (ohm cm) -> sigma (S/cm)
                ref = five[five_panel][x]
                ratio = v / ref
                rec.append({"series": ser, "x": x, "property": prop, "figure": fig, "marker_T_K": round(Tm, 1), "n_markers_of_series": len(d[panel]), "fig5_value": ref, "end_marker_value": v,
                            "ratio_end_marker_over_fig5": ratio, "flag": bool(not 0.8 <= ratio <= 1.25)})
    return rec


def pf_check(f5):
    """Panel c is the power factor S^2 sigma in nW cm^-1 K^-2 (S in uV/K, sigma in S/cm: S^2 sigma = X * 1e-3 nW/cm K^2); the three panels of a marker must agree (reported, not asserted)."""
    out = []
    for ser in ("Ni", "Mg"):
        a = {m[0]: m[1] for m in f5["series"][ser]["a"]}
        b = {m[0]: m[1] for m in f5["series"][ser]["b"]}
        c = {m[0]: m[1] for m in f5["series"][ser]["c"]}
        for x in sorted(a):
            calc = b[x] ** 2 * a[x] * 1e-3
            ratio = calc / c[x] if c[x] else None
            out.append({"series": ser, "x": x, "ratio_S2sigma_over_panel_c": ratio, "flag": None if ratio is None else bool(not 0.8 <= ratio <= 1.25)})
    return out


def composition_of(ser, x):
    if x == 0.0:
        return "LaRhO3"
    return f"LaRh{1 - x:g}{ser}{x:g}O3"


def axis_range(cal):
    return max(cal["labels"]) - min(cal["labels"])


def make_rows(f5, f2, f4, cal_text):
    rows = []
    unreadable = []
    for panel in "ab":
        rng = axis_range(f5["y"][panel])
        for ser in ("Ni", "Mg"):
            for x, val, *_ in f5["series"][ser][panel]:
                if abs(val) < UNREADABLE_FRACTION * rng:
                    unreadable.append((panel, ser, x))
    assert sorted(unreadable) == [("a", "Mg", 0.0), ("a", "Ni", 0.0)], ("amendment 15 applies to the undoped conductivity of Fig. 5(a) and to nothing else", unreadable)
    loc = f"{JOURNAL}, Fig. 5 (article page 1016, PDF page 4)"
    for prop, panel, unit, what in (("sigma", "a", "S/cm", "the conductivity 1/rho of panel (a), ohm^-1 cm^-1 = S/cm"), ("S", "b", "uV/K", "the thermopower S of panel (b), uV/K")):
        for ser, fillname in (("Ni", "filled"), ("Mg", "open")):
            for x, val, xr, cx, cy in f5["series"][ser][panel]:
                undoped = x == 0.0
                comp = composition_of(ser, x)
                label = UNDOPED_LABEL if undoped else f"{comp} (x = {x:g}, {ser} series, as stated for Fig. 5)"
                notes = []
                if not undoped:
                    notes.append(NOT_COMPARED)
                if (panel, ser, x) in unreadable:
                    notes.append(NOT_COMPARED_UNREADABLE)
                notes.append(f"T = 300 K as stated in the caption of Fig. 5 (amendment 1, item 2; the window edge, not the room-temperature value); {ser} series ({fillname} symbols, legend 'A = {ser}'); marker at "
                             f"x = {xr:.4f} for the composition {x:g}")
                if undoped:
                    notes.append("one of two x = 0 markers of this figure (Ni and Mg series), recorded with the same stated sample so that amendment 2 aggregates them")
                if not undoped and x > 0.2 and ser == "Ni":
                    notes.append("Ni at x = 0.3 is 6 at%, a chemistry cluster of its own")
                rows.append({"lit_id": f"{ID_PREFIX}{prop}_{ser}_x{str(x).replace('.', 'p')}", "compound_label": label, "composition": comp, "property": prop, "T_K": f"{T_STATED:g}",
                             "value": f"{val:.3g}", "unit": unit, "doi": DOI, "location": loc + f", panel ({panel})", "value_kind": "digitised",
                             "digitisation_method": ("vector marker (filled path) read from the PDF content stream with thesis_paper/scripts/na12_extract_shibasaki.py; marker centre = centre of its bounding box; "
                                                     f"{what}; axes calibrated on the printed major ticks ({cal_text}); value to 3 significant digits"),
                             "notes": "; ".join(notes)})
    # amendment 15: the undoped resistivity from the end marker nearest 300 K of the x = 0 series of Fig. 2 (Ni series) and Fig. 4 (Mg series)
    rng_a = axis_range(f5["y"]["a"])
    for fig, ser, axes, page_txt, art in (("Fig. 2", "Ni", f2, "PDF page 2", "article page 1014"), ("Fig. 4", "Mg", f4, "PDF page 3", "article page 1015")):
        pts = axes["series"][0.0]["a"]
        Tm, lg = nearest_300(pts)
        assert abs(Tm - T_STATED) <= 3.0, (fig, Tm)
        five = [m[1] for m in f5["series"][ser]["a"] if m[0] == 0.0][0]
        rows.append({"lit_id": f"{ID_PREFIX}resistivity_{fig.replace('. ', '')}_x0p0", "compound_label": UNDOPED_LABEL_FIG24, "composition": "LaRhO3", "property": "resistivity", "T_K": f"{T_STATED:g}",
                     "value": f"{10.0 ** lg:.3g}", "unit": "ohm cm", "doi": DOI, "location": f"{JOURNAL}, {fig} ({art}, {page_txt}), panel (a)", "value_kind": "digitised",
                     "digitisation_method": ("vector marker (filled path) read from the PDF content stream with thesis_paper/scripts/na12_extract_shibasaki.py; marker centre = centre of its bounding box; the resistivity of "
                                             f"panel (a) on its logarithmic axis, calibrated on the decade ticks and exponent labels ({axes['y']['a']['labels'][0]:g} to {axes['y']['a']['labels'][-1]:g}, maximum residual "
                                             f"{axes['y']['a']['residual_label_steps']:.3f} label steps); the marker of the x = 0 series (black) nearest 300 K; value to 3 significant digits"),
                     "notes": (f"T = 300 K as stated in the text (measured from 4.2 K to 300 K; amendment 1, item 2); the end marker is drawn at {Tm:.1f} K (drawn offset recorded, not used); {ser} series, "
                               f"{len(pts)} markers in the series. Used instead of the Fig. 5 conductivity of the undoped sample ({five:.3g} ohm^-1 cm^-1, {100 * five / rng_a:.2f}% of the axis range of {rng_a:g}: "
                               "within 1%, unreadable; amendment 15)")})
    return rows


def build(pdf):
    f5 = fig5(pdf)
    f2 = curve_axes(pdf, 2, FIG2, (0.0, 0.03, 0.05, 0.1, 0.2, 0.3))
    f4 = curve_axes(pdf, 3, FIG4, (0.0, 0.03, 0.05, 0.1, 0.2))
    cal_text = ("composition axis 0, 0.1, 0.2, 0.3; panel a 0 to 40 ohm^-1 cm^-1 and panel b 0 to 400 uV/K, every labelled major tick, maximum residuals "
                f"{max(f5['y']['a']['residual_label_steps'], f5['y']['b']['residual_label_steps'], f5['x']['residual_label_steps']):.3f} label steps")
    rows = make_rows(f5, f2, f4, cal_text)
    # the DOI printed on page 1
    o, c, r, _ = pv.page(pdf, 1)
    _, t1 = pv.interpret(o, c, r)
    s1 = "".join(t["text"] for t in t1).replace(" ", "")
    assert f"DOI:{DOI}" in s1, "the DOI of page 1 is not the one of the values"
    cc = crosscheck(f5, f2, f4)
    pf = pf_check(f5)
    return {"fig5": {k: v for k, v in f5.items() if k != "series"} | {"n_markers": {s: {n: len(v) for n, v in f5["series"][s].items()} for s in f5["series"]}},
            "fig2": {k: v for k, v in f2.items() if k != "series"} | {"n_markers": {str(x): {n: len(p) for n, p in d.items()} for x, d in f2["series"].items()}},
            "fig4": {k: v for k, v in f4.items() if k != "series"} | {"n_markers": {str(x): {n: len(p) for n, p in d.items()} for x, d in f4["series"].items()}},
            "crosscheck_end_markers_vs_fig5": cc, "power_factor_check": pf, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=PDF)
    ap.add_argument("--append-values", default=None)
    args = ap.parse_args()
    if args.append_values:
        d = Path(args.append_values)
        cfg = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        assert cfg["tree_clean"], "that run was made from a dirty tree"
        with open(d / "values_rows.csv", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        with open(VALUES, encoding="utf-8", newline="") as f:
            have = {r["lit_id"] for r in csv.DictReader(f)}
        assert not (have & {r["lit_id"] for r in rows}), "rows of this run are already in the values file"
        assert VALUES.read_bytes().endswith(b"\n")
        with open(VALUES, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
            for r in rows:
                w.writerow(r)
        print(f"appended {len(rows)} rows from {d} to {VALUES}")
        return 0
    prov = rr.provenance({"pdf": args.pdf, "pdf_vector": "thesis_paper/scripts/pdf_vector.py", "na12_plot_tools": "thesis_paper/scripts/na12_plot_tools.py"}, __file__)
    res = build(REPO / args.pdf)
    out = rr.new_run_dir("na12_extraction")
    rr.write_json(out / "extraction.json", {k: v for k, v in res.items() if k != "rows"})
    with open(out / "values_rows.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(res["rows"])
    rr.write_json(out / "run_config.json", {**prov, "paper": "Shibasaki et al. 2009, Fig. 5 (rows), Figs. 2 and 4 (cross-check)", "rules": "docs/decisions.md pre-registration B + amendments 1, 2, 14, 15 (2026-10-13 to 19)"})
    cc = res["crosscheck_end_markers_vs_fig5"]
    print(f"wrote {out}: {len(res['rows'])} rows; cross-check {len(cc)} pairs, {sum(c['flag'] for c in cc)} flagged; power-factor check {len(res['power_factor_check'])} markers, "
          f"{sum(bool(c['flag']) for c in res['power_factor_check'])} flagged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
