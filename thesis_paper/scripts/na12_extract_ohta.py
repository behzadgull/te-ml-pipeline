"""
NA12: Seebeck coefficient (Fig. 1) and electrical conductivity (Fig. 2) of four heavily La- or Nb-doped SrTiO3 crystals from Ohta, Nomura, Ohta & Koumoto 2005
(J. Appl. Phys. 97:034106, DOI 10.1063/1.1847723).
Rules: docs/decisions.md, pre-registration B with amendments 1 (reading rules), 3 (Ohta compared at the cluster level, sample as stated) and 4 (column-consensus marker temperatures,
edge-row sensitivity). No out-of-fold prediction is read or used here.

The figures are bitmaps, digitised by hand in WebPlotDigitizer (one CSV per series and the project file). The clicks give the value of each marker; the temperature of a marker is the
consensus of its column (amendment 4): the marker centres are located in the figure image by a left-right symmetry fit of the column profile of the dark pixels, converted to temperature on
a straight line through ALL printed tick marks of the temperature axis, and averaged over the series that have a marker in that column.

Marker centre (all figures, amendment 5): the centroid of the enclosed white interior of an open symbol, or of the body of a filled symbol after a morphological opening that removes the
connecting line. A left-right symmetry fit of the column profile is computed as well and recorded as a cross-check (it is biased by the line at several markers of Fig. 2; the two agree
within 2.2 px on the 40 markers of Fig. 1); the markers where they differ by more than 3 px are listed in extraction.json.

Which series is which sample. The letters of the CSV file names must be the samples; the run stops unless the order of the series' first values agrees with the order of the
room-temperature values of Table II (read from the PDF text) and the fill of the symbols (filled or open) agrees with the sample's symbol in the figure (a open circle, b filled circle,
c open triangle, d filled triangle). In Fig. 2 the series are, top to bottom, b, d, c, a.

One temperature per column (amendment 5). The figures of one measurement run (Figs. 1 and 2) are given in one invocation; their columns are matched and each column gets one temperature, the
mean of the figures' column consensus weighted by 1/u^2 (u: calibration uncertainty), with uncertainty 1/sqrt(sum 1/u^2). Bins and edge flags use that value (an edge row: within 2 u of a 25 K bin edge, amendment 6).

Validation (asserted): the series have as many clicks as their CSV; a project file that holds a series agrees with its CSV; the temperature axis has 11 printed ticks (every 100 K) and a
straight-line fit through them (maximum residual below 2.5 K); the value axis ticks reproduce the clicked calibration (Fig. 1: every 0.02 mV/K, within 2 uV/K; Fig. 2: decades and minor
ticks of the log axis, within 0.01 decade); every refined centre lies inside its search window and a majority (at least 3 of 5) of the window settings agree within 3 px; markers of one
column lie within 5 K of each other; each series has at most one marker per column; Fig. 1: the first marker of every sample is within 15 uV/K of the room-temperature S of Table II and
has the larger |S|. Reported, not asserted (flags in extraction.json and in the notes of the rows): Fig. 2 first marker against Table II sigma, Table II sigma against n e mu of its own
row, and the match of the figures' columns (combined_columns in extraction.json).

Usage (repository root, clean tree):
    python thesis_paper/scripts/na12_extract_ohta.py --figure 1 <fig1 folder> <fig1 bitmap> --figure 2 <fig2 folder> <fig2 bitmap>     # writes thesis_paper/results/na12_extraction/<UTC>/
    python thesis_paper/scripts/na12_extract_ohta.py --append-values <dir> [--replace-prefix OH2005_]                                  # appends the rows of that run to docs/na12_literature_values.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import pdf_vector as pv  # noqa: E402
import run_record as rr  # noqa: E402

PDF = "thesis_paper/source/pdfs/ohta2005.pdf"
VALUES = REPO / "thesis_paper" / "docs" / "na12_literature_values.csv"
DOI = "10.1063/1.1847723"
JOURNAL = "J. Appl. Phys. 97:034106 (2005)"
COLUMNS = ["lit_id", "compound_label", "composition", "property", "T_K", "value", "unit", "doi", "location", "value_kind", "digitisation_method", "notes"]
THRESHOLD = 128  # dark pixel: grey level below this
MAX_COLUMN_RANGE_K = 5.0
EDGE_FACTOR = 2.0  # amendment 6: a row is an edge row when its temperature is within EDGE_FACTOR x u of a 25 K bin edge (about 95 % coverage)
TABLE2_TOL_UV = 15.0
SEARCH_PX = 16  # default half-width of the search window around a click (a figure may set its own: the clicks of Fig. 2 are up to about 20 px from the marker centre); the result must not touch its edge
WINDOW_ROWS = (-30, 21)  # rows around a click used for the column profile (the markers are about 45 px across; text sits directly under the base of some triangles)
# The symmetry fit of one marker is repeated with five (rows, half-width) settings; a marker line passing through an open triangle can give a second, spurious minimum in one setting.
# The centre is the mean of the estimates within MAX_ALT_DIFF_PX of the median, and at least MIN_AGREE of the five must be in that cluster, otherwise the run stops.
SETTINGS = [((-30, 21), 18), ((-26, 26), 18), ((-26, 26), 20), ((-30, 17), 18), ((-30, 21), 20)]
MAX_ALT_DIFF_PX = 3.0
MIN_AGREE = 3
SAMPLE_DOPING_NOTE = ("sample (b): doping printed 7.9e20 cm^-3 in the captions of Figs. 1 and 3 and 8.4e20 cm^-3 in Tables I and II (S, sigma and n of Table II agree with the latter); "
                      "recorded as in the tables")
SYMBOL_FILLED = {"a": False, "b": True, "c": False, "d": True}  # open or filled symbol of each sample in the figures

FIGURES = {
    1: {"prop": "S", "unit": "uV/K", "scale": 1000.0, "project": "s_project.json", "logy": False, "xticks": [200 + 100 * k for k in range(11)],
        "frame_rows": (1650, 1800), "frame_min": 1000, "x_range": (380, 1910), "x_band": (5, 15), "x_thr": 8, "fx_box": (0, 1700, 380, 420), "y_rows": (0, 1700), "search_px": 16, "estimator": "hole",
        "locate": "located in the image as the centroid of the enclosed white interior of an open symbol or of the opened body of a filled symbol (the connecting line is removed)",
        "location": f"{JOURNAL}, Fig. 1 (article page 034106-2, PDF page 3)", "page": 3, "res_note": "S to the digitiser's click resolution (4.8 uV/K lattice)",
        "id": "S", "method": "clicked S, mV/K converted to uV/K, 3 significant digits", "xlabel": "200 to 1200 K"},
    2: {"prop": "sigma", "unit": "S/cm", "scale": 1.0, "project": "sigma_project.json", "logy": True, "xticks": [300 + 100 * k for k in range(11)],
        "frame_rows": (2600, 2800), "frame_min": 1000, "x_range": (400, 2200), "x_band": (8, 18), "x_thr": 8, "fx_box": (200, 2700, 380, 480), "y_rows": (0, 2712), "search_px": 24, "estimator": "hole",
        "locate": "located in the image as the centroid of the enclosed white interior of an open symbol or of the opened body of a filled symbol (the connecting line is removed)",
        "location": f"{JOURNAL}, Fig. 2 (article page 034106-3, PDF page 4)", "page": 4, "res_note": "sigma read on the log axis from the click",
        "id": "sigma", "method": "clicked sigma in S/cm, 3 significant digits", "xlabel": "300 to 1300 K"},
}


def runs(mask, gap=3):
    """Centres of the runs of True indices (gaps <= gap are bridged)."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    parts = np.split(idx, np.where(np.diff(idx) > gap)[0] + 1)
    return [float(p.mean()) for p in parts]


def axis_ticks(dark, cfg):
    """Bottom frame row, left frame column, temperature-axis tick columns and value-axis tick rows (px), found next to the plot frame."""
    x0, x1 = cfg["x_range"]
    rows = dark[:, x0:x1].sum(axis=1)
    cand = [r for r in range(*cfg["frame_rows"]) if rows[r] > cfg["frame_min"]]
    assert cand, "bottom frame not found"
    fy = cand[len(cand) // 2]
    xt = runs(dark[fy + cfg["x_band"][0]: fy + cfg["x_band"][1], x0:x1].sum(axis=0) >= cfg["x_thr"])
    xt = [x + x0 for x in xt]
    r0, r1, c0, c1 = cfg["fx_box"]
    cols = dark[r0:r1, c0:c1].sum(axis=0)
    fx = int(np.argmax(cols)) + c0
    thr = 9 if not cfg["logy"] else 8
    y0, y1 = cfg["y_rows"]
    yt = runs(dark[y0:y1, fx - 20: fx - 4].sum(axis=1) >= thr)
    yt = [y + y0 for y in yt]
    return fy, fx, xt, yt


def load_csvs(folder, project):
    """{file letter: dict(file, dopant, n_cm3, name, pts [(T, value)])} from the CSVs, the project's axis calibration and the project's datasets (those it holds)."""
    out = {}
    for p in sorted(Path(folder).glob("*.csv")):
        m = re.fullmatch(r"([a-d])_(La|Nb)_(\d\.\d)e(\d\d)_(open|filled)_(circle|triangle)\.csv", p.name)
        assert m, f"unexpected CSV name {p.name}"
        pts = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    a, b = line.split(",")
                    pts.append((float(a), float(b)))
        out[m.group(1)] = {"file": p.name, "dopant": m.group(2), "n_cm3": f"{m.group(3)}e{m.group(4)}", "name": p.stem, "pts": pts}
    assert sorted(out) == ["a", "b", "c", "d"], sorted(out)
    for k, v in out.items():
        assert v["file"].startswith(k + "_"), v["file"]
    proj = json.loads((Path(folder) / project).read_text(encoding="utf-8"))
    ax = proj["axesColl"][0]
    assert not ax["isLogX"]
    cal = ax["calibrationPoints"]
    held, names = {}, {}
    for ds in proj["datasetColl"]:
        pts = [(p["value"][0], p["value"][1]) for p in ds["data"]]
        match = [k for k, v in out.items() if len(v["pts"]) == len(pts) and all(abs(x - c) < 1e-6 and abs(y - d) < 1e-9 * max(1.0, abs(y)) for (x, y), (c, d) in zip(v["pts"], pts))]
        assert len(match) == 1, f"project dataset {ds['name']} equals {len(match)} CSVs"
        held[match[0]] = [(p["x"], p["y"]) for p in ds["data"]]
        names[match[0]] = ds["name"]
    out["_project_dataset_names"] = names
    return out, cal, bool(ax["isLogY"]), held


def click_px(cal, logy, T, v):
    """Pixel position of a clicked value pair, by inverting the project's two-point axis calibration (no rotation: both temperature points share one pixel row)."""
    x1, x2, y1, y2 = cal
    assert abs(x1["py"] - x2["py"]) < 1e-6
    px = x1["px"] + (T - float(x1["dx"])) / (float(x2["dx"]) - float(x1["dx"])) * (x2["px"] - x1["px"])
    f = np.log10 if logy else (lambda z: z)
    py = y1["py"] + (f(v) - f(float(y1["dy"]))) / (f(float(y2["dy"])) - f(float(y1["dy"]))) * (y2["py"] - y1["py"])
    return float(px), float(py)


def _symmetry_centre(dark, cx, cy, rows, half, search):
    cx, cy = int(round(cx)), int(round(cy))
    prof = dark[cy + rows[0]: cy + rows[1], :].sum(axis=0).astype(float)
    grid = np.arange(prof.size)
    d = np.arange(1, half + 1)
    best = None
    for c in np.arange(cx - search, cx + search + 0.01, 0.5):
        cost = np.abs(np.interp(c + d, grid, prof) - np.interp(c - d, grid, prof)).sum()
        if best is None or cost < best[0]:
            best = (cost, float(c))
    return best[1]


def refine_x(dark, cx, cy, search=SEARCH_PX):
    """Marker centre (px): mean of the symmetry-fit estimates (SETTINGS) within MAX_ALT_DIFF_PX of their median; returns (centre, n_agreeing, spread of the agreeing ones)."""
    est = np.array([_symmetry_centre(dark, cx, cy, r, h, search) for r, h in SETTINGS])
    med = float(np.median(est))
    ok = est[np.abs(est - med) <= MAX_ALT_DIFF_PX]
    return float(ok.mean()), int(ok.size), float(ok.max() - ok.min())


def hole_centre(dark, cx, cy, filled, win=34):
    """Marker centre (x, y, area) that does not depend on the line joining the markers: the centroid of the enclosed white interior of an open symbol (the connecting line never enters
    it), or of the body of a filled symbol after a morphological opening with a disc of radius 11 px (wider than the line). None if no such region is near the click."""
    cx, cy = int(round(cx)), int(round(cy))
    sub = dark[cy - win: cy + win + 1, cx - win: cx + win + 1]
    best = None
    if not filled:
        lab, n = ndi.label(~sub)
        for k in range(1, n + 1):
            m = lab == k
            ys, xs = np.nonzero(m)
            touches = ys.min() == 0 or xs.min() == 0 or ys.max() == sub.shape[0] - 1 or xs.max() == sub.shape[1] - 1
            if touches or not 60 <= m.sum() <= 1500:
                continue
            d = float(np.hypot(xs.mean() - win, ys.mean() - win))
            if best is None or d < best[0]:
                best = (d, xs.mean(), ys.mean(), int(m.sum()))
    else:
        r = np.arange(-11, 12)
        op = ndi.binary_opening(sub, structure=(r[:, None] ** 2 + r[None, :] ** 2) <= 121)
        lab, n = ndi.label(op)
        for k in range(1, n + 1):
            m = lab == k
            ys, xs = np.nonzero(m)
            d = float(np.hypot(xs.mean() - win, ys.mean() - win))
            if best is None or d < best[0]:
                best = (d, xs.mean(), ys.mean(), int(m.sum()))
    if best is None:
        return None
    return float(cx - win + best[1]), float(cy - win + best[2]), best[3]


def interior_fill(dark, cx, cy):
    """Fraction of dark pixels in a 13 x 13 window at the marker centre (about 1 for a filled symbol, about 0 for an open one)."""
    cx, cy = int(round(cx)), int(round(cy))
    return float(dark[cy - 6: cy + 7, cx - 6: cx + 7].mean())


def table2(pdf):
    """Table II (room temperature) from the text of the PDF page 3: per sample a to d, sigma (S/cm), n (1e20 cm^-3), mu (cm2/Vs) and S (uV/K)."""
    objs, content, res, _ = pv.page(pdf, 3)
    _, texts = pv.interpret(objs, content, res)
    cells = sorted(((t["y"], t["text"]) for t in texts if re.search(r"\?0\.\d\d", t["text"]) and len(t["text"]) > 12), reverse=True)
    out = []
    for _, s in cells:
        m = re.match(r"^(\d+?)(\d\.\d)(\d\.\d)\?(0\.\d\d)", s)  # the PDF's '?' is the minus sign; S is negative in all cases (n-type, text)
        assert m, s
        out.append({"sigma_Scm": float(m.group(1)), "n_1e20": float(m.group(2)), "mu": float(m.group(3)), "S_uVK": -1000.0 * float(m.group(4))})
    assert len(out) == 4, out
    return out


def table1(pdf):
    """Table I (doping levels) from the text of the PDF page 3: {sample letter: (dopant, 'm.me19')}, samples a to d in the order of the columns."""
    objs, content, res, _ = pv.page(pdf, 3)
    _, texts = pv.interpret(objs, content, res)
    dop = sorted((t["x"], t["text"][:2]) for t in texts if re.fullmatch(r"(La|Nb)-doped", t["text"]) and 680 < t["y"] < 700)
    man = sorted((t["x"], t["text"]) for t in texts if re.fullmatch(r"\d\.\d", t["text"]) and 674 < t["y"] < 680)
    exp = sorted((t["x"], t["text"]) for t in texts if re.fullmatch(r"(19|20)", t["text"]) and 678 < t["y"] < 683)
    assert len(dop) == len(man) == len(exp) == 4, (dop, man, exp)
    out = {}
    for letter, (xd, d), (xm, m), (xe, e) in zip("abcd", dop, man, exp):
        assert abs(xd - xm) < 25 and 0 < xe - xm < 40, (letter, xd, xm, xe)
        out[letter] = (d, f"{m}e{e}")
    return out


def fmt3(v):
    """3 significant digits without exponent notation."""
    return f"{float(f'{v:.3g}'):g}"


def measure(fig, csv_dir, image, pdf):
    cfg = FIGURES[fig]
    data, ucal, logy, held = load_csvs(csv_dir, cfg["project"])
    assert logy == cfg["logy"], "axis type of the project does not match the figure"
    smap = {k: k for k in "abcd"}
    names = data.pop("_project_dataset_names")
    dark = np.array(Image.open(image).convert("L")) < THRESHOLD
    fy, fx, xt, yt = axis_ticks(dark, cfg)
    xticks = cfg["xticks"]
    assert len(xt) == len(xticks), f"{len(xt)} temperature ticks found"
    a, b = np.polyfit(xt, xticks, 1)
    resid = np.array(xticks) - (a * np.array(xt) + b)
    cal_unc = float(np.abs(resid).max())
    assert cal_unc < 2.5, cal_unc  # tick positions of the bitmaps jitter by about 3 px (2 K)
    uy1, uy2 = ucal[2], ucal[3]
    if not logy:  # S axis: ticks every 0.02 mV/K from -0.6 (top) to 0 (bottom)
        assert len(yt) == 31, len(yt)
        ay, by = np.polyfit(yt, [-0.6 + 0.02 * k for k in range(31)], 1)
        for py, dy in ((uy1["py"], float(uy1["dy"])), (uy2["py"], float(uy2["dy"]))):
            assert abs((ay * py + by) - dy) < 0.002, ("S calibration differs from the printed ticks", py, dy)
        yfit = {"slope_mVK_per_px": float(ay), "intercept_mVK": float(by)}
        y_dev = None
    else:  # log axis: ticks at k x 10^n; each detected tick is matched to the nearest expected one using the clicked calibration
        exp = [(k * 10.0 ** n) for n in range(0, 6) for k in range(1, 10)] + [1e6]
        exp = [v for v in exp if v >= 5.0]
        pred = [click_px(ucal, True, 400.0, v)[1] for v in exp]
        assert len(yt) == len(exp), f"{len(yt)} value-axis ticks found, {len(exp)} expected"
        vals = []
        for y in yt:
            j = int(np.argmin([abs(y - p) for p in pred]))
            assert abs(y - pred[j]) < 12.0, ("value tick not matched", y, exp[j], pred[j])
            vals.append(exp[j])
        assert len(set(vals)) == len(vals)
        ay, by = np.polyfit(yt, np.log10(vals), 1)
        r = np.log10(vals) - (ay * np.array(yt) + by)
        assert np.abs(r).max() < 0.01, np.abs(r).max()
        y_dev = [float((ay * py + by) - np.log10(float(dy))) for py, dy in ((uy1["py"], uy1["dy"]), (uy2["py"], uy2["dy"]))]
        assert max(abs(d) for d in y_dev) <= 0.01, ("clicked value calibration differs from the printed ticks", y_dev)
        yfit = {"slope_decade_per_px": float(ay), "intercept_decade": float(by), "max_tick_residual_decade": float(np.abs(r).max()), "n_ticks": len(yt),
                "clicked_calibration_minus_ticks_decade": y_dev}
    ux1, ux2 = ucal[0], ucal[1]
    user_x_off_px = [ux1["px"] - float(np.interp(float(ux1["dx"]), xticks, xt)), ux2["px"] - float(np.interp(float(ux2["dx"]), xticks, xt))]
    t2 = table2(REPO / pdf)
    t1 = table1(REPO / pdf)
    if fig == 2:  # the order of the series in the figure (first clicked values) against the order of Table II
        first_val = {fl: min(data[fl]["pts"])[1] for fl in "abcd"}
        order_series = [smap[fl] for fl in sorted("abcd", key=lambda fl: -first_val[fl])]
        order_t2 = sorted("abcd", key=lambda s: -t2["abcd".index(s)]["sigma_Scm"])
        assert order_series == order_t2, f"the series of the figure ({order_series}, highest sigma first) do not match the order of Table II ({order_t2}); the CSV letters must be the samples (top to bottom in Fig. 2: b, d, c, a)"
    # marker centres, per file letter
    mk = []
    for fl in "abcd":
        s = smap[fl]
        prev = -1.0
        for i, (tc, vc) in enumerate(data[fl]["pts"]):
            px, py = click_px(ucal, logy, tc, vc)
            if fl in held:
                assert abs(held[fl][i][0] - px) < 3.0 and abs(held[fl][i][1] - py) < 1.0, (fl, i, held[fl][i], (px, py))
            c_sym, n_ok, spread = refine_x(dark, px, py, cfg["search_px"])
            h = hole_centre(dark, c_sym if n_ok >= MIN_AGREE else px, py, SYMBOL_FILLED[s])
            if cfg["estimator"] == "symmetry":
                assert n_ok >= MIN_AGREE, f"series {fl} marker at click {tc:.1f} K: only {n_ok} of {len(SETTINGS)} window settings agree"
                c = c_sym
            else:
                assert h is not None, f"series {fl} marker at click {tc:.1f} K: no enclosed interior / filled body found near the click"
                c = h[0]
            assert abs(c - px) < cfg["search_px"] - 0.5, f"series {fl} marker at click {tc:.1f} K: centre at the edge of the search window"
            t_img = float(a * c + b)
            assert t_img > prev
            prev = t_img
            mk.append({"file_letter": fl, "sample": s, "click_T": tc, "click_value": vc, "centre_px": c, "click_px": [px, py], "T_image": t_img, "centre_px_symmetry": c_sym,
                       "n_settings_agreeing": n_ok, "spread_px": spread, "centre_px_hole": None if h is None else h[0], "fill": interior_fill(dark, c, py)})
    for fl in "abcd":
        fills = [m["fill"] for m in mk if m["file_letter"] == fl]
        is_filled = float(np.median(fills)) >= 0.6
        assert is_filled == SYMBOL_FILLED[smap[fl]], f"file {fl} (sample {smap[fl]}): symbols are {'filled' if is_filled else 'open'}, the sample's are {'filled' if SYMBOL_FILLED[smap[fl]] else 'open'}; the CSV letters must be the samples (top to bottom in Fig. 2: b, d, c, a)"
    first = {}
    for fl in "abcd":
        first[fl] = min((m for m in mk if m["file_letter"] == fl), key=lambda m: m["T_image"])
    disagree = [{"sample": m["sample"], "click_T": round(m["click_T"], 1), "symmetry_minus_hole_px": round(m["centre_px_symmetry"] - m["centre_px_hole"], 1)}
                for m in mk if m["centre_px_hole"] is not None and abs(m["centre_px_symmetry"] - m["centre_px_hole"]) > 3.0]
    # columns: markers of all series sorted by temperature; a new column starts after a gap of 20 K
    order = sorted(range(len(mk)), key=lambda i: mk[i]["T_image"])
    cols = []
    for i in order:
        if cols and mk[i]["T_image"] - mk[cols[-1][-1]]["T_image"] < 20.0:
            cols[-1].append(i)
        else:
            cols.append([i])
    for k, c in enumerate(cols, 1):
        samples = [mk[i]["sample"] for i in c]
        assert len(set(samples)) == len(samples), (k, samples)
        ts = [mk[i]["T_image"] for i in c]
        assert max(ts) - min(ts) <= MAX_COLUMN_RANGE_K, (k, ts)
        m = float(np.mean(ts))
        sd = float(np.std(ts, ddof=1)) if len(ts) > 1 else None
        for i in c:
            mk[i].update(column=k, T_col=m, T_col_sd=sd, n_series=len(c), column_range=max(ts) - min(ts))
    checks, sample_flags = [], {s: [] for s in "abcd"}
    for fl in "abcd":
        s = smap[fl]
        f = first[fl]
        t = t2["abcd".index(s)]
        if fig == 1:
            s_uv = cfg["scale"] * f["click_value"]
            d = s_uv - t["S_uVK"]
            assert abs(d) <= TABLE2_TOL_UV and abs(s_uv) >= abs(t["S_uVK"]), (s, s_uv, t["S_uVK"])
            checks.append({"sample": s, "table2_S_uVK_298K": t["S_uVK"], "first_marker_S_uVK": round(s_uv, 1), "first_marker_T_K": round(f["T_col"], 1), "difference_uVK": round(d, 1)})
        else:
            ratio = f["click_value"] / t["sigma_Scm"]
            nemu = t["n_1e20"] * 1e20 * 1.602176634e-19 * t["mu"]
            rec = {"sample": s, "table2_sigma_Scm": t["sigma_Scm"], "first_marker_sigma_Scm": round(f["click_value"], 1), "first_marker_T_K": round(f["T_col"], 1),
                   "ratio_first_marker_over_table2": round(ratio, 3), "n_1e20": t["n_1e20"], "mu_cm2Vs": t["mu"], "n_e_mu_Scm": round(nemu, 1), "table2_over_n_e_mu": round(t["sigma_Scm"] / nemu, 3)}
            flags = []
            if not 0.5 <= ratio <= 1.1:  # sigma decreases with T (inset: n constant, mu ~ T^-1.5 to T^-2), so a marker 50 K above room temperature is at or below the room-temperature value
                flags.append(f"first marker {f['click_value']:.0f} S/cm at {f['T_col']:.0f} K against Table II room-temperature sigma {t['sigma_Scm']:g} S/cm (ratio {ratio:.2f})")
            if not 0.9 <= rec["table2_over_n_e_mu"] <= 1.1:
                flags.append(f"Table II sigma {t['sigma_Scm']:g} S/cm differs from n e mu = {nemu:.0f} S/cm of its own n and mu (ratio {rec['table2_over_n_e_mu']:.2f})")
            rec["flags"] = flags
            sample_flags[s] = flags
            checks.append(rec)
    return {"figure": fig, "project_dataset_names": names,
            "temperature_axis_fit": {"slope_K_per_px": float(a), "intercept_K": float(b), "max_residual_K": cal_unc, "tick_px": xt},
            "value_axis_fit": yfit, "clicked_calibration_offset_px_at_first_and_last_x_point": user_x_off_px, "clicked_calibration_offset_K": [o * a for o in user_x_off_px],
            "n_columns": len(cols), "markers": mk, "estimator": cfg["estimator"], "estimator_disagreements_over_3px": disagree, "table2_checks": checks,
            "_ctx": {"data": data, "t1": t1, "sample_flags": sample_flags, "y_dev": y_dev}}


def combine(results):
    """Amendment 5: columns of the figures that report one measurement run are matched (difference <= 2 sqrt(u1^2 + u2^2), nearest, one per figure) and each gets one temperature, the mean of the
    figures' column consensus weighted by 1/u^2 (u: the figure's calibration uncertainty), with uncertainty 1/sqrt(sum 1/u^2). Returns the groups sorted by temperature."""
    groups = []
    for fig in sorted(results):
        res = results[fig]
        u = res["temperature_axis_fit"]["max_residual_K"]
        cols = {}
        for m in res["markers"]:
            cols.setdefault(m["column"], m)
        for k, m in sorted(cols.items(), key=lambda kv: kv[1]["T_col"]):
            member = {"figure": fig, "column": k, "T": m["T_col"], "u": u, "n_series": m["n_series"], "sd": m["T_col_sd"], "range": m["column_range"]}
            free = [g for g in groups if all(x["figure"] != fig for x in g["members"])]
            best = min(free, key=lambda g: abs(g["T"] - m["T_col"]), default=None)
            if best is not None and abs(best["T"] - m["T_col"]) <= 2.0 * float(np.hypot(best["u"], u)):
                best["members"].append(member)
            else:
                groups.append({"members": [member]})
                best = groups[-1]
            w = np.array([1.0 / x["u"] ** 2 for x in best["members"]])
            best["T"] = float(np.sum(w * np.array([x["T"] for x in best["members"]])) / w.sum())
            best["u"] = float(1.0 / np.sqrt(w.sum()))
    groups.sort(key=lambda g: g["T"])
    for i, g in enumerate(groups, 1):
        g["column"] = i
        edge = round(g["T"] / 25.0) * 25.0
        g["floor_bin"] = float(25.0 * np.floor(g["T"] / 25.0))
        g["edge"] = bool(abs(g["T"] - edge) <= EDGE_FACTOR * g["u"])
        g["edge_distance_K"] = float(abs(g["T"] - edge))
        g["other_bin"] = (edge - 25.0 if g["T"] >= edge else edge) if g["edge"] else None
        for x in g["members"]:
            for m in results[x["figure"]]["markers"]:
                if m["column"] == x["column"]:
                    m["combined_column"] = i
    return groups


def make_rows(results, groups):
    """The values-file rows of every figure, at the combined column temperatures."""
    rows = []
    by_col = {g["column"]: g for g in groups}
    for fig in sorted(results):
        res, cfg = results[fig], FIGURES[fig]
        ctx = res["_ctx"]
        for m in sorted(res["markers"], key=lambda m: (m["sample"], m["combined_column"])):
            s = m["sample"]
            g = by_col[m["combined_column"]]
            label = f"{ctx['t1'][s][0]}-doped SrTiO3, n = {ctx['t1'][s][1]} cm^-3 (sample ({s}), single crystal; Table I)"
            T = g["T"]
            tokens = ["[cluster-level]"]
            if g["edge"]:
                tokens.append(f"[edge-sensitivity: also bin {g['other_bin']:g}]")
            own = "; ".join(f"Fig. {x['figure']}: {x['T']:.1f} K (n = {x['n_series']} series, SD {'n/a' if x['sd'] is None else format(x['sd'], '.1f')} K, calibration u = {x['u']:.1f} K)" for x in g["members"])
            note = (" ".join(tokens) + f" T = combined consensus of column {g['column']} over {len(g['members'])} figure(s) (amendment 5), uncertainty +/- {g['u']:.1f} K, floor bin {g['floor_bin']:g} [{own}]; "
                    f"clicked T {m['click_T']:.1f} K (not used); {cfg['res_note']}")
            if cfg["logy"]:
                note += f"; clicked sigma calibration within {100 * (10 ** max(abs(x) for x in ctx['y_dev']) - 1):.1f} % of the printed ticks"
            if s == "b":
                note += "; " + SAMPLE_DOPING_NOTE
            for fgl in ctx["sample_flags"][s]:
                note += f"; Table II check: {fgl}"
            if T >= 800.0:
                note += "; at or above the 800 K window edge: recorded, not compared"
            rows.append({
                "lit_id": f"OH2005_{cfg['id']}_{s}_c{g['column']:02d}", "compound_label": label, "composition": "SrTiO3", "property": cfg["prop"], "T_K": f"{T:.1f}",
                "value": fmt3(cfg["scale"] * m["click_value"]), "unit": cfg["unit"], "doi": DOI, "location": cfg["location"], "value_kind": "digitised",
                "digitisation_method": (f"bitmap digitised by hand in WebPlotDigitizer ({cfg['method']}); temperature = combined column consensus (amendments 4 and 5) of marker centres "
                                        f"{cfg['locate']}, read on a straight line through the 11 printed major ticks of the figure ({cfg['xlabel']}), with "
                                        "thesis_paper/scripts/na12_extract_ohta.py"),
                "notes": note})
    return rows


def replace_values(d, prefix):
    """Replace the rows of this paper (lit_id starting with prefix) in the values file by the rows of the run d; returns (removed, added)."""
    cfg = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
    assert cfg["tree_clean"], "that run was made from a dirty tree"
    with open(d / "values_rows.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows and all(r["lit_id"].startswith(prefix) for r in rows)
    with open(VALUES, encoding="utf-8", newline="") as f:
        have = list(csv.DictReader(f))
    keep = [r for r in have if not r["lit_id"].startswith(prefix)]
    assert not ({r["lit_id"] for r in keep} & {r["lit_id"] for r in rows}), "duplicate ids"
    with open(VALUES, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(keep + rows)
    return len(have) - len(keep), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", nargs=3, action="append", metavar=("N", "CSV_DIR", "IMAGE"), help="a figure of the paper (1 or 2): folder with its 4 CSVs and project file, and the figure bitmap; repeat per figure")
    ap.add_argument("--pdf", default=PDF)
    ap.add_argument("--append-values", default=None, help="run directory whose rows are appended to the values file")
    ap.add_argument("--replace-prefix", default=None, help="with --append-values: first remove every row whose lit_id starts with this prefix (the earlier rows of the paper)")
    args = ap.parse_args()
    if args.append_values:
        d = Path(args.append_values)
        if args.replace_prefix:
            removed, added = replace_values(d, args.replace_prefix)
            print(f"removed {removed} rows with prefix {args.replace_prefix}, added {added} rows from {d}")
            return 0
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
    assert args.figure, "--figure N CSV_DIR IMAGE is required (repeat per figure)"
    figs = {}
    for n, d, img in args.figure:
        assert int(n) in FIGURES and int(n) not in figs, n
        figs[int(n)] = (d, img)
    inputs = {"pdf": args.pdf, "pdf_vector": "thesis_paper/scripts/pdf_vector.py"}
    for n, (d, img) in sorted(figs.items()):
        inputs[f"fig{n}_image"] = str(Path(img).resolve())
        inputs[f"fig{n}_project"] = str((Path(d) / FIGURES[n]["project"]).resolve())
        for p in sorted(Path(d).glob("*.csv")):
            inputs[f"fig{n}_csv_{p.stem[0]}"] = str(p.resolve())
    prov = rr.provenance(inputs, __file__)
    results = {n: measure(n, d, img, args.pdf) for n, (d, img) in sorted(figs.items())}
    groups = combine(results)
    rows = make_rows(results, groups)
    out = rr.new_run_dir("na12_extraction")
    rr.write_json(out / "extraction.json", {"combined_columns": groups, "figures": {str(n): {k: v for k, v in r.items() if k != "_ctx"} for n, r in results.items()}})
    with open(out / "values_rows.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    rr.write_json(out / "run_config.json", {**prov, "paper": "Ohta et al. 2005, figures " + ", ".join(str(n) for n in sorted(figs)) + ", samples (a) to (d)",
                                            "rules": "docs/decisions.md pre-registration B + amendments 1, 3, 4, 5 (2026-10-13 to 16)"})
    print(f"wrote {out}: {len(rows)} rows, {len(groups)} combined columns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
