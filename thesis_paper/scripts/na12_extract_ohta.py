"""
NA12: Seebeck coefficient (Fig. 1) and electrical conductivity (Fig. 2) of four heavily La- or Nb-doped SrTiO3 crystals from Ohta, Nomura, Ohta & Koumoto 2005
(J. Appl. Phys. 97:034106, DOI 10.1063/1.1847723).
Rules: docs/decisions.md, pre-registration B with amendments 1 (reading rules), 3 (Ohta compared at the cluster level, sample as stated) and 4 (column-consensus marker temperatures,
edge-row sensitivity). No out-of-fold prediction is read or used here.

The figures are bitmaps, digitised by hand in WebPlotDigitizer (one CSV per series and the project file). The clicks give the value of each marker; the temperature of a marker is the
consensus of its column (amendment 4): the marker centres are located in the figure image by a left-right symmetry fit of the column profile of the dark pixels, converted to temperature on
a straight line through ALL printed tick marks of the temperature axis, and averaged over the series that have a marker in that column.

Two ways of locating a marker centre are computed for every marker. Fig. 1 uses the symmetry fit (committed run 20261014T143357); in Fig. 2 the line joining the markers biases the symmetry
fit at several triangles and circles (checked on enlarged crops), so Fig. 2 uses the centre that does not depend on the line: the centroid of the enclosed white interior of an open symbol,
or of the body of a filled symbol after a morphological opening that removes the line. On the 40 markers of Fig. 1 the two agree within 2.2 px; the markers where they differ by more than
3 px are listed in extraction.json.

Which series is which sample. The letters of the CSV file names are the digitiser's own; they are checked, not trusted. --series-map file_letter=sample_letter,... states the sample of each
file (default: the letters are the samples). The run stops unless the order of the series' first values agrees with the order of the room-temperature values of Table II (read from the PDF
text) and the fill of the symbols (filled or open) agrees with the sample's symbol in the figure (a open circle, b filled circle, c open triangle, d filled triangle).

Validation (asserted): the series have as many clicks as their CSV; a project file that holds a series agrees with its CSV; the temperature axis has 11 printed ticks (every 100 K) and a
straight-line fit through them (maximum residual below 2.5 K); the value axis ticks reproduce the clicked calibration (Fig. 1: every 0.02 mV/K, within 2 uV/K; Fig. 2: decades and minor
ticks of the log axis, within 0.01 decade); every refined centre lies inside its search window and a majority (at least 3 of 5) of the window settings agree within 3 px; markers of one
column lie within 5 K of each other; each series has at most one marker per column; Fig. 1: the first marker of every sample is within 15 uV/K of the room-temperature S of Table II and
has the larger |S|. Reported, not asserted (flags in extraction.json and in the notes of the rows): Fig. 2 first marker against Table II sigma, Table II sigma against n e mu of its own
row, and the columns against those of a Fig. 1 run (--fig1-run).

Usage (repository root, clean tree):
    python thesis_paper/scripts/na12_extract_ohta.py --fig 1 --csv-dir <folder> --image <figure bitmap>
    python thesis_paper/scripts/na12_extract_ohta.py --fig 2 --csv-dir <folder> --image <figure bitmap> --series-map a=b,b=d,c=c,d=a --fig1-run <run dir of Fig. 1>
        # writes thesis_paper/results/na12_extraction/<UTC>/
    python thesis_paper/scripts/na12_extract_ohta.py --append-values <dir>      # appends the rows of that run to docs/na12_literature_values.csv
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
        "frame_rows": (1650, 1800), "frame_min": 1000, "x_range": (380, 1910), "x_band": (5, 15), "x_thr": 8, "fx_box": (0, 1700, 380, 420), "y_rows": (0, 1700), "search_px": 16, "estimator": "symmetry",
        "locate": "located in the image by a symmetry fit",
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
    proj = json.loads((Path(folder) / project).read_text(encoding="utf-8"))
    ax = proj["axesColl"][0]
    assert not ax["isLogX"]
    cal = ax["calibrationPoints"]
    held = {}
    for ds in proj["datasetColl"]:
        s = ds["name"][0]
        assert ds["name"] == out[s]["name"], (ds["name"], out[s]["name"])
        assert len(ds["data"]) == len(out[s]["pts"]), (s, len(ds["data"]), len(out[s]["pts"]))
        for p, (t, v) in zip(ds["data"], out[s]["pts"]):
            assert abs(p["value"][0] - t) < 1e-6 and abs(p["value"][1] - v) < 1e-9 * max(1.0, abs(v)), (s, p["value"], (t, v))
        held[s] = [(p["x"], p["y"]) for p in ds["data"]]
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


def build(fig, csv_dir, image, pdf, series_map=None, fig1_run=None):
    cfg = FIGURES[fig]
    data, ucal, logy, held = load_csvs(csv_dir, cfg["project"])
    assert logy == cfg["logy"], "axis type of the project does not match the figure"
    smap = series_map or {k: k for k in "abcd"}
    assert sorted(smap) == list("abcd") and sorted(smap.values()) == list("abcd"), smap
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
        assert order_series == order_t2, f"the series of the figure ({order_series}, highest sigma first) do not match the order of Table II ({order_t2}); check --series-map"
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
        assert is_filled == SYMBOL_FILLED[smap[fl]], f"file {fl} (sample {smap[fl]}): symbols are {'filled' if is_filled else 'open'}, the sample's are {'filled' if SYMBOL_FILLED[smap[fl]] else 'open'}; check --series-map"
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
    # agreement with the columns of a Fig. 1 run
    agreement = None
    if fig1_run:
        ex = json.loads((Path(fig1_run) / "extraction.json").read_text(encoding="utf-8"))
        fig1_cols = sorted({round(m["T_col"], 3) for m in ex["markers"]})
        agreement = []
        for k in range(1, len(cols) + 1):
            tk = next(m["T_col"] for m in mk if m["column"] == k)
            j = min(range(len(fig1_cols)), key=lambda i: abs(fig1_cols[i] - tk))
            agreement.append({"column": k, "T_col": round(tk, 1), "nearest_fig1_column_T": round(fig1_cols[j], 1), "difference_K": round(tk - fig1_cols[j], 1)})
    # rows
    rows = []
    for m in sorted(mk, key=lambda m: (m["sample"], m["column"])):
        s, fl = m["sample"], m["file_letter"]
        label = f"{t1[s][0]}-doped SrTiO3, n = {t1[s][1]} cm^-3 (sample ({s}), single crystal; Table I)"
        T = m["T_col"]
        edge = round(T / 25.0) * 25.0
        tokens = ["[cluster-level]"]
        if abs(T - edge) <= cal_unc:
            other = edge - 25.0 if T >= edge else edge
            tokens.append(f"[edge-sensitivity: also bin {other:g}]")
        floor_bin = 25.0 * np.floor(T / 25.0)
        sd = "" if m["T_col_sd"] is None else f"{m['T_col_sd']:.1f}"
        note = (" ".join(tokens) + f" T = consensus of column {m['column']} ({m['n_series']} series, SD {sd or 'n/a (one series)'} K, range {m['column_range']:.1f} K), calibration "
                f"uncertainty +/- {cal_unc:.1f} K, floor bin {floor_bin:g}; clicked T {m['click_T']:.1f} K (not used); {cfg['res_note']}")
        if logy:
            note += f"; clicked sigma calibration within {100 * (10 ** max(abs(x) for x in y_dev) - 1):.1f} % of the printed ticks"
        if s == "b":
            note += "; " + SAMPLE_DOPING_NOTE
        if fl != s:
            note += f"; digitised in the file of letter {fl} ({data[fl]['file']}), assigned to sample ({s}) by the order of Table II and the symbol (--series-map)"
        for fgl in sample_flags[s]:
            note += f"; Table II check: {fgl}"
        if T >= 800.0:
            note += "; at or above the 800 K window edge: recorded, not compared"
        rows.append({
            "lit_id": f"OH2005_{cfg['id']}_{s}_c{m['column']:02d}", "compound_label": label, "composition": "SrTiO3", "property": cfg["prop"], "T_K": f"{T:.1f}",
            "value": fmt3(cfg["scale"] * m["click_value"]), "unit": cfg["unit"], "doi": DOI, "location": cfg["location"], "value_kind": "digitised",
            "digitisation_method": (f"bitmap digitised by hand in WebPlotDigitizer ({cfg['method']}); temperature = column consensus of marker centres "
                                    f"{cfg['locate']} and read on a straight line through the 11 printed major ticks ({cfg['xlabel']}), with "
                                    "thesis_paper/scripts/na12_extract_ohta.py (amendment 4)"),
            "notes": note})
    return {"figure": fig, "series_map_file_to_sample": smap,
            "temperature_axis_fit": {"slope_K_per_px": float(a), "intercept_K": float(b), "max_residual_K": cal_unc, "tick_px": xt},
            "value_axis_fit": yfit,
            "clicked_calibration_offset_px_at_first_and_last_x_point": user_x_off_px,
            "clicked_calibration_offset_K": [o * a for o in user_x_off_px],
            "n_columns": len(cols), "markers": mk, "estimator": cfg["estimator"], "estimator_disagreements_over_3px": disagree, "table2_checks": checks, "fig1_column_agreement": agreement, "rows": rows}


def parse_map(s):
    if not s:
        return None
    m = dict(p.split("=") for p in s.split(","))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", type=int, choices=sorted(FIGURES), default=1)
    ap.add_argument("--csv-dir", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--pdf", default=PDF)
    ap.add_argument("--series-map", default=None, help="file letter = sample letter, comma separated (default: identity)")
    ap.add_argument("--fig1-run", default=None, help="run directory of the Fig. 1 extraction (its columns are compared with this figure's)")
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
    assert args.csv_dir and args.image, "--csv-dir and --image are required"
    cfg = FIGURES[args.fig]
    inputs = {"pdf": args.pdf, "image": str(Path(args.image).resolve()), "project": str((Path(args.csv_dir) / cfg["project"]).resolve()), "pdf_vector": "thesis_paper/scripts/pdf_vector.py"}
    for p in sorted(Path(args.csv_dir).glob("*.csv")):
        inputs[f"csv_{p.stem[0]}"] = str(p.resolve())
    if args.fig1_run:
        inputs["fig1_extraction"] = str((Path(args.fig1_run) / "extraction.json").resolve())
    prov = rr.provenance(inputs, __file__)
    smap = parse_map(args.series_map)
    res = build(args.fig, args.csv_dir, args.image, args.pdf, smap, args.fig1_run)
    out = rr.new_run_dir("na12_extraction")
    rr.write_json(out / "extraction.json", {k: v for k, v in res.items() if k != "rows"})
    with open(out / "values_rows.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(res["rows"])
    rr.write_json(out / "run_config.json", {**prov, "paper": f"Ohta et al. 2005, Fig. {args.fig} ({cfg['prop']}), samples (a) to (d)", "series_map_file_to_sample": smap or "identity",
                                            "rules": "docs/decisions.md pre-registration B + amendments 1, 3, 4 (2026-10-13/14)"})
    print(f"wrote {out}: {len(res['rows'])} rows, {res['n_columns']} columns; temperature-axis residual {res['temperature_axis_fit']['max_residual_K']:.2f} K")
    return 0


if __name__ == "__main__":
    sys.exit(main())
