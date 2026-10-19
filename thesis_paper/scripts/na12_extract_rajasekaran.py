"""
NA12: electrical conductivity (Fig. 10a) and Seebeck coefficient (Fig. 10b) of La-substituted BaSnO3 from Rajasekaran, Arivanandhan, Kumaki, Jayavel, Hayakawa & Shimomura 2020
(CrystEngComm 22(32):5363-5374, DOI 10.1039/d0ce00702a), digitised by hand in WebPlotDigitizer from the raster Fig. 10 (PDF page 10).
Rules: docs/decisions.md, pre-registration B with amendments 1, 2 (aggregation), 4, 5, 6 (marker temperatures, combined columns, 2u edge tolerance), 9, 12 and 13 (legible markers, the image-derived
gates) and 14 (which values are rows: Fig. 10a and 10b; Table 2 states no temperature and is not read; Fig. 10c is derived and is not read). No out-of-fold prediction is read or used here.

Input: the corrected bitmap of Fig. 10 (the PDF holds a CMYK JPEG, Adobe-inverted, that a naive conversion shows on a black background), one WebPlotDigitizer project per panel with one dataset
(the other series is placed by the inverse of the project's calibration) and one CSV per series: BSOLa5 (black squares, x = 0.05) and BSOLa10 (red circles, x = 0.10), six markers each.

The clicked VALUE axis of both projects does not match the printed ticks (panel a: the top frame line, printed 1.5e3, was entered as 1000; panel b: the frame height was entered as 0 to 1000 instead of 0
to -70 uV/K). The clicked values are therefore not used. The pixel position of every click is exact (the projects' linear calibrations are inverted), and the value is read at the click's pixel row on
the PRINTED ticks of the figure (panel a: 0, 500, 1000, 1500 S/m; panel b: 0 to -70 uV/K in steps of 10). The clicked temperature axis (300 and 600 on the first and last printed tick) is correct and is
used for the gate (c) comparison only; the temperature of a marker is the consensus of its column (amendments 4 and 5): the marker centres are the centroids of the opened body of the filled symbols
(black or red mask, a morphological opening removes the joining line), read on the printed temperature ticks (300 to 600 K, step 50) after correcting the small tilt of the frame; the two series and the
two panels (one measurement run) share columns (combined, amendment 5).

Validation (asserted): the CSV of the held dataset equals its project; the inverse map reproduces the four calibration points; the frames and 7 temperature ticks, 4 (a) or 8 (b) value ticks are found
and fit straight lines (temperature residual below 2.5 K, value residual below 0.6% of the axis range); the clicked temperature calibration points sit on their printed ticks; every marker has an opened
body within 25 px of its click, of the colour of its series; each series has 6 markers in each panel; the DOI printed on page 1 is the one of the values; the Seebeck coefficient of the 573 K marker of
BSOLa5 agrees with the abstract's printed value to 3 uV/K. Reported, not asserted: the abstract's conductivities (their temperature is not stated in the sentence; the ratio of the two 573 K markers against
the ratio of the two printed mantissas), and the difference between the clicked value axis and the printed ticks.

Usage (repository root, clean tree):
    python thesis_paper/scripts/na12_extract_rajasekaran.py --image <fig 10 PNG> --panel a <fig10a folder> --panel b <fig10b folder>     # writes thesis_paper/results/na12_extraction/<UTC>/
    python thesis_paper/scripts/na12_extract_rajasekaran.py --append-values <dir>                                                       # appends the rows of that run to docs/na12_literature_values.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import na12_extract_ohta as oh  # noqa: E402
import pdf_vector as pv  # noqa: E402
import run_record as rr  # noqa: E402

PDF = "thesis_paper/source/pdfs/rajasekaran2020.pdf"
VALUES = REPO / "thesis_paper" / "docs" / "na12_literature_values.csv"
DOI = "10.1039/d0ce00702a"
JOURNAL = "CrystEngComm 22(32):5363-5374 (2020)"
ID_PREFIX = "RJ2020_"
COLUMNS = oh.COLUMNS
OPEN_RADIUS = 7  # px: the markers of Fig. 10 are about 20 to 26 px across and the joining line about 4 px wide; the opening disc must be wider than the line and smaller than a marker (the 11 px of the other figures removes the squares)
X_TICKS = [300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0]
SERIES = {  # CSV key -> the sample of the paper (La at the Ba site, Ba1-xLaxSnO3), the colour of its symbol
    "BSOLa5": {"x": 0.05, "composition": "Ba0.95La0.05SnO3", "colour": "black", "shape": "squares"},
    "BSOLa10": {"x": 0.10, "composition": "Ba0.9La0.1SnO3", "colour": "red", "shape": "circles"},
}
PANELS = {
    "a": {"prop": "sigma", "unit": "S/m", "project": "fig10a_project.json", "csv": {"BSOLa5": "la5.csv", "BSOLa10": "la10.csv"}, "held": "BSOLa10", "tolerance_rel": 0.006,
          "frame_hint": {"left": (140, 175), "right": (745, 775), "bottom": (495, 520), "top": (8, 35)}, "y_ticks": [1500.0, 1000.0, 500.0, 0.0], "what": "electrical conductivity (S/m)"},
    "b": {"prop": "S", "unit": "uV/K", "project": "fig10b_project.json", "csv": {"BSOLa5": "la5_s.csv", "BSOLa10": "la10_s.csv"}, "held": "BSOLa5", "tolerance_rel": 0.006,
          "frame_hint": {"left": (995, 1025), "right": (1620, 1645), "bottom": (515, 540), "top": (10, 40)}, "y_ticks": [0.0, -10.0, -20.0, -30.0, -40.0, -50.0, -60.0, -70.0], "what": "Seebeck coefficient (uV/K)"},
}


def centre(counts, i, w=5):
    lo, hi = max(0, i - w), i + w + 1
    c = counts[lo:hi].astype(float)
    m = c >= 0.5 * c.max()
    return float(np.average(np.arange(lo, hi)[m], weights=c[m]))


def masks_of(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return {"dark": rgb.mean(axis=2) < 128, "black": rgb.max(axis=2) < 90, "red": (r > 180) & (g < 120) & (b < 150) & (r - g > 100)}


def read_csv(path):
    pts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            a, b = line.split(",")
            pts.append((float(a), float(b)))
    return pts


def frame_and_ticks(dark, hint):
    """Frame centres and the printed tick positions (the ticks are drawn outside the frame): the temperature ticks along the bottom, the value ticks along the left edge."""
    cols = dark[:600].sum(axis=0)
    fx = centre(cols, int(np.argmax(cols[hint["left"][0]:hint["left"][1]])) + hint["left"][0])
    rx = centre(cols, int(np.argmax(cols[hint["right"][0]:hint["right"][1]])) + hint["right"][0])
    rows = dark[:600, int(fx) + 8:int(rx) - 8].sum(axis=1)
    fy = centre(rows, int(np.argmax(rows[hint["bottom"][0]:hint["bottom"][1]])) + hint["bottom"][0])
    ty = centre(rows, int(np.argmax(rows[hint["top"][0]:hint["top"][1]])) + hint["top"][0])
    prof = dark[int(fy) + 6:int(fy) + 16, int(fx) - 10:int(rx) + 10].sum(axis=0)
    xt = [x + int(fx) - 10 for x in oh.runs(prof >= max(int(prof.max() * 0.7), 1))]
    prof = dark[int(ty) - 10:int(fy) + 10, int(fx) - 16:int(fx) - 6].sum(axis=1)
    yt = [y + int(ty) - 10 for y in oh.runs(prof >= max(int(prof.max() * 0.7), 1))]
    return {"fx": fx, "rx": rx, "fy": fy, "ty": ty, "x_ticks_px": xt, "y_ticks_px": yt}


def tilt(dark, fr):
    """Slope of the left frame edge (x per row) and of the bottom frame edge (y per column), from the frame's own lines (the figure is slightly rotated)."""
    fx, rx, fy, ty = fr["fx"], fr["rx"], fr["fy"], fr["ty"]
    ys, xs = [], []
    for r in range(int(ty) + 30, int(fy) - 30, 15):
        seg = dark[r, int(fx) - 4:int(fx) + 5].astype(float)
        if seg.sum() >= 2:
            ys.append(r)
            xs.append(int(fx) - 4 + float(np.average(np.arange(len(seg)), weights=seg)))
    s = float(np.polyfit(ys, xs, 1)[0])
    ticks = fr["x_ticks_px"]
    cx, cy = [], []
    for a, b in zip(ticks, ticks[1:]):
        c = int((a + b) / 2)
        seg = dark[int(fy) - 4:int(fy) + 5, c - 3:c + 4].sum(axis=1).astype(float)
        if seg.sum() > 0:
            cx.append(c)
            cy.append(int(fy) - 4 + float(np.average(np.arange(len(seg)), weights=seg)))
    t = float(np.polyfit(cx, cy, 1)[0])
    return s, t


def project_maps(folder, panel):
    """Calibration points, the held dataset's clicks, and the forward and inverse affine maps of the project (the project holds one dataset)."""
    cfg = PANELS[panel]
    proj = json.loads((Path(folder) / cfg["project"]).read_text(encoding="utf-8"))
    ax = proj["axesColl"][0]
    assert not ax["isLogX"] and not ax["isLogY"] and len(proj["datasetColl"]) == 1
    ds = proj["datasetColl"][0]
    held = cfg["held"]
    pts = read_csv(Path(folder) / cfg["csv"][held])
    assert len(ds["data"]) == len(pts) == 6
    for p, (t, v) in zip(ds["data"], pts):
        assert abs(p["value"][0] - t) < 1e-6 and abs(p["value"][1] - v) < 1e-9 * max(1.0, abs(v)), (p["value"], (t, v))
    px = [(p["x"], p["y"]) for p in ds["data"]]
    A = np.array([[x, y, 1.0] for x, y in px])
    cT = np.linalg.lstsq(A, np.array([t for t, v in pts]), rcond=None)[0]
    cV = np.linalg.lstsq(A, np.array([v for t, v in pts]), rcond=None)[0]
    assert np.abs(A @ cT - np.array([t for t, v in pts])).max() < 1e-4 and np.abs(A @ cV - np.array([v for t, v in pts])).max() < 1e-6, "the project's calibration is not affine"
    M = np.array([[cT[0], cT[1]], [cV[0], cV[1]]])

    def fwd(x, y):
        return float(cT[0] * x + cT[1] * y + cT[2]), float(cV[0] * x + cV[1] * y + cV[2])

    def inv(t, v):
        x, y = np.linalg.solve(M, [t - cT[2], v - cV[2]])
        return float(x), float(y)
    cal = ax["calibrationPoints"]
    err = {"x_points_T_error_K": [fwd(c["px"], c["py"])[0] - float(c["dx"]) for c in cal[:2]], "y_points_value_error_user_units": [fwd(c["px"], c["py"])[1] - float(c["dy"]) for c in cal[2:]]}
    assert max(abs(e) for e in err["x_points_T_error_K"]) <= 1.0 and max(abs(e) for e in err["y_points_value_error_user_units"]) <= 1e-3 * 1000, ("the inverse map does not reproduce the calibration", err)
    return cal, px, fwd, inv, held


def panel_measure(panel, folder, rgb, masks):
    cfg = PANELS[panel]
    fr = frame_and_ticks(masks["dark"], cfg["frame_hint"])
    xt, yt = fr["x_ticks_px"], fr["y_ticks_px"]
    assert len(xt) == len(X_TICKS), (panel, len(xt))
    assert len(yt) == len(cfg["y_ticks"]), (panel, len(yt))
    ax_, bx_ = np.polyfit(xt, X_TICKS, 1)
    u_x = float(np.abs(np.array(X_TICKS) - (ax_ * np.array(xt) + bx_)).max())
    assert u_x < 2.5, u_x
    ay_, by_ = np.polyfit(yt, cfg["y_ticks"], 1)
    rng = max(cfg["y_ticks"]) - min(cfg["y_ticks"])
    u_y = float(np.abs(np.array(cfg["y_ticks"]) - (ay_ * np.array(yt) + by_)).max())
    assert u_y < cfg["tolerance_rel"] * rng, u_y
    s, t = tilt(masks["dark"], fr)
    cal, held_px, fwd, inv, held = project_maps(folder, panel)
    # the clicked temperature calibration points sit on their printed ticks (the first and the last)
    for c, tick in ((cal[0], xt[0]), (cal[1], xt[-1])):
        assert abs(c["px"] - tick) <= 6.0, ("a clicked temperature calibration point is not on its printed tick", c["px"], tick)
    ycal_user = [{"py": c["py"], "clicked_value": float(c["dy"]), "printed_tick_value": float(ay_ * c["py"] + by_)} for c in cal[2:]]
    fx, fy = fr["fx"], fr["fy"]
    markers, unplaced = [], []
    for name, sdef in SERIES.items():
        pts = read_csv(Path(folder) / cfg["csv"][name])
        assert len(pts) == 6, (panel, name, len(pts))
        mask = masks[sdef["colour"]]
        for i, (t_click, v_user) in enumerate(pts):
            cx, cy = held_px[i] if name == held else inv(t_click, v_user)
            if name == held:
                assert abs(fwd(cx, cy)[0] - t_click) < 1e-3
            h = oh.body_find(mask, cx, cy, True, win=34, open_radius=OPEN_RADIUS)
            if h is None or abs(h[0] - cx) > 25.0:
                unplaced.append({"panel": panel, "series": name, "click_T_K": t_click, "click_px": [cx, cy], "reason": "no opened body of the series' colour near the click"})
                continue
            c_sym, n_ok, spread = oh.body_symmetry(mask, h, True, win=34, open_radius=OPEN_RADIUS, search=24)  # amendment 17: on the opened body
            fill = oh.interior_fill(mask, h[0], h[1])
            assert fill >= 0.6, (panel, name, t_click, fill)
            xc = h[0] - s * (h[1] - fy)  # the position on the bottom-axis ticks (the frame is slightly rotated)
            t_img = float(ax_ * xc + bx_)
            ypix = cy - t * (cx - fx)  # the click's row on the left-axis ticks
            value = float(ay_ * ypix + by_)
            xs = c_sym - s * (h[1] - fy)
            markers.append({"panel": panel, "series": name, "click_T_K": t_click, "click_value_user": v_user, "click_px": [cx, cy], "centre_px": h[0], "centre_row_px": h[1], "T_image_K": t_img,
                            "value": value, "centre_px_symmetry": c_sym, "n_settings_agreeing": n_ok, "interior_area_px": h[2], "centre_symmetry_minus_used_px": float(xs - xc),
                            "value_at_centre_row": float(ay_ * (h[1] - t * (h[0] - fx)) + by_), "u_K": u_x})
    for n in SERIES:
        assert sum(m["series"] == n for m in markers) + sum(u["series"] == n for u in unplaced) == 6
    return {"frame": fr, "tilt_x_per_row": s, "tilt_y_per_col": t, "x_fit": {"slope_K_per_px": float(ax_), "intercept_K": float(bx_), "max_residual_K": u_x}, "y_fit": {"slope": float(ay_), "intercept": float(by_), "max_residual": u_y,
            "tick_values": cfg["y_ticks"]}, "project_calibration_check": ycal_user, "markers": markers, "unplaceable_clicks": unplaced}


def apply_gates(res):
    """Amendment 13: the image-derived gates, applied at once to the markers of the panel (the median area is that of the series before any marker is removed)."""
    g = oh.GATES
    med = {n: float(np.median([m["interior_area_px"] for m in res["markers"] if m["series"] == n])) for n in SERIES}
    keep, gated = [], []
    for m in res["markers"]:
        why = []
        if m["interior_area_px"] < g["area_ratio"] * med[m["series"]]:
            why.append(f"interior area {m['interior_area_px']} px is below {g['area_ratio']:g} of the series median {med[m['series']]:.0f} px")
        if abs(m["centre_symmetry_minus_used_px"]) > g["sym_px"]:
            why.append(f"the symmetry fit and the interior centre differ by {abs(m['centre_symmetry_minus_used_px']):.1f} px (limit {g['sym_px']:g})")
        if abs(m["T_image_K"] - m["click_T_K"]) > g["click_K"]:
            why.append(f"the image temperature differs from the clicked one by {abs(m['T_image_K'] - m['click_T_K']):.1f} K (limit {g['click_K']:g})")
        (gated if why else keep).append({**m, "reason": "; ".join(why)} if why else m)
    res["markers"], res["gated_markers"] = keep, gated


def columns_of(res):
    """Columns of one panel: the markers of both series within COLUMN_GAP_K of each other; T_col is the mean of the markers' image temperatures (amendment 4)."""
    mk = sorted(res["markers"], key=lambda m: m["T_image_K"])
    cols = []
    for m in mk:
        if cols and m["T_image_K"] - cols[-1][-1]["T_image_K"] < oh.COLUMN_GAP_K:
            cols[-1].append(m)
        else:
            cols.append([m])
    for k, c in enumerate(cols, 1):
        assert len({m["series"] for m in c}) == len(c), (k, [m["series"] for m in c])
        ts = [m["T_image_K"] for m in c]
        assert max(ts) - min(ts) <= oh.MAX_COLUMN_RANGE_K, (k, ts)
        for m in c:
            m.update(column=k, T_col=float(np.mean(ts)), T_col_sd=float(np.std(ts, ddof=1)) if len(ts) > 1 else None, n_series=len(c), column_range=max(ts) - min(ts))
    res["temperature_axis_fit"] = {"max_residual_K": res["x_fit"]["max_residual_K"]}
    return len(cols)


def abstract_check(pdf, markers_b, markers_a):
    """The abstract's S of BSOLa5 and its temperature, and the two conductivity mantissas (page 1)."""
    o, c, r, _ = pv.page(pdf, 1)
    _, texts = pv.interpret(o, c, r)
    lines = {}
    for t in texts:  # the text operators of the page are not in reading order: group them into lines by their vertical position
        lines.setdefault(round(t["y"] / 3), []).append(t)
    s = "".join("".join(t["text"] for t in sorted(ss, key=lambda t: t["x"])) for _, ss in sorted(lines.items(), key=lambda kv: -kv[0])).replace(" ", "").replace("−", "-")
    assert f"DOI:{DOI}" in s, "the DOI of page 1 is not the one of the values"
    m = re.search(r"Seebeckcoefficientof-(\d+(?:\.\d+)?)\D{0,12}wasachievedforBSOLa5at(\d{3})", s)  # the superscripts of the units are interleaved after the temperature
    assert m, "the abstract's Seebeck coefficient was not found"
    s_text, t_text = -float(m.group(1)), float(m.group(2))
    m5 = [m_ for m_ in markers_b if m_["series"] == "BSOLa5"]
    last = max(m5, key=lambda m_: m_["T_image_K"])
    assert abs(last["T_image_K"] - t_text) <= 5.0 and abs(last["value"] - s_text) <= 3.0, ("the 573 K marker of BSOLa5 disagrees with the abstract", last["T_image_K"], last["value"], s_text)
    sig = re.search(r"BSOLa5was(\d+\.\d+).{0,300}?\((\d+\.\d+)\D{0,3}10Sm\)", s)
    out = {"abstract_S_BSOLa5_uVK": s_text, "abstract_T_K": t_text, "marker_T_K": round(last["T_image_K"], 1), "marker_S_uVK": round(last["value"], 2), "difference_uVK": round(last["value"] - s_text, 2)}
    if sig:
        la5 = max((m_ for m_ in markers_a if m_["series"] == "BSOLa5"), key=lambda m_: m_["T_image_K"])
        la10 = max((m_ for m_ in markers_a if m_["series"] == "BSOLa10"), key=lambda m_: m_["T_image_K"])
        out["abstract_sigma_mantissas_BSOLa5_BSOLa10"] = [float(sig.group(1)), float(sig.group(2))]
        out["ratio_of_mantissas"] = float(sig.group(1)) / float(sig.group(2))
        out["ratio_of_573K_markers"] = la5["value"] / la10["value"]
        out["note"] = "the abstract does not state the temperature of the conductivities; reported only"
    return out


def make_rows(results, groups):
    rows = []
    by_col = {(g["run"], g["column"]): g for g in groups}
    for panel in ("a", "b"):
        res, cfg = results[panel], PANELS[panel]
        for m in sorted(res["markers"], key=lambda m: (m["series"], m["combined_column"])):
            sd = SERIES[m["series"]]
            g = by_col[("A", m["combined_column"])]
            T = g["T"]
            tokens = ["[cluster-level]"]
            if g["edge"]:
                tokens.append(f"[edge-sensitivity: also bin {g['other_bin']:g}]")
            own = "; ".join(f"Fig. 10{x['figure']}: {x['T']:.1f} K (n = {x['n_series']} series, SD {'n/a' if x['sd'] is None else format(x['sd'], '.1f')} K, calibration u = {x['u']:.1f} K)" for x in g["members"])
            note = (" ".join(tokens) + f" T = combined consensus of column {g['column']} over {len(g['members'])} panel(s) (amendments 5 and 9), uncertainty +/- {g['u']:.1f} K, floor bin {g['floor_bin']:g} [{own}]; "
                    f"clicked T {m['click_T_K']:.1f} K (not used); the value is read at the click's pixel row on the printed ticks (the clicked value axis of the project does not match them and is not used); "
                    f"La at the Ba site, Ba1-xLaxSnO3 with x = {sd['x']:g}, sample {m['series']} of the paper; Table 2 states no temperature and is not read (amendment 14)")
            if T >= 800.0:
                note += "; at or above the 800 K window edge: recorded, not compared"
            if T < 300.0:
                note += "; below the 300 K window edge: recorded, not compared"
            rows.append({"lit_id": f"{ID_PREFIX}{cfg['prop']}_{m['series']}_c{g['column']:02d}", "compound_label": f"{m['series']} (Ba1-xLaxSnO3, x = {sd['x']:g}, as named in the paper; Fig. 10)",
                         "composition": sd["composition"], "property": cfg["prop"], "T_K": f"{T:.1f}", "value": oh.fmt3(m["value"]), "unit": cfg["unit"], "doi": DOI,
                         "location": f"{JOURNAL}, Fig. 10({panel}) (article page 5372, PDF page 10)", "value_kind": "digitised",
                         "digitisation_method": (f"bitmap digitised by hand in WebPlotDigitizer ({cfg['what']}, 3 significant digits); value at the click's pixel row on the printed ticks of the panel; temperature = combined column "
                                                 f"consensus (amendments 4 and 5) of marker centres located as the centroid of the opened body of the {sd['colour']} {sd['shape']}, read on the printed ticks (300 to 600 K) "
                                                 "with thesis_paper/scripts/na12_extract_rajasekaran.py"),
                         "notes": note})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--panel", nargs=2, action="append", metavar=("P", "DIR"))
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
    assert args.image and args.panel and {p for p, _ in args.panel} == set(PANELS), "--image and --panel a DIR --panel b DIR are required"
    folders = dict(args.panel)
    inputs = {"pdf": args.pdf, "image": str(Path(args.image).resolve())}
    for p, d in folders.items():
        inputs[f"fig10{p}_project"] = str((Path(d) / PANELS[p]["project"]).resolve())
        for n, f in PANELS[p]["csv"].items():
            inputs[f"fig10{p}_csv_{n}"] = str((Path(d) / f).resolve())
    prov = rr.provenance(inputs, __file__)
    rgb = np.array(Image.open(args.image).convert("RGB")).astype(int)
    masks = masks_of(rgb)
    results = {p: panel_measure(p, folders[p], rgb, masks) for p in PANELS}
    ab = abstract_check(REPO / args.pdf, results["b"]["markers"], results["a"]["markers"])
    for p in results:
        apply_gates(results[p])
        results[p]["n_columns"] = columns_of(results[p])
    groups = oh._combine_run(results)
    for g in groups:
        g["run"] = "A"
    rows = make_rows(results, groups)
    nl = []
    for p, res in results.items():
        for gm in res["gated_markers"]:
            nl.append({"panel": p, "series": gm["series"], "column": "", "T_K": f"{gm['T_image_K']:.1f}", "reason": f"digitised marker not legible (amendment 13): {gm['reason']}; T_K is the image temperature"})
        for u in res["unplaceable_clicks"]:
            nl.append({"panel": p, "series": u["series"], "column": "", "T_K": f"{u['click_T_K']:.1f}", "reason": f"clicked marker skipped (amendment 11): {u['reason']}; T_K is the clicked temperature"})
        for g in groups:
            if not 300.0 <= g["T"] < 800.0:
                continue
            have = {m["series"] for m in res["markers"] if m["combined_column"] == g["column"]}
            for n in SERIES:
                if n not in have:
                    nl.append({"panel": p, "series": n, "column": g["column"], "T_K": f"{g['T']:.1f}", "reason": "no digitised marker in this combined column (amendments 9 and 13); derived from the digitised files"})
    out = rr.new_run_dir("na12_extraction")
    slim = {p: {k: v for k, v in r.items()} for p, r in results.items()}
    rr.write_json(out / "extraction.json", {"combined_columns": groups, "panels": slim, "abstract_check": ab, "not_legible": nl, "doi": DOI})
    with open(out / "values_rows.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with open(out / "not_legible.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["panel", "series", "column", "T_K", "reason"], lineterminator="\n")
        w.writeheader()
        w.writerows(nl)
    rr.write_json(out / "run_config.json", {**prov, "paper": "Rajasekaran et al. 2020, Fig. 10(a) and 10(b)", "rules": "docs/decisions.md pre-registration B + amendments 1, 2, 4, 5, 6, 9, 12, 13, 14 (2026-10-13 to 19)"})
    print(f"wrote {out}: {len(rows)} rows, {len(groups)} combined columns; not legible entries {len(nl)}; abstract check: S marker {ab['marker_S_uVK']} against {ab['abstract_S_BSOLa5_uVK']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
