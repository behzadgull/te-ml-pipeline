"""
NA12: thermoelectric values of (Ca0.9M0.1)MnO3 and CaMnO3 from Ohtaki, Koga, Tokunaga, Eguchi & Arai 1995 (J. Solid State Chem. 120:105-111), read from the scan.
Rules: docs/decisions.md, pre-registration B with amendments 1 (reading rules), 4 (column-consensus marker temperatures), 5 (combined columns), 6 (2u edge tolerance), 7 and 9 (legible markers
only, not-legible list derived from the files; Ohtaki Figs. 7, 6, 4, 1 in this order). No out-of-fold prediction is read or used here.

This version reads Fig. 7 (the Bi member): ZT on the right axis (the rows) and Z on the left axis (a check only, amendment 9 item 3). Each axis has its own WebPlotDigitizer project (one dataset
each); both share the temperature axis (deg C, printed ticks 200 to 800). The clicks give the values; the temperature of a column is the consensus of the marker centres (the centroid of the enclosed
white interior of the open Z symbols and of the opened body of the filled ZT symbols, read on a straight line through the printed ticks), over the markers of the column, converted to K
(T_K = T_C + 273.15); its uncertainty is the largest residual of that line.

Validation (asserted): the CSV of each dataset equals the project's dataset; the four interior temperature ticks are found and fit a straight line (residual under 3 deg C); the clicked x
calibration points sit on their printed ticks; the y ticks (four per axis, matched to the positions predicted by the clicked calibration) reproduce the clicked y calibration within 1 % of the axis
range; every marker has an enclosed interior / opened body near its click; each dataset has at most one marker per column and the markers of a column lie within 5 deg C. Reported, not asserted
(flags in extraction.json and in the notes): ZT against Z T(K) at every column (flag outside 0.8 to 1.25, amendment 1 item 3) and the Z markers against the abstract's statement.

Usage (repository root, clean tree):
    python thesis_paper/scripts/na12_extract_ohtaki.py --figure 7 <fig7 folder> <Fig. 7 bitmap>        # writes thesis_paper/results/na12_extraction/<UTC>/
    python thesis_paper/scripts/na12_extract_ohtaki.py --append-values <dir> --replace-prefix OHTAKI1995_
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
import run_record as rr  # noqa: E402

README = "thesis_paper/source/pdfs/README.md"
JOURNAL = "J. Solid State Chem. 120:105-111 (1995)"
ID_PREFIX = "OHTAKI1995_"
C_TO_K = 273.15
# the abstract (article page 105, read from the scan): the figure of merit Z of the Bi member is 0.7 to 0.75 x 1e-4 K^-1 over 600 to 900 deg C, its maximum ZT 0.085 at 900 deg C
ABSTRACT_Z_RANGE = (0.7, 0.75)
ABSTRACT_Z_T_C = (600.0, 900.0)
ABSTRACT_ZT_MAX = (0.085, 900.0)
BI = {"tag": "Bi", "composition": "Ca0.9Bi0.1MnO3", "label": "(Ca0.9Bi0.1)MnO3 (M = Bi, as stated for Fig. {fig})"}
CAMNO3 = {"tag": "CaMnO3", "composition": "CaMnO3", "label": "CaMnO3 (undoped, as stated for Fig. {fig})"}
FIGURES = {
    # max_range_C: the largest spread of the markers of one column (deg C). The Z and ZT markers of Fig. 7 are plotted up to 5.6 deg C apart (measured on this figure; the columns are 90 to 100 deg C
    # apart and the temperature axis is good to 1 deg C), so the gate of the Ohta figures (5 K) is set to 6 deg C here; columns above 5 are flagged in the notes.
    7: {"max_range_C": 6.0, "x_tick_values": [200.0, 400.0, 600.0, 800.0], "location": f"{JOURNAL}, Fig. 7 (article page 110)", "frame_rows": (1500, 2000), "frame_min": 1200,
        "fx_range": (400, 600), "rx_range": (2100, 2300), "scope": "shared", "run": "A", "gates": dict(oh.GATES),
        "datasets": {"ZT": {"project": "zT_project.json", "csv": "zT.csv", "filled": True, "side": "right", "y_ticks": [0.02, 0.04, 0.06, 0.08], "tol": 0.001, "row": True,
                            "prop": "zT", "unit": "1", "id": "zT", "method": "clicked ZT on the right axis", "sample": BI, "estimator": "interior", "axis_range": 0.10, "res_note": "read on the right axis of Fig. 7"},
                     "Z": {"project": "z_project.json", "csv": "z.csv", "filled": False, "side": "left", "y_ticks": [0.2, 0.4, 0.6, 0.8], "tol": 0.01, "row": False,
                           "prop": "Z", "unit": "1e-4 K^-1", "id": "Z", "method": "clicked Z on the left axis", "sample": BI, "estimator": "interior", "axis_range": 1.0, "res_note": ""}}},
    # Fig. 6: the two samples have their own temperatures (amendment 9 item 4), so every sample has its own columns; the symbol of CaMnO3 is a ring with a speckled interior (no enclosed white
    # interior), so its centre is the symmetry fit (an isolated symbol, no line through it)
    6: {"x_tick_values": [200.0, 400.0, 600.0, 800.0, 1000.0], "location": f"{JOURNAL}, Fig. 6 (article page 109)", "frame_rows": (1500, 2000), "frame_min": 1200,
        "fx_range": (300, 520), "rx_range": (2000, 2200), "scope": "per_dataset", "y_frame_values": (0.0, 4.0), "gates": dict(oh.GATES),
        "datasets": {"Bi": {"project": "bi_project.json", "csv": "bi_kappa.csv", "filled": False, "side": "left", "y_ticks": [1.0, 2.0], "tol": 0.04, "row": True, "prop": "kappa", "unit": "W/mK",
                            "id": "kappa", "method": "clicked kappa in W/mK", "sample": BI, "estimator": "interior", "axis_range": 4.0, "res_note": "kappa read on the linear axis of Fig. 6"},
                     "CaMnO3": {"project": "camno3_project.json", "csv": "camno3_kappa.csv", "filled": None, "side": "left", "y_ticks": [1.0, 2.0], "tol": 0.04, "row": True, "prop": "kappa",
                                "unit": "W/mK", "id": "kappa", "method": "clicked kappa in W/mK", "sample": CAMNO3, "estimator": "symmetry", "axis_range": 4.0, "res_note": "kappa read on the linear axis of Fig. 6"}}},
}


def _fig4_series(name, tag, composition, label, csv, filled, estimator, area=(60, 1500)):
    """One series of Fig. 4 (S against temperature; the y ticks are read at the right frame, where no marker covers them; the left-frame ticks sit up to 6 px, 2 uV/K, higher): the legend symbol decides the centre method (amendment 5 item 3; the symmetry fit only for symbols with no enclosed white interior)."""
    return name, {"project": "fig4_project.json", "csv": csv, "filled": filled, "side": "right", "y_ticks": [-100.0, -200.0, -300.0, -400.0], "tol": 5.0, "row": True, "prop": "S", "unit": "uV/K",
                  "id": "S", "method": "clicked S in uV/K", "sample": {"tag": tag, "composition": composition, "label": label}, "estimator": estimator, "axis_range": 500.0, "hole": {"win": 40, "area": area},
                  "res_note": "S read on the linear axis of Fig. 4 (calibration of the one project, applied to every series)"}


# Fig. 4: ten series (nine digitised, Sm not: its speckled triangle cannot be told from Sn's solid one, amendment 9 item 2), one WebPlotDigitizer project with one calibration for all of them (its single dataset,
# "sn", must equal sn_s.csv; the pixel position of every other series' click is the inverse of the affine map fitted on that dataset's 6 clicks and checked against the project's four calibration points).
# The series are measured at their own temperatures (the clicked temperatures of the series differ by tens of deg C), so, as for Fig. 6, every marker is its own column (amendment 9 item 4).
FIGURES[4] = {
    "x_tick_values": [200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0], "location": f"{JOURNAL}, Fig. 4 (article page 108)", "frame_rows": (1500, 2000), "frame_min": 1200,
    "fx_range": (400, 600), "rx_range": (2100, 2300), "scope": "per_dataset", "y_frame_values": (-500.0, 0.0), "shared_project": "Sn", "top_tick_check": True,
    "undigitised_series": ["Sm"], "column_gap_K": oh.COLUMN_GAP_K, "skip_unplaceable": True, "gates": dict(oh.GATES),
    "datasets": dict([
        _fig4_series("Bi", "Bi", "Ca0.9Bi0.1MnO3", "(Ca0.9Bi0.1)MnO3 (M = Bi, as stated for Fig. {fig})", "bi_s.csv", False, "interior"),
        _fig4_series("Ce", "Ce", "Ca0.9Ce0.1MnO3", "(Ca0.9Ce0.1)MnO3 (M = Ce, as stated for Fig. {fig})", "ce_s.csv", True, "interior"),
        _fig4_series("Y", "Y", "Ca0.9Y0.1MnO3", "(Ca0.9Y0.1)MnO3 (M = Y, as stated for Fig. {fig})", "y_s.csv", True, "interior"),
        _fig4_series("Sb", "Sb", "Ca0.9Sb0.1MnO3", "(Ca0.9Sb0.1)MnO3 (M = Sb, as stated for Fig. {fig})", "sb_s.csv", False, "interior", area=(60, 3200)),
        _fig4_series("La", "La", "Ca0.9La0.1MnO3", "(Ca0.9La0.1)MnO3 (M = La, as stated for Fig. {fig})", "la_s.csv", False, "interior"),
        _fig4_series("Pb", "Pb", "Ca0.9Pb0.1MnO3", "(Ca0.9Pb0.1)MnO3 (M = Pb, as stated for Fig. {fig})", "pb_s.csv", None, "symmetry"),
        _fig4_series("In", "In", "Ca0.9In0.1MnO3", "(Ca0.9In0.1)MnO3 (M = In, as stated for Fig. {fig})", "in_s.csv", None, "symmetry"),
        _fig4_series("Sn", "Sn", "Ca0.9Sn0.1MnO3", "(Ca0.9Sn0.1)MnO3 (M = Sn, as stated for Fig. {fig})", "sn_s.csv", True, "interior"),
        _fig4_series("CaMnO3", "CaMnO3", "CaMnO3", "CaMnO3 (undoped, as stated for Fig. {fig})", "camno3_s.csv", None, "symmetry"),
    ]),
}


def _fig1_series(name, tag, composition, label, csv, filled, estimator, area=(60, 1500)):
    """One series of Fig. 1 (log10 sigma, S/cm, against 1000/T): the same symbols as Fig. 4; the click value is log10(sigma / S cm^-1), converted to S/m (x 100) in the row."""
    n, d = _fig4_series(name, tag, composition, label, csv, filled, estimator, area)
    d.update({"project": "fig1_project.json", "side": "left", "y_ticks": [2.0, 1.0, 0.0], "tol": 0.05, "prop": "sigma", "unit": "S/m", "id": "sigma", "axis_range": 4.0, "value_transform": "log10_Scm_to_Sm",
              "method": "clicked log10(sigma / S cm^-1) on the linear log axis, converted to S/m", "res_note": "log10 sigma read on the linear axis of Fig. 1 (calibration of the one project, applied to every series), converted from S/cm to S/m (x 100)"})
    return n, d


# Fig. 1: ten series (nine digitised, Sm not), one project and one calibration, as Fig. 4. The x axis is 1000/T (linear, printed ticks every 0.5 from 1.0 to 4.0): T = 1000/x, and u(T) = T^2/1000 u(x) with u(x) the
# largest residual of the straight lines through the bottom and the top ticks. The figure is tilted (the top ticks sit up to 6 px left of the bottom ones), so x is interpolated between the two tick lines at the
# marker's height. The y axis is log10(sigma / S cm^-1) as plain numbers (ticks 2, 1, 0; the frame lines are -1 and 3).
FIGURES[1] = {
    "x_kind": "inv_T", "x_tick_values": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], "u_max": 0.02, "x_cal_tol": 0.02, "top_tick_check": True, "use_top_ticks": True,
    "location": f"{JOURNAL}, Fig. 1 (article page 107)", "frame_rows": (1500, 2000), "frame_min": 1200,
    "fx_range": (400, 600), "rx_range": (2100, 2300), "scope": "per_dataset", "y_frame_values": (-1.0, 3.0), "shared_project": "Ce",
    "undigitised_series": ["Sm"], "reference_series": "Bi", "column_gap_K": oh.COLUMN_GAP_K, "skip_unplaceable": True, "gates": dict(oh.GATES),
    "datasets": dict([
        _fig1_series("Bi", "Bi", "Ca0.9Bi0.1MnO3", "(Ca0.9Bi0.1)MnO3 (M = Bi, as stated for Fig. {fig})", "bi_sigma.csv", False, "interior"),
        _fig1_series("Ce", "Ce", "Ca0.9Ce0.1MnO3", "(Ca0.9Ce0.1)MnO3 (M = Ce, as stated for Fig. {fig})", "ce_sigma.csv", True, "interior"),
        _fig1_series("Y", "Y", "Ca0.9Y0.1MnO3", "(Ca0.9Y0.1)MnO3 (M = Y, as stated for Fig. {fig})", "y_sigma.csv", True, "interior"),
        _fig1_series("Sb", "Sb", "Ca0.9Sb0.1MnO3", "(Ca0.9Sb0.1)MnO3 (M = Sb, as stated for Fig. {fig})", "sb_sigma.csv", False, "interior", area=(60, 3200)),
        _fig1_series("La", "La", "Ca0.9La0.1MnO3", "(Ca0.9La0.1)MnO3 (M = La, as stated for Fig. {fig})", "la_sigma.csv", False, "interior"),
        _fig1_series("Pb", "Pb", "Ca0.9Pb0.1MnO3", "(Ca0.9Pb0.1)MnO3 (M = Pb, as stated for Fig. {fig})", "pb_sigma.csv", None, "symmetry"),
        _fig1_series("In", "In", "Ca0.9In0.1MnO3", "(Ca0.9In0.1)MnO3 (M = In, as stated for Fig. {fig})", "in_sigma.csv", None, "symmetry"),
        _fig1_series("Sn", "Sn", "Ca0.9Sn0.1MnO3", "(Ca0.9Sn0.1)MnO3 (M = Sn, as stated for Fig. {fig})", "sn-redo_sigma.csv", True, "interior"),
        _fig1_series("CaMnO3", "CaMnO3", "CaMnO3", "CaMnO3 (undoped, as stated for Fig. {fig})", "camno3_sigma.csv", None, "symmetry"),
    ]),
}


def doi_from_readme():
    """The DOI of ohtaki1995.pdf from the PDF README (the scan prints none)."""
    for line in (REPO / README).read_text(encoding="utf-8").splitlines():
        if "ohtaki1995.pdf" in line:
            m = re.search(r"(10\.\d{4}/[A-Za-z0-9.()/_-]+?)\s*\|", line)
            if m:
                return m.group(1)
    raise AssertionError("DOI of ohtaki1995.pdf not found in the PDF README")


def frame_and_ticks(dark, cfg):
    """Frame rows/columns and the interior temperature ticks (px) of the bitmap."""
    r0, r1 = cfg["frame_rows"]
    rows = dark[:, 400:2300].sum(axis=1)
    bot = [r for r in range(r0, r1) if rows[r] > cfg["frame_min"]]
    top = [r for r in range(150, 400) if rows[r] > cfg["frame_min"]]
    fy, ty = bot[len(bot) // 2], top[len(top) // 2]
    fa, fb = cfg["fx_range"]
    ra, rb = cfg["rx_range"]
    fx = int(np.argmax(dark[200:1800, fa:fb].sum(axis=0))) + fa
    rx = int(np.argmax(dark[200:1800, ra:rb].sum(axis=0))) + ra
    prof = dark[fy - 28: fy - 8, fx + 30: rx - 30].sum(axis=0)
    xt = [x + fx + 30 for x in oh.runs(prof >= int(prof.max() * 0.7))]
    return {"fy": fy, "ty": ty, "fx": fx, "rx": rx, "x_ticks_px": xt}


def y_ticks(dark, fr, side, values, cal):
    """Pixel rows of the printed y ticks of one axis, matched to the positions predicted by the clicked calibration (cal: its two y points)."""
    fx, rx, ty, fy = fr["fx"], fr["rx"], fr["ty"], fr["fy"]
    sl = slice(fx + 8, fx + 30) if side == "left" else slice(rx - 30, rx - 8)
    prof = dark[ty + 8: fy - 8, sl].sum(axis=1)
    found = [y + ty + 8 for y in oh.runs(prof >= int(max(prof.max() * 0.7, 1)))]
    (y1, y2) = cal
    v1, v2 = float(y1["dy"]), float(y2["dy"])
    out = []
    for v in values:
        pred = y1["py"] + (v - v1) / (v2 - v1) * (y2["py"] - y1["py"])
        j = int(np.argmin([abs(f - pred) for f in found]))
        assert abs(found[j] - pred) <= 15.0, (side, v, pred, found)
        out.append(found[j])
    assert len(set(out)) == len(out)
    return out


def load_dataset(folder, d):
    """The dataset's CSV points, the project's calibration points and the pixel position of every click (the project holds exactly this dataset)."""
    pts = []
    for line in (Path(folder) / d["csv"]).read_text(encoding="utf-8").splitlines():
        if line.strip():
            a, b = line.split(",")
            pts.append((float(a), float(b)))
    proj = json.loads((Path(folder) / d["project"]).read_text(encoding="utf-8"))
    ax = proj["axesColl"][0]
    assert not ax["isLogX"] and not ax["isLogY"]
    assert len(proj["datasetColl"]) == 1, "the project must hold exactly one dataset"
    ds = proj["datasetColl"][0]
    assert len(ds["data"]) == len(pts)
    for p, (t, v) in zip(ds["data"], pts):
        assert abs(p["value"][0] - t) < 1e-6 and abs(p["value"][1] - v) < 1e-9 * max(1.0, abs(v)), (p["value"], (t, v))
    return pts, ax["calibrationPoints"], [(p["x"], p["y"]) for p in ds["data"]], ds["name"]


def _read_csv(path):
    pts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            a, b = line.split(",")
            pts.append((float(a), float(b)))
    return pts


def shared_project_maps(folder, cfg):
    """One project, one calibration, several series (Fig. 4). The project's single dataset must equal its CSV; the affine map (px, py) -> (T, value) is fitted exactly on that dataset's clicks (WebPlotDigitizer
    rotated the axes: the two temperature points are at different pixel rows), checked against the project's four calibration points, and inverted to give every other series' clicks their pixel positions.
    Returns (calibration points, held dataset name, held clicks px, forward map, inverse map, check dict)."""
    held = cfg["shared_project"]
    d0 = cfg["datasets"][held]
    proj = json.loads((Path(folder) / d0["project"]).read_text(encoding="utf-8"))
    ax = proj["axesColl"][0]
    assert not ax["isLogX"] and not ax["isLogY"] and len(proj["datasetColl"]) == 1
    ds = proj["datasetColl"][0]
    pts = _read_csv(Path(folder) / d0["csv"])
    assert len(ds["data"]) == len(pts) >= 6
    for p, (t, v) in zip(ds["data"], pts):
        assert abs(p["value"][0] - t) < 1e-6 and abs(p["value"][1] - v) < 1e-9 * max(1.0, abs(v)), (p["value"], (t, v))
    px = [(p["x"], p["y"]) for p in ds["data"]]
    A = np.array([[x, y, 1.0] for x, y in px])
    cT = np.linalg.lstsq(A, np.array([t for t, v in pts]), rcond=None)[0]
    cV = np.linalg.lstsq(A, np.array([v for t, v in pts]), rcond=None)[0]
    assert np.abs(A @ cT - np.array([t for t, v in pts])).max() < 1e-4 and np.abs(A @ cV - np.array([v for t, v in pts])).max() < 1e-8, "calibration is not affine"
    M = np.array([[cT[0], cT[1]], [cV[0], cV[1]]])

    def fwd(x, y):
        return float(cT[0] * x + cT[1] * y + cT[2]), float(cV[0] * x + cV[1] * y + cV[2])

    def inv(t, v):
        x, y = np.linalg.solve(M, [t - cT[2], v - cV[2]])
        return float(x), float(y)
    cal = ax["calibrationPoints"]
    chk = {"x_points_T_error_K": [fwd(c["px"], c["py"])[0] - float(c["dx"]) for c in cal[:2]], "y_points_value_error": [fwd(c["px"], c["py"])[1] - float(c["dy"]) for c in cal[2:]]}
    assert max(abs(e) for e in chk["x_points_T_error_K"]) <= cfg.get("x_cal_tol", 3.0) and max(abs(e) for e in chk["y_points_value_error"]) <= d0["tol"], ("the fitted map disagrees with the project's calibration points", chk)
    return cal, held, px, fwd, inv, chk


def load_dataset_fig4(folder, cfg, name, maps):
    """Points of one Fig. 4 series and the pixel position of each click (the held dataset: its own clicks; the others: the inverse map)."""
    cal, held, hpx, fwd, inv, chk = maps
    pts = _read_csv(Path(folder) / cfg["datasets"][name]["csv"])
    px = hpx if name == held else [inv(t, v) for t, v in pts]
    return pts, cal, px, ("held in the project" if name == held else "not held: pixel positions from the inverse map")


def measure(fig, folder, image):
    cfg = FIGURES[fig]
    maps = shared_project_maps(folder, cfg) if cfg.get("shared_project") else None
    dark = np.array(Image.open(image).convert("L")) < oh.THRESHOLD
    fr = frame_and_ticks(dark, cfg)
    xt, xv = fr["x_ticks_px"], cfg["x_tick_values"]
    assert len(xt) == len(xv), f"{len(xt)} interior temperature ticks found"
    a, b = np.polyfit(xt, xv, 1)
    u_c = float(np.abs(np.array(xv) - (a * np.array(xt) + b)).max())
    inv = cfg.get("x_kind") == "inv_T"  # Fig. 1: the axis is 1000/T (linear); u_c and the tick fits are then in units of 1000/T and T = 1000/x
    assert u_c < cfg.get("u_max", 3.0), u_c
    top = None
    if cfg.get("top_tick_check"):  # reported only: the printed ticks along the top frame, to see whether the figure's temperature axis is the same at the top as at the bottom
        r_top = dark[fr["ty"] + 8: fr["ty"] + 28, fr["fx"] + 30: fr["rx"] - 30].sum(axis=0)
        tt = [x + fr["fx"] + 30 for x in oh.runs(r_top >= int(r_top.max() * 0.7))]
        assert len(tt) == len(xv), f"{len(tt)} top ticks found"
        a_t, b_t = np.polyfit(tt, xv, 1)
        top = {"top_ticks_px": tt, "bottom_minus_top_tick_px": [float(p - q) for p, q in zip(xt, tt)], "max_residual_K": float(np.abs(np.array(xv) - (a_t * np.array(tt) + b_t)).max()), "line": (float(a_t), float(b_t))}
    u_x = max(u_c, top["max_residual_K"]) if (inv and top) else u_c  # the tick residual of the axis (both tick lines when both are used)
    mk, info, unplaced = [], {}, []
    for name, d in cfg["datasets"].items():
        pts, cal, px, pname = load_dataset_fig4(folder, cfg, name, maps) if maps else load_dataset(folder, d)
        for c in cal[:2]:  # clicked x calibration points sit on their printed ticks
            j = int(np.argmin([abs(c["px"] - t) for t in xt]))
            assert abs(c["px"] - xt[j]) <= 10.0 and abs(xv[j] - float(c["dx"])) < 1e-6, ("x calibration point not on its tick", c)
        yt = y_ticks(dark, fr, d["side"], d["y_ticks"], cal[2:])
        yv = list(d["y_ticks"])
        yp = list(yt)
        if "y_frame_values" in cfg:  # the frame lines are the 0 and the top of the axis (a tick hidden by a marker, or none inside)
            yp += [fr["fy"], fr["ty"]]
            yv += list(cfg["y_frame_values"])
        ay, by = np.polyfit(yp, yv, 1)
        rng = float(d["axis_range"])
        dev = [float(ay * c["py"] + by - float(c["dy"])) for c in cal[2:]]
        assert max(abs(x) for x in dev) <= d.get("tol", 0.01 * rng), ("clicked y calibration differs from the printed ticks", name, dev)
        marker_dev = [float(v - (ay * cy + by)) for (t_c, v), (cx, cy) in zip(pts, px)]  # the clicked values against the printed ticks at every marker (a rotated calibration drifts)
        assert max(abs(x) for x in marker_dev) <= 0.03 * rng, ("clicked values differ from the printed ticks", name, marker_dev)
        info[name] = {"project_dataset_name": pname, "y_ticks_px": yt, "y_fit_slope_per_px": float(ay), "y_fit_intercept": float(by), "clicked_y_calibration_minus_ticks": dev, "clicked_value_minus_ticks_at_markers_max": max(abs(x) for x in marker_dev),
                      "x_calibration_points_py_difference_px": float(cal[0]["py"] - cal[1]["py"]),
                      "x_calibration_offsets_px": [float(c["px"] - xt[int(np.argmin([abs(c["px"] - t) for t in xt]))]) for c in cal[:2]]}
        prev = -1e9
        for (t_c, v), (cx, cy) in zip(pts, px):
            if inv:
                t_c = 1000.0 / t_c - C_TO_K  # the click's own temperature (deg C) from its 1000/T
            c_sym, n_ok, spread = oh.refine_x(dark, cx, cy, 24)  # the raw column-profile fit: used for the symbols with no enclosed interior only (amendment 17)
            c_raw = c_sym
            area = None
            cyc = None
            try:
                if d["estimator"] == "interior":
                    h = oh.body_find(dark, cx, cy, d["filled"], **d.get("hole", {}))
                    assert h is not None and abs(h[0] - cx) <= 25.0, f"{name} marker at {t_c:.0f} deg C: no enclosed interior / body near the click"
                    c_sym, n_ok, spread = oh.body_symmetry(dark, h, d["filled"], win=d.get("hole", {}).get("win", 34), search=24)  # amendment 17: on the opened body
                    c, area, cyc = h[0], h[2], h[1]
                else:  # an isolated symbol without an enclosed interior: the symmetry fit, which must be a majority of the settings and near the click
                    assert n_ok >= oh.MIN_AGREE and abs(c_sym - cx) <= 25.0, f"{name} marker at {t_c:.0f} deg C: the symmetry fit is not reliable ({n_ok} of {len(oh.SETTINGS)} settings, {c_sym - cx:.1f} px from the click)"
                    c = c_sym
            except AssertionError as e:
                if not cfg.get("skip_unplaceable"):
                    raise
                unplaced.append({"dataset": name, "click_T_C": t_c, "click_value": v, "click_px": [cx, cy], "reason": str(e)})  # amendment 9: the centre cannot be placed, so the marker is not legible
                continue
            t_img = float(a * c + b)
            t_alt = None
            u_K = None
            if inv:  # x = 1000/T read on the bottom tick line, or interpolated between the bottom and the top tick lines at the marker's height (the figure is tilted: the top ticks sit up to 6 px left of the bottom ones)
                x_img = t_img
                if top and cfg.get("use_top_ticks"):
                    frac = (fr["fy"] - cy) / (fr["fy"] - fr["ty"])
                    x_img = float((1 - frac) * t_img + frac * (top["line"][0] * c + top["line"][1]))
                T_K = 1000.0 / x_img
                t_img, u_K = float(T_K - C_TO_K), float(T_K ** 2 / 1000.0 * u_x)  # u(T) = T^2/1000 u(x)
                t_alt = float(1000.0 / (a * c + b) - C_TO_K)  # reported, not used: T on the bottom tick line alone
            elif top:  # the temperature at the marker's own height, interpolating between the bottom-tick line and the top-tick line (reported, not used)
                frac = (fr["fy"] - cy) / (fr["fy"] - fr["ty"])
                t_alt = float((1 - frac) * t_img + frac * (top["line"][0] * c + top["line"][1]))
            assert c > prev  # the markers of a series are clicked in order along the temperature axis
            prev = c
            mk.append({"dataset": name, "click_T_C": t_c, "click_value": v, "centre_px": c, "click_px": [cx, cy], "T_image_C": t_img, "centre_px_symmetry": c_sym, "centre_px_symmetry_raw": c_raw, "n_settings_agreeing": n_ok,
                       "centre_method": d["estimator"], "fill": oh.interior_fill(dark, c, cy), "sample": d["sample"]["tag"], "T_image_topbottom_interpolated_C": t_alt,
                       "centre_symmetry_minus_used_px": float(c_sym - c), "interior_area_px": area, "u_K": u_K, "value_from_centre_row_minus_click": (None if cyc is None else float(ay * cyc + by - v)),
                       "value_out": (100.0 * 10.0 ** v if d.get("value_transform") == "log10_Scm_to_Sm" else v)})
        if d["filled"] is not None:
            fills = float(np.median([m["fill"] for m in mk if m["dataset"] == name]))
            assert (fills >= 0.6) == d["filled"], f"{name}: symbols are {'filled' if fills >= 0.6 else 'open'}, expected {'filled' if d['filled'] else 'open'}"
    gated = []
    if cfg.get("gates"):  # amendment 12: image-derived legibility gates, applied to all placed markers at once (the median area is that of the series before any marker is removed)
        g = cfg["gates"]
        med = {n: float(np.median([m["interior_area_px"] for m in mk if m["dataset"] == n and m["interior_area_px"]] or [np.nan])) for n in cfg["datasets"]}
        keep = []
        for m in mk:
            why = []
            if m["interior_area_px"] and m["interior_area_px"] < g["area_ratio"] * med[m["dataset"]]:
                why.append(f"interior area {m['interior_area_px']} px is below {g['area_ratio']:g} of the series median {med[m['dataset']]:.0f} px")
            if m["centre_method"] == "interior" and abs(m["centre_symmetry_minus_used_px"]) > g["sym_px"]:
                why.append(f"the symmetry fit and the interior centre differ by {abs(m['centre_symmetry_minus_used_px']):.1f} px (limit {g['sym_px']:g})")
            dT = m["T_image_C"] - m["click_T_C"]
            if abs(dT) > g["click_K"]:
                why.append(f"the image temperature differs from the clicked one by {abs(dT):.1f} K (limit {g['click_K']:g})")
            if why:
                gated.append({"dataset": m["dataset"], "T_image_K": m["T_image_C"] + C_TO_K, "click_T_C": m["click_T_C"], "click_value": m["click_value"], "reason": "; ".join(why)})
            else:
                keep.append(m)
        mk = keep
    scopes = [list(range(len(mk)))] if cfg["scope"] == "shared" else [[i for i, m in enumerate(mk) if m["dataset"] == n] for n in cfg["datasets"]]
    n_cols = 0
    for scope in scopes:
        order = sorted(scope, key=lambda i: mk[i]["T_image_C"])
        cols = []
        for i in order:
            if cfg["scope"] == "shared" and cols and mk[i]["T_image_C"] - mk[cols[-1][-1]]["T_image_C"] < oh.COLUMN_GAP_K:
                cols[-1].append(i)
            else:  # a sample with its own temperatures: every marker is its own column
                cols.append([i])
        for k, c in enumerate(cols, 1):
            names = [mk[i]["dataset"] for i in c]
            assert len(set(names)) == len(names), (k, names)
            ts = [mk[i]["T_image_C"] for i in c]
            assert max(ts) - min(ts) <= cfg.get("max_range_C", oh.MAX_COLUMN_RANGE_K), (k, ts)
            m_k = float(np.mean(ts)) + C_TO_K
            sd = float(np.std(ts, ddof=1)) if len(ts) > 1 else None
            run = cfg["run"] if cfg["scope"] == "shared" else f"{fig}:{mk[c[0]]['dataset']}"
            for i in c:
                mk[i].update(column=k, T_col=m_k, T_col_sd=sd, n_series=len(c), column_range=max(ts) - min(ts), column_basis="interior", run=run)
        n_cols += len(cols)
    out = {"figure": fig, "temperature_axis_fit": {"slope_C_per_px": float(a), "intercept_C": float(b), "max_residual_K": (max(m["u_K"] for m in mk) if inv else u_c), "tick_px": xt, "tick_values_C": xv,
                                                    **({"x_axis": "1000/T (linear)", "max_residual_in_1000_over_T": u_x} if inv else {})}, "frame": fr,
           "datasets": info, "n_columns": n_cols, "markers": mk, "unplaceable_clicks": unplaced, "gated_markers": gated}
    if top:
        alts = [abs(m["T_image_topbottom_interpolated_C"] - m["T_image_C"]) for m in mk if m["T_image_topbottom_interpolated_C"] is not None]
        out["top_tick_check"] = {**{k: v for k, v in top.items() if k != "line"}, "max_abs_T_difference_K": max(alts), "note": ("T of every marker on the bottom tick line alone, against the interpolated T used (reported, not used)" if inv else "reported, not used: T of every marker if the temperature axis is interpolated between the bottom and the top ticks")}
    if maps:
        out["shared_project_calibration_check"] = maps[5]
    return out


def z_check(res):  # Fig. 7 only
    """ZT = Z T(K): the ZT marker of a column against Z (1e-4 K^-1) times the combined column temperature; ratio flagged outside 0.8 to 1.25 (reported, not asserted)."""
    out = []
    by = {}
    for m in res["markers"]:
        by.setdefault(m["combined_column"], {})[m["dataset"]] = m
    for col, d in sorted(by.items()):
        if "ZT" in d and "Z" in d:
            T = d["ZT"]["T_comb"]
            zt_calc = d["Z"]["click_value"] * 1e-4 * T
            ratio = d["ZT"]["click_value"] / zt_calc
            out.append({"column": col, "T_K": round(T, 1), "Z_1e-4_per_K": d["Z"]["click_value"], "ZT_marker": d["ZT"]["click_value"], "ZT_from_Z_T": zt_calc, "ratio": ratio, "flag": bool(not 0.8 <= ratio <= 1.25)})
        else:
            out.append({"column": col, "reason": "only one of Z and ZT digitised at this column"})
    return out


def text_checks(res):
    """The abstract's statements against the digitised markers (information, not asserted)."""
    zs = [m for m in res["markers"] if m["dataset"] == "Z"]
    zt = [m for m in res["markers"] if m["dataset"] == "ZT"]
    lo, hi = ABSTRACT_Z_T_C
    inr = [m for m in zs if lo <= m["T_image_C"] <= hi]
    out = {"abstract_Z_range_1e-4_per_K": ABSTRACT_Z_RANGE, "abstract_Z_temperature_range_C": ABSTRACT_Z_T_C,
           "Z_markers_in_that_range": [{"T_C": round(m["T_image_C"], 1), "Z": round(m["click_value"], 4)} for m in inr],
           "abstract_max_ZT": ABSTRACT_ZT_MAX}
    near = [m for m in zt if abs(m["T_image_C"] - ABSTRACT_ZT_MAX[1]) <= 30.0]
    out["ZT_marker_near_900_C"] = [{"T_C": round(m["T_image_C"], 1), "ZT": round(m["click_value"], 4)} for m in near] or "not digitised: the stated maximum cannot be checked with these files"
    out["max_ZT_digitised"] = {"T_C": round(max(zt, key=lambda m: m["click_value"])["T_image_C"], 1), "ZT": round(max(m["click_value"] for m in zt), 4)}
    return out


def make_rows(results, groups, doi, checks):
    rows = []
    by_col = {(g["run"], g["column"]): g for g in groups}
    for fig in sorted(results):
        res, cfg = results[fig], FIGURES[fig]
        for name, d in cfg["datasets"].items():
            if not d["row"]:
                continue
            smp = d["sample"]
            for m in sorted((m for m in res["markers"] if m["dataset"] == name), key=lambda m: m["combined_column"]):
                g = by_col[(m["run"], m["combined_column"])]
                T = g["T"]
                tokens = []
                if g["edge"]:
                    tokens.append(f"[edge-sensitivity: also bin {g['other_bin']:g}]")
                own = "; ".join(f"Fig. {x['figure']}: {x['T']:.1f} K (n = {x['n_series']} markers, SD {'n/a' if x['sd'] is None else format(x['sd'], '.1f')} K, calibration u = {x['u']:.1f} K)" for x in g["members"])
                zc = next((c for c in checks if c.get("column") == m["combined_column"] and "ratio" in c), None) if fig == 7 else None
                note = (" ".join(tokens) + (" " if tokens else "") + f"T = combined consensus of column {g['column']} over {len(g['members'])} figure(s) (amendments 5 and 9), uncertainty +/- {g['u']:.1f} K, "
                        f"floor bin {g['floor_bin']:g} [{own}]; clicked T {m['click_T_C']:.1f} deg C (not used); {d['res_note']}")
                if d["estimator"] != "interior":
                    note += f"; marker centre by the symmetry fit ({m['n_settings_agreeing']} of {len(oh.SETTINGS)} settings agree): the symbol is a ring with a speckled interior"
                wide = [x for x in g["members"] if x["range"] > oh.MAX_COLUMN_RANGE_K]
                if wide:
                    note += "; the Z and ZT markers of this column are plotted " + ", ".join(f"{x['range']:.1f} deg C" for x in wide) + " apart (the column temperature is their mean)"
                if zc:
                    note += f"; ZT / (Z T) check with the Z marker of the same column: ratio {zc['ratio']:.2f}" + (" [FLAG outside 0.8 to 1.25]" if zc["flag"] else "")
                if T >= 800.0:
                    note += "; at or above the 800 K window edge: recorded, not compared"
                if T < 300.0:
                    note += "; below the 300 K window edge: recorded, not compared (room temperature)"
                rows.append({
                    "lit_id": f"{ID_PREFIX}{d['id']}_{smp['tag']}_c{g['column']:02d}", "compound_label": smp["label"].format(fig=fig), "composition": smp["composition"], "property": d["prop"],
                    "T_K": f"{T:.1f}", "value": oh.fmt3(m["value_out"]), "unit": d["unit"], "doi": doi, "location": cfg["location"], "value_kind": "digitised",
                    "digitisation_method": (f"1-bit scan digitised by hand in WebPlotDigitizer ({d['method']}, 3 significant digits); temperature = column consensus (amendments 4, 5 and 9) of "
                                            "marker centres " + ("located as the centroid of the enclosed interior / opened body of the symbols" if d["estimator"] == "interior" else "located by the symmetry fit")
                                            + (f", read as 1000/T on the printed ticks ({cfg['x_tick_values'][0]:g} to {cfg['x_tick_values'][-1]:g}), interpolated between the bottom and the top tick lines at the marker's height, T = 1000/x, "
                                               if cfg.get("x_kind") == "inv_T" else f", read on a straight line through the printed ticks ({cfg['x_tick_values'][0]:g} to {cfg['x_tick_values'][-1]:g} deg C) and converted to K, ")
                                            + "with thesis_paper/scripts/na12_extract_ohtaki.py"),
                    "notes": note})
    return rows


def not_legible_rows(results, groups):
    """Amendment 9: every column of a figure inside the window with each row dataset that has no digitised marker in it (derived from the digitised files)."""
    rows = []
    for fig, res in sorted(results.items()):
        cfg = FIGURES[fig]
        for gm in res.get("gated_markers", []):  # amendments 12 and 13: markers removed by the image-derived gates
            rows.append({"figure": fig, "series": f"{cfg['datasets'][gm['dataset']]['sample']['tag']} ({gm['dataset']})", "column": "", "T_K": f"{gm['T_image_K']:.1f}",
                         "reason": f"digitised marker not legible (amendments 12 and 13): {gm['reason']}; T_K is the image temperature"})
        if cfg.get("reference_series"):  # Fig. 1 (amendment 12): the series share measurement columns; the columns are those of the reference series plus any column where another series has a marker more than the gap from every reference column
            gap = cfg["column_gap_K"]
            ref = sorted(m["T_col"] for m in res["markers"] if m["dataset"] == cfg["reference_series"])
            cols = list(ref)
            extra = sorted(m["T_col"] for m in res["markers"] if m["dataset"] != cfg["reference_series"] and all(abs(m["T_col"] - r) > gap for r in ref))
            grp = []
            for t in extra:
                if grp and t - grp[-1][-1] <= gap:
                    grp[-1].append(t)
                else:
                    grp.append([t])
            cols += [float(np.mean(g)) for g in grp]
            series = [(d["sample"]["tag"], n) for n, d in cfg["datasets"].items() if d["row"]] + [(t, None) for t in cfg.get("undigitised_series", [])]
            for k, T in enumerate(sorted(cols), 1):
                if not 300.0 <= T < 800.0:
                    continue
                for tag, name in series:
                    if name is None or not any(m["dataset"] == name and abs(m["T_col"] - T) <= gap for m in res["markers"]):
                        rows.append({"figure": fig, "series": f"{tag} ({'no file: not digitised' if name is None else name})", "column": k, "T_K": f"{T:.1f}",
                                     "reason": f"no digitised marker within {gap:g} K: not legible (amendments 9 and 12); column of the reference series {cfg['reference_series']}" + (" (or an extra column of another series)" if any(abs(T - float(np.mean(g))) < 1e-9 for g in grp) else "")})
            for u in res.get("unplaceable_clicks", []):
                rows.append({"figure": fig, "series": f"{cfg['datasets'][u['dataset']]['sample']['tag']} ({u['dataset']})", "column": "", "T_K": f"{u['click_T_C'] + C_TO_K:.1f}",
                             "reason": f"clicked marker skipped (amendment 11): {u['reason']}; T_K is the clicked temperature"})
            continue
        if cfg.get("column_gap_K"):  # Fig. 4: the series have no common columns, so the columns are derived by single-linkage clustering of all digitised markers of the figure (gap below column_gap_K)
            mks = sorted(res["markers"], key=lambda m: m["T_col"])
            clusters = []
            for m in mks:
                if clusters and m["T_col"] - clusters[-1][-1]["T_col"] < cfg["column_gap_K"]:
                    clusters[-1].append(m)
                else:
                    clusters.append([m])
            series = [(d["sample"]["tag"], n) for n, d in cfg["datasets"].items() if d["row"]] + [(t, None) for t in cfg.get("undigitised_series", [])]
            for k, c in enumerate(clusters, 1):
                T = float(np.mean([m["T_col"] for m in c]))
                if not 300.0 <= T < 800.0:
                    continue
                have = {m["dataset"] for m in c}
                if 2 * len(have) < len(series):  # amendment 11: a column of the figure needs markers of at least half of the series (a column of one or two series is those series' own extra temperature)
                    continue
                for tag, name in series:
                    if name is None or name not in have:
                        rows.append({"figure": fig, "series": f"{tag} ({'no file: not digitised' if name is None else name})", "column": k, "T_K": f"{T:.1f}",
                                     "reason": f"no digitised marker: the centre could not be placed (amendment 9); derived from the digitised files; column = cluster of {len(c)} markers of {len(have)} of {len(series)} series "
                                               f"(markers within {cfg['column_gap_K']:g} K of each other), range {max(m['T_col'] for m in c) - min(m['T_col'] for m in c):.1f} K"})
            for u in res.get("unplaceable_clicks", []):  # amendment 11: clicks whose centre the estimator could not place (the marker is not legible)
                rows.append({"figure": fig, "series": f"{FIGURES[fig]['datasets'][u['dataset']]['sample']['tag']} ({u['dataset']})", "column": "", "T_K": f"{u['click_T_C'] + C_TO_K:.1f}",
                             "reason": f"clicked marker skipped (amendment 11): {u['reason']}; T_K is the clicked temperature"})
            continue
        if FIGURES[fig]["scope"] != "shared":
            continue  # the samples of this figure have their own temperatures: a skipped marker leaves no column to derive it from (all markers were digitised)
        for name, d in FIGURES[fig]["datasets"].items():
            if not d["row"]:
                continue
            have = {m["combined_column"] for m in res["markers"] if m["dataset"] == name}
            for g in groups:
                if g["run"] == FIGURES[fig]["run"] and 300.0 <= g["T"] < 800.0 and g["column"] not in have:
                    rows.append({"figure": fig, "series": f"{d['sample']['tag']} ({name})", "column": g["column"], "T_K": f"{g['T']:.1f}",
                                 "reason": "no digitised marker: the centre could not be placed (amendment 9); derived from the digitised files"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", nargs=3, action="append", metavar=("N", "DIR", "IMAGE"))
    ap.add_argument("--append-values", default=None)
    ap.add_argument("--replace-prefix", default=None)
    args = ap.parse_args()
    if args.append_values:
        removed, added = oh.replace_values(Path(args.append_values), args.replace_prefix or ID_PREFIX)
        print(f"removed {removed} rows with prefix {args.replace_prefix or ID_PREFIX}, added {added} rows from {args.append_values}")
        return 0
    assert args.figure, "--figure N DIR IMAGE is required"
    figs = {}
    for n, d, img in args.figure:
        assert int(n) in FIGURES and int(n) not in figs, n
        figs[int(n)] = (d, img)
    inputs = {"readme": README}
    for n, (d, img) in sorted(figs.items()):
        inputs[f"fig{n}_image"] = str(Path(img).resolve())
        for name, ds in FIGURES[n]["datasets"].items():
            inputs[f"fig{n}_{name}_project"] = str((Path(d) / ds["project"]).resolve())
            inputs[f"fig{n}_{name}_csv"] = str((Path(d) / ds["csv"]).resolve())
    prov = rr.provenance(inputs, __file__)
    doi = doi_from_readme()
    results = {n: measure(n, d, img) for n, (d, img) in sorted(figs.items())}
    runs = {}
    for n, res in results.items():
        for run in sorted({m["run"] for m in res["markers"]}):
            runs.setdefault(run, {})[n] = {**res, "markers": [m for m in res["markers"] if m["run"] == run]}
    groups = []
    for run, rs in sorted(runs.items()):
        for g in oh._combine_run(rs):
            g["run"] = run
            groups.append(g)
            for x in g["members"]:
                for m in results[x["figure"]]["markers"]:
                    if m["run"] == run and m["column"] == x["column"]:
                        m["T_comb"] = g["T"]
    checks = z_check(results[7]) if 7 in results else []
    rows = make_rows(results, groups, doi, checks)
    nl = not_legible_rows(results, groups)
    out = rr.new_run_dir("na12_extraction")
    rr.write_json(out / "extraction.json", {"combined_columns": groups, "figures": {str(n): r for n, r in results.items()}, "z_check": checks,
                                            "abstract_checks": text_checks(results[7]) if 7 in results else None, "not_legible": nl, "doi": doi})
    with open(out / "values_rows.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=oh.COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with open(out / "not_legible.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["figure", "series", "column", "T_K", "reason"], lineterminator="\n")
        w.writeheader()
        w.writerows(nl)
    rr.write_json(out / "run_config.json", {**prov, "paper": "Ohtaki et al. 1995, figures " + ", ".join(str(n) for n in sorted(figs)),
                                            "rules": "docs/decisions.md pre-registration B + amendments 1, 4, 5, 6, 7, 9, 10, 11, 12, 13 (2026-10-13 to 17)"})
    print(f"wrote {out}: {len(rows)} rows, {len(groups)} columns; not legible (in the window): {len(nl)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
