"""
NA12: values of Flahaut, Funahashi, Lee, Ohta & Koumoto 2006, "Effect of the Yb substitutions on the thermoelectric properties of CaMnO3" (2006 25th International Conference on
Thermoelectrics, pp. 103-106, DOI 10.1109/ICT.2006.331291), read from the PDF. Rules: docs/decisions.md, pre-registration B and its amendment 1 (2026-10-13).
No out-of-fold prediction is read or used here.

This is a TRAINING-SOURCE paper: its curves are in Starrydata2, so a comparison with the out-of-fold prediction restates the cross-validation error and is not an
independent check. Every row says so.

What it reads (vector figures, PDF pages 2 and 3; axes calibrated from the tick marks and the tick labels, which are text here)
  Fig. 2  resistivity against temperature, one curve per x (x = 0.05, 0.1, 0.15, 0.3, 0.4); the curve of each label is the one nearest to the tip of the arrow that starts at the
          label; read at 300, 400, ..., 800 K by linear interpolation along the curve (continuous curves: the 100 K grid of the pre-registration), where the curve covers the temperature.
  Fig. 3  Seebeck coefficient, markers at 373, 573, 773 and 973 K (the paper: S measured at 373-973 K; the drawn position is about 5 K higher, recorded). Only the series whose marker
          shape is unique in the legend are read (CaMnO3 and x = 0.15); x = 0.05/0.4 and x = 0.1/0.5 share a shape and are not read.
  Fig. 4  thermal conductivity, open markers (the closed markers are the electronic part and are not read) at the drawn temperatures (293, 373, ..., 973 K); x = 0.05, 0.1, 0.15.
          The x = 0 series is the data of Ohtaki et al. (the paper says so) and is not read.
  Fig. 5  Z in 1e-4 K^-1, markers at 373, 573, 773 and 973 K; zT = Z T with the nominal temperature; x = 0, 0.05, 0.1, 0.15 (diamond, square, circle, triangle, as in the caption).
Validation (asserted): zT of x = 0.05 at 973 K is the text's 0.2 (one digit); the ratio of that zT to the one of x = 0 is the text's factor 4; the lowest resistivity of x = 0.15 is the
abstract's 3 mohm cm. Consistency: zT = S^2 T / (rho kappa) is recomputed for x = 0.15 where S, rho and kappa are all read, and compared with Z T (flag outside 0.8-1.25).

Usage (repository root, clean tree):
    python thesis_paper/scripts/na12_extract_flahaut.py                        # writes thesis_paper/results/na12_extraction/<UTC>/
    python thesis_paper/scripts/na12_extract_flahaut.py --append-values <dir>  # appends the rows of that run to docs/na12_literature_values.csv
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import na12_plot_tools as T  # noqa: E402
import pdf_vector as pv  # noqa: E402
import run_record as rr  # noqa: E402

PDF = "thesis_paper/source/pdfs/flahaut2006.pdf"
VALUES = REPO / "thesis_paper" / "docs" / "na12_literature_values.csv"
DOI = "10.1109/ICT.2006.331291"
COLUMNS = ["lit_id", "compound_label", "composition", "property", "T_K", "value", "unit", "doi", "location", "value_kind", "digitisation_method", "notes"]
BOOK = "2006 25th Int. Conf. on Thermoelectrics, pp. 103-106"
COMPOSITION = {0.0: "CaMnO3", 0.05: "Ca0.95Yb0.05MnO3", 0.1: "Ca0.9Yb0.1MnO3", 0.15: "Ca0.85Yb0.15MnO3", 0.3: "Ca0.7Yb0.3MnO3", 0.4: "Ca0.6Yb0.4MnO3"}
SHAPE_X = {"diamond": 0.0, "square": 0.05, "circle": 0.1, "triangle": 0.15}  # captions of Figs. 4 and 5
GRID_K = (300, 400, 500, 600, 700, 800)
TRAINING = "TRAINING-SOURCE PAPER (its curves are in Starrydata2; a comparison restates the cross-validation error, it is not independent)"


def calibrate(paths, texts, box, xmult=1):
    """Frame, y fit and x fit of one plot from its tick marks and tick labels (labels are text in this PDF)."""
    segs = T.segments(paths, box)
    fr = T.frame_of(segs, min_len=100.0)
    yt = T.left_ticks(segs, fr, lo=2.3, hi=3.0)
    xt = T.bottom_ticks(segs, fr, lo=2.3, hi=3.0)
    ylab = sorted((t["y"], T.number(t["text"])) for t in texts
                  if fr[0] - 32 <= t["x"] < fr[0] and fr[1] - 8 <= t["y"] <= fr[3] + 8 and T.number(t["text"]) is not None and t["size"] >= 6)
    run = [t["text"].strip() for t in texts if fr[0] - 20 <= t["x"] <= fr[2] and (fr[1] - 30 <= t["y"] <= fr[1] + 3 or fr[3] - 3 <= t["y"] <= fr[3] + 30)
           and re.fullmatch(r"\d{6,}", t["text"].strip())]
    assert len(ylab) == len(yt), (len(ylab), len(yt))
    ya, yb, yr = T.linear_fit(yt, [v for _, v in ylab])
    xv = T.split_run(run[0], len(xt))
    xa, xb, xr = T.linear_fit(xt, xv)
    assert yr < 0.05 and xr < 0.05, (yr, xr)
    return {"frame": fr, "ya": ya, "yb": yb, "xa": xa, "xb": xb, "y_ticks": len(yt), "x_ticks": len(xt), "y_first_label": ylab[0][1], "y_last_label": ylab[-1][1],
            "x_first_label": xv[0], "x_last_label": xv[-1], "y_residual_label_steps": round(yr, 4), "x_residual_label_steps": round(xr, 4)}


def fig_markers(paths, cal, legend_width=0.4):
    """Markers inside the frame, legend symbols (heavier line) excluded: [(shape, T_drawn_K, value, size)]."""
    fr = cal["frame"]
    out = []
    for shape, cx, cy, size, width in T.markers(paths, fr):
        if width >= legend_width:
            continue
        out.append((shape, cal["xa"] * cx + cal["xb"], cal["ya"] * cy + cal["yb"], size))
    return out


def read_all(pdf):
    o2, c2, r2, _ = pv.page(pdf, 2)
    p2, t2 = pv.interpret(o2, c2, r2)
    o3, c3, r3, _ = pv.page(pdf, 3)
    p3, t3 = pv.interpret(o3, c3, r3)
    res = {"calibration": {}}
    # ---- Fig. 2: resistivity curves
    box2 = (330.0, 540.0, 590.0, 715.0)
    segs = T.segments(p2, box2)
    fr = T.frame_of(segs, min_len=100.0)
    yt = T.left_ticks(segs, fr, lo=1.0, hi=3.5)
    xt = T.bottom_ticks(segs, fr, lo=1.0, hi=3.5)
    ylab = sorted((t["y"], T.number(t["text"])) for t in t2 if 320 <= t["x"] < fr[0] and fr[1] - 8 <= t["y"] <= fr[3] + 8 and T.number(t["text"]) is not None and t["size"] >= 6)
    # the labelled major ticks are the ticks nearest to (label baseline + the common offset)
    off = sorted(min(yt, key=lambda y: abs(y - (ly + 1.7))) - ly for ly, _ in ylab)
    majors = [min(yt, key=lambda y: abs(y - (ly + 1.7))) for ly, _ in ylab]
    ya, yb, yr = T.linear_fit(majors, [v for _, v in ylab])
    run = [t["text"].strip() for t in t2 if fr[0] - 20 <= t["x"] <= fr[2] and fr[1] - 30 <= t["y"] <= fr[1] + 3 and re.fullmatch(r"\d{6,}", t["text"].strip())][0]
    xv = T.split_run(run, 11)
    major_x = xt[::2]
    assert len(major_x) == len(xv)
    xa, xb, xr = T.linear_fit(major_x, xv)
    assert yr < 0.05 and xr < 0.05, (yr, xr)
    res["calibration"]["Fig2_resistivity"] = {"frame": [round(v, 2) for v in fr], "y_labels": [v for _, v in ylab], "y_residual_label_steps": round(yr, 4), "x_labels": f"{xv[0]}..{xv[-1]}",
                                              "x_residual_label_steps": round(xr, 4)}
    curves = []
    for p in p2:
        for sp in p["pts"]:
            xs = [a for a, b in sp]
            ys = [b for a, b in sp]
            if len(sp) >= 200 and fr[0] <= min(xs) and max(xs) <= fr[2] and fr[1] <= min(ys) and max(ys) <= fr[3]:
                curves.append(sp)
    assert len(curves) == 5, len(curves)
    labels = [(t["text"].strip(), t["x"], t["y"]) for t in t2 if re.fullmatch(r"x = [\d.]+", t["text"].strip()) and fr[0] <= t["x"] <= fr[2] + 20 and fr[1] <= t["y"] <= fr[3] + 20]
    arrows = [(sp[0], sp[1]) for p in p2 for sp in p["pts"] if len(sp) == 2]
    assign = {}
    for name, lx, ly in labels:
        x = float(name.split("=")[1])
        if x == 0.0:
            continue
        cand = []
        for a, b in arrows:
            L = math.hypot(a[0] - b[0], a[1] - b[1])
            for s, e in ((a, b), (b, a)):
                d = math.hypot(s[0] - (lx + 9), s[1] - (ly + 2))
                if d < 9 and L > 6:
                    cand.append((d, e))
        cand.sort()
        assert cand, f"no arrow for {name}"
        tip = cand[0][1]
        dists = [min(math.hypot(tip[0] - a, tip[1] - b) for a, b in cv) for cv in curves]
        order = sorted(range(len(curves)), key=lambda k: dists[k])
        assert dists[order[0]] < 6.0 and dists[order[1]] > 2 * dists[order[0]], f"{name}: the arrow tip does not single out one curve ({dists})"
        assign[x] = (order[0], round(dists[order[0]], 2), round(dists[order[1]], 2))
    assert len({v[0] for v in assign.values()}) == len(assign) == 5, "two labels point to one curve"
    res["fig2_assignment"] = {str(x): {"curve": v[0], "tip_distance_pt": v[1], "second_nearest_pt": v[2]} for x, v in assign.items()}
    rho = {}
    for x, (k, _, _) in assign.items():
        pts = sorted((xa * a + xb, ya * b + yb) for a, b in curves[k])
        rho[x] = pts
    res["rho_curves"] = rho
    # ---- Figs. 3, 4, 5
    for name, box in {"Fig3_S": (60, 430, 300, 600), "Fig4_kappa": (40, 130, 300, 330), "Fig5_Z": (330, 150, 560, 335)}.items():
        cal = calibrate(p3, t3, box)
        res["calibration"][name] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in cal.items() if k != "frame"}
        res[name] = fig_markers(p3, cal)
    return res


def interp(pts, T_K):
    """Linear interpolation of a polyline [(T, value)] at T_K; None outside its range."""
    if T_K < pts[0][0] or T_K > pts[-1][0]:
        return None
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t0 <= T_K <= t1 and t1 > t0:
            return v0 + (v1 - v0) * (T_K - t0) / (t1 - t0)
    return None


def build(pdf):
    d = read_all(pdf)
    rows, detail = [], {}
    loc = {"rho": f"{BOOK}, Fig. 2 (article page 104, PDF page 2)", "S": f"{BOOK}, Fig. 3 (article page 105, PDF page 3)", "kappa": f"{BOOK}, Fig. 4 (article page 105, PDF page 3)",
           "Z": f"{BOOK}, Fig. 5 (article page 105, PDF page 3)"}

    def add(prop, x, T_K, value, unit, kind, where, method, note):
        rows.append({"lit_id": f"FL2006_{prop}_x{x:g}_{int(round(T_K))}K", "compound_label": f"{COMPOSITION[x]} (Ca1-xYbxMnO3, x = {x:g}; training-source paper)",
                     "composition": COMPOSITION[x], "property": prop, "T_K": f"{T_K:g}", "value": f"{value:.3g}", "unit": unit, "doi": DOI, "location": where, "value_kind": kind,
                     "digitisation_method": method, "notes": TRAINING + ("; " + note if note else "")})

    cal = d["calibration"]
    # resistivity: continuous curves, the 100 K grid
    rho_grid = {}
    for x in (0.05, 0.1, 0.15, 0.3, 0.4):
        for Tk in GRID_K:
            v = interp(d["rho_curves"][x], float(Tk))
            if v is None:
                continue
            rho_grid[(x, Tk)] = v
            note = "800 K is recorded, not compared" if Tk >= 800 else ""
            add("resistivity", x, float(Tk), v, "ohm cm", "digitised", loc["rho"],
                "vector polyline read from the PDF content stream (scripts/na12_extract_flahaut.py); the curve of the label is the one nearest to the label's arrow tip; "
                "linear interpolation along the polyline at the grid temperature; axes fitted on the labelled major ticks; value to 3 significant digits", note)
    # Seebeck coefficient: markers with a unique shape in the legend
    S_by = {"diamond": 0.0, "triangle": 0.15}
    S_vals = {}
    nominal = (373, 573, 773, 973)
    for shape, x in S_by.items():
        ms = sorted(m for m in d["Fig3_S"] if m[0] == shape)
        assert len(ms) == 4, (shape, len(ms))
        for (s, Td, val, size), Tn in zip(ms, nominal):
            assert abs(Td - Tn) < 8
            S_vals[(x, Tn)] = val
            add("S", x, float(Tn), val, "uV/K", "digitised", loc["S"],
                "vector marker (open " + shape + ") read from the PDF content stream; marker centre; axes fitted on the labelled major ticks; value to 3 significant digits",
                f"marker drawn at {Td:.0f} K ({Td - Tn:+.0f} K from the nominal measurement temperature); the nominal temperature is used" + ("; 973 K is above the window, recorded not compared" if Tn >= 800 else ""))
    # thermal conductivity: open markers (the upper cluster), x = 0.05, 0.1, 0.15
    kappa_vals = {}
    for shape in ("square", "circle", "triangle"):
        x = SHAPE_X[shape]
        ms = [m for m in d["Fig4_kappa"] if m[0] == shape]
        top = sorted((m for m in ms if m[2] > 1.0), key=lambda m: m[1])
        assert len(top) == 8, (shape, len(top))
        for s, Td, val, size in top:
            Tn = round(Td)
            kappa_vals[(x, Tn)] = val
            note = f"marker drawn at {Td:.1f} K; no measurement temperature is stated for the thermal conductivity, the drawn position is used"
            if Tn < 300:
                note += "; below the 300 K window edge: recorded, not compared"
            if Tn >= 800:
                note += "; above the window, recorded not compared"
            add("kappa", x, float(Tn), val, "W/mK", "digitised", loc["kappa"],
                "vector marker (open " + shape + ") read from the PDF content stream; the upper cluster is the thermal conductivity (the closed symbols of the lower cluster are its electronic part); "
                "axes fitted on the labelled major ticks; value to 3 significant digits", note)
    # zT = Z T
    zT_vals = {}
    for shape, x in SHAPE_X.items():
        ms = sorted(m for m in d["Fig5_Z"] if m[0] == shape)
        assert len(ms) == 4, (shape, len(ms))
        for (s, Td, Z, size), Tn in zip(ms, nominal):
            assert abs(Td - Tn) < 8
            zt = Z * 1e-4 * Tn
            zT_vals[(x, Tn)] = zt
            note = f"Z read from Fig. 5 in 1e-4 K^-1 ({Z:.3g}) at the marker drawn at {Td:.0f} K; zT = Z T with the nominal temperature {Tn} K"
            if x == 0.0:
                note += "; the thermal conductivity of x = 0 in this paper is the data of Ohtaki et al., so this zT mixes sources"
            if Tn >= 800:
                note += "; 973 K is above the window, recorded not compared"
            add("zT", x, float(Tn), zt, "1", "digitised", loc["Z"],
                "Z read as a vector marker from the PDF content stream (marker centre, axes fitted on the labelled major ticks) and multiplied by the nominal temperature", note)
    # validation against the paper's own statements
    validation = []
    zt973 = zT_vals[(0.05, 973)]
    assert 0.15 <= zt973 < 0.25, zt973
    validation.append({"statement": "ZT = 0.2 for x = 0.05 at 1000 K (abstract; page 105)", "read": round(zt973, 4), "accepted_range": [0.15, 0.25]})
    fac = zt973 / zT_vals[(0.0, 973)]
    assert 3.5 <= fac <= 4.5, fac
    validation.append({"statement": "the substitution of 5% Yb enhances by a factor 4 the ZT at 1000 K of CaMnO3 (page 105)", "read": round(fac, 2), "accepted_range": [3.5, 4.5]})
    rmin = min(v for _, v in d["rho_curves"][0.15])
    assert 0.0025 <= rmin <= 0.0035, rmin
    validation.append({"statement": "the lowest resistivity reaches 3 mohm cm for x = 0.15 (abstract)", "read_ohm_cm": round(rmin, 5), "accepted_range_ohm_cm": [0.0025, 0.0035]})
    # consistency: zT = S^2 T / (rho kappa) for x = 0.15, where S, rho and kappa are all read
    consistency = []
    for Tn in (373, 573, 773, 973):
        if (0.15, Tn) in S_vals and (0.15, Tn) in kappa_vals:
            rho = interp(d["rho_curves"][0.15], float(Tn))
            if rho is None:
                continue
            Sv, ka = S_vals[(0.15, Tn)] * 1e-6, kappa_vals[(0.15, Tn)]
            calc = Sv ** 2 * Tn / ((rho * 1e-2) * ka)
            ratio = calc / zT_vals[(0.15, Tn)]
            consistency.append({"T_K": Tn, "zT_read": zT_vals[(0.15, Tn)], "zT_recomputed": calc, "ratio_recomputed_to_read": round(ratio, 3), "outside_0.8_1.25": not (0.8 <= ratio <= 1.25)})
    return {"calibration": cal, "fig2_assignment": d["fig2_assignment"], "validation": validation, "consistency_zT_x0.15": consistency, "rows": rows}


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
    rr.write_json(out / "run_config.json", {**prov, "paper": "Flahaut et al. 2006 (ICT), training-source", "rules": "docs/decisions.md pre-registration B + amendment 1 (2026-10-13)"})
    print(f"wrote {out}: {len(res['rows'])} rows")
    for v in res["validation"]:
        print("validation:", v)
    for c in res["consistency_zT_x0.15"]:
        print("zT consistency (x = 0.15):", c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
