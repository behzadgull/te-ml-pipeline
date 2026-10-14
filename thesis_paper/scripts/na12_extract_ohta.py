"""
NA12: Seebeck coefficient of four heavily La- or Nb-doped SrTiO3 crystals from Ohta, Nomura, Ohta & Koumoto 2005 (J. Appl. Phys. 97:034106, DOI 10.1063/1.1847723), Fig. 1.
Rules: docs/decisions.md, pre-registration B with amendments 1 (reading rules), 3 (Ohta compared at the cluster level, sample as stated) and 4 (column-consensus marker temperatures,
edge-row sensitivity). No out-of-fold prediction is read or used here.

The figure is a bitmap, digitised by hand in WebPlotDigitizer (one CSV per series and the project file). The clicks give the S value of each marker; the temperature of a marker is the
consensus of its column (amendment 4): the marker centres are located in the figure image by a left-right symmetry fit of the column profile of the dark pixels, converted to temperature on
a straight line through ALL printed tick marks of the temperature axis, and averaged over the series that have a marker in that column.

Validation (asserted): 4 series of 10 clicks, CSV equal to the project file; the temperature axis has 11 major ticks (200 to 1200 K every 100 K) and a linear fit through them; the S axis
ticks (every 0.02 mV/K) reproduce the click calibration within 2 uV/K; every refined centre lies inside its search window and a majority (at least 3 of 5) of the window settings agree within 3 px; markers of one column lie within 5 K of each other; each series has at
most one marker per column; the first marker of every sample is within 15 uV/K of the room-temperature S of Table II (read from the PDF text) and has the larger |S| (|S| increases with T, text).

Usage (repository root, clean tree):
    python thesis_paper/scripts/na12_extract_ohta.py --csv-dir <folder with the 4 CSVs + project JSON> --image <figure bitmap>      # writes thesis_paper/results/na12_extraction/<UTC>/
    python thesis_paper/scripts/na12_extract_ohta.py --append-values <dir>                                                      # appends the rows of that run to docs/na12_literature_values.csv
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
import pdf_vector as pv  # noqa: E402
import run_record as rr  # noqa: E402

PDF = "thesis_paper/source/pdfs/ohta2005.pdf"
VALUES = REPO / "thesis_paper" / "docs" / "na12_literature_values.csv"
DOI = "10.1063/1.1847723"
JOURNAL = "J. Appl. Phys. 97:034106 (2005)"
COLUMNS = ["lit_id", "compound_label", "composition", "property", "T_K", "value", "unit", "doi", "location", "value_kind", "digitisation_method", "notes"]
PROJECT = "s_project.json"
THRESHOLD = 128  # dark pixel: grey level below this
X_TICKS = [200 + 100 * k for k in range(11)]  # printed major ticks of the temperature axis, K
MAX_COLUMN_RANGE_K = 5.0
TABLE2_TOL_UV = 15.0
SEARCH_PX = 16  # the clicks are within about 14 px of the marker centre; the result must not touch the edge of the search window
WINDOW_ROWS = (-30, 21)  # rows around a click used for the column profile (the markers are about 45 px across; text sits directly under the base of some triangles)
# The symmetry fit of one marker is repeated with five (rows, half-width) settings; a marker line passing through an open triangle can give a second, spurious minimum in one setting.
# The centre is the mean of the estimates within MAX_ALT_DIFF_PX of the median, and at least MIN_AGREE of the five must be in that cluster, otherwise the run stops.
SETTINGS = [((-30, 21), 18), ((-26, 26), 18), ((-26, 26), 20), ((-30, 17), 18), ((-30, 21), 20)]
MAX_ALT_DIFF_PX = 3.0
MIN_AGREE = 3
SAMPLE_DOPING_NOTE = ("sample (b): doping printed 7.9e20 cm^-3 in the captions of Figs. 1 and 3 and 8.4e20 cm^-3 in Tables I and II (S, sigma and n of Table II agree with the latter); "
                      "recorded as in the tables")


def runs(mask, gap=3):
    """Centres of the runs of True indices (gaps <= gap are bridged)."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    parts = np.split(idx, np.where(np.diff(idx) > gap)[0] + 1)
    return [float(p.mean()) for p in parts]


def axis_ticks(dark):
    """Pixel positions of the temperature-axis major ticks (columns) and of the S-axis ticks (rows), found next to the plot frame."""
    rows = dark[:, 380:1900].sum(axis=1)
    cand = [r for r in range(1650, 1800) if rows[r] > 1000]
    assert cand, "bottom frame not found"
    fy = cand[len(cand) // 2]
    xt = runs(dark[fy + 5: fy + 15, 380:1910].sum(axis=0) >= 8)
    xt = [x + 380 for x in xt]
    cols = dark[:1700, 380:420].sum(axis=0)
    fx = int(np.argmax(cols)) + 380
    yt = runs(dark[:1700, fx - 20: fx - 4].sum(axis=1) >= 9)
    return fy, fx, xt, yt


def load_csvs(folder):
    """{sample letter: dict(doping label, rows [(T_click, S_click)])} from the four CSVs; checked against the project file."""
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
    proj = json.loads((Path(folder) / PROJECT).read_text(encoding="utf-8"))
    for ds in proj["datasetColl"]:
        s = ds["name"][0]
        assert ds["name"] == out[s]["name"], (ds["name"], out[s]["name"])
        assert len(ds["data"]) == len(out[s]["pts"]) == 10, (s, len(ds["data"]), len(out[s]["pts"]))
        for p, (t, v) in zip(ds["data"], out[s]["pts"]):
            assert abs(p["value"][0] - t) < 1e-6 and abs(p["value"][1] - v) < 1e-9, (s, p["value"], (t, v))
            out[s].setdefault("px", []).append((p["x"], p["y"]))
    cal = proj["axesColl"][0]["calibrationPoints"]
    assert not proj["axesColl"][0]["isLogX"] and not proj["axesColl"][0]["isLogY"]
    return out, cal


def _symmetry_centre(dark, cx, cy, rows, half):
    cx, cy = int(round(cx)), int(round(cy))
    prof = dark[cy + rows[0]: cy + rows[1], :].sum(axis=0).astype(float)
    grid = np.arange(prof.size)
    d = np.arange(1, half + 1)
    best = None
    for c in np.arange(cx - SEARCH_PX, cx + SEARCH_PX + 0.01, 0.5):
        cost = np.abs(np.interp(c + d, grid, prof) - np.interp(c - d, grid, prof)).sum()
        if best is None or cost < best[0]:
            best = (cost, float(c))
    return best[1]


def refine_x(dark, cx, cy):
    """Marker centre (px): mean of the symmetry-fit estimates (SETTINGS) within MAX_ALT_DIFF_PX of their median; returns (centre, n_agreeing, spread of the agreeing ones)."""
    est = np.array([_symmetry_centre(dark, cx, cy, r, h) for r, h in SETTINGS])
    med = float(np.median(est))
    ok = est[np.abs(est - med) <= MAX_ALT_DIFF_PX]
    return float(ok.mean()), int(ok.size), float(ok.max() - ok.min())


def table2_S(pdf):
    """The four room-temperature S values (uV/K, sample a to d) of Table II, from the text of the PDF page."""
    objs, content, res, _ = pv.page(pdf, 3)
    _, texts = pv.interpret(objs, content, res)
    cells = sorted(((t["y"], t["text"]) for t in texts if re.search(r"\?0\.\d\d", t["text"]) and len(t["text"]) > 12), reverse=True)
    vals = []
    for _, s in cells:
        m = re.findall(r"\?(0\.\d\d)", s)
        assert len(m) == 1, s
        vals.append(-1000.0 * float(m[0]))  # the PDF's '?' is the minus sign; S is negative in all cases (n-type, text)
    assert len(vals) == 4, vals
    return vals


def build(csv_dir, image, pdf):
    data, ucal = load_csvs(csv_dir)
    dark = np.array(Image.open(image).convert("L")) < THRESHOLD
    fy, fx, xt, yt = axis_ticks(dark)
    assert len(xt) == len(X_TICKS), f"{len(xt)} temperature ticks found"
    a, b = np.polyfit(xt, X_TICKS, 1)
    resid = np.array(X_TICKS) - (a * np.array(xt) + b)
    cal_unc = float(np.abs(resid).max())
    assert cal_unc < 2.0, cal_unc
    # S axis: ticks every 0.02 mV/K from -0.6 (top) to 0 (bottom)
    assert len(yt) == 31, len(yt)
    ay, by = np.polyfit(yt, [-0.6 + 0.02 * k for k in range(31)], 1)
    uy1, uy2 = ucal[2], ucal[3]
    for py, dy in ((uy1["py"], float(uy1["dy"])), (uy2["py"], float(uy2["dy"]))):
        assert abs((ay * py + by) - dy) < 0.002, ("S calibration differs from the printed ticks", py, dy)
    ux1, ux2 = ucal[0], ucal[1]
    user_x_off_px = [ux1["px"] - xt[0], ux2["px"] - xt[-1]]
    mk = []  # per marker
    for s in "abcd":
        prev = -1.0
        for (tc, vc), (px, py) in zip(data[s]["pts"], data[s]["px"]):
            c, n_ok, spread = refine_x(dark, px, py)
            assert n_ok >= MIN_AGREE, f"sample {s} marker at click {tc:.1f} K: only {n_ok} of {len(SETTINGS)} window settings agree"
            assert abs(c - px) < SEARCH_PX - 0.5, f"sample {s} marker at click {tc:.1f} K: centre at the edge of the search window"
            t_img = float(a * c + b)
            assert t_img > prev
            prev = t_img
            mk.append({"sample": s, "click_T": tc, "click_S_mVK": vc, "centre_px": c, "T_image": t_img, "n_settings_agreeing": n_ok, "spread_px": spread})
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
    # Table II consistency
    t2 = table2_S(REPO / pdf)
    checks = []
    for s, ref in zip("abcd", t2):
        first = min((m for m in mk if m["sample"] == s), key=lambda m: m["T_image"])
        s_uv = 1000.0 * first["click_S_mVK"]
        d = s_uv - ref
        assert abs(d) <= TABLE2_TOL_UV and abs(s_uv) >= abs(ref), (s, s_uv, ref)
        checks.append({"sample": s, "table2_S_uVK_298K": ref, "first_marker_S_uVK": round(s_uv, 1), "first_marker_T_K": round(first["T_col"], 1), "difference_uVK": round(d, 1)})
    # rows
    rows = []
    loc = f"{JOURNAL}, Fig. 1 (article page 034106-2, PDF page 3)"
    for m in sorted(mk, key=lambda m: (m["sample"], m["column"])):
        s = m["sample"]
        d = data[s]
        label = f"{d['dopant']}-doped SrTiO3, n = {d['n_cm3']} cm^-3 (sample ({s}), single crystal; Table I)"
        T = m["T_col"]
        edge = round(T / 25.0) * 25.0
        tokens = ["[cluster-level]"]
        if abs(T - edge) <= cal_unc:
            other = edge - 25.0 if T >= edge else edge
            tokens.append(f"[edge-sensitivity: also bin {other:g}]")
        floor_bin = 25.0 * np.floor(T / 25.0)
        sd = "" if m["T_col_sd"] is None else f"{m['T_col_sd']:.1f}"
        note = (" ".join(tokens) + f" T = consensus of column {m['column']} ({m['n_series']} series, SD {sd or 'n/a (one series)'} K, range {m['column_range']:.1f} K), calibration "
                f"uncertainty +/- {cal_unc:.1f} K, floor bin {floor_bin:g}; clicked T {m['click_T']:.1f} K (not used); S to the digitiser's click resolution (4.8 uV/K lattice)")
        if s == "b":
            note += "; " + SAMPLE_DOPING_NOTE
        if T >= 800.0:
            note += "; at or above the 800 K window edge: recorded, not compared"
        rows.append({
            "lit_id": f"OH2005_S_{s}_c{m['column']:02d}", "compound_label": label, "composition": "SrTiO3", "property": "S", "T_K": f"{T:.1f}",
            "value": f"{1000.0 * m['click_S_mVK']:.3g}", "unit": "uV/K", "doi": DOI, "location": loc, "value_kind": "digitised",
            "digitisation_method": ("bitmap digitised by hand in WebPlotDigitizer (clicked S, mV/K converted to uV/K, 3 significant digits); temperature = column consensus of marker centres "
                                    "located in the image by a symmetry fit and read on a straight line through the 11 printed major ticks (200 to 1200 K), with "
                                    "thesis_paper/scripts/na12_extract_ohta.py (amendment 4)"),
            "notes": note})
    return {"temperature_axis_fit": {"slope_K_per_px": float(a), "intercept_K": float(b), "max_residual_K": cal_unc, "tick_px": xt},
            "S_axis_fit": {"slope_mVK_per_px": float(ay), "intercept_mVK": float(by)},
            "clicked_calibration_offset_px_at_200K_and_1200K": user_x_off_px,
            "clicked_calibration_offset_K": [o * a for o in user_x_off_px],
            "n_columns": len(cols), "markers": mk, "table2_checks": checks, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default=None)
    ap.add_argument("--image", default=None)
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
    assert args.csv_dir and args.image, "--csv-dir and --image are required"
    inputs = {"pdf": args.pdf, "image": str(Path(args.image).resolve()), "project": str((Path(args.csv_dir) / PROJECT).resolve()), "pdf_vector": "thesis_paper/scripts/pdf_vector.py"}
    for p in sorted(Path(args.csv_dir).glob("*.csv")):
        inputs[f"csv_{p.stem[0]}"] = str(p.resolve())
    prov = rr.provenance(inputs, __file__)
    res = build(args.csv_dir, args.image, args.pdf)
    out = rr.new_run_dir("na12_extraction")
    rr.write_json(out / "extraction.json", {k: v for k, v in res.items() if k != "rows"})
    with open(out / "values_rows.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(res["rows"])
    rr.write_json(out / "run_config.json", {**prov, "paper": "Ohta et al. 2005, Fig. 1 (S), samples (a) to (d)",
                                            "rules": "docs/decisions.md pre-registration B + amendments 1, 3, 4 (2026-10-13/14)"})
    print(f"wrote {out}: {len(res['rows'])} rows, {res['n_columns']} columns; temperature-axis residual {res['temperature_axis_fit']['max_residual_K']:.2f} K")
    return 0


if __name__ == "__main__":
    sys.exit(main())
