"""
NA12: values of Sebastia-Luna et al. 2022 (ACS Appl. Energy Mater. 5:10216-10223, DOI 10.1021/acsaem.2c01936) for CsSnI3, read from the PDF.
Rules: docs/decisions.md, pre-registration B (2026-10-05) and its amendment 1 (2026-10-13). No out-of-fold prediction is read or used here.

What it does
  1. Reads the vector markers of Fig. 2a (B-gamma CsSnI3; PDF page 3) from the page content stream (pdf_vector.py): the markers are filled paths, the axes are rectangles
     (the tick labels are outlines, not text). Each y axis is calibrated by a linear fit to its major tick marks, whose labels were read from the rendered figure
     (first label and step are constants below); the temperature axis likewise. The marker centre is the centre of its 16-point body path.
  2. Reads the printed maxima of Table 2 (PDF page 5) from the text layer.
  3. Validates: every digitised marker that Table 2 also prints must agree with the printed value within the printed uncertainty (asserted).
  4. Internal consistency: where S, sigma, kappa and zT are given at one temperature, zT = S^2 sigma T / kappa is recomputed and the ratio to the read zT is recorded
     (flag outside 0.8 to 1.25).
  5. Applies amendment 1: the nominal measurement temperature of the series (25, 50, 75, 100 C) is used, the drawn plotting offset is recorded and not used; a Table 2 value
     replaces the digitised one where it is tabulated; the printed uncertainty is carried only where printed. The 25 C values (298.15 K) are below the window edge and
     are recorded, not compared (the comparison script does that by itself).

Usage (repository root, clean tree):
    python thesis_paper/scripts/na12_extract_sebastialuna.py                       # writes thesis_paper/results/na12_extraction/<UTC>/
    python thesis_paper/scripts/na12_extract_sebastialuna.py --append-values <dir>  # appends the rows of that run to docs/na12_literature_values.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import pdf_vector as pv  # noqa: E402
import run_record as rr  # noqa: E402

PDF = "thesis_paper/source/pdfs/sebastialuna2022.pdf"
VALUES = REPO / "thesis_paper" / "docs" / "na12_literature_values.csv"
DOI = "10.1021/acsaem.2c01936"
COLUMNS = ["lit_id", "compound_label", "composition", "property", "T_K", "value", "unit", "doi", "location", "value_kind", "digitisation_method", "notes"]
COL = {"sigma": (0.0, 0.0, 0.0), "kappa": (1.0, 0.0, 0.0), "S": (0.0, 0.5, 0.0), "zT": (0.0, 0.0, 1.0)}
COLNAME = {"sigma": "black", "kappa": "red", "S": "green", "zT": "blue"}
UNIT = {"sigma": "S/cm", "kappa": "W/mK", "S": "uV/K", "zT": "1"}
# axis labels read from the rendered figure: (first label, step between labels, factor to the unit above)
FIG2A = dict(page=3, box=(100.0, 100.0, 340.0, 232.0),
             axes={"sigma": (0.0, 10.0, 1.0), "kappa": (0.5, 0.1, 1.0), "S": (90.0, 10.0, 1.0), "zT": (1.6, 0.2, 1e-2)}, xaxis=(20.0, 10.0))
NOMINAL_C = (25, 50, 75, 100)
TABLE2_ORDER = ("sigma", "kappa", "S", "zT")  # columns of the B-gamma CsSnI3 row of Table 2
LOCATION_FIG = "ACS Appl. Energy Mater. 2022, 5, 10216-10223, Fig. 2a (article page 10218, PDF page 3)"
LOCATION_TAB = "ACS Appl. Energy Mater. 2022, 5, 10216-10223, Table 2 (article page 10220, PDF page 5)"


def rects(paths, box):
    """Axis-aligned rectangle subpaths (x0, y0, x1, y1, colour) of the filled paths inside the box."""
    out = []
    bx0, by0, bx1, by1 = box
    for p in paths:
        if p["kind"] != "f":
            continue
        for sp in p["pts"]:
            if 4 <= len(sp) <= 6:
                xs = [a for a, b in sp]
                ys = [b for a, b in sp]
                x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
                if all(abs(a - x0) < 1e-3 or abs(a - x1) < 1e-3 for a in xs) and all(abs(b - y0) < 1e-3 or abs(b - y1) < 1e-3 for b in ys):
                    if bx0 <= x0 and x1 <= bx1 and by0 <= y0 and y1 <= by1:
                        out.append((x0, y0, x1, y1, tuple(round(c, 2) for c in p["color"])))
    return out


def fit(pos, first, step):
    """Linear map value = a * position + b through the major ticks (one per label); returns a, b, the maximum residual in tick steps."""
    n = len(pos)
    vals = [first + step * i for i in range(n)]
    mp, mv = sum(pos) / n, sum(vals) / n
    a = sum((p - mp) * (v - mv) for p, v in zip(pos, vals)) / sum((p - mp) ** 2 for p in pos)
    b = mv - a * mp
    return a, b, max(abs(a * p + b - v) / abs(step) for p, v in zip(pos, vals)), vals


def read_markers(objs, content, res, cfg):
    """{property: [(T_drawn_C, value, x_pt, y_pt)] sorted by temperature} and the calibration summary of Fig. 2a."""
    paths, _ = pv.interpret(objs, content, res)
    R = rects(paths, cfg["box"])
    cal, summary = {}, {}
    for prop, c in COL.items():
        lines = sorted((r for r in R if r[4] == c and (r[3] - r[1]) > 60 and (r[2] - r[0]) < 1.2), key=lambda r: r[0])
        assert lines, f"no axis line for {prop}"
        ax = lines[0]
        axx = (ax[0] + ax[2]) / 2
        ticks = [r for r in R if r[4] == c and (r[2] - r[0]) <= 2.6 and (r[3] - r[1]) <= 1.0 and abs((r[0] + r[2]) / 2 - axx) < 2.0 and ax[1] - 0.5 <= r[1] <= ax[3] + 0.5]
        major_len = max(round(r[2] - r[0], 1) for r in ticks)
        majors = sorted((r[1] + r[3]) / 2 for r in ticks if round(r[2] - r[0], 1) == major_len)
        first, step, scale = cfg["axes"][prop]
        a, b, resid, vals = fit(majors, first, step)
        assert resid < 0.1, f"{prop}: the major ticks are not uniformly spaced (residual {resid})"
        cal[prop] = (a, b, scale)
        summary[prop] = {"colour": COLNAME[prop], "axis_x_pt": round(axx, 2), "major_ticks": len(majors), "first_label": vals[0], "last_label": vals[-1], "step": step,
                         "max_residual_tick_steps": round(resid, 4)}
    frame = [r for r in R if r[4] == COL["sigma"] and (r[3] - r[1]) > 60 and (r[2] - r[0]) < 1.2]
    bottom = min(r[1] for r in frame)
    xt = [r for r in R if r[4] == COL["sigma"] and 1.5 <= (r[3] - r[1]) <= 3.0 and (r[2] - r[0]) <= 1.0 and abs(r[1] - bottom) < 1.0]
    longest = max(r[3] - r[1] for r in xt)
    xs = sorted((r[0] + r[2]) / 2 for r in xt if abs((r[3] - r[1]) - longest) < 0.1)
    ax_, bx_, rx, valsx = fit(xs, *cfg["xaxis"])
    assert rx < 0.1
    summary["temperature_axis"] = {"major_ticks": len(xs), "first_label": valsx[0], "last_label": valsx[-1], "max_residual_tick_steps": round(rx, 4)}
    mk = {p: [] for p in COL}
    bx0, by0, bx1, by1 = cfg["box"]
    for pth in paths:
        if pth["kind"] != "f":
            continue
        col = tuple(round(v, 2) for v in pth["color"])
        for sp in pth["pts"]:
            xs2 = [a for a, b in sp]
            ys2 = [b for a, b in sp]
            w, h = max(xs2) - min(xs2), max(ys2) - min(ys2)
            cx, cy = (max(xs2) + min(xs2)) / 2, (max(ys2) + min(ys2)) / 2
            if bx0 <= cx <= bx1 and by0 <= cy <= by1 and 4.4 <= w <= 4.9 and 4.4 <= h <= 4.9 and abs(w - h) < 0.15 and len(sp) >= 10:
                for prop, c in COL.items():
                    if col == c:
                        mk[prop].append((cx, cy))
    out = {}
    for prop in COL:
        a, b, scale = cal[prop]
        pts = sorted(set((round(x, 3), round(y, 3)) for x, y in mk[prop]))
        out[prop] = sorted(((ax_ * x + bx_, (a * y + b) * scale, x, y) for x, y in pts))
        assert len(out[prop]) == 4, f"{prop}: {len(out[prop])} markers found, expected 4"
    return out, summary


def read_table2(objs, content, res):
    """The four printed (value, uncertainty, temperature C) triples of the B-gamma CsSnI3 row of Table 2."""
    _, texts = pv.interpret(objs, content, res)
    s = " ".join(t["text"] for t in texts)
    i = s.index("Table 2. Comparison")
    row = s[i:]
    m = re.search(r"CsSnI\s*3\s+(.*?)\s+Cs\s*2\s*SnI\s*6", row)
    assert m, "the B-gamma CsSnI3 row of Table 2 was not found"
    triples = re.findall(r"(\d+(?:\.\d+)?)\s*±\s*(\d+(?:\.\d+)?)\s+at\s+(\d+)\s*°\s*C", m.group(1))
    assert len(triples) == 4, f"expected 4 printed maxima, found {len(triples)}"
    return {prop: (float(v), float(e), int(t), f"{v} ± {e} at {t} C") for prop, (v, e, t) in zip(TABLE2_ORDER, triples)}


def consistency(rows, low=0.8, high=1.25):
    """Where S, sigma, kappa and zT are all given at one temperature, recompute zT = S^2 sigma T / kappa (SI units) and compare it with the zT of the row.
    Returns one record per temperature with the ratio (recomputed / read) and a flag when it lies outside [low, high]."""
    SI = {"uV/K": 1e-6, "S/cm": 100.0, "W/mK": 1.0, "1": 1.0}
    by_T = {}
    for r in rows:
        by_T.setdefault(r["T_K"], {})[r["property"]] = float(r["value"]) * SI[r["unit"]]
    out = []
    for T, d in sorted(by_T.items(), key=lambda kv: float(kv[0])):
        if all(k in d for k in ("S", "sigma", "kappa", "zT")):
            calc = d["S"] ** 2 * d["sigma"] * float(T) / d["kappa"]
            ratio = calc / d["zT"]
            out.append({"T_K": float(T), "zT_read": d["zT"], "zT_recomputed": calc, "ratio_recomputed_to_read": round(ratio, 3), "outside_0.8_1.25": not (low <= ratio <= high)})
    return out


def build(pdf):
    objs3, c3, r3, _ = pv.page(pdf, 3)
    markers, summary = read_markers(objs3, c3, r3, FIG2A)
    objs5, c5, r5, _ = pv.page(pdf, 5)
    table = read_table2(objs5, c5, r5)
    rows, detail, validation = [], [], []
    for prop in COL:
        for nom, (Td, val, x_pt, y_pt) in zip(NOMINAL_C, markers[prop]):
            assert abs(Td - nom) <= 3.0, f"{prop}: the marker at {Td:.1f} C does not belong to the nominal {nom} C"
            detail.append({"property": prop, "nominal_C": nom, "drawn_C": round(Td, 2), "value_digitised": val, "x_pt": x_pt, "y_pt": y_pt})
    for prop, (tv, te, tT, printed) in table.items():
        d = next(x for x in detail if x["property"] == prop and x["nominal_C"] == tT)
        diff = abs(d["value_digitised"] - tv)
        assert diff <= te, f"{prop}: the digitised marker ({d['value_digitised']}) disagrees with Table 2 ({printed})"
        validation.append({"property": prop, "table2": printed, "digitised": d["value_digitised"], "drawn_C": d["drawn_C"], "difference": diff, "difference_in_printed_uncertainties": round(diff / te, 3)})
    for d in detail:
        prop, nom = d["property"], d["nominal_C"]
        T_K = round(nom + 273.15, 2)
        tab = table.get(prop)
        use_table = tab is not None and tab[2] == nom
        note = []
        if T_K < 300:
            note.append("below the 300 K window edge: recorded, not compared")
        if abs(d["drawn_C"] - nom) > 0.75:
            note.append(f"marker drawn at {d['drawn_C']:.1f} C ({d['drawn_C'] - nom:+.1f} C from the nominal temperature); the nominal temperature is used")
        if use_table:
            val, err = tab[0], tab[1]
            kind, loc = "table", LOCATION_TAB
            method = ""
            unit = "" if UNIT[prop] == "1" else " " + UNIT[prop]
            note.append(f"maximum of the series as printed in Table 2, printed uncertainty +/- {err:g}{unit}; the Fig. 2a marker reads {d['value_digitised']:.4g} "
                        f"(difference {abs(d['value_digitised'] - val):.3g}, inside the printed uncertainty)")
        else:
            val, kind, loc = float(f"{d['value_digitised']:.3g}"), "digitised", LOCATION_FIG
            s = summary[prop]
            method = (f"vector marker read from the PDF content stream with thesis_paper/scripts/na12_extract_sebastialuna.py; marker centre = centre of the 16-point body path; "
                      f"the {s['colour']} axis fitted linearly through its {s['major_ticks']} major ticks (labels {s['first_label']:g} to {s['last_label']:g}, step {s['step']:g}, "
                      f"maximum residual {s['max_residual_tick_steps']:.3f} tick steps); temperature axis through {summary['temperature_axis']['major_ticks']} major ticks; "
                      f"value rounded to 3 significant digits")
        rows.append({"lit_id": f"SL2022_{prop}_{nom}C", "compound_label": "CsSnI3 (black phase, vacuum-deposited thin film)", "composition": "CsSnI3", "property": prop,
                     "T_K": f"{T_K:.2f}", "value": f"{val:g}" if kind == "table" else f"{val:g}", "unit": UNIT[prop], "doi": DOI, "location": loc, "value_kind": kind,
                     "digitisation_method": method, "notes": "; ".join(note)})
    return {"calibration": summary, "table2_printed": {k: v[3] for k, v in table.items()}, "markers": detail, "validation": validation, "consistency_zT": consistency(rows), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=PDF)
    ap.add_argument("--append-values", default=None, help="results folder of an earlier run whose rows are appended to docs/na12_literature_values.csv")
    args = ap.parse_args()
    if args.append_values:
        d = Path(args.append_values)
        cfg = __import__("json").loads((d / "run_config.json").read_text(encoding="utf-8"))
        assert cfg["tree_clean"], "that run was made from a dirty tree"
        with open(d / "values_rows.csv", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        with open(VALUES, encoding="utf-8", newline="") as f:
            have = {r["lit_id"] for r in csv.DictReader(f)}
        assert not (have & {r["lit_id"] for r in rows}), "rows of this run are already in the values file"
        raw = VALUES.read_bytes()
        assert raw.endswith(b"\n")
        with open(VALUES, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
            for r in rows:
                w.writerow(r)
        print(f"appended {len(rows)} rows from {d} to {VALUES}")
        return 0
    prov = rr.provenance({"pdf": args.pdf}, __file__)
    res = build(REPO / args.pdf)
    out = rr.new_run_dir("na12_extraction")
    rr.write_json(out / "extraction.json", {k: v for k, v in res.items() if k != "rows"})
    with open(out / "values_rows.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(res["rows"])
    rr.write_json(out / "run_config.json", {**prov, "figure": "Fig. 2a (B-gamma CsSnI3), Table 2", "rules": "docs/decisions.md pre-registration B + amendment 1 (2026-10-13)"})
    print(f"wrote {out}")
    for v in res["validation"]:
        print("validation:", v)
    for c in res["consistency_zT"]:
        print("zT consistency:", c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
