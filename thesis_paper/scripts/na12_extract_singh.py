"""
NA12: experimental Seebeck coefficient of LaCoO3 from Singh & Pandey 2017 (Philosophical Magazine 97(6):451-463, DOI 10.1080/14786435.2016.1263404), read from the PDF.
Rules: docs/decisions.md, pre-registration B and its amendment 1 (2026-10-13). No out-of-fold prediction is read or used here.

Only the experimental data are read: the markers of Fig. 1 (PDF page 5; the measurement of a polycrystalline LaCoO3 sample, 300-600 K) and the 300 K value stated in the abstract.
The DFT and Boltzmann-transport results of the paper (S against chemical potential, calculated S(T), power factor, ZT; Figs. 4 to 7) are not experimental and are not read.

Each marker is a measurement at its drawn temperature (amendment 1, rule a): the temperature of a marker is read from the temperature axis, which is calibrated on the 7 labelled major ticks
(300 to 600 K); the S axis on the 6 labelled major ticks (100 to 600 uV/K, the tick of 0 and 700 are on the same grid and are checked). The markers of the inset (a zoom of the same data) are
not read twice; the legend symbol is not a datum.

Validation (asserted): the marker at 300 K is the abstract's 635 uV/K; the marker at 600 K is its 46 uV/K; the change between 300 and 540 K is the text's 576 uV/K and between 540 and 600 K
its 13 uV/K (all printed with a tilde, so they are accepted within 3 uV/K).

Usage (repository root, clean tree):
    python thesis_paper/scripts/na12_extract_singh.py                        # writes thesis_paper/results/na12_extraction/<UTC>/
    python thesis_paper/scripts/na12_extract_singh.py --append-values <dir>  # appends the rows of that run to docs/na12_literature_values.csv
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

PDF = "thesis_paper/source/pdfs/singh2016.pdf"
VALUES = REPO / "thesis_paper" / "docs" / "na12_literature_values.csv"
DOI = "10.1080/14786435.2016.1263404"
COLUMNS = ["lit_id", "compound_label", "composition", "property", "T_K", "value", "unit", "doi", "location", "value_kind", "digitisation_method", "notes"]
RED = (1.0, 0.0, 0.0)
JOURNAL = "Philos. Mag. 97(6):451-463"


def read(pdf):
    objs, content, res, _ = pv.page(pdf, 5)
    paths, texts = pv.interpret(objs, content, res)
    segs = T.segments(paths)
    fr = T.frame_of(segs, min_len=100.0)
    yt = T.left_ticks(segs, fr, lo=4.0, hi=6.0)
    xt = T.bottom_ticks(segs, fr, lo=4.0, hi=6.0)
    # y labels: 100..600; each label's tick is the one nearest to (baseline + offset); the offset is the one that gives the best straight line
    ylab = sorted((t["y"], T.number(t["text"])) for t in texts if fr[0] - 32 <= t["x"] < fr[0] and fr[1] - 8 <= t["y"] <= fr[3] + 8 and T.number(t["text"]) is not None and t["size"] >= 6)
    best = None
    for k in range(0, 60):
        off = k * 0.1
        tk = [min(yt, key=lambda y: abs(y - (ly + off))) for ly, _ in ylab]
        if len(set(tk)) != len(tk):
            continue
        a, b, r = T.linear_fit(tk, [v for _, v in ylab])
        if best is None or r < best[0]:
            best = (r, off, a, b, tk)
    r, off, ya, yb, majors = best
    assert r < 0.02, r
    # x labels: the run '300350...600' (7 labels, step 50); the first tick of the run is the tick nearest to the centre of the first label
    runs = [t for t in texts if fr[0] - 5 <= t["x"] <= fr[2] and fr[1] - 30 <= t["y"] <= fr[1] and re.fullmatch(r"\d{6,}", t["text"].strip())]
    assert len(runs) == 1
    xv = T.split_run(runs[0]["text"].strip(), 7)
    centre = runs[0]["x"] + 1.5 * runs[0]["size"] * 0.56
    start = min(range(len(xt) - 6), key=lambda i: abs(xt[i] - centre))
    xa, xb, xr = T.linear_fit(xt[start:start + 7], xv)
    assert xr < 0.02, xr
    # the inset: the other (smaller) black rectangle inside the frame
    inset = None
    for p in paths:
        if p["kind"] == "S" and p["color"] == (0.0, 0.0, 0.0):
            for sp in p["pts"]:
                if len(sp) == 5:
                    xs = [a for a, b in sp]
                    ys = [b for a, b in sp]
                    w, h = max(xs) - min(xs), max(ys) - min(ys)
                    if 60 < w < 90 and 40 < h < 60:
                        inset = (min(xs), min(ys), max(xs), max(ys))
    assert inset is not None
    mk = []
    for p in paths:
        if p["kind"] == "S" and p["color"] == RED and len(p["pts"]) == 2:
            pts = [q for sp in p["pts"] for q in sp]
            xs = [a for a, b in pts]
            ys = [b for a, b in pts]
            if max(xs) - min(xs) < 4 and max(ys) - min(ys) < 4:
                cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
                if fr[0] <= cx <= fr[2] and fr[1] <= cy <= fr[3] and not (inset[0] <= cx <= inset[2] and inset[1] <= cy <= inset[3]):
                    mk.append((xa * cx + xb, ya * cy + yb, cx, cy))
    mk.sort()
    # the legend symbol sits inside the inset's x range below it: it is a single closed subpath and is not among the two-subpath data markers, so nothing to remove
    cal = {"frame": [round(v, 2) for v in fr], "y_major_ticks": len(yt), "y_labels": [v for _, v in ylab], "y_label_tick_offset_pt": round(off, 2), "y_residual_label_steps": round(r, 4),
           "x_major_ticks": len(xt), "x_labels": f"{xv[0]}..{xv[-1]}", "x_first_tick_used": round(xt[start], 2), "x_residual_label_steps": round(xr, 4), "inset": [round(v, 1) for v in inset]}
    # text values from the abstract (PDF page 2)
    o2, c2, r2, _ = pv.page(pdf, 2)
    _, t2 = pv.interpret(o2, c2, r2)
    s = re.sub(r"\s+", " ", " ".join(t["text"] for t in t2))
    m = re.search(r"experimentalvalueofSeebeck.{0,20}at300Kisfoundtobe∼(\d+)", s.replace(" ", ""))
    assert m, "the abstract's 300 K value was not found"
    t300 = float(m.group(1))
    m2 = re.search(r"reachesto∼(\d+).{0,3}V/Kat∼600K", s.replace(" ", ""))
    assert m2, "the abstract's 600 K value was not found"
    t600 = float(m2.group(1))
    # text of page 5: the changes between 300 and 540 K and between 540 and 600 K
    s5 = re.sub(r"\s+", "", " ".join(t["text"] for t in texts))
    c1 = re.search(r"change.{0,6}of∼(\d+)", s5)
    c2_ = re.search(r"540–600K,thechangeinthevalue.{0,12}is∼(\d+)", s5)
    assert c1 and c2_, "the changes quoted in the text were not found"
    return {"calibration": cal, "markers": mk, "abstract_300K": t300, "abstract_600K": t600, "change_300_540": float(c1.group(1)), "change_540_600": float(c2_.group(1))}


def at(markers, T_K):
    """The marker nearest to the temperature."""
    return min(markers, key=lambda m: abs(m[0] - T_K))


def build(pdf):
    d = read(pdf)
    mk = d["markers"]
    m300, m600, m540 = at(mk, 300.0), at(mk, 600.0), at(mk, 540.0)
    validation = [
        {"statement": "S at 300 K is ~635 uV/K (abstract, page 451)", "text": d["abstract_300K"], "read": round(m300[1], 2), "T_K": round(m300[0], 2), "tolerance": 3.0},
        {"statement": "S reaches ~46 uV/K at ~600 K (abstract)", "text": d["abstract_600K"], "read": round(m600[1], 2), "T_K": round(m600[0], 2), "tolerance": 3.0},
        {"statement": "the change of S between 300 and 540 K is ~576 uV/K (page 454)", "text": d["change_300_540"], "read": round(m300[1] - m540[1], 2), "tolerance": 3.0},
        {"statement": "the change of S between 540 and 600 K is ~13 uV/K (page 454)", "text": d["change_540_600"], "read": round(m540[1] - m600[1], 2), "tolerance": 3.0},
    ]
    for v in validation:
        assert abs(v["read"] - v["text"]) <= v["tolerance"], v
    rows = []
    loc = f"{JOURNAL}, Fig. 1 (article page 454, PDF page 5)"
    base = {"compound_label": "LaCoO3 (polycrystalline, solution combustion; experimental)", "composition": "LaCoO3", "property": "S", "unit": "uV/K", "doi": DOI}
    rows.append({**base, "lit_id": "SG2017_S_300K_text", "T_K": "300", "value": f"{d['abstract_300K']:g}", "location": f"{JOURNAL}, abstract (article page 451, PDF page 2)",
                 "value_kind": "text", "digitisation_method": "", "notes": "experimental value at 300 K as stated in the abstract (printed with a tilde)"})
    cal = d["calibration"]
    for i, (Tk, S, cx, cy) in enumerate(mk, 1):
        note = ""
        if Tk < 300.0:
            note = "below the 300 K window edge: recorded, not compared"
        rows.append({**base, "lit_id": f"SG2017_S_Fig1_{i:03d}", "T_K": f"{Tk:.1f}", "value": f"{S:.3g}", "location": loc, "value_kind": "digitised",
                     "digitisation_method": ("vector marker (open square) read from the PDF content stream with thesis_paper/scripts/na12_extract_singh.py; marker centre; "
                                             f"temperature axis fitted on {cal['x_major_ticks'] - 1} labelled major ticks ({cal['x_labels']}), S axis on {len(cal['y_labels'])} labelled major ticks "
                                             f"({cal['y_labels'][0]:g} to {cal['y_labels'][-1]:g}); maximum residual {max(cal['y_residual_label_steps'], cal['x_residual_label_steps']):.3f} label steps; "
                                             "S to 3 significant digits, T to 0.1 K"),
                     "notes": note})
    return {"calibration": cal, "n_markers": len(mk), "validation": validation, "rows": rows}


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
    rr.write_json(out / "run_config.json", {**prov, "paper": "Singh & Pandey 2017, experimental S only (Fig. 1 and the 300 K text value)",
                                            "rules": "docs/decisions.md pre-registration B + amendment 1 (2026-10-13)"})
    print(f"wrote {out}: {len(res['rows'])} rows ({res['n_markers']} markers)")
    for v in res["validation"]:
        print("validation:", v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
