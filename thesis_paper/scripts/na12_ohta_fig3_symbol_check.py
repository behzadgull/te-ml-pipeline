"""
NA12: are the (b) and (d) series of Fig. 3 of Ohta et al. 2005 (thermal conductivity) swapped? Decided from the image only.

The clicks of the digitised files `_b_filled circles.csv` and `_d_filled triangles.csv` are tested against the shape of the symbol at the clicked position. Templates are the legend symbols of the
figure itself ((b) filled circle, (d) filled triangle), cut from the bitmap. In each column that holds clicks of both series, the best circle and the best triangle are found as local maxima of
the fraction of template pixels that are dark (>= 0.80), the pair (or single symbol) that explains most of the dark pixels with the least dark-free template area is kept, and each click is
assigned to the nearest fitted symbol. A column is separable when both a circle and a triangle were fitted and their centres are at least 20 px apart; otherwise the filled symbols are
merged into one body and the clicks there cannot be assigned. A swap would show as separable columns whose b clicks sit on the triangle and d clicks on the circle.

The Table II room-temperature kappa (b 9.1 < d 9.6) is not used: the decision is from the shapes.

Usage (repository root, clean tree):
    python thesis_paper/scripts/na12_ohta_fig3_symbol_check.py --csv-dir <fig3 folder> --image <Fig. 3 bitmap>     # writes thesis_paper/results/na12_symbol_check/<UTC>/symbol_check.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import na12_extract_ohta as o  # noqa: E402
import run_record as rr  # noqa: E402

LEGEND_B = (1785, 1657)  # centre of the legend symbol (b), filled circle, in the x3 bitmap of Fig. 3
LEGEND_D = (1778, 1912)  # centre of the legend symbol (d), filled triangle
MIN_COVERAGE = 0.80
MIN_SEPARATION_PX = 20.0
COLUMN_PX = 120.0  # clicks of one column are within this many px of each other


def legend_template(dark, cx, cy, half=26):
    """Binary template of a legend symbol: the connected dark component at (cx, cy), cropped to its bounding box."""
    sub = dark[cy - half: cy + half + 1, cx - half: cx + half + 1]
    lab, n = ndi.label(sub)
    k = lab[half, half] or max(range(1, n + 1), key=lambda i: (lab == i).sum())
    m = lab == k
    ys, xs = np.nonzero(m)
    return m[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def coverage_map(dark, T, x0, x1, y0, y1):
    """Fraction of the template's pixels that are dark, for template centres in [x0, x1] x [y0, y1] (stride 2)."""
    h, w = T.shape
    out = {}
    for cy in range(y0, y1 + 1, 2):
        for cx in range(x0, x1 + 1, 2):
            win = dark[cy - h // 2: cy - h // 2 + h, cx - w // 2: cx - w // 2 + w]
            if win.shape == T.shape:
                out[(cx, cy)] = float((win & T).sum() / T.sum())
    return out


def peaks(sm, n=6, sep=10):
    chosen = []
    for p, v in sorted(sm.items(), key=lambda kv: -kv[1]):
        if v < MIN_COVERAGE:
            break
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= sep for q, _ in chosen):
            chosen.append((p, v))
        if len(chosen) >= n:
            break
    return chosen


def mask_at(T, cx, cy, shape):
    m = np.zeros(shape, bool)
    h, w = T.shape
    m[cy - h // 2: cy - h // 2 + h, cx - w // 2: cx - w // 2 + w] = T
    return m


def check(csv_dir, image):
    dark = np.array(Image.open(image).convert("L")) < o.THRESHOLD
    TC, TT = legend_template(dark, *LEGEND_B), legend_template(dark, *LEGEND_D)
    data, ucal, logy, held = o.load_csvs(csv_dir, o.FIGURES[3]["project"])
    data.pop("_project_dataset_names")
    clicks = {fl: [held[fl][i] if fl in held else o.click_px(ucal, False, *pt) for i, pt in enumerate(data[fl]["pts"])] for fl in "bd"}
    cols = {}
    for fl in "bd":
        for i, (px, py) in enumerate(clicks[fl]):
            cols.setdefault(int(round(px / COLUMN_PX)), []).append((fl, i, px, py, data[fl]["pts"][i][1]))
    out = []
    for key, items in sorted(cols.items()):
        cx0 = int(np.mean([it[2] for it in items]))
        ys = [it[3] for it in items]
        y0, y1 = int(min(ys)) - 50, int(max(ys)) + 50
        x0, x1 = cx0 - 18, cx0 + 18
        pc, pt = peaks(coverage_map(dark, TC, x0, x1, y0, y1)), peaks(coverage_map(dark, TT, x0, x1, y0, y1))
        best = None
        reg = np.zeros(dark.shape, bool)
        reg[y0 - 30: y1 + 30, x0 - 30: x1 + 30] = True
        for c in [None] + [p for p, _ in pc]:
            for t in [None] + [p for p, _ in pt]:
                if c is None and t is None:
                    continue
                U = np.zeros(dark.shape, bool)
                if c is not None:
                    U |= mask_at(TC, *c, dark.shape)
                if t is not None:
                    U |= mask_at(TT, *t, dark.shape)
                sc = int((dark & U & reg).sum()) - 1.5 * int((U & ~dark & reg).sum())
                if best is None or sc > best[0]:
                    best = (sc, c, t)
        _, c, t = best
        both = c is not None and t is not None
        separable = bool(both and np.hypot(c[0] - t[0], c[1] - t[1]) >= MIN_SEPARATION_PX)
        has_b = any(it[0] == "b" for it in items)
        has_d = any(it[0] == "d" for it in items)
        for fl, i, px, py, kv in items:
            dc = float(np.hypot(px - c[0], py - c[1])) if c else None
            dt = float(np.hypot(px - t[0], py - t[1])) if t else None
            near = "circle" if (dt is None or (dc is not None and dc < dt)) else "triangle"
            expect = "circle" if fl == "b" else "triangle"
            if separable:
                status = "consistent" if near == expect else "SWAPPED"
            elif has_b and has_d:
                status = "not separable (the filled symbols of this column are merged)"
            else:  # one filled series only in this column: it is consistent if the nearest fitted symbol, within 20 px, is of its expected shape
                d_near = dc if near == "circle" else dt
                status = "consistent (single filled symbol)" if (near == expect and d_near is not None and d_near <= 20.0) else "not separable"
            out.append({"column_px": key * int(COLUMN_PX), "series_file": fl, "marker": i, "kappa_WmK": round(kv, 3), "click_px": round(px, 1), "click_py": round(py, 1),
                        "circle_fit": "" if c is None else f"{c[0]},{c[1]}", "triangle_fit": "" if t is None else f"{t[0]},{t[1]}",
                        "dist_to_circle_px": "" if dc is None else round(dc, 1), "dist_to_triangle_px": "" if dt is None else round(dt, 1), "nearest_symbol": near,
                        "label_expects": expect, "column_separable": separable, "status": status})
    return out, {"legend_circle_px": int(TC.sum()), "legend_triangle_px": int(TT.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    inputs = {"image": str(Path(args.image).resolve()), "project": str((Path(args.csv_dir) / o.FIGURES[3]["project"]).resolve())}
    for p in sorted(Path(args.csv_dir).glob("*.csv")):
        inputs[f"csv_{p.stem.lstrip('_')[0]}"] = str(p.resolve())
    prov = rr.provenance(inputs, __file__)
    rows, info = check(args.csv_dir, args.image)
    out = rr.new_run_dir("na12_symbol_check")
    with open(out / "symbol_check.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    swapped = [r for r in rows if r["status"] == "SWAPPED"]
    verdict = ("no separable column shows a swap" if not swapped else f"{len(swapped)} markers on the wrong symbol: SWAPPED")
    rr.write_json(out / "run_config.json", {**prov, "paper": "Ohta et al. 2005, Fig. 3: are series (b) and (d) swapped?", "legend_templates": info, "n_clicks": len(rows),
                                            "n_consistent": sum(r["status"].startswith("consistent") for r in rows), "n_not_separable": sum(r["status"].startswith("not separable") for r in rows),
                                            "n_swapped": len(swapped), "verdict": verdict})
    print(f"wrote {out}: {len(rows)} clicks; {verdict}")
    for r in rows:
        print(f"  {r['series_file']} {r['marker']}  kappa {r['kappa_WmK']:5.2f}  nearest {r['nearest_symbol']:8s} (to circle {r['dist_to_circle_px']}, to triangle {r['dist_to_triangle_px']})  {r['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
