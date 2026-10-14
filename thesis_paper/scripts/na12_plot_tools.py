"""
Helpers for reading vector plots of the NA12 source papers (pdf_vector.py): stroke segments, plot frame, tick marks and tick-label calibration, marker shapes.
Used by na12_extract_flahaut.py and na12_extract_singh.py (docs/decisions.md, 2026-10-13, amendment 1 to pre-registration B).
"""

import re

import pdf_vector as pv


def segments(paths, box=None):
    """All stroked straight segments (xa, ya, xb, yb) of the paths, optionally inside box=(x0, y0, x1, y1)."""
    out = []
    for p in paths:
        if p["kind"] != "S":
            continue
        for sp in p["pts"]:
            for (xa, ya), (xb, yb) in zip(sp, sp[1:]):
                if box is None or (box[0] <= min(xa, xb) and max(xa, xb) <= box[2] and box[1] <= min(ya, yb) and max(ya, yb) <= box[3]):
                    out.append((xa, ya, xb, yb))
    return out


def frame_of(segs, min_len=40.0):
    """Bounding rectangle (x0, y0, x1, y1) of the long horizontal and vertical segments."""
    h = [s for s in segs if abs(s[1] - s[3]) < 1e-6 and abs(s[2] - s[0]) >= min_len]
    v = [s for s in segs if abs(s[0] - s[2]) < 1e-6 and abs(s[3] - s[1]) >= min_len]
    assert h and v, "no frame found"
    return (min(s[0] for s in v), min(s[1] for s in h), max(s[0] for s in v), max(s[1] for s in h))


def left_ticks(segs, frame, lo=2.3, hi=3.0, tol=0.6):
    """y positions of the major tick marks on the left axis (horizontal segments of length lo..hi that end on the frame's left edge)."""
    x0 = frame[0]
    out = []
    for xa, ya, xb, yb in segs:
        if abs(ya - yb) < 1e-6 and lo <= abs(xb - xa) <= hi and (abs(max(xa, xb) - x0) < tol or abs(min(xa, xb) - x0) < tol):
            out.append(round(ya, 3))
    return sorted(set(out))


def bottom_ticks(segs, frame, lo=2.3, hi=3.0, tol=0.6):
    """x positions of the major tick marks on the bottom axis."""
    y0 = frame[1]
    out = []
    for xa, ya, xb, yb in segs:
        if abs(xa - xb) < 1e-6 and lo <= abs(yb - ya) <= hi and (abs(max(ya, yb) - y0) < tol or abs(min(ya, yb) - y0) < tol):
            out.append(round(xa, 3))
    return sorted(set(out))


def number(s):
    """A tick label as a float ('0,5' with a decimal comma, '-100', '1.2'); None if it is not a number."""
    t = s.strip().replace(",", ".").replace("−", "-")
    return float(t) if re.fullmatch(r"-?\d{1,5}(?:\.\d+)?", t) else None


def split_run(run, n):
    """Split a run of concatenated, equally spaced integer labels ('3004005006007008009001000') into the n numbers whose digits make up the run."""
    for first in range(0, 2001, 50):
        for step in (10, 20, 25, 50, 100, 200, 250, 500):
            seq = [first + step * k for k in range(n)]
            if "".join(str(v) for v in seq) == run:
                return seq
    raise AssertionError(f"cannot split {run!r} into {n} equally spaced labels")


def linear_fit(pos, vals):
    """value = a * pos + b by least squares; returns a, b and the maximum residual in units of the median label step."""
    n = len(pos)
    mp, mv = sum(pos) / n, sum(vals) / n
    a = sum((p - mp) * (v - mv) for p, v in zip(pos, vals)) / sum((p - mp) ** 2 for p in pos)
    b = mv - a * mp
    steps = sorted(abs(vals[i + 1] - vals[i]) for i in range(n - 1))
    step = steps[len(steps) // 2]
    return a, b, max(abs(a * p + b - v) for p, v in zip(pos, vals)) / step


def shape_class(sp, ncurve):
    """open marker shape from its subpath: circle (curved), triangle (3 points), square (axis-aligned corners), diamond (points at the mid-edges)."""
    xs = [a for a, b in sp]
    ys = [b for a, b in sp]
    n = len(sp)
    if ncurve >= 3:
        return "circle"
    if n == 3:
        return "triangle"
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    corners = sum(1 for a, b in sp if (abs(a - x0) < 1e-6 or abs(a - x1) < 1e-6) and (abs(b - y0) < 1e-6 or abs(b - y1) < 1e-6))
    mids = sum(1 for a, b in sp if abs(a - cx) < 0.06 * (x1 - x0 + 1e-9) or abs(b - cy) < 0.06 * (y1 - y0 + 1e-9))
    if n in (4, 5) and corners >= 4:
        return "square"
    if n in (4, 5) and mids >= 4:
        return "diamond"
    return f"other{n}"


def markers(paths, box, min_size=2.5, max_size=8.0):
    """[(shape, cx, cy, size, line width)] of the small closed stroked shapes inside box."""
    out = []
    for p in paths:
        for sp, nc in zip(p["pts"], p["ncurve"]):
            xs = [a for a, b in sp]
            ys = [b for a, b in sp]
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
            if box[0] <= cx <= box[2] and box[1] <= cy <= box[3] and min_size < w < max_size and h > min_size and len(sp) >= 3:
                out.append((shape_class(sp, nc), cx, cy, round(w, 1), round(p["width"], 2)))
    return out
