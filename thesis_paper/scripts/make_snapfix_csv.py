"""
Rebuild the snapfix featurized CSV (featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv) from File A's featurized CSV.

History: after the two grouping fixes of commit c1c6873 (SNAP(0.05) in src/canonicalization.chemistry_cluster_id, scheme S5 in nested_cv.randomized_group_kfold; CLAUDE.md, "Grouping Fixes"), the
chemistry_cluster_id column of File A's featurized CSV was regenerated and nothing else was changed (same rows, same other 406 columns; results/noise_floor/20260917T172251 records that
data_cleaning.py was not rerun). The script that did this was not committed. This script is its reconstruction: it streams File A's CSV, replaces the chemistry_cluster_id field of each row by
src.canonicalization.chemistry_cluster_id(parse_formula(composition)) with the module's default threshold and snap tolerance, checks that composition_id is unchanged, and writes every other field
untouched. With --compare-to it checks the result against an existing snapfix CSV (SHA256 and size, else the first differing byte, else column-by-column).

Usage (repository root):
    python thesis_paper/scripts/make_snapfix_csv.py --input <File A featurized CSV> --out <new CSV> [--compare-to <existing snapfix CSV>] [--record <run_config.json>]

File A: SHA256 e97406fa5223efa466d1e3fe4ffb2b1bc0303dc94b6adac6a824a643a9156f8d, 975,062,560 bytes (the script refuses any other input).
Expected snapfix: SHA256 d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489, 974,854,507 bytes.
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")
from src.canonicalization import DEFAULT_DOPANT_THRESHOLD_FRAC, DEFAULT_SNAP_TOLERANCE, chemistry_cluster_id, composition_id, parse_formula  # noqa: E402

FILE_A_SHA256 = "e97406fa5223efa466d1e3fe4ffb2b1bc0303dc94b6adac6a824a643a9156f8d"
FILE_A_BYTES = 975_062_560
SNAPFIX_SHA256 = "d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489"
SNAPFIX_BYTES = 974_854_507
FORMULA_COL, CLUSTER_COL, COMPOSITION_ID_COL = "composition", "chemistry_cluster_id", "composition_id"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


SNAPFIX_LINE_TERMINATOR = "\r\n"  # the snapfix CSV was written on Windows by pandas (os.linesep); File A has "\n" (first run of this script: every cell equal, 280,174 bytes fewer = one per line)


def rebuild(src, dst, term=SNAPFIX_LINE_TERMINATOR):
    """Stream src to dst, replacing only the chemistry_cluster_id field and writing line terminator `term`. Returns counts."""
    cache, n, changed, comp_id_mismatch, failed = {}, 0, 0, 0, 0
    with open(src, encoding="utf-8", newline="") as fi, open(dst, "w", encoding="utf-8", newline="") as fo:
        rd, wr = csv.reader(fi), csv.writer(fo, lineterminator=term)
        header = next(rd)
        wr.writerow(header)
        i_f, i_c, i_id = header.index(FORMULA_COL), header.index(CLUSTER_COL), header.index(COMPOSITION_ID_COL)
        for row in rd:
            formula = row[i_f]
            if formula not in cache:
                comp, err = parse_formula(formula)
                cache[formula] = (None, None) if comp is None else (composition_id(comp), chemistry_cluster_id(comp, DEFAULT_DOPANT_THRESHOLD_FRAC, DEFAULT_SNAP_TOLERANCE))
            cid, cluster = cache[formula]
            if cluster is None:
                failed += 1
            elif cid != row[i_id]:
                comp_id_mismatch += 1
            if cluster is not None and cluster != row[i_c]:
                changed += 1
            row[i_c] = row[i_c] if cluster is None else cluster
            wr.writerow(row)
            n += 1
    return {"rows": n, "unique_formulas": len(cache), "cluster_id_changed_vs_input": changed, "composition_id_mismatch": comp_id_mismatch, "formulas_failed_to_parse": failed, "line_terminator": repr(term)}


def compare(new, ref):
    """SHA256 and size first; if they differ, the first differing byte, then a column-by-column comparison."""
    out = {"new_sha256": sha256(new), "new_bytes": Path(new).stat().st_size, "ref_sha256": sha256(ref), "ref_bytes": Path(ref).stat().st_size}
    out["byte_identical"] = out["new_sha256"] == out["ref_sha256"]
    if not out["byte_identical"]:
        with open(new, "rb") as a, open(ref, "rb") as b:
            pos = 0
            while True:
                x, y = a.read(1 << 20), b.read(1 << 20)
                if x != y:
                    k = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
                    out["first_difference_at_byte"] = pos + k
                    break
                if not x:
                    break
                pos += len(x)
        with open(new, encoding="utf-8", newline="") as fa, open(ref, encoding="utf-8", newline="") as fb:
            ra, rb = csv.reader(fa), csv.reader(fb)
            ha, hb = next(ra), next(rb)
            diff_cols, rows = {}, 0
            for x, y in zip(ra, rb):
                rows += 1
                for j, (u, v) in enumerate(zip(x, y)):
                    if u != v:
                        diff_cols[ha[j]] = diff_cols.get(ha[j], 0) + 1
            out.update({"same_header": ha == hb, "rows_compared": rows, "cells_differing_by_column": diff_cols})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare-to", default=None)
    ap.add_argument("--record", default=None, help="write the run record (JSON) here")
    a = ap.parse_args()
    t0 = time.time()
    size = Path(a.input).stat().st_size
    in_sha = sha256(a.input)
    if (in_sha, size) != (FILE_A_SHA256, FILE_A_BYTES):
        sys.exit(f"input is not File A: sha256 {in_sha}, {size} bytes")
    res = rebuild(a.input, a.out)
    rec = {"script": "thesis_paper/scripts/make_snapfix_csv.py", "script_sha256": sha256(__file__), "input_sha256": in_sha, "input_bytes": size, "out_sha256": sha256(a.out), "out_bytes": Path(a.out).stat().st_size,
           "expected_snapfix_sha256": SNAPFIX_SHA256, "expected_snapfix_bytes": SNAPFIX_BYTES, "threshold_frac": DEFAULT_DOPANT_THRESHOLD_FRAC, "snap_tolerance": DEFAULT_SNAP_TOLERANCE,
           "git_head": git("rev-parse", "HEAD"), "tree_clean": git("status", "--porcelain") == "", "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "elapsed_s": round(time.time() - t0, 1), **res}
    rec["matches_expected_snapfix"] = (rec["out_sha256"], rec["out_bytes"]) == (SNAPFIX_SHA256, SNAPFIX_BYTES)
    if a.compare_to:
        rec["comparison_with_existing"] = compare(a.out, a.compare_to)
    print(json.dumps(rec, indent=1))
    if a.record:
        Path(a.record).parent.mkdir(parents=True, exist_ok=True)
        Path(a.record).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return 0 if rec["matches_expected_snapfix"] else 1


if __name__ == "__main__":
    sys.exit(main())
