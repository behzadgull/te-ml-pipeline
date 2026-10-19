"""
NA12: the not-legible counts that accompany the comparison (docs/decisions.md, amendments 7, 9, 11, 12, 13 and 17). For each digitised scanned paper (Ohta et al., Ohtaki et al., Rajasekaran et al.) the official
extraction run that produced its rows in the values file is found (the latest run folder of thesis_paper/results/na12_extraction whose values_rows.csv holds rows with the paper's lit_id prefix and whose rows are
identical to those in the values file), and the entries of its not_legible.csv are counted by kind and by whether the temperature is inside the window 300 <= T < 800 K:
  removed by a gate           "digitised marker not legible" (amendments 12, 13, 17)
  unplaceable click           "clicked marker skipped" (amendment 11)
  empty derived column        "no digitised marker" (a column of the figure in which a series has no marker; amendments 7, 9, 13)
The vector papers (Sebastia-Luna, Singh, Flahaut, Shibasaki) have no marker centres to place and no list. Writes not_legible_counts.json into --out-dir.

Usage (repository root):  python thesis_paper/scripts/na12_not_legible_counts.py --out-dir <dir>
"""

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_record as rr  # noqa: E402

RUNS = REPO / "thesis_paper" / "results" / "na12_extraction"
VALUES = REPO / "thesis_paper" / "docs" / "na12_literature_values.csv"
PAPERS = {"OH2005_": "Ohta et al. 2005", "OHTAKI1995_": "Ohtaki et al. 1995", "RJ2020_": "Rajasekaran et al. 2020"}
KINDS = (("removed by a gate", "digitised marker not legible"), ("unplaceable click", "clicked marker skipped"), ("empty derived column", "no digitised marker"))


def rows_of(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def find_run(prefix, values):
    """The latest run folder whose rows for the prefix are exactly the rows of the values file."""
    want = sorted((r["lit_id"], r["value"], r["T_K"]) for r in values if r["lit_id"].startswith(prefix))
    hits = []
    for d in sorted(RUNS.iterdir()):
        f = d / "values_rows.csv"
        if not f.exists():
            continue
        got = sorted((r["lit_id"], r["value"], r["T_K"]) for r in rows_of(f) if r["lit_id"].startswith(prefix))
        if got and got == want:
            hits.append(d)
    assert hits, f"no run folder reproduces the rows of {prefix}"
    return hits[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    prov = rr.provenance({"values": "thesis_paper/docs/na12_literature_values.csv"}, __file__)  # commit, tree state and the SHA256 of the values file, before anything is written
    values = rows_of(VALUES)
    out = {}
    for prefix, name in PAPERS.items():
        d = find_run(prefix, values)
        nl = rows_of(d / "not_legible.csv")
        counts = {k: {"total": 0, "inside_window": 0} for k, _ in KINDS}
        other = 0
        for r in nl:
            for k, key in KINDS:
                if key in r["reason"]:
                    counts[k]["total"] += 1
                    t = float(r["T_K"]) if r["T_K"] else None
                    counts[k]["inside_window"] += int(t is not None and 300.0 <= t < 800.0)
                    break
            else:
                other += 1
        assert other == 0, (name, other)
        out[name] = {"run_folder": str(d.relative_to(REPO)).replace("\\", "/"), "run_config_sha256": rr.sha256_file(d / "run_config.json"), "not_legible_csv_sha256": rr.sha256_file(d / "not_legible.csv"),
                     "entries": len(nl), "by_kind": counts, "rows_in_values_file": sum(1 for r in values if r["lit_id"].startswith(prefix))}
    outd = Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    rr.write_json(outd / "not_legible_counts.json", out)
    rr.write_json(outd / "not_legible_counts_run_config.json", prov)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
