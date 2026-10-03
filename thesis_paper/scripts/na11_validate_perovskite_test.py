"""
Validation of the structural perovskite test (perovskite_test.py) before it is used on any screening candidate.

Cases
  - synthetic prototypes built in this file: the cubic perovskite SrTiO3 (expected perovskite), the hexagonal BaNiO3 type with
    face-sharing octahedra (expected not), and the cubic anti-perovskite Sr3BiN (expected anti-perovskite, not perovskite);
  - real structures from the JARVIS dft_3d dataset downloaded by na10_jarvis_prep.py, picked by formula and space-group number, for
    compounds whose structure type is established: GdFeO3-type, rhombohedral and cubic perovskites (expected perovskite) and the
    hexagonal CsNiCl3 / BaNiO3 types (expected not).
A case whose entry is absent from the dataset is reported as missing, never silently skipped. The verdicts are not tuned: the script
stops with a non-zero exit code if any expectation fails, and the full table is written to the results folder either way.

Usage (from the repository root):
    PYTHONPATH=C:/Users/choha/py_extra/jarvis_2026.6.12 python thesis_paper/scripts/na11_validate_perovskite_test.py
"""

import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from jarvis.core.atoms import Atoms  # noqa: E402
from pymatgen.core import Lattice, Structure  # noqa: E402

import perovskite_test as pt  # noqa: E402
import run_record as rr  # noqa: E402

JARVIS_ZIP = "data/external/jarvis/jdft_3d-9-24-2025.json.zip"
JARVIS_JSON = "jdft_3d-9-24-2025.json"

# (formula, space-group number, expected perovskite verdict, structure type)
REAL_CASES = [
    ("SrTiO3", 221, True, "cubic perovskite"), ("BaTiO3", 221, True, "cubic perovskite"), ("KTaO3", 221, True, "cubic perovskite"),
    ("BaZrO3", 221, True, "cubic perovskite"), ("KMgF3", 221, True, "cubic perovskite"), ("CsSnI3", 221, True, "cubic perovskite"),
    ("CaTiO3", 62, True, "GdFeO3-type perovskite"), ("SrZrO3", 62, True, "GdFeO3-type perovskite"), ("CaMnO3", 62, True, "GdFeO3-type perovskite"),
    ("NaMgF3", 62, True, "GdFeO3-type perovskite"), ("BaZrS3", 62, True, "GdFeO3-type chalcogenide perovskite"),
    ("LaAlO3", 167, True, "rhombohedral perovskite"),
    ("BaNiO3", 194, False, "hexagonal, face-sharing octahedra"), ("CsNiCl3", 194, False, "hexagonal, face-sharing octahedra"),
    ("BaTiO3", 194, False, "hexagonal, face-sharing octahedra"), ("BaMnO3", 194, False, "hexagonal, face-sharing octahedra"),
]


def synthetic():
    """Prototype structures with their expected verdicts."""
    cubic = Structure.from_spacegroup("Pm-3m", Lattice.cubic(3.905), ["Sr", "Ti", "O"], [[0.5, 0.5, 0.5], [0, 0, 0], [0.5, 0, 0]])
    hexa = Structure.from_spacegroup("P6_3/mmc", Lattice.hexagonal(5.63, 4.82), ["Ba", "Ni", "O"], [[1 / 3, 2 / 3, 0.25], [0, 0, 0], [0.15, 0.30, 0.25]])
    anti = Structure.from_spacegroup("Pm-3m", Lattice.cubic(5.0), ["Bi", "N", "Sr"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]])
    return [("SrTiO3 (synthetic cubic)", cubic, True, False), ("BaNiO3 (synthetic hexagonal)", hexa, False, False), ("Sr3BiN (synthetic anti-perovskite)", anti, False, True)]


def main():
    """Entry point."""
    rows = []
    for name, s, expect, expect_anti in synthetic():
        v = pt.classify(s)
        rows.append({"case": name, "source": "synthetic", "expected_perovskite": expect, "perovskite": v["perovskite"], "anti_perovskite": v["anti_perovskite"],
                     "expected_anti": expect_anti, "dimensionality": v["dimensionality"], "reason": v["reason"],
                     "ok": v["perovskite"] == expect and v["anti_perovskite"] == expect_anti})
    with zipfile.ZipFile(REPO / JARVIS_ZIP) as z:
        data = json.loads(z.read(JARVIS_JSON))
    index = {}
    for d in data:
        index.setdefault((d["formula"], int(d["spg_number"])), []).append(d)
    for formula, spg, expect, kind in REAL_CASES:
        entries = index.get((formula, spg), [])
        if not entries:
            rows.append({"case": f"{formula} spg {spg}", "source": "JARVIS", "expected_perovskite": expect, "perovskite": None, "reason": "no such entry in the dataset", "ok": None, "kind": kind})
            continue
        for d in entries:
            s = Atoms.from_dict(d["atoms"]).pymatgen_converter()
            v = pt.classify(s)
            rows.append({"case": f"{formula} spg {spg} {d['jid']}", "source": "JARVIS", "kind": kind, "expected_perovskite": expect, "perovskite": v["perovskite"],
                         "anti_perovskite": v["anti_perovskite"], "dimensionality": v["dimensionality"], "reason": v["reason"], "ok": v["perovskite"] == expect})
    prov = rr.provenance({"jarvis_zip": JARVIS_ZIP, "perovskite_test": "thesis_paper/scripts/perovskite_test.py"}, __file__)
    out = rr.new_run_dir("na11_validation")
    summary = {"n_cases": len(rows), "n_ok": sum(1 for r in rows if r["ok"] is True), "n_failed": sum(1 for r in rows if r["ok"] is False),
               "n_missing": sum(1 for r in rows if r["ok"] is None), "rows": rows,
               "parameters": {"BOND_FACTOR": pt.BOND_FACTOR, "MIN_OPPOSITE_ANGLE": pt.MIN_OPPOSITE_ANGLE, "SEARCH_RADIUS": pt.SEARCH_RADIUS}}
    rr.write_json(out / "run_config.json", prov)
    rr.write_json(out / "validation.json", summary)
    print(f"wrote {out}")
    for r in rows:
        print(("OK  " if r["ok"] else "FAIL" if r["ok"] is False else "MISS"), r["case"], "expected", r["expected_perovskite"], "got", r["perovskite"], r["reason"][:90])
    print({k: v for k, v in summary.items() if k.startswith("n_")})
    return 1 if summary["n_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
