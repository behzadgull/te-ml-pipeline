"""
NA11 step 1: query the Materials Project and store the result. Run in its own environment: mp-api needs newer numpy, pandas and pymatgen
than the project's pinned versions, so it is installed under C:/Users/choha/py_extra/mp_api and put on PYTHONPATH for this script only.
The query output is plain JSON, read by na11_mp_filter_featurize.py in the project environment.

The API key is read from the environment variable MP_API_KEY. It is never printed, logged, written to the output or committed; the script
stops with exit code 2 if the variable is not set.

Query A (light, every material): energy_above_hull <= 0.05 eV/atom (no band-gap restriction), fields: id, formula, elements, E_hull, gap.
  The filter chain is applied locally so that the count after every filter can be reported from one stored result.
Query B (heavy): the same stability limit, three-element compounds only, with the structure and space group, for the ABX3 and structural
  tests. Its band-gap window is the thesis window 0.1 <= Eg <= 3.0 eV with the upper bound widened by GAP_SENS_FACTOR (1.5, to 4.5 eV), so that the
  filter script can report a sensitivity row; the main criteria are applied locally to the stored values and are not changed by the wider query.
  MP band gaps are PBE values and underestimate experimental gaps (the reason for the sensitivity row).

Output: data/external/mp/mp_query_<UTC stamp>.json (gitignored; SHA256 recorded) and data/external/mp/mp_query_<UTC stamp>.config.json (the
parameters, versions, counts, the output's SHA256, git HEAD and tree_clean; never the key). The config is kept beside the query, not in the repository
tree, so that the working tree stays clean for the next step; na11_mp_filter_featurize.py copies it into its own results folder.

Usage (from the repository root):
    set MP_API_KEY=...            (in the shell, not in a file)
    PYTHONPATH=C:/Users/choha/py_extra/mp_api python thesis_paper/scripts/na11_mp_query.py
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_record as rr  # noqa: E402  (standard library only)
STORE = REPO / "data" / "external" / "mp"
MP_API_VERSION = "0.46.5"
E_HULL_MAX = 0.05
GAP_WINDOW = (0.1, 3.0)  # the thesis window
GAP_SENS_FACTOR = 1.5
GAP_WINDOW_B = (GAP_WINDOW[0], GAP_WINDOW[1] * GAP_SENS_FACTOR)


def main():
    """Entry point."""
    key = os.environ.get("MP_API_KEY")
    if not key:
        print("MP_API_KEY is not set: set it in the shell (not in a file) and rerun. Nothing was queried.", file=sys.stderr)
        return 2
    assert version("mp-api") == MP_API_VERSION, version("mp-api")
    from mp_api.client import MPRester

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    STORE.mkdir(parents=True, exist_ok=True)
    with MPRester(key) as mpr:
        db_version = str(mpr.get_database_version())
        docs_a = mpr.materials.summary.search(
            energy_above_hull=(0, E_HULL_MAX),
            fields=["material_id", "formula_pretty", "elements", "energy_above_hull", "band_gap"],
        )
        a = [{"material_id": str(d.material_id), "formula": d.formula_pretty, "elements": sorted(str(e) for e in d.elements),
              "ehull": d.energy_above_hull, "gap": d.band_gap} for d in docs_a]
        docs_b = mpr.materials.summary.search(
            energy_above_hull=(0, E_HULL_MAX), band_gap=GAP_WINDOW_B, num_elements=(3, 3),
            fields=["material_id", "formula_pretty", "energy_above_hull", "band_gap", "is_stable", "symmetry", "structure"],
        )
        b = [{"material_id": str(d.material_id), "formula": d.formula_pretty, "ehull": d.energy_above_hull, "gap": d.band_gap,
              "is_stable": bool(d.is_stable), "spg_number": int(d.symmetry.number), "spg_symbol": d.symmetry.symbol,
              "structure": d.structure.as_dict()} for d in docs_b]
    out = STORE / f"mp_query_{stamp}.json"
    payload = {"meta": {"utc_stamp": stamp, "mp_api_version": MP_API_VERSION, "mp_database_version": db_version, "e_hull_max": E_HULL_MAX,
                        "gap_window_main": list(GAP_WINDOW), "gap_window_B": list(GAP_WINDOW_B), "gap_sens_factor": GAP_SENS_FACTOR, "n_A": len(a), "n_B": len(b)}, "A": a, "B": b}
    out.write_text(json.dumps(payload), encoding="utf-8")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    prov = rr.provenance({}, __file__)
    cfg = {**payload["meta"], "output": str(out.relative_to(REPO)), "output_sha256": sha, "output_bytes": out.stat().st_size, "api_key_recorded": False,
           "git_head": prov["git_head"], "tree_clean": prov["tree_clean"], "dirty_files": prov["dirty_files"], "script": prov["script"],
           "script_sha256": prov["script_sha256"], "python": prov["python"],
           "versions": {k: version(k) for k in ("mp-api", "pymatgen", "numpy", "pandas")}}
    (STORE / f"mp_query_{stamp}.config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"stored {len(a)} stable-enough materials (A) and {len(b)} three-element compounds with structures (B); sha256 {sha}; tree_clean {prov['tree_clean']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
