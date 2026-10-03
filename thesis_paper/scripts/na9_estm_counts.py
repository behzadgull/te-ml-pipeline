"""
NA9: ESTM formula counts and the seen/unseen split, from the Paper A external-validation module's own data path (temperature filter,
canonicalisation, featurisation, the two deduplication passes), pointed at the snapfix training CSV the committed ESTM results used.
No model is fitted.

The ESTM row count after featurisation must equal the committed one (dropped plus surviving rows of either deduplication pass in
results/external_snapfix/20260917T160553/estm_results.json), and both deduplication passes must reproduce its dropped counts.

Usage (from the repository root):  python thesis_paper/scripts/na9_estm_counts.py
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import paper_a_values as pav  # noqa: E402
import run_record as rr  # noqa: E402
from src import external_validation as ev  # noqa: E402

TRAIN = "data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv"
ESTM_XLSX = "data/external/estm.xlsx"


def main():
    """Entry point."""
    prov = rr.provenance({"training_csv": TRAIN, "estm_xlsx": ESTM_XLSX, "estm_results": pav.ESTM, "src/external_validation.py": "src/external_validation.py"}, __file__)
    assert prov["inputs"]["training_csv"]["sha256"] == pav._json(pav.ESTM)["provenance"]["training_csv_sha256"]
    report, estm, train = ev.dry_run_inventory(estm_path=REPO / ESTM_XLSX, training_csv=REPO / TRAIN)
    e = pav._json(pav.ESTM)
    a, b = e["dedup_a_source_doi"], e["dedup_b_chemistry_cluster"]
    n_committed = a["n_dropped"] + a["n_surviving"]
    assert len(estm) == n_committed, (len(estm), n_committed)
    assert report["n_would_drop_doi_dedup"] == a["n_dropped"] and report["n_would_drop_cluster_dedup"] == b["n_dropped"]

    train_formulas = set(train["composition_id"].dropna())
    train_clusters = set(train["chemistry_cluster_id"].dropna())
    seen = estm["composition_id"].isin(train_formulas)
    seen_cluster = estm["chemistry_cluster_id"].isin(train_clusters)
    res = {
        "inventory": {k: v for k, v in report.items() if k != "unit_handling"},
        "estm_rows_in_scope": len(estm),
        "estm_unique_formulas": int(estm["composition_id"].nunique()),
        "estm_unique_clusters": int(estm["chemistry_cluster_id"].nunique()),
        "formulas_seen_in_training": int(estm.loc[seen, "composition_id"].nunique()),
        "formulas_unseen_in_training": int(estm.loc[~seen, "composition_id"].nunique()),
        "rows_seen_formula": int(seen.sum()),
        "rows_unseen_formula": int((~seen).sum()),
        "clusters_seen_in_training": int(estm.loc[seen_cluster, "chemistry_cluster_id"].nunique()),
        "clusters_unseen_in_training": int(estm.loc[~seen_cluster, "chemistry_cluster_id"].nunique()),
        "rows_with_property": {t: int(estm[t].notna().sum()) for t in pav.TARGETS if t in estm.columns},
        "rows_surviving_pass_a": a["n_surviving"],
        "rows_surviving_pass_b": b["n_surviving"],
    }
    out = rr.new_run_dir("na9")
    rr.write_json(out / "run_config.json", prov)
    rr.write_json(out / "results.json", res)
    print(f"wrote {out}")
    print(pd.Series(res).to_string())


if __name__ == "__main__":
    main()
