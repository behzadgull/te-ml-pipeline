"""
NA10 data preparation (no predictions): the JARVIS dft_3d dataset, restricted to entries with a computed n- or p-type Seebeck
coefficient, featurised with the same featurisers as the training data and mapped to the training data's chemistry clusters.

Steps
  1. Download dft_3d with jarvis-tools (version pinned below; it is installed outside the project environment, under
     C:/Users/choha/py_extra/jarvis_<version>, and put on PYTHONPATH). The downloaded zip and the extracted json are hashed.
  2. Keep entries with a numeric n-Seebeck or p-Seebeck. JARVIS Seebeck values are BoltzTraP constant-relaxation-time values at 600 K
     and a fixed doping; they are comparable to the training S only as a cross-domain test, which is how the thesis uses them. Electrical
     and thermal conductivity are not usable: JARVIS reports conductivity divided by an unknown relaxation time, and no zT.
  3. Parse each formula, assign composition_id and chemistry_cluster_id with the training canonicalisation (src.canonicalization through
     the Paper A external-validation helpers), featurise with MAGPIE and CBFV (Oliynyk) exactly as ESTM was, and assert that the feature
     columns equal the training columns in the same order. temperature_bin is set to the 600 K training bin.
  4. Flag each entry: composition seen in the training data; chemistry cluster seen in the training data; ABX3 stoichiometry (three
     elements in a 1:1:3 ratio; a stoichiometry flag only, not a structure test).

Outputs: thesis_paper/results/na10/<UTC stamp>/{run_config.json, summary.json, jarvis_entries.csv}; the featurised matrix is written
to data/external/jarvis/ (gitignored) and its SHA256 is recorded in run_config.json. Nothing is predicted.

Usage (from the repository root):
    PYTHONPATH=C:/Users/choha/py_extra/jarvis_2026.6.12 python thesis_paper/scripts/na10_jarvis_prep.py
"""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import jarvis  # noqa: E402
import jarvis.db.figshare as figshare  # noqa: E402

import paper_a_values as pav  # noqa: E402
import run_record as rr  # noqa: E402
from src import external_validation as ev  # noqa: E402
from src.nested_cv import get_feature_columns  # noqa: E402

JARVIS_TOOLS_VERSION = "2026.6.12"
DATASET = "dft_3d"
STORE = REPO / "data" / "external" / "jarvis"
TRAIN = "data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv"
SEEBECK_TEMPERATURE_K = 600  # JARVIS BoltzTraP Seebeck values are at 600 K
TRAINING_BIN = 600


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def numeric(x):
    """JARVIS stores missing values as the string 'na'."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def is_abx3(formula):
    """Three elements in a 1:1:3 ratio after reduction (a stoichiometry flag only)."""
    from pymatgen.core import Composition

    amounts = sorted(Composition(formula).reduced_composition.get_el_amt_dict().values())
    return len(amounts) == 3 and np.allclose(amounts, [1, 1, 3])


def main():
    """Entry point."""
    from importlib.metadata import version

    assert version("jarvis-tools") == JARVIS_TOOLS_VERSION, version("jarvis-tools")
    STORE.mkdir(parents=True, exist_ok=True)
    info = figshare.get_db_info()[DATASET]
    url, js_tag = info[0], info[1]
    data = figshare.data(DATASET, store_dir=str(STORE))
    zips = sorted(STORE.glob("*.zip"))
    assert zips, f"no zip stored under {STORE}"
    zip_path = next((z for z in zips if js_tag.replace(".json", "") in z.name or DATASET in z.name), zips[0])
    with zipfile.ZipFile(zip_path) as z:
        json_bytes = z.read(js_tag)
    dataset_info = {"jarvis_tools_version": JARVIS_TOOLS_VERSION, "dataset": DATASET, "url": url, "json_name": js_tag,
                    "zip_file": str(zip_path.relative_to(REPO)), "zip_sha256": sha_bytes(zip_path.read_bytes()), "zip_bytes": zip_path.stat().st_size,
                    "json_sha256": sha_bytes(json_bytes), "json_bytes": len(json_bytes), "n_entries_total": len(data)}
    assert len(json.loads(json_bytes)) == len(data)

    df = pd.DataFrame(data)
    df["n_seebeck"] = df["n-Seebeck"].map(numeric)
    df["p_seebeck"] = df["p-Seebeck"].map(numeric)
    keep = df["n_seebeck"].notna() | df["p_seebeck"].notna()
    sel = df.loc[keep, ["jid", "formula", "spg_number", "n_seebeck", "p_seebeck", "optb88vdw_bandgap", "ehull"]].copy()
    sel["optb88vdw_bandgap"] = sel["optb88vdw_bandgap"].map(numeric)
    sel["ehull"] = sel["ehull"].map(numeric)
    counts = {"entries_total": len(df), "with_n_seebeck": int(df["n_seebeck"].notna().sum()), "with_p_seebeck": int(df["p_seebeck"].notna().sum()),
              "with_n_or_p_seebeck": int(keep.sum()), "with_both": int((df["n_seebeck"].notna() & df["p_seebeck"].notna()).sum())}

    # canonicalise and featurise like ESTM (same helpers, same featurisers)
    sel["Formula"] = sel["formula"]
    sel["temperature_bin"] = TRAINING_BIN
    canonical, n_parse_failures = ev.canonicalize_estm(sel)
    feat, n_feat_failures = ev.featurize_estm(canonical)
    train_head = pd.read_csv(REPO / TRAIN, nrows=5)
    feature_cols = get_feature_columns(train_head)
    assert get_feature_columns(feat) == feature_cols, "JARVIS feature columns differ from the training columns"
    counts.update({"parse_failures": n_parse_failures, "featurisation_failed_unique_formulas": n_feat_failures, "featurised_entries": len(feat),
                   "n_feature_columns": len(feature_cols)})

    train = pd.read_csv(REPO / TRAIN, usecols=["composition_id", "chemistry_cluster_id"])
    feat["composition_seen"] = feat["composition_id"].isin(set(train["composition_id"].dropna()))
    feat["cluster_seen"] = feat["chemistry_cluster_id"].isin(set(train["chemistry_cluster_id"].dropna()))
    feat["abx3_stoichiometry"] = feat["formula"].map(is_abx3)
    counts.update({"composition_seen": int(feat["composition_seen"].sum()), "composition_unseen": int((~feat["composition_seen"]).sum()),
                   "cluster_seen": int(feat["cluster_seen"].sum()), "cluster_unseen": int((~feat["cluster_seen"]).sum()),
                   "abx3_stoichiometry": int(feat["abx3_stoichiometry"].sum()),
                   "abx3_stoichiometry_cluster_unseen": int((feat["abx3_stoichiometry"] & ~feat["cluster_seen"]).sum()),
                   "unique_composition_ids": int(feat["composition_id"].nunique()), "unique_clusters": int(feat["chemistry_cluster_id"].nunique())})

    feat_path = STORE / "jarvis_dft3d_seebeck_featurized.csv"
    feat.to_csv(feat_path, index=False)
    prov = rr.provenance({"training_csv": TRAIN, "src/external_validation.py": "src/external_validation.py"}, __file__)
    prov["jarvis"] = dataset_info
    prov["featurised_matrix"] = {"path": str(feat_path.relative_to(REPO)), "sha256": rr.sha256_file(feat_path), "bytes": feat_path.stat().st_size}
    prov["seebeck_temperature_K"] = SEEBECK_TEMPERATURE_K
    out = rr.new_run_dir("na10")
    entries = feat[["jid", "formula", "composition_id", "chemistry_cluster_id", "spg_number", "n_seebeck", "p_seebeck", "optb88vdw_bandgap", "ehull",
                    "composition_seen", "cluster_seen", "abx3_stoichiometry"]]
    entries.to_csv(out / "jarvis_entries.csv", index=False, lineterminator="\n")
    prov["entries_csv_sha256"] = rr.sha256_file(out / "jarvis_entries.csv", lf=True)
    rr.write_json(out / "run_config.json", prov)
    rr.write_json(out / "summary.json", counts)
    print(f"wrote {out}")
    print(json.dumps(counts, indent=1))
    print(json.dumps(dataset_info, indent=1))


if __name__ == "__main__":
    main()
