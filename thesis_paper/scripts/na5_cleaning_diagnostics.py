"""
NA5: cleaning-step diagnostics that the committed funnel does not hold, from a rerun of src/data_cleaning.py's steps on the saved raw
pull of the Paper A snapshot (never src/data_acquisition.py, which pulls the live database).

Reports: raw data points before range filtering; mean zT before and after step 6; number of spikes set to NaN in step 11; and the effect
of a pre-2005 publication filter and a minimum-property-count filter on the final rows. The rerun is accepted only if its row count
after every step equals the committed funnel (results/cleaning_funnel/20260914T100914/funnel_counts.json) and its final row count
equals the committed cleaned CSV.

Usage (from the repository root):  python thesis_paper/scripts/na5_cleaning_diagnostics.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import paper_a_values as pav  # noqa: E402
import run_record as rr  # noqa: E402
from src import data_cleaning as dc  # noqa: E402

RAW = "checkpoints/saved_predictions/te-ml-pipeline/data/raw"
CLEANED = "checkpoints/saved_predictions/te-ml-pipeline/data/processed/cleaned_ThermoelectricMaterials_2026-08-15.csv"
FUNNEL = pav.FUNNEL
PROPS = list(dc.WIDE_PROPERTIES)


def count_points(curves):
    """Number of (x, y) points in a curves table (arrays stored as JSON strings)."""
    n = 0
    for x in curves["x"]:
        try:
            n += len(json.loads(x))
        except (json.JSONDecodeError, TypeError):
            pass
    return n


def main():
    """Entry point."""
    raw_meta = pav._json(pav.RAWMETA)["files"]
    inputs = {f"raw_{k}": f"{RAW}/ThermoelectricMaterials_{k}.csv.gz" for k in ("papers", "samples", "curves")}
    inputs["funnel_counts"] = FUNNEL
    inputs["cleaned_csv_file_a"] = CLEANED
    inputs["src/data_cleaning.py"] = "src/data_cleaning.py"
    prov = rr.provenance(inputs, __file__)
    for k in ("papers", "samples", "curves"):
        assert prov["inputs"][f"raw_{k}"]["sha256"] == raw_meta[k]["sha256"], f"raw {k} file is not the pinned pull"

    curves = dc.load_raw_curves(REPO / RAW)
    papers = dc.load_raw_papers(REPO / RAW)
    res = {"raw_curves_rows": len(curves), "raw_papers_rows": len(papers)}
    target_curves = curves[curves["prop_x"].isin(dc.TEMP_PROP_X) & curves["prop_y"].isin(dc.TARGET_PROP_Y)]
    res["points_all_curves"] = count_points(curves)
    res["points_target_curves_before_range_filter"] = count_points(target_curves)
    res["target_curve_rows"] = len(target_curves)

    steps = {}
    df = dc.step1_extract_and_filter_properties(curves)
    steps[1] = len(df)
    df = dc.step2_integrate_and_consolidate(df)
    steps[2] = len(df)
    df = dc.step3_filter_temperature(df)
    steps[3] = len(df)
    df = dc.step4_pivot_wide(df)
    steps[4] = len(df)
    df = dc.step5_clean_formulas(df)
    steps[5] = len(df)
    zt_before6 = df["zT"].dropna()
    df = dc.step6_zt_self_consistency(df)
    steps[6] = len(df)
    zt_after6 = df["zT"].dropna()
    df = dc.step7_remove_dft(df, papers)
    steps[7] = len(df)
    df = dc.step8_multi_source_consistency(df)
    steps[8] = len(df)
    df = dc.step9_mad_outlier_filter(df)
    steps[9] = len(df)
    df = dc.step10_min_temperature_coverage(df)
    steps[10] = len(df)
    nn_before = df[PROPS].notna().sum()
    df11 = dc.step11_smoothness_filter(df)
    steps[11] = len(df11)

    committed = {int(s["step"].split("_")[0]): s["rows"] for s in pav._json(FUNNEL)["funnel"]}
    assert steps == committed, {k: (steps[k], committed[k]) for k in steps if steps[k] != committed[k]}
    cleaned_rows = sum(1 for _ in open(REPO / CLEANED, encoding="utf-8")) - 1
    assert len(df11) == cleaned_rows, (len(df11), cleaned_rows)

    res["rows_after_step"] = steps
    res["zT_before_step6"] = {"n": int(zt_before6.size), "mean": float(zt_before6.mean())}
    res["zT_after_step6"] = {"n": int(zt_after6.size), "mean": float(zt_after6.mean())}
    zt_final = df11["zT"].dropna()
    res["zT_final"] = {"n": int(zt_final.size), "mean": float(zt_final.mean()), "n_above_3": int((zt_final > 3.0).sum())}
    nn_after = df11[PROPS].notna().sum()
    res["step11_spikes_set_to_nan"] = {p: int(nn_before[p] - nn_after[p]) for p in PROPS}
    res["step11_spikes_set_to_nan_total"] = int(sum(res["step11_spikes_set_to_nan"].values()))
    res["step11_rows_removed"] = steps[10] - steps[11]

    # candidate extra filters, evaluated on the final rows: how many rows would each remove?
    def issue_year(s):
        """Publication year from the Crossref-style 'issued' JSON, NaN if it is absent."""
        try:
            parts = json.loads(s)
            parts = parts.get("date_parts") or parts.get("date-parts")
            return float(parts[0][0])
        except (TypeError, ValueError, KeyError, IndexError, AttributeError):
            return np.nan

    years = papers.drop_duplicates("DOI").set_index("DOI")["issued"].map(issue_year)
    row_year = df11["DOI"].map(years)
    res["extra_filters_on_final_rows"] = {
        "published_before_2005": int((row_year < 2005).sum()),
        "year_unknown": int(row_year.isna().sum()),
        "fewer_than_2_properties": int((df11[PROPS].notna().sum(axis=1) < 2).sum()),
        "final_rows": len(df11),
    }
    out = rr.new_run_dir("na5")
    rr.write_json(out / "run_config.json", prov)
    rr.write_json(out / "results.json", res)
    print(f"wrote {out}")
    keys = ("points_all_curves", "points_target_curves_before_range_filter", "zT_before_step6", "zT_after_step6", "zT_final",
            "step11_spikes_set_to_nan_total", "extra_filters_on_final_rows")
    print(json.dumps({k: res[k] for k in keys}, indent=1))


if __name__ == "__main__":
    main()
