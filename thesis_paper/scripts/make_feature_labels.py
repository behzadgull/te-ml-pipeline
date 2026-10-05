"""
Builds thesis_paper/docs/feature_labels.csv: a readable label for each of the 397 model features, used by every figure, table and sentence that names a feature (feature_labels.py reads the CSV;
no label is typed in a plot or in the text).

A feature is a statistic of an ELEMENTAL property of the formula (or the temperature), never a measured property of the material. Label format: "Elemental <property>, <statistic> (<set>)", set MAGPIE or CBFV;
the temperature feature is "Temperature".
  Sets and column names: MAGPIE columns "MagpieData <statistic> <Property>" (matminer 0.9.2 Magpie featuriser, Ward et al. 2016), CBFV columns "CBFV_<statistic>_<property>" (CBFV 1.1.0, the Oliynyk element table).
  Statistics (verified in the code of the packages): MAGPIE mean = atomic-fraction-weighted mean, avg_dev = atomic-fraction-weighted mean absolute deviation about that mean, minimum, maximum, range,
  mode = the value of the most abundant element. CBFV avg = weighted mean, dev = weighted MEAN ABSOLUTE deviation about the weighted mean (CBFV composition.py: sum of f_i * |x_i - mean|; it is not a standard
  deviation), range = max - min (numpy ptp), max, min, mode = the value of the most abundant element (the smallest among equally abundant ones).
Typos of the source column names are fixed in the label only ("electonegativity" -> "electronegativity", "Metalliod" -> "metalloid", "Rockow" -> "Rochow"); the column names themselves are unchanged. Units of the
elemental properties are kept in the `unit` column of the table and not in the labels.

Usage (from the repository root):  python thesis_paper/scripts/make_feature_labels.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
COLUMNS_FROM = "thesis_paper/results/na3_a/20261004T201639/results.json"  # the 397 feature columns of every final model, in order
OUT = REPO / "thesis_paper" / "docs" / "feature_labels.csv"

MAGPIE_STAT = {"mean": "mean", "avg_dev": "mean abs. deviation", "minimum": "minimum", "maximum": "maximum", "range": "range", "mode": "mode"}
CBFV_STAT = {"avg": "mean", "dev": "mean abs. deviation", "min": "minimum", "max": "maximum", "range": "range", "mode": "mode"}
MAGPIE_PROP = {  # token: (label, unit)
    "Number": ("atomic number", ""), "MendeleevNumber": ("Mendeleev number", ""), "AtomicWeight": ("atomic weight", "u"), "MeltingT": ("melting temperature", "K"),
    "Column": ("periodic-table column", ""), "Row": ("periodic-table row", ""), "CovalentRadius": ("covalent radius", "pm"), "Electronegativity": ("electronegativity", ""),
    "NsValence": ("s valence electrons", ""), "NpValence": ("p valence electrons", ""), "NdValence": ("d valence electrons", ""), "NfValence": ("f valence electrons", ""),
    "NValence": ("valence electrons (total)", ""), "NsUnfilled": ("unfilled s valence electrons", ""), "NpUnfilled": ("unfilled p valence electrons", ""),
    "NdUnfilled": ("unfilled d valence electrons", ""), "NfUnfilled": ("unfilled f valence electrons", ""), "NUnfilled": ("unfilled valence electrons (total)", ""),
    "GSvolume_pa": ("DFT ground-state volume per atom", "A^3"), "GSbandgap": ("DFT ground-state band gap", "eV"), "GSmagmom": ("DFT ground-state magnetic moment", "muB"),
    "SpaceGroupNumber": ("space-group number", ""),
}
CBFV_PROP = {
    "1st_ionization_potential_(kJ/mol)": ("1st ionization potential", "kJ/mol"), "Allred-Rockow_electronegativity": ("Allred-Rochow electronegativity", ""),
    "Atomic_Number": ("atomic number", ""), "Atomic_Radius": ("atomic radius", ""), "Atomic_Weight": ("atomic weight", "u"), "Boiling_Point_(K)": ("boiling point", "K"),
    "Cohesive_energy": ("cohesive energy", ""), "Covalent_Radius": ("covalent radius", ""), "Density_(g/mL)": ("density", "g/mL"), "Gordy_electonegativity": ("Gordy electronegativity", ""),
    "MB_electonegativity": ("MB electronegativity", ""), "Melting_point_(K)": ("melting point", "K"), "Mendeleev_Number": ("Mendeleev number", ""), "Metal": ("metal flag", ""),
    "Metalliod": ("metalloid flag", ""), "Miracle_Radius_[pm]": ("Miracle radius", "pm"), "Mulliken_EN": ("Mulliken electronegativity", ""), "Nonmetal": ("nonmetal flag", ""),
    "Number_of_unfilled_d_valence_electrons": ("unfilled d valence electrons", ""), "Number_of_unfilled_f_valence_electrons": ("unfilled f valence electrons", ""),
    "Number_of_unfilled_p_valence_electrons": ("unfilled p valence electrons", ""), "Number_of_unfilled_s_valence_electrons": ("unfilled s valence electrons", ""),
    "Pauling_Electronegativity": ("Pauling electronegativity", ""), "Period": ("period", ""), "Zunger_radii_sum": ("Zunger radii sum", ""), "crystal_radius": ("crystal radius", ""),
    "families": ("family code", ""), "gilmor_number_of_valence_electron": ("Gilmor number of valence electrons", ""), "group": ("group", ""),
    "heat_atomization(kJ/mol)": ("heat of atomization", "kJ/mol"), "heat_of_fusion_(kJ/mol)_": ("heat of fusion", "kJ/mol"), "heat_of_vaporization_(kJ/mol)_": ("heat of vaporization", "kJ/mol"),
    "ionic_radius": ("ionic radius", ""), "l_quantum_number": ("orbital quantum number l", ""), "metallic_valence": ("metallic valence", ""),
    "number_of_valence_electrons": ("number of valence electrons", ""), "outer_shell_electrons": ("outer-shell electrons", ""), "polarizability(A^3)": ("polarizability", "A^3"),
    "specific_heat_(J/g_K)_": ("specific heat", "J/(g K)"), "thermal_conductivity_(W/(m_K))_": ("thermal conductivity", "W/(m K)"), "valence_d": ("d valence electrons", ""),
    "valence_f": ("f valence electrons", ""), "valence_p": ("p valence electrons", ""), "valence_s": ("s valence electrons", ""),
}


def label_of(col):
    """(set, property token, statistic token, property label, unit, statistic label, label) of one feature column."""
    if col == "temperature_bin":
        return ("temperature", "temperature_bin", "", "temperature", "K", "", "Temperature")
    if col.startswith("MagpieData "):
        _, stat, prop = col.split(" ", 2)
        pl, unit = MAGPIE_PROP[prop]
        sl = MAGPIE_STAT[stat]
        return ("MAGPIE", prop, stat, pl, unit, sl, f"Elemental {pl}, {sl} (MAGPIE)")
    if col.startswith("CBFV_"):
        _, stat, prop = col.split("_", 2)
        pl, unit = CBFV_PROP[prop]
        sl = CBFV_STAT[stat]
        return ("CBFV", prop, stat, pl, unit, sl, f"Elemental {pl}, {sl} (CBFV)")
    raise KeyError(col)


def main():
    """Entry point."""
    cols = json.loads((REPO / COLUMNS_FROM).read_text(encoding="utf-8"))["feature_columns"]
    assert len(cols) == 397 and len(set(cols)) == 397
    rows = [{"column": c, **dict(zip(("feature_set", "property_token", "statistic_token", "property_label", "unit", "statistic_label", "label"), label_of(c)))} for c in cols]
    d = pd.DataFrame(rows)
    assert d["label"].is_unique, "two features share a label"
    d.to_csv(OUT, index=False, lineterminator="\n", encoding="utf-8")
    print("wrote", OUT, len(d), "labels; longest:", d["label"].str.len().max())


if __name__ == "__main__":
    sys.exit(main())
