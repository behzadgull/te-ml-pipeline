"""
Facts for the Paper B preview manuscript, read only from committed artifacts and checked before use.

Used by make_figures_b.py, make_tables_b.py and check_paper_b.py, so that every figure, table and
number in paper_b/paper/paper.md traces to one place. No model results exist yet and none are read.

Artifacts (all under paper_b/):
  reports/family_labels/20260926T181709/   final family labels (v2.2, produced from a clean commit)
  reports/super_family_qualification/20260926T181743/units.csv   super-family and standalone units
  config/paper_b.yaml                      a priori family-size threshold
  config/super_families.yaml               pre-registered super-family definitions
  results/splits/20260929T113055/run_config.json   folds and repeats of the committed splits

Provenance hashes recorded in the run configurations were computed on Windows working copies (CRLF) on
2026-09-26 and 2026-09-29, before .gitattributes forced LF (2026-10-01). A file therefore passes if its LF
form or its CRLF form matches the recorded hash; which one matched is returned in `hash_forms`.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PB = ROOT / "paper_b"
LABELS_RUN = PB / "reports" / "family_labels" / "20260926T181709"
SUPER_RUN = PB / "reports" / "super_family_qualification" / "20260926T181743"
SPLITS_RUN = PB / "results" / "splits" / "20260929T113055"
THRESHOLD_YAML = PB / "config" / "paper_b.yaml"
SUPER_YAML = PB / "config" / "super_families.yaml"
FAMILIES_YAML = PB / "config" / "families.yaml"

TARGETS = ("S", "sigma", "kappa", "zT")
TARGET_SYMBOL = {"S": "S", "sigma": "σ", "kappa": "κ", "zT": "zT"}
NEVER_HELD_OUT = ("unassignable", "other_oxide")

# Display names. Plain text for the manuscript; MATH for matplotlib (mathtext subscripts). Keys are asserted to
# equal the families in family_summary.csv, so a rule added or renamed upstream fails here, not silently.
FAMILY_NAMES = {
    "iv_vi_rocksalt": ("IV–VI rock salt", "IV–VI rock salt"),
    "skutterudite": ("Skutterudite", "Skutterudite"),
    "half_heusler": ("Half-Heusler", "Half-Heusler"),
    "cobaltite": ("Cobaltite", "Cobaltite"),
    "tetradymite": ("Tetradymite", "Tetradymite"),
    "mg2x": ("Mg₂X", "Mg$_2$X"),
    "manganite": ("Manganite", "Manganite"),
    "perovskite_titanate": ("Perovskite titanate", "Perovskite titanate"),
    "gete_type": ("GeTe-type", "GeTe-type"),
    "tm_silicide": ("Transition-metal silicide", "TM silicide"),
    "cu2x_ag2x_liquid_like": ("Cu₂X/Ag₂X liquid-like", "Cu$_2$X/Ag$_2$X liquid-like"),
    "bicuseo": ("BiCuSeO-type", "BiCuSeO-type"),
    "zinc_antimonide": ("Zinc antimonide", "Zinc antimonide"),
    "diamond_like_cu": ("Diamond-like Cu chalcogenide", "Diamond-like Cu"),
    "clathrate_type_i": ("Type-I clathrate", "Type-I clathrate"),
    "zintl_122": ("Zintl 1-2-2", "Zintl 1-2-2"),
    "full_heusler": ("Full-Heusler", "Full-Heusler"),
    "zno_based": ("ZnO-based", "ZnO-based"),
    "snse_type": ("SnSe-type", "SnSe-type"),
    "sige": ("SiGe", "SiGe"),
    "mg3x2_zintl": ("Mg₃X₂ Zintl", "Mg$_3$X$_2$ Zintl"),
    "i_v_vi2": ("I–V–VI₂", "I–V–VI$_2$"),
    "zintl_14_1_11": ("Zintl 14-1-11", "Zintl 14-1-11"),
    "in2o3_based": ("In₂O₃-based", "In$_2$O$_3$-based"),
    "gete_sb2te3_pseudobinary": ("GeTe–Sb₂Te₃ pseudobinary", "GeTe–Sb$_2$Te$_3$ pseudobinary"),
    "tetrahedrite": ("Tetrahedrite", "Tetrahedrite"),
    "mgagsb": ("MgAgSb", "MgAgSb"),
    "other_oxide": ("Other oxide", "Other oxide"),
    "unassignable": ("Unassignable", "Unassignable"),
}
SUPER_NAMES = {"iv_vi": ("IV–VI", "IV–VI"), "oxides": ("Oxides", "Oxides"), "zintl": ("Zintl", "Zintl")}

# One-line rule summaries for Table 1, in plain words. Each is checked against rules_plain_language.txt (the rule
# tag must be present and its elements, site counts and tolerance must appear in the matching rule text).
RULE_WORDS = {
    "tetradymite": "Bi/Sb on one site and Te/Se/S on another in 2:3 ratio",
    "snse_type": "Sn-majority cation with Se or S, 1:1",
    "gete_type": "Ge-majority cation with Te, 1:1; Sb/Bi substitution below one third of the cations",
    "iv_vi_rocksalt": "Pb/Sn/Ge with Te/Se/S, 1:1, after the SnSe-type and GeTe-type rules",
    "skutterudite": "Co/Rh/Ir/Fe/Ni with Sb/As/P in 1:3 ratio; optional filler up to 8% of atoms",
    "half_heusler": "XYZ in 1:1:1 ratio: early transition or rare-earth metal, late transition metal, p-block element",
    "mg2x": "Mg with Si/Ge/Sn in 2:1 ratio",
    "mg3x2_zintl": "Mg with Sb/Bi in 3:2 ratio",
    "cu2x_ag2x_liquid_like": "Cu/Ag with Se/S/Te in 2:1 ratio; tolerance 20%",
    "zintl_122": "Ca/Sr/Ba/Eu/Yb, Mg/Zn/Cd/Mn and Sb/Bi/As in 1:2:2 ratio",
    "zintl_14_1_11": "Yb/Ca/Sr/Eu with Sb/Bi/As in 14:11 ratio; optional Mn/Al/Zn/Cd/Ga/In/Mg up to 10% of atoms",
    "clathrate_type_i": "Ba/Sr/Eu/Cs/K/Na, Ga/Al/In/Zn/Cd/Cu/Ni/Ag/Au and Ge/Si/Sn in 8:16:30 ratio",
    "tetrahedrite": "Cu-site metals, Sb/As/Bi and S/Se in 12:4:13 ratio",
    "bicuseo": "Bi-site cation, Cu/Ag, Se/Te/S and O in 1:1:1:1 ratio",
    "zinc_antimonide": "Zn with Sb in 4:3 or 1:1 ratio",
    "tm_silicide": "Mn/Cr/Fe/Ru/Re silicide with 1.7 to 2.353 Si per metal atom",
    "sige": "Si and Ge only",
    "i_v_vi2": "Ag/Cu, Sb/Bi and Te/Se/S in 1:1:2 ratio",
    "gete_sb2te3_pseudobinary": "Ge/Pb/Sn with Sb/Bi and Te, charge-balanced; Sb/Bi at least one third of the cations",
    "diamond_like_cu": "Cu with In/Ga and a chalcogen in 1:1:2, Cu with Sb and a chalcogen in 3:1:4, or Cu with Sn and a chalcogen in 2:1:3",
    "full_heusler": "X₂YZ in 2:1:1 ratio: late transition metal, early transition metal, p-block element",
    "mgagsb": "Mg, Ag and Sb in 1:1:1 ratio",
    "cobaltite": "Oxide (O at least 20% of atoms) with Co and at least one of Ca/Na/Bi/Sr; Co outweighs Mn",
    "perovskite_titanate": "Oxide with Ti and at least one of Sr/Ca/Ba",
    "manganite": "Oxide with Mn and at least one of Ca/La/Sr; Mn outweighs Co",
    "zno_based": "Oxide in which Zn is the largest non-oxygen element",
    "in2o3_based": "Oxide in which In is the largest non-oxygen element",
    "other_oxide": "Any other oxide (O at least 20% of atoms)",
}
# Rules in the order the first-match-wins list tries them (rules_plain_language.txt numbers them 1..28).
RULE_ORDER = (
    "tetradymite snse_type gete_type iv_vi_rocksalt skutterudite half_heusler mg2x mg3x2_zintl cu2x_ag2x_liquid_like "
    "zintl_122 zintl_14_1_11 clathrate_type_i tetrahedrite bicuseo zinc_antimonide tm_silicide sige i_v_vi2 "
    "gete_sb2te3_pseudobinary diamond_like_cu full_heusler mgagsb cobaltite perovskite_titanate manganite zno_based "
    "in2o3_based other_oxide"
).split()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def hash_form_matching(path, expected):
    """'lf' or 'crlf' if that form of the file's bytes hashes to `expected`; AssertionError otherwise."""
    raw = Path(path).read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    for form, data in (("lf", lf), ("crlf", lf.replace(b"\n", b"\r\n"))):
        if sha256_bytes(data) == expected:
            return form
    raise AssertionError(f"{path}: neither its LF nor its CRLF form matches the recorded SHA256 {expected}")


def format_host(host):
    """Host formula as pandoc markdown with subscripts: 'Bi2Te2.7Se0.3' -> 'Bi~2~Te~2.7~Se~0.3~'; amounts of exactly 1 are dropped."""
    import re

    s = re.sub(r"(?<=[A-Za-z\)])1(?![\d.])", "", host)
    return re.sub(r"(?<=[A-Za-z\)])(\d+(?:\.\d+)?)", r"~\1~", s)


def parse_rules_text():
    """{tag: rule text} from rules_plain_language.txt, in file order (lines 'N. [tag] description')."""
    import re

    rules = {}
    order = []
    for block in (LABELS_RUN / "rules_plain_language.txt").read_text(encoding="utf-8").split("\n\n"):
        m = re.match(r"(\d+)\. \[(\w+)\] (.*)", block.strip(), flags=re.S)
        if m:
            assert int(m.group(1)) == len(order) + 1, (m.group(1), len(order))
            order.append(m.group(2))
            rules[m.group(2)] = m.group(3)
    return rules, order


def check_rule_words(rules):
    """
    Every element list ('Co/Rh/Ir'), percentage and site ratio in a RULE_WORDS entry must be present in
    that rule's own text from the final run. Ratios are compared with the site `count`s of each variant
    (variants are separated by '); or (' in the rule text); rules without counts are skipped for ratios.
    """
    import re

    for tag, words in RULE_WORDS.items():
        text = rules[tag]
        for lst in re.findall(r"\b(?:[A-Z][a-z]?/)+[A-Z][a-z]?\b", words):
            assert lst in text, (tag, lst)
        for pct in re.findall(r"\d+(?:\.\d+)?%", words):
            assert pct in text, (tag, pct)
        for num in re.findall(r"\d+\.\d+", words):
            assert num in text, (tag, num)
        variants = re.split(r"\); or \(", text)
        ratios = {":".join(re.findall(r"\(count (\d+)", v)) for v in variants}
        ratios.discard("")
        for ratio in re.findall(r"\b\d+(?::\d+)+\b", words):
            assert ratio in ratios, (tag, ratio, ratios)


def load_facts():
    """Read, validate and summarise the artifacts. Returns a dict; raises AssertionError on any inconsistency."""
    rules_text, rules_order = parse_rules_text()
    assert rules_order == list(RULE_ORDER), "rules_plain_language.txt order differs from RULE_ORDER"
    check_rule_words(rules_text)

    labels_cfg = json.loads((LABELS_RUN / "run_config.json").read_text(encoding="utf-8"))
    super_cfg = json.loads((SUPER_RUN / "run_config.json").read_text(encoding="utf-8"))
    splits_cfg = json.loads((SPLITS_RUN / "run_config.json").read_text(encoding="utf-8"))

    # ---- provenance: every artifact matches the hash its own downstream run recorded
    hash_forms = {
        "host_family_labels.csv": hash_form_matching(LABELS_RUN / "host_family_labels.csv", splits_cfg["labels_sha256"]),
        "units.csv": hash_form_matching(SUPER_RUN / "units.csv", splits_cfg["super_units_sha256"]),
        "super_families.yaml": hash_form_matching(SUPER_YAML, splits_cfg["super_families_sha256"]),
        "families.yaml": hash_form_matching(FAMILIES_YAML, labels_cfg["rules_sha256"]),
        "labels run_config.json": hash_form_matching(LABELS_RUN / "run_config.json", super_cfg["labels_run_config_sha256"]),
    }
    assert super_cfg["labels_sha256"] == splits_cfg["labels_sha256"], "super-family and splits runs disagree on the labels hash"
    assert labels_cfg["tree_clean"] is True and labels_cfg["version_label"] == "v2.2", labels_cfg["version_label"]

    thr = yaml.safe_load(THRESHOLD_YAML.read_text(encoding="utf-8"))["paper_b"]["min_family_sample_size"]
    assert thr == labels_cfg["threshold"] == super_cfg["threshold"], (thr, labels_cfg["threshold"])
    min_clusters, min_rows = thr["min_clusters"], thr["min_rows_per_target"]

    supers_cfg = yaml.safe_load(SUPER_YAML.read_text(encoding="utf-8"))
    super_members = supers_cfg["super_families"]
    assert tuple(supers_cfg["never_held_out"]) == NEVER_HELD_OUT, supers_cfg["never_held_out"]

    summary = pd.read_csv(LABELS_RUN / "family_summary.csv", keep_default_na=False)
    hosts = pd.read_csv(LABELS_RUN / "host_family_labels.csv", keep_default_na=False)
    audit = pd.read_csv(LABELS_RUN / "audit_sample.csv", keep_default_na=False)
    units = pd.read_csv(SUPER_RUN / "units.csv", keep_default_na=False)

    assert set(summary.family) == set(FAMILY_NAMES), set(summary.family) ^ set(FAMILY_NAMES)
    assert set(RULE_ORDER) == set(FAMILY_NAMES) - {"unassignable"} and len(RULE_ORDER) == 28, len(RULE_ORDER)
    assert set(RULE_WORDS) == set(RULE_ORDER), set(RULE_WORDS) ^ set(RULE_ORDER)

    # ---- the label table is internally consistent with the host-level file
    n_hosts, n_rows = len(hosts), int(hosts.rows.sum())
    assert n_hosts == labels_cfg["n_hosts"] == int(summary.clusters_all.sum()), (n_hosts, labels_cfg["n_hosts"])
    assert n_rows == labels_cfg["dataset"]["n_rows"] == int(summary.rows_all.sum()), n_rows
    by_family = hosts.groupby("family").agg(clusters=("host", "size"), rows=("rows", "sum"))
    for _, r in summary.iterrows():
        assert int(by_family.loc[r.family, "clusters"]) == int(r.clusters_all), r.family
        assert int(by_family.loc[r.family, "rows"]) == int(r.rows_all), r.family
        for t in TARGETS:
            sub = hosts[hosts.family == r.family]
            assert int(sub[t].sum()) == int(r[f"rows_{t}"]), (r.family, t)
            assert int((sub[t] > 0).sum()) == int(r[f"clusters_{t}"]), (r.family, t)

    # ---- the qualifies flags are what the pre-registered threshold gives (>= on both counts)
    fam = summary[~summary.family.isin(NEVER_HELD_OUT)].copy()
    assert len(fam) == 27, len(fam)
    for t in TARGETS:
        recomputed = (fam[f"clusters_{t}"] >= min_clusters) & (fam[f"rows_{t}"] >= min_rows)
        flags = fam[f"qualifies_{t}"].astype(str).eq("True")
        assert (recomputed == flags).all(), t
    qualifying = {t: int(fam[f"qualifies_{t}"].astype(str).eq("True").sum()) for t in TARGETS}
    assert sum(qualifying.values()) == splits_cfg["n_family_pairs"], (qualifying, splits_cfg["n_family_pairs"])

    def failing(t):
        rows_ok = fam[f"rows_{t}"] >= min_rows
        cl_ok = fam[f"clusters_{t}"] >= min_clusters
        return {
            "rows_only": sorted(fam.family[~rows_ok & cl_ok]),
            "clusters_only": sorted(fam.family[rows_ok & ~cl_ok]),
            "both": sorted(fam.family[~rows_ok & ~cl_ok]),
        }

    fail = {t: failing(t) for t in TARGETS}

    unas = summary[summary.family == "unassignable"].iloc[0]
    oth = summary[summary.family == "other_oxide"].iloc[0]
    target_rows = {t: int(summary[f"rows_{t}"].sum()) for t in TARGETS}
    unas_row_share_target = {t: int(unas[f"rows_{t}"]) / target_rows[t] for t in TARGETS}

    # ---- super-families: units.csv agrees with the definitions and with the family table
    supers = {}
    for name, members in super_members.items():
        u = units[(units.unit == name) & (units.kind == "super_family")].iloc[0]
        assert u.members.split(";") == members, (name, u.members)
        sub = summary[summary.family.isin(members)]
        assert int(u.clusters_all) == int(sub.clusters_all.sum()), name
        for t in TARGETS:
            assert int(u[f"rows_{t}"]) == int(sub[f"rows_{t}"].sum()), (name, t)
            assert int(u[f"clusters_{t}"]) == int(sub[f"clusters_{t}"].sum()), (name, t)
            assert (int(u[f"clusters_{t}"]) >= min_clusters and int(u[f"rows_{t}"]) >= min_rows) == (u[f"qualifies_{t}"] in (True, "True")), (name, t)
            alone = fam[fam.family.isin(members) & ~fam[f"qualifies_{t}"].astype(str).eq("True")].family
            assert sorted(alone) == sorted(x for x in str(u[f"members_below_threshold_alone_{t}"]).split(";") if x), (name, t)
        supers[name] = {
            "members": members, "clusters_all": int(u.clusters_all),
            "rows": {t: int(u[f"rows_{t}"]) for t in TARGETS}, "clusters": {t: int(u[f"clusters_{t}"]) for t in TARGETS},
            "qualifies": {t: u[f"qualifies_{t}"] in (True, "True") for t in TARGETS},
            "below_alone": {t: [x for x in str(u[f"members_below_threshold_alone_{t}"]).split(";") if x] for t in TARGETS},
        }
    in_super = {m for ms in super_members.values() for m in ms}
    standalone = sorted(set(fam.family) - in_super)
    stand_units = units[units.kind == "family"]
    assert sorted(stand_units.unit) == standalone, "standalone units in units.csv are not the families outside every super-family"
    standalone_qualifying = {t: int(stand_units[f"qualifies_{t}"].astype(str).eq("True").sum()) for t in TARGETS}
    assert sum(supers[s]["qualifies"][t] for s in supers for t in TARGETS) == splits_cfg["n_super_family_pairs"]
    assert sum(standalone_qualifying.values()) == splits_cfg["n_standalone_pairs"], (standalone_qualifying, splits_cfg)

    # ---- audit sample: up to 15 hosts per family (all of them if fewer), 30 unassignable
    audit_n = audit.groupby("assigned_family").size().to_dict()
    assert set(audit_n) == set(summary.family)
    for f, n in audit_n.items():
        expect = int(summary.loc[summary.family == f, "clusters_all"].iloc[0])
        if f == "unassignable":
            assert n == 30, n
        else:
            assert n == min(15, expect), (f, n, expect)
    assert (audit.set_index("host").assigned_family.reindex(audit.host) == audit.assigned_family.values).all()

    return {
        "min_clusters": min_clusters, "min_rows": min_rows, "n_folds": splits_cfg["n_folds"], "repeats": splits_cfg["repeats"],
        "n_hosts": n_hosts, "n_rows": n_rows, "target_rows": target_rows,
        "n_named_families": len(fam), "n_rules": len(RULE_ORDER),
        "unassignable": {
            "clusters": int(unas.clusters_all), "rows": int(unas.rows_all),
            "cluster_share": int(unas.clusters_all) / n_hosts, "row_share": int(unas.rows_all) / n_rows,
            "row_share_by_target": unas_row_share_target,
        },
        "other_oxide": {"clusters": int(oth.clusters_all), "rows": int(oth.rows_all)},
        "assigned_to_named_family": {"clusters": n_hosts - int(unas.clusters_all) - int(oth.clusters_all),
                                     "rows": n_rows - int(unas.rows_all) - int(oth.rows_all)},
        "qualifying": qualifying, "failing": fail,
        "summary": summary, "families": fam, "hosts": hosts, "audit": audit, "audit_n": audit_n, "units": units,
        "supers": supers, "super_members": super_members, "standalone": standalone, "standalone_qualifying": standalone_qualifying,
        "hash_forms": hash_forms, "labels_cfg": labels_cfg, "splits_cfg": splits_cfg,
        "labels_git_head": labels_cfg["git_head"], "n_hosts_moved_v2_1_to_v2_2": labels_cfg["n_hosts_moved_from_prev"],
    }


if __name__ == "__main__":
    f = load_facts()
    print({k: f[k] for k in ("n_hosts", "n_rows", "qualifying", "n_named_families", "hash_forms")})
