"""
Super-family qualification per target (methodology doc, sections 7.3 and 7.4).

Reads a family-labelling run's host_family_labels.csv, the a priori threshold
(paper_b/config/paper_b.yaml) and the super-family definitions
(paper_b/config/super_families.yaml). The held-out units are the super-families
plus every other named family that is not in never_held_out. A unit qualifies
for a target if it has at least min_clusters clusters holding that target and at
least min_rows_per_target rows for it, counted over the union of its members;
a member below the threshold on its own still counts toward the union.

Writes paper_b/reports/super_family_qualification/<UTC>/units.csv and
run_config.json (labels file SHA256, git HEAD, whether the tree was clean
ignoring the labels run folder itself, threshold, definitions file SHA256).
Run from the repository root:

    python paper_b/scripts/super_family_qualification.py --run-dir paper_b/reports/family_labels/<UTC>

Imports nothing from top-level src/.
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

THRESHOLD_PATH = Path("paper_b/config/paper_b.yaml")
SUPER_PATH = Path("paper_b/config/super_families.yaml")
OUT_DIR = Path("paper_b/reports/super_family_qualification")
TARGETS = ("S", "sigma", "kappa", "zT")


def sha256_file(path):
    """SHA256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit_row(name, kind, members, labelled, threshold):
    """Counts and qualification for one unit: clusters and rows per target over its members."""
    group = labelled[labelled["family"].isin(members)]
    row = {"unit": name, "kind": kind, "members": ";".join(members), "clusters_all": len(group)}
    for target in TARGETS:
        row[f"clusters_{target}"] = int((group[target] > 0).sum())
        row[f"rows_{target}"] = int(group[target].sum())
        row[f"qualifies_{target}"] = bool(
            row[f"clusters_{target}"] >= threshold["min_clusters"]
            and row[f"rows_{target}"] >= threshold["min_rows_per_target"]
        )
    return row


def build_units(labelled, definitions, threshold):
    """
    One row per held-out unit: each super-family, then every named family that is
    in no super-family and not in never_held_out. For super-families, also lists
    (per target) the members that would not qualify on their own.
    """
    families = set(labelled["family"])
    supers = definitions["super_families"]
    never = set(definitions["never_held_out"])
    members = [m for group in supers.values() for m in group]
    missing = set(members) - families
    if missing:
        raise ValueError(f"super-family members not in the labels: {sorted(missing)}")
    if len(members) != len(set(members)) or set(members) & never:
        raise ValueError("a family is in two super-families or is both a member and never_held_out")
    rows = []
    for name, group in supers.items():
        row = unit_row(name, "super_family", group, labelled, threshold)
        for target in TARGETS:
            alone = [unit_row(m, "family", [m], labelled, threshold)[f"qualifies_{target}"] for m in group]
            row[f"members_below_threshold_alone_{target}"] = ";".join(m for m, ok in zip(group, alone) if not ok)
        rows.append(row)
    for family in sorted(families - set(members) - never):
        rows.append(unit_row(family, "family", [family], labelled, threshold))
    return pd.DataFrame(rows)


def git_state(exclude_dir):
    """Return (HEAD, clean ignoring exclude_dir, dirty files ignoring exclude_dir)."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", ".", f":(exclude){exclude_dir.as_posix()}"],
        capture_output=True, text=True, check=True,
    ).stdout
    dirty = [line[3:] for line in status.splitlines()]
    return head, not dirty, dirty


def main(argv=None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, help="a family-labelling run folder holding host_family_labels.csv")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    labels_path = run_dir / "host_family_labels.csv"
    labelled = pd.read_csv(labels_path, keep_default_na=False)
    threshold = yaml.safe_load(THRESHOLD_PATH.read_text(encoding="utf-8"))["paper_b"]["min_family_sample_size"]
    definitions = yaml.safe_load(SUPER_PATH.read_text(encoding="utf-8"))
    units = build_units(labelled, definitions, threshold)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = OUT_DIR / stamp
    out_dir.mkdir(parents=True)
    head, clean, dirty = git_state(run_dir)
    config = {
        "labels_file": str(labels_path),
        "labels_sha256": sha256_file(labels_path),
        "labels_run_config": str(run_dir / "run_config.json"),
        "labels_run_config_sha256": sha256_file(run_dir / "run_config.json"),
        "definitions_file": str(SUPER_PATH),
        "definitions_sha256": sha256_file(SUPER_PATH),
        "threshold": threshold,
        "git_head": head,
        "tree_clean_excluding_labels_run_dir": clean,
        "dirty_files_excluding_labels_run_dir": dirty,
        "utc_stamp": stamp,
    }
    units.to_csv(out_dir / "units.csv", index=False)
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    print(f"Wrote {out_dir}\n")
    for kind, title in (("super_family", "Super-families"), ("family", "Standalone families")):
        part = units[units["kind"] == kind]
        columns = ["unit", "clusters_all"] + [f"{p}_{t}" for t in TARGETS for p in ("clusters", "rows")]
        print(f"{title}:")
        print(part[columns].to_string(index=False))
    print("\nQualification (unit: targets passed):")
    for _, row in units.iterrows():
        passed = [t for t in TARGETS if row[f"qualifies_{t}"]]
        print(f"  {row['unit']} [{row['kind']}]: {', '.join(passed) or 'none'}")
    for _, row in units[units["kind"] == "super_family"].iterrows():
        for target in TARGETS:
            below = row[f"members_below_threshold_alone_{target}"]
            if below:
                print(f"  {row['unit']} / {target}: members below the threshold alone, counted in the union: {below}")


if __name__ == "__main__":
    main()
