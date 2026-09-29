"""
Pre-generated splits for the Paper B modelling harness (methodology doc,
section 8). No model is trained; this only decides which rows go where.

For every qualifying (target, unit) pair -- family-level, super-family-level,
and the super-family analysis's standalone-family C3 reruns, using the exact
same pairs and G choices as paper_b/scripts/compute_estimate.py's
`build_pairs` (imported, not reimplemented, so the splits and the compute
estimate never disagree about which pairs exist or which G was chosen) -- and
every repeat r in 1..R (R=3, methodology doc section 8.7):

  * a chemistry-cluster-respecting 5-fold assignment of the unit's own rows
    for that target, via src/nested_cv.py's `randomized_group_kfold` (the
    project's frozen, bug-fixed fold assignor, not a fresh implementation);
  * for each fold, the C2 control's removed row set: a seeded random sample
    of non-unit rows, sized so C2's training set matches C1's (all of the
    unit removed) exactly;
  * the C3 control's removed row set: every row belonging to G (the whole
    family or super-family `build_pairs` chose), the same set for every fold
    and repeat of that pair, so it needs no seed.

Every random draw's seed is a deterministic SHA256-derived 32-bit integer
from (level, target, unit, repeat[, fold, purpose]) and is recorded in that
pair's own JSON sidecar, not left implicit. Three assertions run on every
pair before its files are written: the 5 folds partition the unit's rows
exactly; no chemistry cluster's rows split across folds; C2's resulting
training size equals C1's.

Writes paper_b/results/splits/<UTC>/{family,super_family,super_analysis_standalone}/
<target>/<unit>.npz (+ .json sidecar) and a top-level manifest.json listing
every file's SHA256, plus run_config.json (dataset SHA, labels SHA,
super-units SHA, super_families.yaml SHA, git HEAD, tree_clean, R, N_FOLDS).

Run from the repository root:
    python -m paper_b.scripts.make_splits --csv <snapfix csv> \
        --labels-run paper_b/reports/family_labels/<UTC> \
        --super-units paper_b/reports/super_family_qualification/<UTC>/units.csv
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from paper_b.scripts.compute_estimate import SUPER_PATH, TARGETS, build_pairs
from src.nested_cv import randomized_group_kfold

MANIFEST_PATH = Path("paper_b/SHARED_DEPENDENCIES.md")
OUT_DIR = Path("paper_b/results/splits")
N_FOLDS = 5
R = 3  # methodology doc section 8.7


def sha256_file(path):
    """SHA256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_dataset_identity():
    """(sha256, bytes) of the snapfix CSV from the manifest's data table."""
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0] == "snapfix featurized CSV":
            return cells[2], int(cells[3].replace(",", ""))
    raise ValueError(f"no snapfix data row in {MANIFEST_PATH}")


def load_rows(csv_path, labels_run):
    """
    Load the snapfix CSV (SHA-checked) and join in each host's family label.
    Returns (df with row_id [the row's position in the raw file, 0-based, the
    identifier every split file uses], chemistry_cluster_id, family, and the
    four targets; dataset identity dict).
    """
    expected_sha, expected_bytes = expected_dataset_identity()
    size = csv_path.stat().st_size
    if size != expected_bytes:
        raise ValueError(f"{csv_path}: {size} bytes, manifest says {expected_bytes}")
    sha = sha256_file(csv_path)
    if sha != expected_sha:
        raise ValueError(f"{csv_path}: SHA256 {sha} differs from the manifest {expected_sha}")
    df = pd.read_csv(csv_path, usecols=["chemistry_cluster_id", *TARGETS])
    df["row_id"] = np.arange(len(df))
    labels = pd.read_csv(labels_run / "host_family_labels.csv", keep_default_na=False)[["host", "family"]]
    df = df.merge(labels.rename(columns={"host": "chemistry_cluster_id"}), on="chemistry_cluster_id", how="left")
    missing = df["family"].isna().sum()
    if missing:
        raise ValueError(f"{missing} rows have a chemistry_cluster_id not present in {labels_run}/host_family_labels.csv")
    identity = {"path": str(csv_path), "sha256": sha, "bytes": size, "n_rows": int(len(df))}
    return df, identity


def seed_for(*parts):
    """Deterministic 32-bit seed from a stable string of `parts`."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def members_of(unit_name, supers):
    """The plain family name(s) making up `unit_name`'s rows: its super-family's members, or itself."""
    return supers.get(unit_name, [unit_name])


def make_fold_assignment(df, target, members, level, unit, repeat):
    """
    The unit's own rows for `target` (family in `members`), and a 5-fold,
    chemistry-cluster-respecting assignment via randomized_group_kfold.
    Returns (row_ids, chemistry_cluster_id per row, fold_of, seed).
    """
    sub = df.loc[df["family"].isin(members) & df[target].notna(), ["row_id", "chemistry_cluster_id"]].reset_index(drop=True)
    seed = seed_for(level, target, unit, repeat, "folds")
    rng = np.random.default_rng(seed)
    fold_of = np.full(len(sub), -1, dtype=int)
    for fold_idx, (_, test_pos) in enumerate(randomized_group_kfold(sub["chemistry_cluster_id"].to_numpy(), N_FOLDS, rng)):
        fold_of[test_pos] = fold_idx
    if (fold_of < 0).any():
        raise AssertionError(f"{level}/{target}/{unit} repeat {repeat}: not every row was assigned a fold")
    return sub["row_id"].to_numpy(), sub["chemistry_cluster_id"].to_numpy(), fold_of, seed


def assert_folds_valid(row_ids, cluster_of_row, fold_of, n_family):
    """
    The 5 folds partition `row_ids` exactly (every row assigned to exactly
    one fold, matching `n_family`), and no chemistry cluster's rows land in
    more than one fold. A unit with fewer chemistry clusters than N_FOLDS
    may leave some fold indices empty; that is not itself an error.
    """
    if len(row_ids) != n_family or len(set(row_ids)) != n_family:
        raise AssertionError("fold assignment does not cover the unit's rows exactly once each")
    by_cluster = pd.DataFrame({"cluster": cluster_of_row, "fold": fold_of}).groupby("cluster")["fold"].nunique()
    if (by_cluster > 1).any():
        bad = by_cluster[by_cluster > 1].index.tolist()
        raise AssertionError(f"chemistry cluster(s) split across folds: {bad}")


def make_c2_removed(df, target, members, pool_row_ids, n_family, fold_test_size, level, unit, repeat, fold):
    """A seeded sample of `n_family - fold_test_size` rows from `pool_row_ids` (non-unit rows), and its seed."""
    count = n_family - fold_test_size
    seed = seed_for(level, target, unit, repeat, fold, "c2")
    rng = np.random.default_rng(seed)
    removed = rng.choice(pool_row_ids, size=count, replace=False) if count > 0 else np.array([], dtype=pool_row_ids.dtype)
    return removed, seed, count


def assert_c2_matches_c1(n_total, n_family, fold_test_size, c2_removed_count):
    """C0's training size for this fold, minus the C2 removal, must equal C1's training size (n_total - n_family)."""
    c0_train = n_total - fold_test_size
    c2_train = c0_train - c2_removed_count
    c1_train = n_total - n_family
    if c2_train != c1_train:
        raise AssertionError(f"C2 training size {c2_train} != C1 training size {c1_train}")


def build_one_pair(df, target, unit, members, g, supers, level, n_total, out_dir):
    """Generate, assert and write one pair-record's split file (+ sidecar); returns the two file paths."""
    unit_mask = df["family"].isin(members) & df[target].notna()
    n_family = int(unit_mask.sum())
    pool_row_ids = df.loc[~df["family"].isin(members) & df[target].notna(), "row_id"].to_numpy()

    npz_payload = {}
    sidecar = {"level": level, "target": target, "unit": unit, "members": members, "g": g,
              "g_members": members_of(g, supers), "n_family": n_family, "n_total": int(n_total),
              "n_folds": N_FOLDS, "repeats": R, "fold_seeds": {}, "c2_seeds": {}, "c2_removed_counts": {}}

    g_row_ids = df.loc[df["family"].isin(members_of(g, supers)) & df[target].notna(), "row_id"].to_numpy()
    npz_payload["g_row_ids"] = g_row_ids
    sidecar["n_g"] = int(len(g_row_ids))

    for repeat in range(1, R + 1):
        row_ids, cluster_of_row, fold_of, fold_seed = make_fold_assignment(df, target, members, level, unit, repeat)
        assert_folds_valid(row_ids, cluster_of_row, fold_of, n_family)
        npz_payload[f"repeat{repeat}_row_ids"] = row_ids
        npz_payload[f"repeat{repeat}_fold_of"] = fold_of
        sidecar["fold_seeds"][str(repeat)] = fold_seed
        for fold in range(N_FOLDS):
            fold_test_size = int((fold_of == fold).sum())
            removed, c2_seed, count = make_c2_removed(df, target, members, pool_row_ids, n_family, fold_test_size,
                                                       level, unit, repeat, fold)
            assert_c2_matches_c1(n_total, n_family, fold_test_size, count)
            npz_payload[f"repeat{repeat}_fold{fold}_c2_removed_row_ids"] = removed
            sidecar["c2_seeds"][f"{repeat}_{fold}"] = c2_seed
            sidecar["c2_removed_counts"][f"{repeat}_{fold}"] = count

    unit_dir = out_dir / level / target
    unit_dir.mkdir(parents=True, exist_ok=True)
    npz_path = unit_dir / f"{unit}.npz"
    json_path = unit_dir / f"{unit}.json"
    np.savez_compressed(npz_path, **npz_payload)
    json_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return npz_path, json_path


def git_state(exclude_dir):
    """Return (HEAD, tree clean ignoring exclude_dir, dirty files ignoring exclude_dir). exclude_dir is this
    run's own output directory: it is necessarily untracked while this run writes into it, which is not what
    tree_clean is meant to report (whether the CODE and INPUTS were clean when the run started)."""
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
    parser.add_argument("--csv", required=True)
    parser.add_argument("--labels-run", required=True)
    parser.add_argument("--super-units", required=True)
    args = parser.parse_args(argv)
    labels_run = Path(args.labels_run)

    df, identity = load_rows(Path(args.csv), labels_run)
    summary = pd.read_csv(labels_run / "family_summary.csv", keep_default_na=False)
    never = set(yaml.safe_load(SUPER_PATH.read_text(encoding="utf-8"))["never_held_out"])
    supers = yaml.safe_load(SUPER_PATH.read_text(encoding="utf-8"))["super_families"]
    units = pd.read_csv(args.super_units, keep_default_na=False)
    totals, pairs, super_pairs, standalone = build_pairs(summary, units, supers, never)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = OUT_DIR / stamp
    written = []
    for level, frame in (("family", pairs), ("super_family", super_pairs), ("super_analysis_standalone", standalone)):
        for _, row in frame.iterrows():
            members = [row.unit] if level != "super_family" else supers[row.unit]
            npz_path, json_path = build_one_pair(df, row.target, row.unit, members, row.g, supers, level,
                                                 totals[row.target], out_dir)
            written.append(npz_path)
            written.append(json_path)
            print(f"{level}/{row.target}/{row.unit}: n_family={row.n_family}, G={row.g} (n_g={row.n_g}) -- ok", flush=True)

    manifest = {str(p.relative_to(out_dir)): sha256_file(p) for p in sorted(written)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    head, clean, dirty = git_state(out_dir)
    config = {
        "dataset": identity, "labels_run": str(labels_run), "labels_sha256": sha256_file(labels_run / "host_family_labels.csv"),
        "super_units": args.super_units, "super_units_sha256": sha256_file(args.super_units),
        "super_families_sha256": sha256_file(SUPER_PATH), "n_folds": N_FOLDS, "repeats": R,
        "n_family_pairs": int(len(pairs)), "n_super_family_pairs": int(len(super_pairs)),
        "n_standalone_pairs": int(len(standalone)), "n_files_written": len(written),
        "git_head": head, "tree_clean": clean, "dirty_files": dirty, "utc_stamp": stamp,
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_dir} ({len(written)} files, manifest + run_config)")


if __name__ == "__main__":
    main()
