"""
Leave-one-family-out evaluation and controls for Paper B (methodology doc,
section 8). Runs the pre-registered design exactly: C0 pooled, C1 LOFO, C2
size-matched random, C3 structured (against G), and the specialist with its
own nested tuning -- for every (target, unit) pair the splits folder covers,
every repeat, every fold, both models (xgboost, ridge).

Inputs: the snapfix CSV (SHA-checked against paper_b/SHARED_DEPENDENCIES.md),
the final family-labels run (referenced for provenance only -- the splits
folder already encodes every row set the harness needs), and a splits folder
from paper_b/scripts/make_splits.py (its manifest.json is checked file by
file before anything is read from it).

Units, for checkpointing and resume: one (level, target, unit, model,
condition, repeat, fold) triple per prediction file (condition C1 has no
fold, since its training set does not depend on one; condition tuning_once
has no repeat/fold/condition, one file per (level, target, unit, model)).
Every write goes through `write_if_absent`, so a killed and restarted run
recomputes nothing that already has a checkpoint file on disk.

Hyperparameters: tuned once per (level, target, unit, model) on every row
outside the unit (inner 3-fold chemistry-cluster CV, TUNING_TRIALS, via
src/nested_cv.py's own `tune_hyperparameters` -- not reimplemented), reused
unchanged by C0/C1/C2/C3. `super_analysis_standalone` units (the C3-only
reruns of section 7.3/8.5) reuse the plain `family`-level pair's tuning_once
checkpoint for the same (target, unit) rather than retuning, since they are
the same rows. The specialist tunes fresh (SPECIALIST_TRIALS) inside each of
its own outer training folds, since its training rows differ by fold.

Roles (--role): `gpu-pooled` runs tuning_once + C0-C3 for every (level,
target, unit); `cpu-specialist` runs the specialist, sequenced per
(repeat, fold) for finer-grained load balancing across --workers processes;
`gpu-specialist-control` reruns the specialist on GPU for a small, named set
of pairs (methodology doc section 8.7's 3 zT device-control pairs by
default), for comparison against the CPU specialist run on the same pairs.
--workers > 1 spawns that many subprocesses pulling units off one shared
iterator (multiprocessing.Pool.imap_unordered); for the GPU roles, each
worker is pinned to one GPU index from --gpu-index (a comma list, cycled
across workers), so concurrent GPU workers use *different* physical GPUs --
section 8.6 found no benefit, and a probable contention cause, when they did
not.

Role independence (methodology doc section 8.1): `cpu-specialist` (and
`gpu-specialist-control`) never read anything `gpu-pooled` writes.
`process_specialist_unit` calls `tune_once` fresh, every (repeat, fold) --
it never opens a tuning_once/C0/C1/C2/C3 checkpoint. This is not just a
claim to trust from reading the code: the recommended Kaggle usage (section
8.7) points each role at its OWN --checkpoint-dir, never shared (e.g.
/kaggle/working/ckpt_gpu_pooled, ckpt_cpu_specialist, ckpt_gpu_control), so
a specialist directory that never held a single gpu-pooled file, yet
completes every unit, is a structural proof, checked directly by
smoke_test_lofo.py's check_11. An analysis/aggregation step reading
role-separated directories uses `load_checkpoint_dirs_for_analysis` below,
which merges them and checks identity agreement ACROSS directories, not
only within each one.

No cupy: unlike Paper A's src/nested_cv.py, this module does not convert X/y
to cupy arrays before a device='cuda' fit. paper_b/SHARED_DEPENDENCIES.md
does not list nested_cv.py's private `_to_device`, and section 8.6's
calibration was itself run without cupy (unavailable on that Kaggle image),
so pricing and reality agree: every GPU fit here takes XGBoost's slower,
still-correct DMatrix-fallback path (see `_to_device`'s own docstring in
src/nested_cv.py), not a hidden extra assumption.

Run from the repository root, e.g. session 1 (GPU, two workers):
    python -m paper_b.src.lofo_paperb --role gpu-pooled --workers 2 --gpu-index 0,1 \
        --csv <snapfix csv> --labels-run paper_b/reports/family_labels/<UTC> \
        --splits-dir paper_b/results/splits/<UTC>
and a CPU session (specialists, four workers):
    python -m paper_b.src.lofo_paperb --role cpu-specialist --workers 4 \
        --csv <snapfix csv> --labels-run paper_b/reports/family_labels/<UTC> \
        --splits-dir paper_b/results/splits/<UTC>
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

from src.nested_cv import MODEL_REGISTRY, N_INNER_FOLDS, N_OPTUNA_TRIALS, get_feature_columns, tune_hyperparameters
from paper_b.src.metrics_paperb import transform_target

SHARED_MANIFEST_PATH = Path("paper_b/SHARED_DEPENDENCIES.md")
TARGETS = ("S", "sigma", "kappa", "zT")
MODELS = ("xgboost", "ridge")
LEVELS = ("family", "super_family", "super_analysis_standalone")
N_FOLDS = 5
R = 3  # methodology doc section 8.7
N_INNER = N_INNER_FOLDS  # 3, from src/nested_cv.py
TUNING_TRIALS = N_OPTUNA_TRIALS  # pooled tuning, section 8.1
SPECIALIST_TRIALS = 10  # nested inside each outer training fold, section 8.1
DEFAULT_CHECKPOINT_DIR = Path("paper_b/checkpoints/lofo")
# The 3 device-control pairs, methodology doc section 8.7 (largest/median/smallest qualifying family for zT).
DEFAULT_DEVICE_CONTROL_PAIRS = [["family", "zT", "iv_vi_rocksalt"], ["family", "zT", "zintl_122"], ["family", "zT", "manganite"]]
# Unit ordering, methodology doc section 8.7: zT first, its 3 device-control pairs first within zT, then S,
# sigma, kappa; family level before super_family before the super_analysis_standalone C3-only reruns (which
# in any case each depend on their own (target, unit)'s family-level tuning_once already existing).
TARGET_ORDER = ("zT", "S", "sigma", "kappa")
LEVEL_ORDER = ("family", "super_family", "super_analysis_standalone")
DEVICE_CONTROL_UNITS_ZT = [unit for level, target, unit in DEFAULT_DEVICE_CONTROL_PAIRS if target == "zT" and level == "family"]

assert TUNING_TRIALS == 20, "methodology doc section 8.1 fixes pooled tuning at 20 trials"


# ---------------------------------------------------------------------------
# Provenance and I/O
# ---------------------------------------------------------------------------

def sha256_file(path, chunk=1 << 24):
    """SHA256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def expected_dataset_identity():
    """(sha256, bytes) of the snapfix CSV from the manifest's data table."""
    for line in SHARED_MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0] == "snapfix featurized CSV":
            return cells[2], int(cells[3].replace(",", ""))
    raise ValueError(f"no snapfix data row in {SHARED_MANIFEST_PATH}")


def normalize_rel_path(rel_path):
    """
    A manifest/run_config relative path, forward-slash always, regardless of
    which OS wrote it. PureWindowsPath parses both '/' and '\\' as
    separators, so this is a no-op on an already-POSIX path and a real fix
    on one written by a Windows os.path.relpath/str(Path) call (as
    make_splits.py's splits manifest was, before that write site was fixed
    to call .as_posix() too) -- read-side normalisation so a manifest
    committed with backslashes still resolves correctly on Linux (Kaggle),
    without rewriting or re-hashing the committed file.
    """
    return PureWindowsPath(rel_path).as_posix()


def check_splits_manifest(splits_dir):
    """Verify every file the splits manifest lists against its recorded SHA256; raise on the first mismatch."""
    manifest = json.loads((splits_dir / "manifest.json").read_text(encoding="utf-8"))
    for rel_path, expected in manifest.items():
        path = splits_dir / normalize_rel_path(rel_path)
        if not path.exists():
            raise FileNotFoundError(f"splits manifest lists {rel_path}, missing under {splits_dir}")
        got = sha256_file(path)
        if got != expected:
            raise ValueError(f"{path}: SHA256 {got} != manifest's {expected}")
    return manifest


def load_dataset(csv_path):
    """
    Load the snapfix CSV (SHA-checked) once: X (float64, Paper A's feature
    columns via src/nested_cv.py's own `get_feature_columns`), y_by_target
    (raw values, NaN where absent), chemistry_cluster_id, and row_id (the
    0-based position in the raw file -- the same identifier the splits
    folder's row_ids use). Returns (X, y_by_target, cluster_ids, identity).
    """
    expected_sha, expected_bytes = expected_dataset_identity()
    size = csv_path.stat().st_size
    if size != expected_bytes:
        raise ValueError(f"{csv_path}: {size} bytes, manifest says {expected_bytes}")
    sha = sha256_file(csv_path)
    if sha != expected_sha:
        raise ValueError(f"{csv_path}: SHA256 {sha} differs from the manifest {expected_sha}")
    header = pd.read_csv(csv_path, nrows=0)
    features = get_feature_columns(header)
    df = pd.read_csv(csv_path, usecols=[*features, "chemistry_cluster_id", *TARGETS])
    X = df[features].to_numpy(dtype=np.float64)
    y_by_target = {t: df[t].to_numpy(dtype=np.float64) for t in TARGETS}
    cluster_ids = df["chemistry_cluster_id"].to_numpy()
    identity = {"path": Path(csv_path).as_posix(), "sha256": sha, "bytes": size, "n_rows": int(len(df)), "n_features": len(features)}
    return X, y_by_target, cluster_ids, identity


def load_pair_bundle(splits_dir, level, target, unit):
    """The sidecar JSON and npz arrays for one (level, target, unit) pair."""
    path = splits_dir / level / target / unit
    sidecar = json.loads(Path(f"{path}.json").read_text(encoding="utf-8"))
    npz = np.load(f"{path}.npz")
    return sidecar, npz


def list_pairs(splits_dir, levels=LEVELS, targets=TARGETS, units=None):
    """Every (level, target, unit) with a split file under `splits_dir`, optionally filtered."""
    pairs = []
    for level in levels:
        level_dir = splits_dir / level
        if not level_dir.exists():
            continue
        for target_dir in sorted(level_dir.glob("*")):
            target = target_dir.name
            if target not in targets:
                continue
            for json_path in sorted(target_dir.glob("*.json")):
                unit = json_path.stem
                if units is not None and unit not in units:
                    continue
                pairs.append((level, target, unit))
    return pairs


def pair_sort_key(pair):
    """
    Sort key for a (level, target, unit) pair implementing the section 8.7
    order: target (zT, S, sigma, kappa), then level (family, super_family,
    super_analysis_standalone), then, only for zT's family-level pairs, the
    3 device-control units first (in DEFAULT_DEVICE_CONTROL_PAIRS's own
    order), then every other unit alphabetically.
    """
    level, target, unit = pair
    target_rank = TARGET_ORDER.index(target) if target in TARGET_ORDER else len(TARGET_ORDER)
    level_rank = LEVEL_ORDER.index(level) if level in LEVEL_ORDER else len(LEVEL_ORDER)
    if target == "zT" and level == "family" and unit in DEVICE_CONTROL_UNITS_ZT:
        control_rank = DEVICE_CONTROL_UNITS_ZT.index(unit)
    else:
        control_rank = len(DEVICE_CONTROL_UNITS_ZT)
    return (target_rank, level_rank, control_rank, unit)


def order_pairs(pairs):
    """Sort (level, target, unit) pairs into the section 8.7 processing order."""
    return sorted(pairs, key=pair_sort_key)


# The run_config identity fields that must agree across every session that wrote into one checkpoint directory,
# before its units are aggregated into any analysis (a mix of sessions run against different data/code is not
# one analysis).
ANALYSIS_IDENTITY_FIELDS = ("dataset_sha256", "splits_dir", "labels_sha256", "git_head")


def load_checkpoint_dir_for_analysis(checkpoint_dir):
    """
    The production guard before any analysis/aggregation step reads a
    checkpoint directory's units: every session's run_config
    (checkpoint_dir/run_configs/*.json, excluding the *_results.json log
    files) must agree on ANALYSIS_IDENTITY_FIELDS, and no unit's sidecar may
    have smoke_cap=true (a unit fit with --smoke-search-space-cap is not
    representative of anything but the smoke test that made it). Raises
    ValueError on either violation; returns the sorted list of unit sidecar
    paths (every *.json under checkpoint_dir except run_configs/) on success.
    """
    checkpoint_dir = Path(checkpoint_dir)
    identities = set()
    for path in sorted((checkpoint_dir / "run_configs").glob("*.json")):
        if path.name.endswith("_results.json"):
            continue
        config = json.loads(path.read_text(encoding="utf-8"))
        identities.add(tuple(config.get(field) for field in ANALYSIS_IDENTITY_FIELDS))
    if len(identities) > 1:
        raise ValueError(
            f"{checkpoint_dir}: sessions disagree on {ANALYSIS_IDENTITY_FIELDS}: {sorted(identities)}"
        )
    unit_paths = sorted(p for p in checkpoint_dir.rglob("*.json") if "run_configs" not in p.parts and p.name != "manifest.json")
    smoke_capped = []
    for path in unit_paths:
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        if "smoke_cap" not in sidecar:
            raise ValueError(f"{path}: no smoke_cap field recorded; this unit predates the production guard, refit it")
        if sidecar["smoke_cap"]:
            smoke_capped.append(path)
    if smoke_capped:
        raise ValueError(
            f"{checkpoint_dir}: {len(smoke_capped)} unit(s) were fit with --smoke-search-space-cap and are not "
            f"usable for analysis, e.g. {smoke_capped[0]}"
        )
    return unit_paths


def load_checkpoint_dirs_for_analysis(checkpoint_dirs):
    """
    Merge several role-separated checkpoint directories (methodology doc
    section 8.7's Kaggle plan: one directory per --role -- gpu-pooled's
    tuning_once/C0-C3, cpu-specialist's specialist/, gpu-specialist-control's
    specialist_gpu_control/) into one analysis-ready unit list.

    Runs load_checkpoint_dir_for_analysis on each directory unchanged first
    (so each directory's own smoke_cap check and its own within-directory
    identity check still apply, independently), then additionally requires
    every directory's sessions to agree with every OTHER directory's on
    ANALYSIS_IDENTITY_FIELDS -- so a gpu-pooled directory from one dataset or
    commit can never be silently combined with a cpu-specialist directory
    built against a different one. Raises ValueError on any violation (from
    either the per-directory check or this cross-directory one); returns the
    concatenated, sorted list of unit sidecar paths across all directories on
    success.
    """
    checkpoint_dirs = [Path(d) for d in checkpoint_dirs]
    all_identities = set()
    all_unit_paths = []
    for checkpoint_dir in checkpoint_dirs:
        unit_paths = load_checkpoint_dir_for_analysis(checkpoint_dir)  # per-directory smoke_cap + within-dir identity
        for path in sorted((checkpoint_dir / "run_configs").glob("*.json")):
            if path.name.endswith("_results.json"):
                continue
            config = json.loads(path.read_text(encoding="utf-8"))
            all_identities.add(tuple(config.get(field) for field in ANALYSIS_IDENTITY_FIELDS))
        all_unit_paths.extend(unit_paths)
    if len(all_identities) > 1:
        raise ValueError(
            f"role checkpoint dirs disagree on {ANALYSIS_IDENTITY_FIELDS} across "
            f"{[str(d) for d in checkpoint_dirs]}: {sorted(all_identities)}"
        )
    return sorted(all_unit_paths)


def write_checkpoint_manifest(checkpoint_dir):
    """
    Write manifest.json (relative path -> SHA256) for every file under
    checkpoint_dir except manifest.json itself -- every unit's .npz/.json and
    every session's run_configs/*.json. Called at the end of every session
    (main() below) and again before packaging for upload/download, so a
    manifest is always current for --restore-from's tamper check.
    """
    checkpoint_dir = Path(checkpoint_dir)
    manifest = {}
    for path in sorted(checkpoint_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest[str(path.relative_to(checkpoint_dir).as_posix())] = sha256_file(path)
    (checkpoint_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def restore_from(restore_path, checkpoint_dir, current_identity):
    """
    Merge a prior session's checkpoint output into checkpoint_dir, before any
    new unit runs. `restore_path` is a directory (an already-extracted prior
    checkpoint dir) or a .tar.gz of one; either must hold a manifest.json
    (from `write_checkpoint_manifest`) and a run_configs/ folder.

    Two checks, in this order, BEFORE any file is copied:
      1. identity: the prior session's own run_configs must all agree on
         ANALYSIS_IDENTITY_FIELDS, and that identity must equal
         `current_identity` (this session's dataset/labels/splits/git_head) --
         refuses a restore from a different dataset, labels run, splits
         folder or code commit.
      2. tamper: every file the manifest lists must still hash to what the
         manifest recorded -- refuses if anything was altered since the
         manifest was written.
    Only after both pass does it copy every manifest-listed file into
    checkpoint_dir (skipping any that already exist there). Returns a dict
    with counts; raises ValueError/FileNotFoundError on either check's failure.
    """
    restore_path = Path(restore_path)
    checkpoint_dir = Path(checkpoint_dir)
    tmp_extract = None
    if restore_path.is_file():
        tmp_extract = Path(tempfile.mkdtemp(prefix="lofo_restore_"))
        with tarfile.open(restore_path, "r:gz") as tar:
            tar.extractall(tmp_extract)
        entries = list(tmp_extract.iterdir())
        source_dir = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmp_extract
    elif restore_path.is_dir():
        source_dir = restore_path
    else:
        raise FileNotFoundError(f"--restore-from {restore_path}: neither a directory nor a file")

    try:
        manifest_path = source_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"{restore_path}: no manifest.json (every session must write one; see write_checkpoint_manifest)")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        prior_identities = set()
        for path in sorted((source_dir / "run_configs").glob("*.json")):
            if path.name.endswith("_results.json"):
                continue
            config = json.loads(path.read_text(encoding="utf-8"))
            prior_identities.add(tuple(config.get(field) for field in ANALYSIS_IDENTITY_FIELDS))
        if not prior_identities:
            raise FileNotFoundError(f"{restore_path}: no session run_config under run_configs/ to verify identity against")
        if len(prior_identities) > 1:
            raise ValueError(f"{restore_path}: its own prior sessions disagree on {ANALYSIS_IDENTITY_FIELDS}: {sorted(prior_identities)}")
        prior_identity = next(iter(prior_identities))
        if prior_identity != current_identity:
            raise ValueError(
                f"--restore-from identity mismatch on {ANALYSIS_IDENTITY_FIELDS}: "
                f"prior session {prior_identity} != this session {current_identity}"
            )

        for rel_path, expected_sha in manifest.items():
            source_file = source_dir / normalize_rel_path(rel_path)
            if not source_file.exists():
                raise FileNotFoundError(f"{restore_path}: manifest lists {rel_path}, missing from the archive/directory")
            got = sha256_file(source_file)
            if got != expected_sha:
                raise ValueError(
                    f"{restore_path}: {rel_path} does not match its manifest SHA256 "
                    f"(expected {expected_sha}, got {got}); refusing a possibly tampered restore"
                )

        copied = 0
        for rel_path in manifest:
            norm_rel_path = normalize_rel_path(rel_path)
            dest_file = checkpoint_dir / norm_rel_path
            if dest_file.exists():
                continue
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_dir / norm_rel_path, dest_file)
            copied += 1
        return {"source": str(restore_path), "n_manifest_entries": len(manifest), "n_copied": copied,
                "n_already_present": len(manifest) - copied}
    finally:
        if tmp_extract is not None:
            shutil.rmtree(tmp_extract, ignore_errors=True)


def write_if_absent(path, arrays, sidecar):
    """Write `path`.npz (from `arrays`) and `path`.json (from `sidecar`) unless a checkpoint already exists; returns whether it wrote."""
    if path.with_suffix(".npz").exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, default=float) + "\n", encoding="utf-8")
    return True


def load_checkpoint(path):
    """The (arrays, sidecar) of an existing checkpoint, or (None, None) if it does not exist."""
    if not path.with_suffix(".npz").exists():
        return None, None
    arrays = dict(np.load(path.with_suffix(".npz")))
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8")) if path.with_suffix(".json").exists() else {}
    return arrays, sidecar


def seed_for(*parts):
    """Deterministic 32-bit seed from a stable string of `parts` (same scheme as make_splits.py's)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_and_predict(model_type, device, params, X_train, y_train, X_test):
    """Build the model from MODEL_REGISTRY, fit on (X_train, y_train), predict on X_test."""
    model = MODEL_REGISTRY[model_type]["build"](params, device)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def tune_once(model_type, device, seed, X_train, y_train, groups_train, n_trials):
    """`src.nested_cv.tune_hyperparameters`, inner 3-fold chemistry-cluster CV; returns (best_params, inner_cv_r2)."""
    return tune_hyperparameters(X_train, y_train, groups_train, model_type=model_type,
                                n_trials=n_trials, n_inner_folds=N_INNER, seed=seed, device=device)


def _capped_xgb_search_space(trial):
    """
    A tiny search space, for --smoke-search-space-cap ONLY: never the source
    of a reported result. tuning_once/C0-C3 fit on nearly the whole dataset
    (every row outside a small unit, often 99%+ of it), so Paper A's full
    search space (max_depth up to 10, n_estimators up to 600) makes even a
    2-trial local CPU smoke test take much longer than is useful for
    verification -- the exact "infeasible on local CPU" case CLAUDE.md
    already documents for Paper A's own tuning, now inherited by this
    harness's tuning_once. This mirrors that file's own precedent: "a capped
    range is acceptable only for a clearly-labeled calibration/smoke-test
    run, never as the default the real reported numbers come from."
    src/nested_cv.py itself is never edited; this only monkey-patches the
    MODEL_REGISTRY dict this process imported, and only when explicitly asked.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 10, 20, step=10),
        "max_depth": trial.suggest_int("max_depth", 2, 2),
        "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.8, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 3),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 1.0, log=True),
    }


def apply_smoke_search_space_cap():
    """Apply `_capped_xgb_search_space` in place of Paper A's full search space, for this process only."""
    MODEL_REGISTRY["xgboost"]["search_space"] = _capped_xgb_search_space


def unit_row_ids(npz):
    """F's own row_ids (the same set across repeats; repeat1's is as good as any)."""
    return npz["repeat1_row_ids"]


def exclude_unit_mask(npz, n_rows):
    """A mask over all n_rows that is False on the unit's own rows, True elsewhere."""
    mask = np.ones(n_rows, dtype=bool)
    mask[unit_row_ids(npz)] = False
    return mask


# ---------------------------------------------------------------------------
# Leakage assertions (checked, not just designed to hold by construction)
# ---------------------------------------------------------------------------

def assert_disjoint(mask_or_ids_a, mask_or_ids_b, n_rows, context):
    """Raise if the two row sets (each a boolean mask over n_rows, or an array of row ids) share any row."""
    def to_ids(x):
        x = np.asarray(x)
        return np.where(x)[0] if x.dtype == bool else x
    a, b = set(to_ids(mask_or_ids_a).tolist()), set(to_ids(mask_or_ids_b).tolist())
    overlap = a & b
    if overlap:
        raise AssertionError(f"{context}: {len(overlap)} row(s) leaked, e.g. row_id {next(iter(overlap))}")


def assert_subset(mask_or_ids, allowed_ids, context):
    """Raise unless every row in `mask_or_ids` is also in `allowed_ids`."""
    ids = np.asarray(mask_or_ids)
    ids = np.where(ids)[0] if ids.dtype == bool else ids
    extra = set(ids.tolist()) - set(np.asarray(allowed_ids).tolist())
    if extra:
        raise AssertionError(f"{context}: {len(extra)} row(s) outside the allowed set, e.g. row_id {next(iter(extra))}")


def assert_size_equal(a, b, context):
    """Raise unless the two training sizes are equal."""
    if a != b:
        raise AssertionError(f"{context}: sizes differ ({a} != {b})")


# ---------------------------------------------------------------------------
# gpu-pooled: tuning_once + C0 + C1 + C2 + C3
# ---------------------------------------------------------------------------

def process_pooled_pair(level, target, unit, model_type, device, X, y_by_target, cluster_ids, checkpoint_dir,
                        splits_dir, tuning_trials=TUNING_TRIALS, repeats=None, folds=None, smoke_cap=False,
                        deadline=None, on_unit_done=None):
    """
    tuning_once + C0/C1/C2/C3 for one (level, target, unit, model) pair.
    Returns (log, deadline_hit): `log` is a list of (component, path, wrote)
    for what ran (anything already checkpointed is skipped); `deadline_hit`
    is True if this call stopped early because `deadline` (an absolute
    `time.time()` value, or None for no budget) had passed.

    A single call can span tuning_once + C1 + C0/C2/C3 across every repeat
    and fold -- potentially a long-running unit of work, since `run_pool`'s
    own --time-budget-hours check only gates which TASK gets dispatched
    next, not what happens inside one already-dispatched task (found
    2026-09-30: a task that started before the deadline ran ~70+ minutes
    unbroken against a 5-minute budget). This function therefore checks the
    SAME deadline itself, immediately after every piece of real work it
    does (tuning_once, C1, and each (component, repeat, fold) fit) -- never
    mid-fit (a single model.fit() call is not preemptable), only between
    them -- and returns as soon as it finds the deadline passed, leaving
    whatever remains for a later invocation to pick up via the usual
    write_if_absent skip-logic. `repeats`/`folds` restrict which repeat
    numbers (1-based) and fold indices (0-based) are processed this call,
    same as before. `on_unit_done(component, repeat, fold, seconds)`, if
    given, is called immediately after each piece of real work (not after a
    skip) for progress logging -- see log_unit_progress.
    """
    def deadline_passed():
        return deadline is not None and time.time() >= deadline

    y = transform_target(y_by_target[target], target)
    n_rows = len(y)
    not_nan = ~np.isnan(y)
    sidecar, npz = load_pair_bundle(splits_dir, level, target, unit)
    base = checkpoint_dir / level / target / unit / model_type
    log = []

    if level == "super_analysis_standalone":
        family_base = checkpoint_dir / "family" / target / unit / model_type
        tuning_arrays, tuning_sidecar = load_checkpoint(family_base / "tuning_once")
        if tuning_sidecar is None:
            raise FileNotFoundError(
                f"{level}/{target}/{unit}: needs the family-level pair's tuning_once checkpoint first ({family_base})"
            )
        best_params = tuning_sidecar["best_params"]
        c1_train_size = None
    else:
        _, tuning_sidecar = load_checkpoint(base / "tuning_once")
        if tuning_sidecar is None:
            unit_started = time.perf_counter()
            excl_mask = exclude_unit_mask(npz, n_rows) & not_nan
            assert_disjoint(excl_mask, unit_row_ids(npz), n_rows, f"tuning_once {level}/{target}/{unit}: contains a row of F")
            best_params, inner_cv_r2 = tune_once(model_type, device, seed_for(level, target, unit, model_type, "tuning"),
                                                 X[excl_mask], y[excl_mask], cluster_ids[excl_mask], tuning_trials)
            wrote = write_if_absent(base / "tuning_once", arrays={"n_train_rows": np.array([int(excl_mask.sum())])},
                                    sidecar={"level": level, "target": target, "unit": unit, "model": model_type,
                                            "device": device, "n_trials": tuning_trials,
                                            "best_params": best_params, "inner_cv_r2": inner_cv_r2, "smoke_cap": smoke_cap})
            log.append(("tuning_once", base / "tuning_once", wrote))
            if on_unit_done is not None:
                on_unit_done("tuning_once", None, None, time.perf_counter() - unit_started)
            if deadline_passed():
                return log, True
        else:
            best_params = tuning_sidecar["best_params"]

        c1_train_size = int((exclude_unit_mask(npz, n_rows) & not_nan).sum())
        c1_path = base / "C1"
        if not c1_path.with_suffix(".npz").exists():
            unit_started = time.perf_counter()
            excl_mask = exclude_unit_mask(npz, n_rows) & not_nan
            assert_disjoint(excl_mask, unit_row_ids(npz), n_rows, f"C1 {level}/{target}/{unit}: training contains a row of F")
            f_row_ids = unit_row_ids(npz)
            f_mask = np.zeros(n_rows, dtype=bool)
            f_mask[f_row_ids] = True
            y_pred = fit_and_predict(model_type, device, best_params, X[excl_mask], y[excl_mask], X[f_mask])
            wrote = write_if_absent(c1_path, arrays={"row_ids": f_row_ids, "y_true": y[f_mask], "y_pred": y_pred,
                                                     "train_mean": np.array([y[excl_mask].mean()])},
                                    sidecar={"params": best_params, "n_train": int(excl_mask.sum()), "smoke_cap": smoke_cap})
            log.append(("C1", c1_path, wrote))
            if on_unit_done is not None:
                on_unit_done("C1", None, None, time.perf_counter() - unit_started)
            if deadline_passed():
                return log, True

    repeat_range = repeats if repeats is not None else range(1, R + 1)
    fold_range = folds if folds is not None else range(N_FOLDS)
    for repeat in repeat_range:
        row_ids = npz[f"repeat{repeat}_row_ids"]
        fold_of = npz[f"repeat{repeat}_fold_of"]
        for fold in fold_range:
            test_row_ids = row_ids[fold_of == fold]
            if len(test_row_ids) == 0:
                continue
            test_mask = np.zeros(n_rows, dtype=bool)
            test_mask[test_row_ids] = True
            c0_train_mask = not_nan & ~test_mask  # C0: every non-null row for the target except this fold's test rows

            # super_analysis_standalone is a C3-only rerun (section 7.3/8.5): its C0/C1/C2/tuning_once are the
            # family-level pair's, already computed for this same (target, unit); only G may differ here.
            components = (("C3", "g_row_ids"),) if level == "super_analysis_standalone" else (
                ("C0", None), ("C2", f"repeat{repeat}_fold{fold}_c2_removed_row_ids"), ("C3", "g_row_ids"))
            for component, extra_removed_key in components:
                path = base / component / f"r{repeat}_f{fold}"
                if path.with_suffix(".npz").exists():
                    continue
                unit_started = time.perf_counter()
                train_mask = c0_train_mask.copy()
                if extra_removed_key is not None:
                    train_mask[npz[extra_removed_key]] = False
                assert_disjoint(train_mask, test_mask, n_rows,
                                f"{component} {level}/{target}/{unit} r{repeat}f{fold}: training contains a test-fold row")
                if component == "C2" and c1_train_size is not None:
                    assert_size_equal(int(train_mask.sum()), c1_train_size,
                                      f"C2 {level}/{target}/{unit} r{repeat}f{fold}: training size != C1's")
                if component == "C3":
                    assert_disjoint(train_mask, npz["g_row_ids"], n_rows,
                                    f"C3 {level}/{target}/{unit} r{repeat}f{fold}: training contains a row of G")
                y_pred = fit_and_predict(model_type, device, best_params, X[train_mask], y[train_mask], X[test_mask])
                wrote = write_if_absent(path, arrays={"row_ids": test_row_ids, "y_true": y[test_mask], "y_pred": y_pred,
                                                      "train_mean": np.array([y[train_mask].mean()])},
                                        sidecar={"params": best_params, "n_train": int(train_mask.sum()), "smoke_cap": smoke_cap})
                log.append((component, path, wrote))
                if on_unit_done is not None:
                    on_unit_done(component, repeat, fold, time.perf_counter() - unit_started)
                if deadline_passed():
                    return log, True
    return log, False


# ---------------------------------------------------------------------------
# cpu-specialist / gpu-specialist-control
# ---------------------------------------------------------------------------

def process_specialist_unit(level, target, unit, model_type, device, repeat, fold, X, y_by_target, cluster_ids,
                            checkpoint_dir, splits_dir, trials=SPECIALIST_TRIALS, checkpoint_suffix="", smoke_cap=False):
    """
    One specialist fit: F's own training folds (repeat, all folds but
    `fold`), nested tuning (inner 3-fold chemistry-cluster CV over those
    training rows), fit, predict on F's `fold` test rows. Compared with C0
    of the same (level, target, unit, model, repeat, fold). Returns
    (component_name, path, wrote), or None if this (repeat, fold) has no
    test rows (a unit smaller than N_FOLDS chemistry clusters can leave a
    fold empty; see make_splits.py's assert_folds_valid).
    """
    y = transform_target(y_by_target[target], target)
    name = f"specialist{checkpoint_suffix}"
    path = checkpoint_dir / level / target / unit / model_type / name / f"r{repeat}_f{fold}"
    if path.with_suffix(".npz").exists():
        return name, path, False
    _, npz = load_pair_bundle(splits_dir, level, target, unit)
    row_ids = npz[f"repeat{repeat}_row_ids"]
    fold_of = npz[f"repeat{repeat}_fold_of"]
    test_row_ids = row_ids[fold_of == fold]
    train_row_ids = row_ids[fold_of != fold]
    if len(test_row_ids) == 0 or len(train_row_ids) == 0:
        return None
    assert_subset(train_row_ids, row_ids, f"specialist {level}/{target}/{unit} r{repeat}f{fold}: training has a row outside F")
    assert_disjoint(train_row_ids, test_row_ids, len(cluster_ids),
                    f"specialist {level}/{target}/{unit} r{repeat}f{fold}: training contains a test-fold row")
    train_groups = cluster_ids[train_row_ids]
    best_params, inner_cv_r2 = tune_once(model_type, device, seed_for(level, target, unit, model_type, repeat, fold, "specialist"),
                                         X[train_row_ids], y[train_row_ids], train_groups, trials)
    y_pred = fit_and_predict(model_type, device, best_params, X[train_row_ids], y[train_row_ids], X[test_row_ids])
    wrote = write_if_absent(path, arrays={"row_ids": test_row_ids, "y_true": y[test_row_ids], "y_pred": y_pred,
                                          "train_mean": np.array([y[train_row_ids].mean()])},
                            sidecar={"params": best_params, "inner_cv_r2": inner_cv_r2, "n_train": int(len(train_row_ids)),
                                    "device": device, "smoke_cap": smoke_cap})
    return name, path, wrote


# ---------------------------------------------------------------------------
# Worker pool plumbing
# ---------------------------------------------------------------------------

_WORKER_STATE = {}


def _pool_initializer(csv_path, splits_dir, checkpoint_dir, gpu_indices, counter, tuning_trials, specialist_trials,
                      repeats, folds, smoke_cap=False, deadline=None):
    """Runs once per worker process (or once, directly, for --workers<=1): load the dataset, resolve this worker's slot/GPU index."""
    if smoke_cap:
        apply_smoke_search_space_cap()
    X, y_by_target, cluster_ids, identity = load_dataset(Path(csv_path))
    with counter.get_lock():
        slot = counter.value
        counter.value += 1
    worker_gpu = gpu_indices[slot % len(gpu_indices)] if gpu_indices else None
    _WORKER_STATE.update(X=X, y_by_target=y_by_target, cluster_ids=cluster_ids, identity=identity,
                         splits_dir=Path(splits_dir), checkpoint_dir=Path(checkpoint_dir), worker_gpu=worker_gpu,
                         worker_slot=slot, tuning_trials=tuning_trials, specialist_trials=specialist_trials,
                         repeats=repeats, folds=folds, smoke_cap=smoke_cap, deadline=deadline)


def _device():
    return f"cuda:{_WORKER_STATE['worker_gpu']}" if _WORKER_STATE["worker_gpu"] is not None else "cpu"


def log_unit_progress(worker_slot, gpu_index, level, target, unit, condition, repeat, fold, model_type, seconds):
    """
    One flushed stdout line per finished unit: UTC time, worker, GPU index,
    target (t), unit (F), condition, repeat, fold, model, seconds. Kaggle
    buffers stdout by default, which is why the 70+-minute hang (2026-09-30)
    produced no visible output at all until cancelled -- run every Kaggle
    cell with `python -u` so these lines appear as they happen, not only at
    process exit.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{ts} worker={worker_slot} gpu={gpu_index} level={level} target={target} unit={unit} "
          f"condition={condition} repeat={repeat} fold={fold} model={model_type} seconds={seconds:.2f}", flush=True)


def _pooled_task(task):
    """One gpu-pooled task: (level, target, unit, model_type)."""
    level, target, unit, model_type = task
    device = _device()
    worker_slot, gpu_index = _WORKER_STATE["worker_slot"], _WORKER_STATE["worker_gpu"]
    started = time.perf_counter()

    def on_unit_done(component, repeat, fold, seconds):
        log_unit_progress(worker_slot, gpu_index, level, target, unit, component, repeat, fold, model_type, seconds)

    log, deadline_hit = process_pooled_pair(
        level, target, unit, model_type, device, _WORKER_STATE["X"], _WORKER_STATE["y_by_target"],
        _WORKER_STATE["cluster_ids"], _WORKER_STATE["checkpoint_dir"], _WORKER_STATE["splits_dir"],
        tuning_trials=_WORKER_STATE["tuning_trials"], repeats=_WORKER_STATE["repeats"],
        folds=_WORKER_STATE["folds"], smoke_cap=_WORKER_STATE["smoke_cap"],
        deadline=_WORKER_STATE.get("deadline"), on_unit_done=on_unit_done)
    return {"task": list(task), "device": device, "seconds": time.perf_counter() - started,
            "wrote": [str(p) for _, p, wrote in log if wrote], "gpu_index": gpu_index,
            "worker_slot": worker_slot, "deadline_hit": deadline_hit}


def _specialist_task(task):
    """One cpu-specialist / gpu-specialist-control task: (level, target, unit, model_type, repeat, fold, suffix)."""
    level, target, unit, model_type, repeat, fold, suffix = task
    device = _device()
    started = time.perf_counter()
    result = process_specialist_unit(level, target, unit, model_type, device, repeat, fold, _WORKER_STATE["X"],
                                     _WORKER_STATE["y_by_target"], _WORKER_STATE["cluster_ids"],
                                     _WORKER_STATE["checkpoint_dir"], _WORKER_STATE["splits_dir"],
                                     trials=_WORKER_STATE["specialist_trials"], checkpoint_suffix=suffix,
                                     smoke_cap=_WORKER_STATE["smoke_cap"])
    wrote = bool(result[2]) if result is not None else False
    elapsed = time.perf_counter() - started
    log_unit_progress(_WORKER_STATE["worker_slot"], _WORKER_STATE["worker_gpu"], level, target, unit,
                      f"specialist{suffix}", repeat, fold, model_type, elapsed)
    return {"task": list(task), "device": device, "seconds": elapsed, "wrote": [] if not wrote else [str(result[1])],
            "gpu_index": _WORKER_STATE["worker_gpu"], "worker_slot": _WORKER_STATE["worker_slot"]}


def _budgeted(tasks, deadline):
    """
    Yield `tasks` in order, stopping (a plain StopIteration, no partial task
    yielded) once `time.time() >= deadline`. `deadline` of None means no
    budget: yield everything. This is how --time-budget-hours "starts no new
    unit" between TASKS -- a task already pulled by an idle worker used to
    always run to completion regardless of the deadline (this only gated
    what got pulled NEXT); as of 2026-09-30 that is no longer the whole
    story for gpu-pooled, since process_pooled_pair now also checks the
    SAME deadline between the components inside one task (see its own
    docstring) -- both checks share this one wall-clock deadline value.
    `time.time()`, not `time.perf_counter()`, because this deadline is
    computed once in the parent process and must also be read correctly
    inside spawned worker processes (see main()'s own note on this).  With
    workers > 1 a multiprocessing.Pool may have a small number of tasks
    already buffered for dispatch at the moment the deadline is crossed, so
    the cutoff is close to H hours, not exact to the second.
    """
    for task in tasks:
        if deadline is not None and time.time() >= deadline:
            return
        yield task


def run_pool(tasks, task_fn, csv_path, splits_dir, checkpoint_dir, gpu_indices, workers, tuning_trials, specialist_trials,
            repeats, folds, smoke_cap=False, deadline=None):
    """
    Run `tasks` through `task_fn`, in this process if workers<=1, else via a
    `workers`-process pool (a shared work queue). `deadline` (an absolute
    `time.time()` value; see --time-budget-hours) is passed into every
    worker's `_WORKER_STATE` too (not just used to gate `_budgeted` here),
    so gpu-pooled's `process_pooled_pair` can also check it between the
    components INSIDE one task -- a single (level, target, unit, model) task
    can run tuning_once + C1 + C0/C2/C3 across every repeat and fold, which
    without that finer check could run for a long time uninterrupted even
    after the deadline passed (found 2026-09-30: a Kaggle smoke run hung for
    70+ minutes against a 5-minute budget, one gpu-pooled task running
    unbroken). Returns (results, n_not_started): n_not_started is how many
    of `tasks` were never even attempted this session because the deadline
    had already passed before `_budgeted` reached them. A task that WAS
    attempted but stopped partway through (gpu-pooled only, deadline hit
    between components) still appears in `results`, with `deadline_hit=True`
    in its dict -- see main()'s session_summary, which folds these into
    units_remaining too.
    """
    # Every multiprocessing primitive here is built from THIS SAME explicit context, not the module-level
    # mp.Value/mp.Lock/etc (which use the platform DEFAULT context -- 'fork' on Linux, 'spawn' on Windows and
    # macOS). Found 2026-10-01 on Kaggle (Linux): a counter built via the bare mp.Value("i", 0) call carries a
    # fork-context SemLock, and handing that to a Pool explicitly constructed with mp.get_context("spawn")
    # raised "A SemLock created in a fork context is being shared with a process in a spawn context" at pool
    # start. Windows has no fork context at all, so mp.Value's default there already happened to be spawn --
    # this local smoke test cannot reproduce the crash no matter what it does, only the Kaggle Linux run can.
    context = mp.get_context("spawn")
    tasks = list(tasks)
    if workers <= 1:
        _pool_initializer(csv_path, splits_dir, checkpoint_dir, gpu_indices, context.Value("i", 0), tuning_trials,
                          specialist_trials, repeats, folds, smoke_cap, deadline)
        # check the deadline live, right before each task, not once up front against the whole list
        results = [task_fn(task) for task in _budgeted(tasks, deadline)]
    else:
        budgeted = _budgeted(tasks, deadline)
        counter = context.Value("i", 0)
        with context.Pool(processes=workers, initializer=_pool_initializer,
                          initargs=(csv_path, splits_dir, checkpoint_dir, gpu_indices, counter, tuning_trials,
                                   specialist_trials, repeats, folds, smoke_cap, deadline)) as pool:
            results = list(pool.imap_unordered(task_fn, budgeted))
    return results, len(tasks) - len(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def git_state():
    """Return (HEAD, tree clean, dirty files)."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    dirty = [line[3:] for line in status.splitlines()]
    return head, not dirty, dirty


def library_versions():
    """Versions of every library this module's fits depend on, for the run_config."""
    out = {"python": sys.version, "platform": platform.platform()}
    for name in ("numpy", "pandas", "sklearn", "xgboost", "optuna"):
        try:
            out[name] = __import__(name).__version__
        except Exception as exc:
            out[name] = f"unavailable ({type(exc).__name__})"
    return out


def main(argv=None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--role", required=True, choices=("gpu-pooled", "cpu-specialist", "gpu-specialist-control"))
    parser.add_argument("--csv", required=True)
    parser.add_argument("--labels-run", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--gpu-index", default=None, help="comma list, e.g. '0,1'; cycled across --workers")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--targets", default=None, help="comma list; default all four")
    parser.add_argument("--units", default=None, help="comma list of unit names; default every qualifying unit")
    parser.add_argument("--levels", default=None, help="comma list from family,super_family,super_analysis_standalone")
    parser.add_argument("--repeats", default=None, help="comma list of repeat numbers (1-based); default 1..R")
    parser.add_argument("--folds", default=None, help="comma list of fold indices (0-based); default 0..N_FOLDS-1")
    parser.add_argument("--models", default=None, help="comma list from xgboost,ridge; default both")
    parser.add_argument("--tuning-trials", type=int, default=TUNING_TRIALS)
    parser.add_argument("--specialist-trials", type=int, default=SPECIALIST_TRIALS)
    parser.add_argument("--device-control-pairs", default=None,
                        help="JSON list of [level,target,unit]; default the 3 zT pairs, section 8.7")
    parser.add_argument("--smoke-search-space-cap", action="store_true",
                        help="tiny XGBoost search space (see apply_smoke_search_space_cap); local smoke tests only, "
                             "never for a reported result")
    parser.add_argument("--time-budget-hours", type=float, default=None,
                        help="after H hours, start no new unit, finish the running ones, write a session summary, exit 0")
    parser.add_argument("--restore-from", default=None,
                        help="a prior session's checkpoint directory or .tar.gz (see restore_from); merged in before any unit runs")
    args = parser.parse_args(argv)

    csv_path, splits_dir, checkpoint_dir = Path(args.csv), Path(args.splits_dir), Path(args.checkpoint_dir)
    if args.smoke_search_space_cap and "smoke" not in str(checkpoint_dir).lower():
        raise SystemExit(
            f"--smoke-search-space-cap requires --checkpoint-dir to contain 'smoke' (got {checkpoint_dir}); "
            "this stops a capped, non-representative fit from being written where a real result might read it."
        )
    manifest = check_splits_manifest(splits_dir)
    splits_config = json.loads((splits_dir / "run_config.json").read_text(encoding="utf-8"))
    targets = tuple(args.targets.split(",")) if args.targets else TARGETS
    models = tuple(args.models.split(",")) if args.models else MODELS
    repeats = [int(r) for r in args.repeats.split(",")] if args.repeats else None
    folds = [int(f) for f in args.folds.split(",")] if args.folds else None
    units = set(args.units.split(",")) if args.units else None
    gpu_indices = [int(g) for g in args.gpu_index.split(",")] if args.gpu_index else None

    # Identity is fixed at the start of the session (code, dataset, labels, splits), before any restore or fit,
    # matching how tree_clean is meant to be read elsewhere in this project (state at run start, not run end).
    head, clean, dirty = git_state()
    dataset_sha, dataset_bytes = expected_dataset_identity()
    labels_sha = sha256_file(Path(args.labels_run) / "host_family_labels.csv")
    current_identity = (dataset_sha, str(splits_dir), labels_sha, head)  # must match ANALYSIS_IDENTITY_FIELDS's order
    restore_result = None
    if args.restore_from:
        restore_result = restore_from(args.restore_from, checkpoint_dir, current_identity)
        print(f"--restore-from {args.restore_from}: {restore_result['n_copied']} file(s) copied, "
              f"{restore_result['n_already_present']} already present, of {restore_result['n_manifest_entries']} listed")

    # time.time(), not time.perf_counter(): this deadline is read inside spawned worker processes too (run_pool
    # passes it into _WORKER_STATE, for process_pooled_pair's own in-pair check), and perf_counter()'s reference
    # point is documented as undefined -- only valid to diff within the process that produced it. time.time() is
    # an ordinary wall-clock timestamp, unambiguous across processes on the same machine.
    deadline = time.time() + args.time_budget_hours * 3600 if args.time_budget_hours is not None else None

    started = time.perf_counter()
    if args.role == "gpu-pooled":
        levels = tuple(args.levels.split(",")) if args.levels else LEVELS
        pairs = order_pairs(list_pairs(splits_dir, levels=levels, targets=targets, units=units))
        tasks = [(level, target, unit, model) for level, target, unit in pairs for model in models]
        results, n_not_started = run_pool(tasks, _pooled_task, csv_path, splits_dir, checkpoint_dir, gpu_indices,
                                          args.workers, args.tuning_trials, args.specialist_trials, repeats, folds,
                                          args.smoke_search_space_cap, deadline)
    elif args.role in ("cpu-specialist", "gpu-specialist-control"):
        if args.role == "gpu-specialist-control":
            pairs = (json.loads(args.device_control_pairs) if args.device_control_pairs else DEFAULT_DEVICE_CONTROL_PAIRS)
            pairs = order_pairs([tuple(p) for p in pairs])
            suffix = "_gpu_control"
            if not gpu_indices:
                gpu_indices = [0]
        else:
            pairs = order_pairs(list_pairs(splits_dir, levels=("family", "super_family"), targets=targets, units=units))
            suffix = ""
            gpu_indices = None
        rep_range = repeats or list(range(1, R + 1))
        fold_range = folds or list(range(N_FOLDS))
        tasks = [(level, target, unit, model, repeat, fold, suffix)
                for level, target, unit in pairs for model in models for repeat in rep_range for fold in fold_range]
        results, n_not_started = run_pool(tasks, _specialist_task, csv_path, splits_dir, checkpoint_dir, gpu_indices,
                                          args.workers, args.tuning_trials, args.specialist_trials, None, None,
                                          args.smoke_search_space_cap, deadline)
    else:
        raise ValueError(args.role)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_config_dir = checkpoint_dir / "run_configs"
    run_config_dir.mkdir(parents=True, exist_ok=True)

    per_worker = {}
    for r in results:
        slot = r.get("worker_slot")
        entry = per_worker.setdefault(slot, {"n_units": 0, "total_seconds": 0.0, "gpu_index": r.get("gpu_index")})
        entry["n_units"] += 1
        entry["total_seconds"] += r.get("seconds", 0.0)
    for entry in per_worker.values():
        entry["mean_seconds_per_unit"] = entry["total_seconds"] / entry["n_units"] if entry["n_units"] else None
    # A gpu-pooled task that hit the deadline mid-pair (process_pooled_pair's own in-pair check, not just
    # run_pool's between-task one) is neither "not started" nor fully done -- it appears in `results` (some
    # component(s) checkpointed, wrote may be non-empty) but n_not_started never counted it. Fold it into
    # units_remaining so the summary reflects real remaining work, not just whole never-dispatched tasks.
    n_partial = sum(1 for r in results if r.get("deadline_hit"))
    session_summary = {
        "units_done_this_session": len(results) - n_partial, "units_partial_this_session": n_partial,
        "units_remaining": n_not_started + n_partial,
        "time_budget_hours": args.time_budget_hours,
        "budget_exhausted": (n_not_started > 0 or n_partial > 0) and deadline is not None,
        "per_worker": {str(k): v for k, v in sorted(per_worker.items(), key=lambda kv: (kv[0] is None, kv[0]))},
    }

    config = {
        "role": args.role, "workers": args.workers, "gpu_index": gpu_indices,
        "dataset_sha256": dataset_sha, "dataset_bytes": dataset_bytes,
        "labels_run": args.labels_run, "labels_sha256": labels_sha,
        "splits_dir": str(splits_dir), "splits_manifest_n_files": len(manifest), "splits_run_config": splits_config,
        "targets": list(targets), "models": list(models), "repeats_filter": repeats, "folds_filter": folds,
        "tuning_trials": args.tuning_trials, "specialist_trials": args.specialist_trials,
        "smoke_search_space_cap": args.smoke_search_space_cap, "time_budget_hours": args.time_budget_hours,
        "restore_from": args.restore_from, "restore_result": restore_result,
        "task_order": [f"{level}/{target}/{unit}" for level, target, unit in pairs],  # section 8.7, logged
        "n_tasks": len(tasks), "n_wrote": sum(1 for r in results if r.get("wrote")), "total_seconds": time.perf_counter() - started,
        "session_summary": session_summary,
        "git_head": head, "tree_clean": clean, "dirty_files": dirty, "library_versions": library_versions(), "utc_stamp": stamp,
    }
    config_path = run_config_dir / f"{stamp}_{args.role}.json"
    config_path.write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    (run_config_dir / f"{stamp}_{args.role}_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    write_checkpoint_manifest(checkpoint_dir)
    print(f"role={args.role} tasks={len(tasks)} wrote={config['n_wrote']} seconds={config['total_seconds']:.1f}")
    print(f"session summary: done={session_summary['units_done_this_session']} "
          f"partial={session_summary['units_partial_this_session']} "
          f"remaining={session_summary['units_remaining']} budget_exhausted={session_summary['budget_exhausted']}")
    for slot, entry in session_summary["per_worker"].items():
        print(f"  worker {slot} (gpu_index={entry['gpu_index']}): {entry['n_units']} unit(s), "
              f"mean {entry['mean_seconds_per_unit']:.2f} s/unit")
    print(f"run_config: {config_path}")


if __name__ == "__main__":
    main()
