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
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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


def check_splits_manifest(splits_dir):
    """Verify every file the splits manifest lists against its recorded SHA256; raise on the first mismatch."""
    manifest = json.loads((splits_dir / "manifest.json").read_text(encoding="utf-8"))
    for rel_path, expected in manifest.items():
        path = splits_dir / rel_path
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
    identity = {"path": str(csv_path), "sha256": sha, "bytes": size, "n_rows": int(len(df)), "n_features": len(features)}
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
    unit_paths = sorted(p for p in checkpoint_dir.rglob("*.json") if "run_configs" not in p.parts)
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
                        splits_dir, tuning_trials=TUNING_TRIALS, repeats=None, folds=None, smoke_cap=False):
    """
    tuning_once + C0/C1/C2/C3 for one (level, target, unit, model) pair.
    Returns a list of (component, path, wrote) for what ran; anything already
    checkpointed is skipped. `repeats`/`folds` restrict which repeat numbers
    (1-based) and fold indices (0-based) are processed this call, leaving the
    rest for a later invocation.
    """
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
            excl_mask = exclude_unit_mask(npz, n_rows) & not_nan
            assert_disjoint(excl_mask, unit_row_ids(npz), n_rows, f"tuning_once {level}/{target}/{unit}: contains a row of F")
            best_params, inner_cv_r2 = tune_once(model_type, device, seed_for(level, target, unit, model_type, "tuning"),
                                                 X[excl_mask], y[excl_mask], cluster_ids[excl_mask], tuning_trials)
            wrote = write_if_absent(base / "tuning_once", arrays={"n_train_rows": np.array([int(excl_mask.sum())])},
                                    sidecar={"level": level, "target": target, "unit": unit, "model": model_type,
                                            "device": device, "n_trials": tuning_trials,
                                            "best_params": best_params, "inner_cv_r2": inner_cv_r2, "smoke_cap": smoke_cap})
            log.append(("tuning_once", base / "tuning_once", wrote))
        else:
            best_params = tuning_sidecar["best_params"]

        c1_train_size = int((exclude_unit_mask(npz, n_rows) & not_nan).sum())
        c1_path = base / "C1"
        if not c1_path.with_suffix(".npz").exists():
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
    return log


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
                      repeats, folds, smoke_cap=False):
    """Runs once per worker process (or once, directly, for --workers<=1): load the dataset, resolve this worker's GPU index."""
    if smoke_cap:
        apply_smoke_search_space_cap()
    X, y_by_target, cluster_ids, identity = load_dataset(Path(csv_path))
    worker_gpu = None
    if gpu_indices:
        with counter.get_lock():
            slot = counter.value
            counter.value += 1
        worker_gpu = gpu_indices[slot % len(gpu_indices)]
    _WORKER_STATE.update(X=X, y_by_target=y_by_target, cluster_ids=cluster_ids, identity=identity,
                         splits_dir=Path(splits_dir), checkpoint_dir=Path(checkpoint_dir), worker_gpu=worker_gpu,
                         tuning_trials=tuning_trials, specialist_trials=specialist_trials, repeats=repeats, folds=folds,
                         smoke_cap=smoke_cap)


def _device():
    return f"cuda:{_WORKER_STATE['worker_gpu']}" if _WORKER_STATE["worker_gpu"] is not None else "cpu"


def _pooled_task(task):
    """One gpu-pooled task: (level, target, unit, model_type)."""
    level, target, unit, model_type = task
    device = _device()
    started = time.perf_counter()
    log = process_pooled_pair(level, target, unit, model_type, device, _WORKER_STATE["X"], _WORKER_STATE["y_by_target"],
                              _WORKER_STATE["cluster_ids"], _WORKER_STATE["checkpoint_dir"], _WORKER_STATE["splits_dir"],
                              tuning_trials=_WORKER_STATE["tuning_trials"], repeats=_WORKER_STATE["repeats"],
                              folds=_WORKER_STATE["folds"], smoke_cap=_WORKER_STATE["smoke_cap"])
    return {"task": list(task), "device": device, "seconds": time.perf_counter() - started,
            "wrote": [str(p) for _, p, wrote in log if wrote], "gpu_index": _WORKER_STATE["worker_gpu"]}


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
    return {"task": list(task), "device": device, "seconds": time.perf_counter() - started, "wrote": [] if not wrote else [str(result[1])],
            "gpu_index": _WORKER_STATE["worker_gpu"]}


def run_pool(tasks, task_fn, csv_path, splits_dir, checkpoint_dir, gpu_indices, workers, tuning_trials, specialist_trials,
            repeats, folds, smoke_cap=False):
    """Run `tasks` through `task_fn`, in this process if workers<=1, else via a `workers`-process pool (a shared work queue)."""
    if workers <= 1:
        _pool_initializer(csv_path, splits_dir, checkpoint_dir, gpu_indices, mp.Value("i", 0), tuning_trials,
                          specialist_trials, repeats, folds, smoke_cap)
        return [task_fn(task) for task in tasks]
    counter = mp.Value("i", 0)
    context = mp.get_context("spawn")
    with context.Pool(processes=workers, initializer=_pool_initializer,
                      initargs=(csv_path, splits_dir, checkpoint_dir, gpu_indices, counter, tuning_trials,
                               specialist_trials, repeats, folds, smoke_cap)) as pool:
        return list(pool.imap_unordered(task_fn, tasks))


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

    started = time.perf_counter()
    if args.role == "gpu-pooled":
        levels = tuple(args.levels.split(",")) if args.levels else LEVELS
        pairs = list_pairs(splits_dir, levels=levels, targets=targets, units=units)
        tasks = [(level, target, unit, model) for level, target, unit in pairs for model in models]
        results = run_pool(tasks, _pooled_task, csv_path, splits_dir, checkpoint_dir, gpu_indices, args.workers,
                           args.tuning_trials, args.specialist_trials, repeats, folds, args.smoke_search_space_cap)
    elif args.role in ("cpu-specialist", "gpu-specialist-control"):
        if args.role == "gpu-specialist-control":
            pairs = (json.loads(args.device_control_pairs) if args.device_control_pairs else DEFAULT_DEVICE_CONTROL_PAIRS)
            suffix = "_gpu_control"
            if not gpu_indices:
                gpu_indices = [0]
        else:
            pairs = list_pairs(splits_dir, levels=("family", "super_family"), targets=targets, units=units)
            suffix = ""
            gpu_indices = None
        rep_range = repeats or list(range(1, R + 1))
        fold_range = folds or list(range(N_FOLDS))
        tasks = [(level, target, unit, model, repeat, fold, suffix)
                for level, target, unit in pairs for model in models for repeat in rep_range for fold in fold_range]
        results = run_pool(tasks, _specialist_task, csv_path, splits_dir, checkpoint_dir, gpu_indices, args.workers,
                           args.tuning_trials, args.specialist_trials, None, None, args.smoke_search_space_cap)
    else:
        raise ValueError(args.role)

    head, clean, dirty = git_state()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_config_dir = checkpoint_dir / "run_configs"
    run_config_dir.mkdir(parents=True, exist_ok=True)
    dataset_sha, dataset_bytes = expected_dataset_identity()
    config = {
        "role": args.role, "workers": args.workers, "gpu_index": gpu_indices,
        "dataset_sha256": dataset_sha, "dataset_bytes": dataset_bytes,
        "labels_run": args.labels_run, "labels_sha256": sha256_file(Path(args.labels_run) / "host_family_labels.csv"),
        "splits_dir": str(splits_dir), "splits_manifest_n_files": len(manifest), "splits_run_config": splits_config,
        "targets": list(targets), "models": list(models), "repeats_filter": repeats, "folds_filter": folds,
        "tuning_trials": args.tuning_trials, "specialist_trials": args.specialist_trials,
        "smoke_search_space_cap": args.smoke_search_space_cap,
        "n_tasks": len(tasks), "n_wrote": sum(1 for r in results if r.get("wrote")), "total_seconds": time.perf_counter() - started,
        "git_head": head, "tree_clean": clean, "dirty_files": dirty, "library_versions": library_versions(), "utc_stamp": stamp,
    }
    config_path = run_config_dir / f"{stamp}_{args.role}.json"
    config_path.write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")
    (run_config_dir / f"{stamp}_{args.role}_results.json").write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"role={args.role} tasks={len(tasks)} wrote={config['n_wrote']} seconds={config['total_seconds']:.1f}")
    print(f"run_config: {config_path}")


if __name__ == "__main__":
    main()
