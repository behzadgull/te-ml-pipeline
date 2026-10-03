"""
Common harness for the thesis-paper compute scripts that run on Kaggle (and on any Linux or Windows machine).

Every script built on it gets the same guarantees:
  * code identity: the run records the git HEAD and whether the working tree was clean, and REFUSES to start unless HEAD equals --expect-commit
    and the tree is clean (--allow-dirty exists for local development and smoke tests; a run made with it records allow_dirty = true and is
    never accepted as a result);
  * data identity: the dataset path is given explicitly or found by file name under /kaggle/input, and its SHA256 and size are checked against
    the pinned snapfix CSV before any model is fitted;
  * units: the work is a list of units (for example one outer fold). Each finished unit is written at once (npz plus json) and listed in
    manifest.json with its SHA256, so a killed session loses at most one unit;
  * resume: --restore-from takes a previous session's output directory or .tar.gz, verifies every file against its manifest, refuses a restore
    whose recorded identity (analysis, dataset, code commit, analysis parameters) differs from this run's, and copies the finished units in;
  * budget: --time-budget-hours stops starting new units after that time and writes status.json with complete = false, so the next session continues;
  * bundle: at the end <out-dir>.tar.gz holds everything, ready to download or to attach as the next session's input.

This module imports nothing from the project except through the explicit paths it is given; scripts add the repository root to sys.path themselves.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_NAME = "featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv"
DATASET_SHA256 = "d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489"
DATASET_BYTES = 974854507
IDENTITY_FIELDS = ("analysis", "dataset_sha256", "git_head", "params")


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def repo_root():
    return Path(__file__).resolve().parents[3]


def _git(*args):
    return subprocess.run(["git", *args], cwd=repo_root(), capture_output=True, text=True, check=True).stdout


def git_state():
    """(head, tree_clean, dirty files)."""
    dirty = [l[3:] for l in _git("status", "--porcelain").splitlines()]
    return _git("rev-parse", "HEAD").strip(), not dirty, dirty


def find_dataset(explicit=None):
    """The snapfix CSV: the given path, or the one file of that name under /kaggle/input."""
    if explicit:
        return Path(explicit)
    hits = sorted(Path("/kaggle/input").rglob(DATASET_NAME)) if Path("/kaggle/input").exists() else []
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one {DATASET_NAME} under /kaggle/input, found {len(hits)}; pass --dataset")
    return hits[0]


def add_common_args(ap):
    """Arguments every script has."""
    ap.add_argument("--out-dir", required=True, help="this session's output directory (units, manifest, status, run_config)")
    ap.add_argument("--dataset", default=None, help=f"path to {DATASET_NAME} (default: searched under /kaggle/input)")
    ap.add_argument("--expect-commit", default=None, help="full git SHA this run must be at")
    ap.add_argument("--allow-dirty", action="store_true", help="local development and smoke tests only; the run is recorded as not accepted")
    ap.add_argument("--restore-from", default=None, help="previous session's output directory or .tar.gz; several, comma-separated, are merged (each is verified and identity-checked)")
    ap.add_argument("--time-budget-hours", type=float, default=None, help="stop starting new units after this many hours")
    ap.add_argument("--smoke", action="store_true", help="tiny subset and tiny models, for plumbing tests (never a result)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--max-units", type=int, default=None, help="testing only: stop after computing this many units in this session")


def smoke_subset(df, group_col, n_rows, seed=0):
    """A deterministic subset made of whole chemistry clusters, so that grouped folds still work."""
    rng = np.random.default_rng(seed)
    groups = df[group_col].to_numpy()
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    sizes = pd.Series(groups).value_counts()
    chosen, total = [], 0
    for g in uniq:
        chosen.append(g)
        total += int(sizes[g])
        if total >= n_rows and len(chosen) >= 20:
            break
    return df[df[group_col].isin(set(chosen))].reset_index(drop=True)


class Session:
    """One run of one analysis: identity, units, restore, budget, bundle."""

    def __init__(self, analysis, args, params):
        self.analysis, self.args, self.params = analysis, args, dict(params)
        self.t0 = time.perf_counter()
        self.out = Path(args.out_dir)
        self.units_dir = self.out / "units"
        self.out.mkdir(parents=True, exist_ok=True)
        self.units_dir.mkdir(exist_ok=True)
        self.restored, self.computed = [], []
        head, clean, dirty = git_state()
        if not args.allow_dirty:
            if not args.expect_commit:
                raise SystemExit("--expect-commit is required (or --allow-dirty for local development)")
            if head != args.expect_commit:
                raise SystemExit(f"HEAD is {head}, expected {args.expect_commit}")
            if not clean:
                raise SystemExit(f"working tree is not clean: {dirty[:5]}")
        self.ds_path = find_dataset(args.dataset)
        sha, size = sha256_file(self.ds_path), self.ds_path.stat().st_size
        if (sha, size) != (DATASET_SHA256, DATASET_BYTES):
            raise SystemExit(f"dataset {self.ds_path} is not the pinned snapfix CSV: sha256 {sha}, {size} bytes")
        script = Path(sys.argv[0]).resolve()
        self.identity = {"analysis": analysis, "dataset_sha256": sha, "git_head": head, "params": self.params}
        self.run_config = {**self.identity, "dataset_path": str(self.ds_path), "dataset_bytes": size, "tree_clean": clean, "dirty_files": dirty,
                           "allow_dirty": bool(args.allow_dirty), "expect_commit": args.expect_commit, "smoke": bool(args.smoke), "device": args.device,
                           "script": str(script.relative_to(repo_root())).replace("\\", "/"), "script_sha256": sha256_file(script),
                           "python": sys.version.split()[0], "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "numpy": np.__version__, "pandas": pd.__version__}
        for mod in ("xgboost", "sklearn", "lightgbm", "optuna"):
            try:
                self.run_config[mod] = __import__(mod).__version__
            except Exception:
                pass
        self.run_config["gpu"] = _gpu()
        sessions = self.out / "run_configs"
        sessions.mkdir(exist_ok=True)
        n_prev = len(list(sessions.glob("session_*.json")))
        self._session_file = sessions / f"session_{n_prev + 1:02d}.json"
        print(f"[{analysis}] HEAD {head[:10]} clean={clean} allow_dirty={args.allow_dirty} dataset sha ok ({size:,} bytes) smoke={args.smoke}", flush=True)
        for src in [x for x in (args.restore_from or "").split(",") if x]:
            self.restore(src)

    # -- identity ------------------------------------------------------------------------------------------------------
    def _check_identity(self, other):
        bad = [k for k in IDENTITY_FIELDS if other.get(k) != self.identity.get(k)]
        if bad:
            raise SystemExit(f"restore refused: the previous session differs on {bad}")

    # -- units ---------------------------------------------------------------------------------------------------------
    def _paths(self, uid):
        safe = uid.replace("/", "__")
        return self.units_dir / f"{safe}.json", self.units_dir / f"{safe}.npz"

    def done(self, uid):
        meta, _ = self._paths(uid)
        return meta.exists()

    def save(self, uid, meta=None, arrays=None):
        """Write a finished unit (arrays first, then the json that marks it done) and refresh the manifest."""
        jpath, npath = self._paths(uid)
        if arrays:
            tmp = npath.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, npath)
        tmpj = jpath.with_suffix(".tmp.json")
        tmpj.write_text(json.dumps({"unit": uid, "meta": meta or {}, "has_arrays": bool(arrays)}, indent=1), encoding="utf-8")
        os.replace(tmpj, jpath)
        self.computed.append(uid)
        self.write_manifest()

    def load(self, uid):
        """(meta dict, arrays dict or None) of a finished unit."""
        jpath, npath = self._paths(uid)
        rec = json.loads(jpath.read_text(encoding="utf-8"))
        arrays = None
        if rec["has_arrays"]:
            with np.load(npath, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
        return rec["meta"], arrays

    def write_manifest(self):
        entries = {}
        for p in sorted(self.out.rglob("*")):
            if p.is_file() and p.name not in ("manifest.json",) and not p.name.endswith(".tmp.npz") and not p.name.endswith(".tmp.json"):
                entries[str(p.relative_to(self.out)).replace("\\", "/")] = sha256_file(p)
        tmp = self.out / "manifest.json.tmp"
        tmp.write_text(json.dumps({"files": entries, "identity": self.identity}, indent=1), encoding="utf-8")
        os.replace(tmp, self.out / "manifest.json")

    # -- restore -------------------------------------------------------------------------------------------------------
    def restore(self, path):
        path = Path(path)
        tmp = None
        if path.is_file():
            tmp = Path(tempfile.mkdtemp(prefix="restore_"))
            with tarfile.open(path, "r:gz") as tar:
                for m in tar.getmembers():
                    target = (tmp / m.name).resolve()
                    if not str(target).startswith(str(tmp.resolve())) or m.issym() or m.islnk():
                        raise SystemExit(f"unsafe member in {path}: {m.name}")
                tar.extractall(tmp)
            roots = [p for p in tmp.iterdir() if p.is_dir()]
            src = roots[0] if len(roots) == 1 and (roots[0] / "manifest.json").exists() else tmp
        else:
            src = path
        mf = src / "manifest.json"
        if not mf.exists():
            raise SystemExit(f"{path}: no manifest.json (every session writes one)")
        manifest = json.loads(mf.read_text(encoding="utf-8"))
        self._check_identity(manifest["identity"])
        copied = 0
        for rel, sha in manifest["files"].items():
            f = src / rel
            if not f.exists():
                raise SystemExit(f"{path}: manifest lists {rel} but it is missing")
            if sha256_file(f) != sha:
                raise SystemExit(f"{path}: {rel} does not match its manifest SHA256; refusing a possibly altered restore")
            dst = self.out / rel
            if rel.startswith("run_configs/") and dst.exists():
                k = 1
                while (self.out / f"{rel[:-5]}_restored{k}.json").exists():
                    k += 1
                dst = self.out / f"{rel[:-5]}_restored{k}.json"
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, dst)
                copied += 1
                if rel.startswith("units/") and rel.endswith(".json"):
                    self.restored.append(Path(rel).stem)
        n_prev = len(list((self.out / "run_configs").glob("session_*.json")))
        self._session_file = self.out / "run_configs" / f"session_{n_prev + 1:02d}.json"
        print(f"restored {copied} files ({len(self.restored)} units) from {path}", flush=True)
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    # -- budget, finish ------------------------------------------------------------------------------------------------
    def out_of_time(self):
        if self.args.max_units is not None and len(self.computed) >= self.args.max_units:
            return True
        b = self.args.time_budget_hours
        return b is not None and (time.perf_counter() - self.t0) / 3600.0 >= b

    def finish(self, units_total, results=None, bundle=True):
        """Write status.json, this session's run_config, the manifest, and the bundle; return True if complete."""
        all_done = [u for u in self._all_units()]
        complete = len(all_done) >= units_total
        self.run_config["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.run_config["elapsed_hours"] = (time.perf_counter() - self.t0) / 3600.0
        self._session_file.write_text(json.dumps(self.run_config, indent=1), encoding="utf-8")
        status = {"complete": complete, "units_total": units_total, "units_done": len(all_done), "units_computed_this_session": len(self.computed),
                  "units_restored_this_session": len(self.restored), "accepted_as_result": complete and not self.args.allow_dirty and not self.args.smoke}
        (self.out / "status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
        if results is not None:
            (self.out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        self.write_manifest()
        if bundle:
            tgz = self.out.with_suffix(".tar.gz") if self.out.suffix == "" else Path(str(self.out) + ".tar.gz")
            with tarfile.open(tgz, "w:gz") as tar:
                tar.add(self.out, arcname=self.out.name)
            print(f"bundle: {tgz} sha256 {sha256_file(tgz)}", flush=True)
        print(f"status: {json.dumps(status)}", flush=True)
        return complete

    def _all_units(self):
        return [p.stem for p in self.units_dir.glob("*.json") if not p.name.endswith(".tmp.json")]


def _gpu():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def load_frame(session, columns=None):
    """The snapfix CSV (whole, or the given columns); in smoke mode reduced by the caller with smoke_subset."""
    return pd.read_csv(session.ds_path, usecols=columns)


def r2(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return 1.0 - float(np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
