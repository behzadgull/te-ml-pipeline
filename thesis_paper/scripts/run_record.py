"""
Provenance record for every thesis-paper run (the project's standing rule): the SHA256 and size of each input file, the code commit,
whether the working tree was clean (with the list of dirty files), and the SHA256 of the running script itself, because this
folder's scripts are committed only after the run. Run folders are named with a UTC timestamp.
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GIT = r"C:\Program Files\Git\cmd\git.exe"


def sha256_file(path, lf=False):
    """SHA256 of a file; with lf=True CRLF is converted to LF first (text files)."""
    data = Path(path).read_bytes()
    if lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _git(*args):
    exe = GIT if Path(GIT).exists() else "git"
    return subprocess.run([exe, *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def utc_stamp():
    """YYYYMMDDTHHMMSS in UTC."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def provenance(inputs, script):
    """
    inputs: {name: path relative to the repository root (or absolute)}. script: __file__ of the caller.
    Returns the dict that goes into run_config.json.
    """
    files = {}
    for name, rel in inputs.items():
        p = Path(rel) if Path(rel).is_absolute() else REPO / rel
        files[name] = {"path": str(rel), "sha256": sha256_file(p), "bytes": p.stat().st_size}
    dirty = [line[3:] for line in _git("status", "--porcelain").splitlines()]
    return {
        "inputs": files,
        "git_head": _git("rev-parse", "HEAD").strip(),
        "tree_clean": not dirty,
        "dirty_files": dirty,
        "script": str(Path(script).resolve().relative_to(REPO)).replace("\\", "/"),
        "script_sha256": sha256_file(script, lf=True),
        "python": sys.version.split()[0],
    }


def new_run_dir(analysis):
    """thesis_paper/results/<analysis>/<UTC stamp>/ (created)."""
    d = REPO / "thesis_paper" / "results" / analysis / utc_stamp()
    d.mkdir(parents=True, exist_ok=False)
    return d


def write_json(path, obj):
    """Write JSON with a trailing newline and a readable layout."""
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
