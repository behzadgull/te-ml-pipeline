"""
Before a restart after a power cut: find checkpoint files that are not readable and move them to a quarantine folder, so the resumable runs recompute them.

NTFS can leave a file that was written and renamed just before the power went as an empty or truncated file. Both harnesses treat "the file exists" as "the unit is done", so such a file
would be skipped forever. Only files modified in the last --since-hours are read (a power cut damages only the latest writes); --since-hours 0 reads everything.

  --stacking-dir   an na2_stacking_nested.py out-dir: every units/*.json must parse and hold {"unit", "meta"} (a stacking unit has no .npz)
  --paperb-dir     a paper_b lofo_paperb.py checkpoint dir (repeatable): every .npz must pass the zip CRC test and every .json sidecar must parse
Exit code 0 always (the point is the quarantine); the number moved is printed.

Usage:  python verify_outputs.py --stacking-dir <dir> --paperb-dir <dir> --quarantine <dir> [--since-hours 72]
"""

import argparse
import json
import shutil
import time
import zipfile
from pathlib import Path

SKIP_JSON_PREFIX = ("status", "run_config", "manifest", "session_", "task_", "worker")


def bad_json(path, need_keys=()):
    """True if the file does not parse as JSON (or lacks the required keys)."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return any(k not in d for k in need_keys) if need_keys else False
    except Exception:
        return True


def bad_npz(path):
    """True if the file is not a complete .npz (zip) archive."""
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is not None
    except Exception:
        return True


def quarantine(path, root, qdir, moved):
    """Move a file under qdir, keeping its path relative to root."""
    dest = Path(qdir) / path.relative_to(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dest))
    moved.append(str(path))


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--stacking-dir", default=None)
    ap.add_argument("--paperb-dir", action="append", default=[])
    ap.add_argument("--quarantine", required=True)
    ap.add_argument("--since-hours", type=float, default=72.0)
    args = ap.parse_args()
    cutoff = time.time() - args.since_hours * 3600 if args.since_hours > 0 else 0.0
    moved, checked = [], 0
    if args.stacking_dir and (Path(args.stacking_dir) / "units").exists():
        root = Path(args.stacking_dir)
        for p in sorted((root / "units").glob("*.json")):
            if p.name.endswith(".tmp.json") or p.stat().st_mtime < cutoff:
                continue
            checked += 1
            if bad_json(p, ("unit", "meta")):
                quarantine(p, root.parent, args.quarantine, moved)
    for d in args.paperb_dir:
        root = Path(d)
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.stat().st_mtime < cutoff:
                continue
            if p.suffix == ".npz" and ".tmp" not in p.name:
                checked += 1
                if bad_npz(p):
                    quarantine(p, root.parent, args.quarantine, moved)
                    side = p.with_suffix(".json")
                    if side.exists():
                        quarantine(side, root.parent, args.quarantine, moved)
            elif p.suffix == ".json" and not p.name.startswith(SKIP_JSON_PREFIX) and p.with_suffix(".npz").exists():
                checked += 1
                if bad_json(p):
                    quarantine(p, root.parent, args.quarantine, moved)
                    npz = p.with_suffix(".npz")
                    if npz.exists():
                        quarantine(npz, root.parent, args.quarantine, moved)
    print(f"verify_outputs: {checked} file(s) read (modified in the last {args.since_hours:g} h), {len(moved)} moved to {args.quarantine}")
    for m in moved:
        print("  quarantined", m)


if __name__ == "__main__":
    main()
