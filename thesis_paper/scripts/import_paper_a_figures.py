"""
Copy the Paper A figures the thesis paper reuses into thesis_paper/figures/paper_a/ (so the thesis folder is self-contained) and record
the source path and SHA256 of each file in thesis_paper/figures/paper_a/MANIFEST.json. The figures are generated from committed
artifacts by scripts/make_figures.py (Paper A); nothing here redraws them.

Usage (from the repository root):
    python thesis_paper/scripts/import_paper_a_figures.py           # copy and write the manifest
    python thesis_paper/scripts/import_paper_a_figures.py --check   # exit 1 if a copy differs from its source
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEST = REPO / "thesis_paper" / "figures" / "paper_a"
FIGURES = {
    "cleaning_funnel": "Eleven-step cleaning funnel (Paper A Figure 3)",
    "fig2_validation_ladder": "Five-way validation ladder (Paper A Figure 5)",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    manifest, bad = {}, []
    for stem, what in FIGURES.items():
        for ext in ("png", "pdf"):
            src = REPO / "figures" / f"{stem}.{ext}"
            dst = DEST / f"{stem}.{ext}"
            manifest[f"{stem}.{ext}"] = {"source": f"figures/{stem}.{ext}", "sha256": sha(src), "what": what}
            if args.check:
                if not dst.exists() or sha(dst) != sha(src):
                    bad.append(dst.name)
            else:
                DEST.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
    if args.check:
        print("copies differ from their sources: " + ", ".join(bad) if bad else "figure copies match their sources")
        return 1 if bad else 0
    (DEST / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"copied {len(manifest)} files to {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
