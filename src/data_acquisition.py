"""
Raw data acquisition from the Starrydata2 bulk dataset export.

Access method (confirmed 2026-08-15 by inspecting starrydata2.org, the
Starrydata GitHub organization, and the release manifest directly):
Starrydata2 exposes three access paths -- a REST API for targeted
per-sample/per-paper/per-figure JSON queries
(https://www.starrydata2.org/api/...), a Figshare project with monthly
archival snapshots, and an official bulk export mirrored to GitHub
Releases at github.com/starrydata/starrydata_datasets, regenerated daily
from the live database and published under the "latest" release tag so
that /releases/download/latest/<file> always resolves to the newest
snapshot. The GitHub release also ships a manifest.json with per-file
row counts and SHA-256 checksums for the upstream db_snapshot. This
module uses the GitHub release path: it is the only one of the three
that hands back complete, checksum-verified table dumps in a single
pull, which is what a "fresh pull, version-dated at extraction" (CLAUDE.md
Phase 0) needs -- the REST API is built for scoped queries, not bulk
extraction, and Figshare's cadence is monthly rather than a controlled
on-demand pull.

Downloads the papers/samples/curves tables for the ThermoelectricMaterials
project split (the project-curated subset relevant to this pipeline's four
target properties), verifies each against the manifest's SHA-256, and
writes the untouched raw files plus an extraction_metadata.json record
(exact UTC extraction timestamp, upstream db_snapshot, source URLs,
checksums, row counts) to data/raw/. No cleaning, filtering, or parsing
of the CSV contents happens here -- see src/data_cleaning.py for the
11-step cleaning pipeline that runs downstream of this raw pull.
"""

import csv
import gzip
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RELEASE_BASE_URL = "https://github.com/starrydata/starrydata_datasets/releases/download/latest"
MANIFEST_URL = f"{RELEASE_BASE_URL}/manifest.json"
PROJECT = "ThermoelectricMaterials"
TABLES = ("papers", "samples", "curves")

RAW_DATA_DIR = Path("data/raw")


def _download(url, dest_path):
    """Stream url to dest_path on disk."""
    with urllib.request.urlopen(url) as response, open(dest_path, "wb") as f:
        f.write(response.read())


def _sha256(path):
    """Compute the SHA-256 hex digest of a file on disk."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_csv_gz_data_rows(path):
    """
    Count data rows (excluding the header) in a gzip CSV via csv.reader,
    not a raw line count -- fields may contain embedded newlines.
    """
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        return sum(1 for _ in reader)


def acquire(raw_data_dir=RAW_DATA_DIR, project=PROJECT):
    """
    Download manifest.json and the papers/samples/curves tables for
    `project` from the Starrydata2 bulk GitHub release, verify each
    downloaded file's SHA-256 against the manifest, and write
    extraction_metadata.json recording the exact extraction timestamp,
    source, upstream db_snapshot, and per-file checksums/row counts.

    Raises ValueError if a downloaded file's checksum does not match
    the manifest. Returns the metadata dict that was written.
    """
    raw_data_dir = Path(raw_data_dir)
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    extraction_timestamp = datetime.now(timezone.utc).isoformat()

    manifest_path = raw_data_dir / "manifest.json"
    _download(MANIFEST_URL, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    project_manifest = manifest["projects"][project]

    files_metadata = {}
    for table in TABLES:
        entry = project_manifest[table]
        filename = entry["filename"]
        url = f"{RELEASE_BASE_URL}/{filename}"
        dest_path = raw_data_dir / filename
        _download(url, dest_path)

        actual_sha256 = _sha256(dest_path)
        if actual_sha256 != entry["sha256"]:
            raise ValueError(
                f"Checksum mismatch for {filename}: "
                f"expected {entry['sha256']}, got {actual_sha256}"
            )

        files_metadata[table] = {
            "filename": filename,
            "url": url,
            "sha256": actual_sha256,
            "manifest_row_count": entry["rows"],
            "counted_row_count": _count_csv_gz_data_rows(dest_path),
        }

    metadata = {
        "source": "Starrydata2 bulk dataset export "
        "(github.com/starrydata/starrydata_datasets, GitHub Releases 'latest' tag)",
        "project": project,
        "extraction_timestamp_utc": extraction_timestamp,
        "upstream_db_snapshot": manifest.get("db_snapshot"),
        "manifest_generated_at": manifest.get("generated_at"),
        "manifest_url": MANIFEST_URL,
        "files": files_metadata,
    }
    (raw_data_dir / "extraction_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


if __name__ == "__main__":
    result = acquire()
    for table, info in result["files"].items():
        print(f"{table}: {info['counted_row_count']} rows ({info['filename']})")
