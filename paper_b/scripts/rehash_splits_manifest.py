"""
Regenerate ONLY a splits directory's manifest.json hash values, after the
2026-10-01 line-ending fix (.gitattributes 'paper_b/**/*.json text eol=lf'
etc., plus a forced working-tree recheckout). The split files THEMSELVES
are unchanged: this script proves it before writing anything, not after.

Root cause: make_splits.py's manifest writer hashed the WORKING COPY at
generation time, which on Windows (core.autocrlf=true) was CRLF, even
though the committed git blob was always plain LF. check_splits_manifest
then re-hashes whatever the CURRENT checkout gives it -- correct (LF) on
Linux, which is why Kaggle's SHA256 never matched a manifest value that had
been computed from a Windows CRLF view of the identical content.

For every entry the OLD manifest lists:
  - a .npz (binary, LFS-tracked, never touched by eol handling): its hash
    must be bit-identical to the old one, or this refuses outright.
  - a text file (every other extension here, all .json): the CURRENT
    working-copy bytes must (a) be byte-identical to the committed blob at
    --old-ref (proving this script changed nothing itself), (b) be pure LF
    (no CRLF) already, (c) reproduce the OLD manifest's hash EXACTLY when
    converted back to CRLF (proving the old, broken hash really was just a
    line-ending view of this same content, not different data), and (d)
    parse to the identical JSON value as the old blob (an independent,
    format-aware equality check). Only if every one of these holds does a
    file's entry get the new, LF-based hash.

Also normalises every manifest KEY to forward slashes while it is at it
(paper_b.src.lofo_paperb.normalize_rel_path), fixing the backslash-path
issue in the data itself for this manifest, on top of the read-side
normalisation already in place (2026-09-30) for any manifest that still
has not been regenerated this way.

Run from the repository root:
    python -m paper_b.scripts.rehash_splits_manifest \
        --splits-dir paper_b/results/splits/20260929T113055 --old-ref <commit>
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_b.src.lofo_paperb import normalize_rel_path  # noqa: E402


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def git_show(ref, posix_path):
    """Raw bytes of `posix_path` (forward-slash, repo-root-relative) at `ref`."""
    result = subprocess.run(["git", "show", f"{ref}:{posix_path}"], capture_output=True, check=True)
    return result.stdout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--old-ref", required=True, help="git commit holding the OLD, CRLF-hashed manifest")
    args = parser.parse_args(argv)

    splits_dir = Path(args.splits_dir)
    old_manifest = json.loads((splits_dir / "manifest.json").read_text(encoding="utf-8"))

    new_manifest = {}
    n_binary, n_text = 0, 0
    for raw_rel_path, old_hash in sorted(old_manifest.items()):
        rel_path = normalize_rel_path(raw_rel_path)
        path = splits_dir / rel_path
        new_bytes = path.read_bytes()

        if path.suffix == ".npz":
            if sha256_bytes(new_bytes) != old_hash:
                raise ValueError(f"{rel_path}: binary file's hash changed unexpectedly (expected no change at all)")
            new_manifest[rel_path] = old_hash
            n_binary += 1
            continue

        repo_posix_path = f"{splits_dir.as_posix()}/{rel_path}"
        old_blob = git_show(args.old_ref, repo_posix_path)
        if old_blob != new_bytes:
            raise ValueError(
                f"{rel_path}: working-copy content differs from the committed blob at {args.old_ref} "
                "(expected byte-identical -- this script must not change any split's data)"
            )
        if b"\r\n" in new_bytes:
            raise ValueError(f"{rel_path}: still CRLF in the working copy; run the .gitattributes recheckout first")
        crlf_bytes = new_bytes.replace(b"\n", b"\r\n")
        reconstructed_old_hash = sha256_bytes(crlf_bytes)
        if reconstructed_old_hash != old_hash:
            raise ValueError(
                f"{rel_path}: the old manifest's hash ({old_hash}) is NOT reproduced by CRLF-converting the "
                f"current content (got {reconstructed_old_hash}) -- this is not a pure line-ending difference, "
                "refusing to silently replace it"
            )
        if json.loads(old_blob.decode("utf-8")) != json.loads(new_bytes.decode("utf-8")):
            raise ValueError(f"{rel_path}: parsed JSON content differs between the old blob and the current file")

        new_manifest[rel_path] = sha256_bytes(new_bytes)
        n_text += 1

    assert n_binary + n_text == len(old_manifest), f"checked {n_binary + n_text} of {len(old_manifest)} entries"
    (splits_dir / "manifest.json").write_text(json.dumps(new_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{splits_dir}/manifest.json: {n_text} text entries re-hashed (LF basis, keys normalised to forward "
          f"slash), {n_binary} binary entries unchanged, every one verified content-identical to {args.old_ref}")


if __name__ == "__main__":
    main()
