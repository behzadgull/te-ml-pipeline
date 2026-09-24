# Raw pull record for the canonical Starrydata2 snapshot (File A)

`extraction_metadata.json` is the record written by `src/data_acquisition.py` when the raw
Starrydata2 export was pulled. It is a byte-for-byte copy of the original, which lives in the
gitignored, local-only raw-data directory:

    checkpoints/saved_predictions/te-ml-pipeline/data/raw/extraction_metadata.json
    (absolute path on the maintainer's machine:
    C:\Users\choha\te-ml-pipeline\checkpoints\saved_predictions\te-ml-pipeline\data\raw\extraction_metadata.json)

The copy exists so that scripts and figures can read the snapshot date and raw counts from a
committed file instead of a gitignored one.

| | |
|---|---|
| SHA256 | `d101d6675ceb15190a435da19783b9bbdba616caebb46009275ca1429653c1e3` |
| Size | 1,575 bytes |
| `extraction_timestamp_utc` | 2026-08-22T07:05:52.366299+00:00 (when the pull ran) |
| `upstream_db_snapshot` | 2026-08-22 02:00:02 UTC+0900 (JST) (the source database's own snapshot label) |
| `manifest_generated_at` | 2026-08-21T18:47:36.105712+00:00 |
| Rows counted | papers 9,494; samples 55,261; curves 156,101 |

02:00 JST on 22 August is 17:00 UTC on 21 August, so "22 August 2026" is the date in the source
database's own JST label and is also the (UTC) date of the pull.

The raw data files themselves (`*.csv.gz`, with their own SHA256 values listed inside the JSON)
are not in the repository. `.gitattributes` marks this JSON `-text` so that line-ending
conversion cannot change its bytes; check the hash above after any checkout.

Read by `make_study_overview` in `scripts/make_figures.py`, which asserts the SHA256.
