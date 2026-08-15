"""
Global data cleaning pipeline for the fresh Starrydata2 pull.

Implements the 11-step cleaning spec (CLAUDE.md, Data Cleaning Pipeline):
property extraction and range filtering, data integration and
consolidation, temperature filtering and binning, pivot long-to-wide,
formula cleaning via pymatgen, zT self-consistency check, DFT data
removal, multi-source consistency filtering (knee-method thresholds),
MAD outlier filtering, minimum temperature coverage, and the smoothness
filter.

Runs once, globally, before any train/test split (see Grouping Key,
"Global cleaning, not fold-local").
"""
