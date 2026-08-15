"""
Pipeline entry point.

Will orchestrate the full run in Build Order (CLAUDE.md): Phase 0 shared
foundation (data cleaning, canonicalization, chemistry-cluster
definition), Phase 1-2 Paper A (validation ladder, noise floor, external
validation, direct-vs-derived zT, screening rediscovery, PCA-split
comparison), and Phase 3 Paper B (family labels, LOFO + controls,
specialist-vs-pooled), reading paths/seeds/thresholds from config.yaml.
"""
