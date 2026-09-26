# Paper B shared dependencies

Paper B lives under `paper_b/` so it can be handed over separately. It may
import exactly the two top-level modules below and nothing else from `src/`.
Every hash is the SHA256 of the file as committed at `6f88f9d` (the last
commit before the one that added this manifest; neither module is changed by
the commit that adds it), taken over the file with CRLF converted to LF, so a
Windows checkout with `core.autocrlf=true` hashes the same as the committed
blob.

`paper_b/scripts/check_shared_dependencies.py` verifies the hashes below and
that no file under `paper_b/` imports any other `src.*` module. Run it from
the repository root before every Paper B commit. If a listed module changes,
update its hash here in the same commit and say why.

## Shared code

| Module | SHA256 | What Paper B may use |
|---|---|---|
| `src/canonicalization.py` | `4d1b427e68a72ad95ffe5588d54b76467e350451e335e97dcb13b326efb9dfb1` | `parse_formula`, `composition_id`, `chemistry_cluster_id`, `is_degenerate_cluster`, `DEFAULT_DOPANT_THRESHOLD_FRAC`, `DEFAULT_SNAP_TOLERANCE`. Imports only pymatgen. |
| `src/nested_cv.py` | `ca3787ff5e36750aaa6062be3fdf5ec293f6584ca103a64a25322a2283b73254` | Modelling machinery: `MODEL_REGISTRY`, `tune_hyperparameters`, `get_feature_columns`, `randomized_group_kfold`, `GROUP_COL`, `TEMPERATURE_COL`, `LOG_TRANSFORM_TARGETS`, `N_INNER_FOLDS`, `N_OPTUNA_TRIALS`, `nadeau_bengio_test`. Imports only third-party packages (numpy, pandas, optuna, xgboost, scikit-learn). |

## Data loading

Paper B does NOT use `src.nested_cv.load_target_data`. That function reads
"the most recent `featurized_<project>_*.csv`" by a glob, so which file it
loads depends on what else sits in `data/processed/`; this is the failure mode
CLAUDE.md records for the cleaned-CSV globs. Paper B loads the dataset by
explicit path with a SHA256 check, in code under `paper_b/src/`, and filters
each target to `df[target].notna()` as Paper A does.

| Dataset | Path | SHA256 | Bytes |
|---|---|---|---|
| snapfix featurized CSV | `data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv` | `d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489` | 974,854,507 |

Row counts per target after the `notna` filter: S 185,064, sigma 182,755,
kappa 121,110, zT 129,419 (CLAUDE.md, Canonical Dataset).

## Not shared

Paper B does not import `data_acquisition`, `data_cleaning`, `featurization`,
`external_validation`, `direct_vs_derived_zt`, `backtransform_check`,
`noise_floor`, `plotting_style`, `screening` or `validation_ladder`, nor any
file under `scripts/`. Paper A's frozen hyperparameters are not used either:
Paper B retunes inside each LOFO fold (methodology doc, section 7 (c)).
