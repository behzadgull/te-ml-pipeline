#!/usr/bin/env bash
# Set up the thesis-paper GPU runs on the department's V100S machine (Linux, NVIDIA driver with CUDA 12 support).
#
#   1. a conda environment with the versions the Paper A runs used (thesis_paper/scripts/gpu/environment.yml);
#   2. a copy of the pinned dataset on persistent storage, SHA256-checked, linked as data/processed/;
#   3. a checkpoint directory on persistent storage (every run is given --checkpoint-dir under it, so a killed run resumes from its per-fold
#      checkpoints);
#   4. a record of the environment (python, packages, GPU, driver) next to the checkpoints.
# It changes nothing outside the repository checkout and PERSIST, and it does not start any training.
#
# Usage, from the repository root (a clone at the commit you intend to run):
#   PERSIST=/data/thesis_persist DATASET_SRC=/path/to/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv \
#       bash thesis_paper/scripts/gpu/setup_v100s.sh
# Then activate the environment (printed at the end) and run the calibration test (calibrate_v100s.sh).
set -euo pipefail

: "${PERSIST:?set PERSIST to a directory on persistent storage}"
: "${DATASET_SRC:?set DATASET_SRC to the snapfix featurized CSV}"
ENV_NAME="${ENV_NAME:-thesis-te}"
DATASET_NAME="featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv"
DATASET_SHA256="d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489"  # the file every Paper A run and thesis run used
DATASET_BYTES=974854507

repo="$(git rev-parse --show-toplevel)"
cd "$repo"
echo "repository: $repo at $(git rev-parse --short HEAD); working tree $(git status --porcelain | wc -l) changed files"

# 1. conda environment ---------------------------------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found: install Miniconda or load the cluster's conda module, then rerun" >&2
  exit 1
fi
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda environment $ENV_NAME already exists (kept as is)"
else
  conda env create -n "$ENV_NAME" -f thesis_paper/scripts/gpu/environment.yml
fi

# 2. dataset copy + SHA256 check ------------------------------------------------------------------------------------------
mkdir -p "$PERSIST/data/processed" "$PERSIST/checkpoints"
dest="$PERSIST/data/processed/$DATASET_NAME"
if [ ! -f "$dest" ]; then
  cp "$DATASET_SRC" "$dest"
fi
actual="$(sha256sum "$dest" | cut -d' ' -f1)"
size="$(stat -c %s "$dest")"
if [ "$actual" != "$DATASET_SHA256" ] || [ "$size" != "$DATASET_BYTES" ]; then
  echo "dataset check FAILED: sha256 $actual, $size bytes (expected $DATASET_SHA256, $DATASET_BYTES bytes)" >&2
  exit 1
fi
echo "dataset OK: $DATASET_NAME sha256 $actual"
# only the pinned dataset may sit in data/processed: src/nested_cv.py loads the most recent featurized_*.csv it finds there
others="$(ls "$PERSIST/data/processed" | grep -v "^$DATASET_NAME$" || true)"
if [ -n "$others" ]; then
  echo "refusing to continue: other files in $PERSIST/data/processed would be picked up by the loader: $others" >&2
  exit 1
fi

# 3. link data/processed to the persistent copy -----------------------------------------------------------------------------
# (data/ is gitignored, so a fresh clone has none.) Checkpoints are NOT linked: the clone's checkpoints/ holds tracked Paper A files
# (the frozen hyperparameters this work reads), so every run is given an explicit --checkpoint-dir under $PERSIST/checkpoints/.
mkdir -p data
if [ -e data/processed ] && [ ! -L data/processed ]; then
  echo "data/processed exists and is not a link; move it away first" >&2
  exit 1
fi
ln -sfn "$PERSIST/data/processed" data/processed
for t in S sigma kappa zT; do
  test -f "checkpoints/saved_predictions/checkpoints/frozen_hyperparams/$t.json" || { echo "missing committed frozen hyperparameters for $t" >&2; exit 1; }
done
echo "frozen hyperparameters present for S, sigma, kappa, zT"

# 4. environment record ------------------------------------------------------------------------------------------------
rec="$PERSIST/checkpoints/environment_$(date -u +%Y%m%dT%H%M%S).txt"
{
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head: $(git rev-parse HEAD)"
  echo "git_status_porcelain_lines: $(git status --porcelain | wc -l)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 || echo "nvidia-smi not available"
  conda run -n "$ENV_NAME" python -c "import sys, numpy, pandas, sklearn, xgboost, optuna; print('python', sys.version.split()[0]); [print(m.__name__, m.__version__) for m in (numpy, pandas, sklearn, xgboost, optuna)]"
  conda run -n "$ENV_NAME" pip freeze
} > "$rec"
echo "environment recorded in $rec"
echo
echo "next: conda activate $ENV_NAME && PERSIST=$PERSIST bash thesis_paper/scripts/gpu/calibrate_v100s.sh"
