#!/usr/bin/env bash
# Calibration test for the V100S machine: run after setup_v100s.sh, inside the conda environment, from the repository root.
#   PERSIST=/data/thesis_persist bash thesis_paper/scripts/gpu/calibrate_v100s.sh
# It (1) times XGBoost fits at three sizes on the GPU and on the CPU with the frozen zT hyperparameters, (2) converts the timings into
# estimated hours for the planned GPU analyses (NA1, NA2, NA3, NA6, NA7), and (3) reproduces repeat 0 of the committed zT chemistry-cluster run
# (five outer folds, frozen hyperparameters, seed 0) and compares its pooled R2 with the committed value, which shows whether this GPU
# gives the same numbers as the Kaggle P100 that produced Paper A. Nothing is kept except the report and the repeat-0 checkpoints.
set -euo pipefail
: "${PERSIST:?set PERSIST to the persistent directory used by setup_v100s.sh}"
out="$PERSIST/checkpoints/calibration_$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$out"
python thesis_paper/scripts/gpu/calibrate_gpu.py \
  --csv data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv \
  --out-dir "$out" "$@" 2>&1 | tee "$out/calibration.log"
echo "report: $out/calibration_report.json"
