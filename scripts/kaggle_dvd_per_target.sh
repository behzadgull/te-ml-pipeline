#!/usr/bin/env bash
# Direct-vs-derived zT rerun with each component model's OWN frozen hyperparameters,
# plus a same-device control with zT's set shared (the published configuration).
# For a Kaggle notebook cell:   !COMMIT=<sha> SNAPFIX_CSV=<path> bash scripts/kaggle_dvd_per_target.sh
# (run it from /kaggle/working, GPU on, internet on). Not yet run.
#
# Required environment:
#   COMMIT      full git SHA of a commit that contains this script and the per_target mode in
#               src/direct_vs_derived_zt.py. Commit and push those changes first.
#   SNAPFIX_CSV path to featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv on Kaggle
#               (NOT the canonical-dataset-a upload: that is File A with the pre-snapfix
#               chemistry_cluster_id).
# Optional: REPO_URL (default: the GitHub repo). If the repo is private, use a token in the URL.
set -euo pipefail

: "${COMMIT:?set COMMIT to the full git SHA to run}"
: "${SNAPFIX_CSV:?set SNAPFIX_CSV to the snapfix featurized CSV path}"
REPO_URL="${REPO_URL:-https://github.com/behzadgull/te-ml-pipeline.git}"
EXPECT_SHA=d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489

# 1. code at the exact commit, clean tree (LFS objects are not needed here)
if [ ! -d te-ml-pipeline ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone "$REPO_URL" te-ml-pipeline
fi
cd te-ml-pipeline
git fetch origin
GIT_LFS_SKIP_SMUDGE=1 git checkout --detach "$COMMIT"
test "$(git rev-parse HEAD)" = "$COMMIT"
test -z "$(git status --porcelain)" || { echo "working tree is not clean"; exit 1; }
test -f checkpoints/saved_predictions/checkpoints/frozen_hyperparams/S.json

# 2. pinned environment (from results/ladder_regen_snapfix/*/environment.txt)
pip install -q numpy==1.26.4 xgboost==2.0.3 scikit-learn==1.4.2 optuna==3.6.1

# 3. the dataset: verify identity before anything runs
echo "$EXPECT_SHA  $SNAPFIX_CSV" | sha256sum -c -
mkdir -p data/processed
ln -sf "$SNAPFIX_CSV" data/processed/featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv
ls data/processed   # must list only the snapfix CSV: load_all_four_subset() takes the last glob match

nvidia-smi -L

# 4. control first (published configuration, same device), then the per-target run
for MODE in zt_shared per_target; do
  CK="checkpoints/direct_vs_derived_zt_snapfix_${MODE}_cuda"
  python -m src.direct_vs_derived_zt --hyperparams "$MODE" --device cuda --checkpoint-dir "$CK" \
    2>&1 | tee "dvd_${MODE}.log"
  python -m src.backtransform_check --checkpoint-dir "$CK" --results-json "$CK/backtransform_check_results.json" \
    --no-figure 2>&1 | tee "btc_${MODE}.log"
done

# 5. bundle everything to download (run_config.json, results.json, the check JSONs, logs, per-fold npz)
tar -czf /kaggle/working/dvd_per_target_bundle.tar.gz \
  checkpoints/direct_vs_derived_zt_snapfix_zt_shared_cuda checkpoints/direct_vs_derived_zt_snapfix_per_target_cuda \
  dvd_zt_shared.log dvd_per_target.log btc_zt_shared.log btc_per_target.log
sha256sum /kaggle/working/dvd_per_target_bundle.tar.gz
