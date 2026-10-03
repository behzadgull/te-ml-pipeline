# Kaggle cells for the thesis-paper compute

The scripts are pinned to code commit `@@COMMIT@@` (every cell checks out exactly that commit, asserts HEAD equals it and the tree is clean, and
every script refuses to start otherwise). This file was committed afterwards, in a later commit that changes nothing but this file, so the cells'
commit is not this file's own commit; that is intended.

## 0. Files to attach or download

| Session | Accelerator | Attach as notebook input | Previous session's output | Download afterwards |
|---|---|---|---|---|
| S0. Linux smoke test | None (CPU) | snapfix dataset | none | `smoke_report.json` (I verify it, then commit it under `thesis_paper/results/smoke_test_linux/<UTC>/`) |
| G1. NA7 | GPU T4 x2 | snapfix dataset | none (first session) | `na7.tar.gz` (merged result); if the status says incomplete, also `na7_a.tar.gz` and `na7_b.tar.gz` |
| C1. NA2 random forest | None (CPU) | snapfix dataset | none (first session) | `na2_random_forest.tar.gz` |

* **Snapfix dataset**: Kaggle dataset `muhammadbehzadgull/te-ml-pipeline-canonical-dataset-a-snapfix`, file
  `featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv`, 974,854,507 bytes, SHA256
  `d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489`. Found by file name under `/kaggle/input`; the scripts refuse a different SHA256 or size.
* **JARVIS featurised CSV** (not needed for S0, G1, C1; needed from the final-models session on): private Kaggle dataset `thesis-jarvis-featurized`
  holding exactly `jarvis_dft3d_seebeck_featurized.csv` (from `data/external/jarvis/`, 72,098,294 bytes), SHA256
  `3c23d5500fa0d3c6fa6a92313ee61e12fb90f93189697966d5de279adfc49c49` (equal to the `sha256` recorded in `thesis_paper/results/na10/20261003T133155/run_config.json`).
  Path on Kaggle: `/kaggle/input/thesis-jarvis-featurized/jarvis_dft3d_seebeck_featurized.csv`, or, where Kaggle mounts datasets under the owner, 
  `/kaggle/input/datasets/muhammadbehzadgull/thesis-jarvis-featurized/jarvis_dft3d_seebeck_featurized.csv`. The cells find it with `find`, so either works.
* **The code is not an input**: the first cell clones the public repository and checks out the pinned commit. Frozen hyperparameters, committed ladder
  metrics and ESTM smear factors are tracked in git. Git LFS files are not fetched (`GIT_LFS_SKIP_SMUDGE=1`); no Kaggle script reads one.
* **Later sessions of the same analysis** attach the previous session's `.tar.gz` (upload it as a Kaggle dataset, or attach the previous notebook
  version's output) and add `--restore-from <path>`. The restore verifies every file against its manifest SHA256 and refuses it if analysis,
  dataset SHA256, code commit or analysis parameters differ.
* **Logs and outputs are in `/kaggle/working/`, outside the clone**, so the tree stays clean and `tree_clean` is true.
* **Use Save Version (Save & Run All)** for the real runs: only that keeps `/kaggle/working` if the session dies. `--time-budget-hours 10.5` stops
  new units after 10.5 h and writes `complete: false`; Kaggle's session limit is 12 h.
* **Two T4s**: the GPU sessions run two processes at once, one per GPU (`CUDA_VISIBLE_DEVICES=0` and `=1`), each with its own output directory, and the
  cell waits for both. Per-unit times on T4 are unmeasured, so no session counts are stated.

## 1. Session S0 (CPU, Accelerator None): Linux smoke test

Run once before G1 and C1. Expected last line: `all_passed: True`. If any line says FAIL, send me the output and do not start G1 or C1.

Cell 1 (Python): clone at the commit, verify, install the pins.

```python
import os, subprocess, sys
COMMIT = "@@COMMIT@@"
os.makedirs("/kaggle/working/logs", exist_ok=True)
env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
os.chdir("/kaggle/working")
subprocess.run("rm -rf te-ml-pipeline", shell=True, check=True)
subprocess.run(["git", "clone", "--quiet", "https://github.com/behzadgull/te-ml-pipeline.git", "te-ml-pipeline"], check=True, env=env)
os.chdir("te-ml-pipeline")
subprocess.run(["git", "checkout", "--quiet", COMMIT], check=True, env=env)
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
assert head == COMMIT, f"HEAD {head} != {COMMIT}"
assert dirty == "", f"working tree not clean:\n{dirty}"
print("HEAD", head, "tree clean")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy==1.26.4", "pandas==2.2.2", "scipy", "scikit-learn==1.4.2", "xgboost==2.0.3",
                "optuna==3.6.1", "lightgbm==4.3.0"], check=True)
print(subprocess.run([sys.executable, "-c", "import numpy,pandas,sklearn,xgboost,optuna,lightgbm as l;"
                      "print(numpy.__version__,pandas.__version__,sklearn.__version__,xgboost.__version__,optuna.__version__,l.__version__)"],
                     capture_output=True, text=True).stdout)
```

Cell 2 (bash): the smoke test, run the way the real sessions run (no `--allow-dirty`). It fits the four final models but skips the JARVIS scoring
unit (the JARVIS CSV is not in the clone).

```bash
%%bash
cd /kaggle/working/te-ml-pipeline
DS=$(find /kaggle/input -name 'featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv' | head -1)
echo "dataset: $DS"
python thesis_paper/scripts/kaggle/smoke_test_kaggle.py --dataset "$DS" --work-dir /kaggle/working/smoke \
    --expect-commit "$(git rev-parse HEAD)" --report /kaggle/working/smoke_report.json 2>&1 | tee /kaggle/working/logs/smoke.log | tail -70
sha256sum /kaggle/working/smoke_report.json
```

Download `/kaggle/working/smoke_report.json` and send me the printed SHA256.

## 2. Session G1 (GPU T4 x2): NA7, direct vs derived zT under a random row-level split

Accelerator: GPU T4 x2. Internet on. Attach the snapfix dataset. No previous output.

Cell 1: the same clone-and-install cell as S0 (copy it unchanged).

Cell 2 (bash): confirm both GPUs, then run the 25 units as two shards, one per GPU, and wait for both.

```bash
%%bash
cd /kaggle/working/te-ml-pipeline
nvidia-smi --query-gpu=index,name,memory.total --format=csv
H=$(git rev-parse HEAD)
S=thesis_paper/scripts/kaggle/na7_random_dvd.py
CUDA_VISIBLE_DEVICES=0 python $S --out-dir /kaggle/working/na7_a --shard 0/2 --expect-commit $H --device cuda --time-budget-hours 10.5 > /kaggle/working/logs/na7_a.log 2>&1 &
PA=$!
CUDA_VISIBLE_DEVICES=1 python $S --out-dir /kaggle/working/na7_b --shard 1/2 --expect-commit $H --device cuda --time-budget-hours 10.5 > /kaggle/working/logs/na7_b.log 2>&1 &
PB=$!
wait $PA; RA=$?
wait $PB; RB=$?
echo "exit codes: shard a $RA, shard b $RB"
tail -3 /kaggle/working/logs/na7_a.log; tail -3 /kaggle/working/logs/na7_b.log
cat /kaggle/working/na7_a/status.json; echo; cat /kaggle/working/na7_b/status.json; echo
```

Cell 3 (bash): merge the two shards into the final result. `--max-units 0` forbids computing anything here, so the merge can only restore; it writes
`results.json` only if all 25 units are present.

```bash
%%bash
cd /kaggle/working/te-ml-pipeline
python thesis_paper/scripts/kaggle/na7_random_dvd.py --out-dir /kaggle/working/na7 \
    --restore-from /kaggle/working/na7_a.tar.gz,/kaggle/working/na7_b.tar.gz \
    --expect-commit "$(git rev-parse HEAD)" --max-units 0 2>&1 | tee /kaggle/working/logs/na7_merge.log | tail -30
cat /kaggle/working/na7/status.json; echo
ls -la /kaggle/working/na7.tar.gz /kaggle/working/na7_a.tar.gz /kaggle/working/na7_b.tar.gz
sha256sum /kaggle/working/na7.tar.gz
```

Download `na7.tar.gz` (and `na7_a.tar.gz`, `na7_b.tar.gz` if `status.json` of the merge says `"complete": false`). To continue an incomplete run,
start G2 with the same Cell 2, adding `--restore-from <that shard's tar.gz>` to each shard's command (shard a restores `na7_a`, shard b restores
`na7_b`), then repeat Cell 3.

## 3. Session C1 (CPU, Accelerator None): NA2 random forest

Accelerator: None. Internet on. Attach the snapfix dataset. No previous output.

Cell 1: the same clone-and-install cell as S0.

Cell 2 (bash):

```bash
%%bash
cd /kaggle/working/te-ml-pipeline
nproc; free -g | head -2
python thesis_paper/scripts/kaggle/na2_trees.py --model random_forest --out-dir /kaggle/working/na2_random_forest \
    --expect-commit "$(git rev-parse HEAD)" --time-budget-hours 10.5 2>&1 | tee /kaggle/working/logs/na2_random_forest.log
cat /kaggle/working/na2_random_forest/status.json; echo
ls -la /kaggle/working/na2_random_forest.tar.gz && sha256sum /kaggle/working/na2_random_forest.tar.gz
```

Download `na2_random_forest.tar.gz`. If `status.json` says incomplete, start C2 with the same cells and `--restore-from
<that tar.gz as an attached dataset>` added. For each target the script first tunes (Optuna, the project's trial count, each trial a checkpointed
unit), freezes the best set, then runs the 25 chemistry-cluster folds; the first fold of each target is asserted to have the same row counts as the
committed XGBoost rung. Memory can be the limit for a forest at the top of the search space; Save Version keeps what was finished.

## 4. Later sessions (the same Cell 1, then these)

| Order | Session | Accelerator | Cell 2 | Attach |
|---|---|---|---|---|
| 2 | Final models, no classifier, plus NA6, concurrently | GPU T4 x2 | process A (GPU 0): `na6_classifier.py --out-dir /kaggle/working/na6 --device cuda`; process B (GPU 1): `na_final_models.py --out-dir /kaggle/working/final_a --device cuda --jarvis-csv $J --jarvis-sha256 3c23d550...9c49` (`J=$(find /kaggle/input -name jarvis_dft3d_seebeck_featurized.csv)`), both with `--expect-commit`, `&`, `wait` as in G1 | snapfix dataset, `thesis-jarvis-featurized` |
| 3 | NA3 SHAP | GPU T4 x2 | A (GPU 0): `na3_shap.py --targets S,kappa --out-dir /kaggle/working/na3_a`; B (GPU 1): `--targets sigma,zT --out-dir /kaggle/working/na3_b` | snapfix dataset |
| C2 | NA2 LightGBM | CPU | `na2_trees.py --model lightgbm --out-dir /kaggle/working/na2_lightgbm` | snapfix dataset |
| C3 | NA13 feature selection | CPU | `na13_feature_selection.py --out-dir /kaggle/working/na13` | snapfix dataset |
| later | Final models again, with the classifier and the MP candidates | GPU | as the final-models command plus `--classifier-dir <na6 dir> --classifier-sha256 <sha of final_classifier.json> --mp-csv ... --mp-sha256 ...` | snapfix, JARVIS, the na6 output, the MP csv |
| local | NA2 stacking | your PC | `python thesis_paper/scripts/kaggle/na2_stacking.py --rf-dir <tar.gz> --lgbm-dir <tar.gz>` | the two NA2 bundles (committed XGBoost predictions are local) |
| held | NA1 nested CV | department V100S | `thesis_paper/scripts/gpu/` | not run on Kaggle |

NA6 and the first final-models run are different analyses and run side by side. NA3's two halves have disjoint targets and therefore their own
identities and results; they are not merged, and are combined at analysis time.

## 5. What the smoke test proves and does not

Proves (local Windows run; S0 repeats it on Linux): every script runs end to end; every output has a manifest whose SHA256 values match the files;
every run_config records dataset SHA256, git HEAD and tree_clean; a script refuses to start without `--expect-commit` or `--allow-dirty`; a killed
session resumes from a directory or tar.gz with nothing recomputed and results identical to an uninterrupted run; two NA7 shards merge into results
identical to a single run; a restore with different parameters or a tampered unit is refused; a zero time budget stops cleanly and is continued.

Does not prove: XGBoost `device="cuda"` behaviour on the T4 or two simultaneous processes on two GPUs; the time per unit; that tiny-subset numbers mean
anything; that the pip pins resolve on the current Kaggle image (S0 is that check).
