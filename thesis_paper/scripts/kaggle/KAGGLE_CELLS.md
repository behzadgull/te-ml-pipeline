# Kaggle cells for the thesis-paper compute

All remaining sessions (S0, the random-forest and LightGBM sessions of section 3, G4 and G5 of section 3b) are pinned to ONE code commit, `7529e7cfb7f14341994d1f57123e11edc75443e4` (every cell checks out exactly that commit,
asserts HEAD equals it and the tree is clean, and every script refuses to start otherwise). Sessions G1, G2 and G3 (sections 2, 2b, 2c) have already run, at the earlier commits stated there. This file was committed afterwards, in a later commit that changes nothing but this file, so the cells'
commit is not this file's own commit; that is intended.

## 0. Files to attach or download

| Session | Accelerator | Attach as notebook input | Previous session's output | Download afterwards |
|---|---|---|---|---|
| S0. Linux smoke test | None (CPU) | snapfix dataset | none | `smoke_report.json` (I verify it, then commit it under `thesis_paper/results/smoke_test_linux/<UTC>/`) |
| G1. NA7 | GPU T4 x2 | snapfix dataset | none (first session) | `na7.tar.gz` (merged result); if the status says incomplete, also `na7_a.tar.gz` and `na7_b.tar.gz` |
| R-S, R-sigma, R-kappa, R-zT, L. NA2 random forest (fixed setting) and LightGBM, see section 3 | None (CPU) | snapfix dataset | none | `na2_rf_<target>.tar.gz` x 4, `na2_lightgbm.tar.gz` |

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

Run once on commit `7529e7cfb7f14341994d1f57123e11edc75443e4` before any remaining session (R-S, R-sigma, R-kappa, R-zT, L, G4, G5). Its smoke test covers the fixed-setting random forest, `na3_rows.py` and `na1_nested_cv.py`.
Expected last line: `all_passed: True`. If any line says FAIL, send me the output and do not start the other sessions.

Cell 1 (Python): clone at the commit, verify, install the pins.

```python
import os, subprocess, sys
COMMIT = "7529e7cfb7f14341994d1f57123e11edc75443e4"
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

# The Kaggle image is Python 3.13 (as of 2026-10-03) and the pinned numpy 1.26.4 / pandas 2.2.2 / scikit-learn 1.4.2 have no cp313 wheels.
# The pins stay (the Paper A results were produced with them), so everything runs in a uv-managed Python 3.12 venv.
VENV = "/kaggle/working/venv"
PY = f"{VENV}/bin/python"
PINS = ["numpy==1.26.4", "pandas==2.2.2", "scipy", "scikit-learn==1.4.2", "xgboost==2.0.3", "optuna==3.6.1", "lightgbm==4.3.0"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
subprocess.run([sys.executable, "-m", "uv", "venv", VENV, "--python", "3.12"], check=True)
subprocess.run([sys.executable, "-m", "uv", "pip", "install", "--python", PY, *PINS], check=True)
probe = subprocess.run([PY, "-c", """
import sys, importlib.metadata as m
print("python", sys.version.split()[0])
assert sys.version_info[:2] == (3, 12), sys.version
want = {"numpy": "1.26.4", "pandas": "2.2.2", "scikit-learn": "1.4.2", "xgboost": "2.0.3", "optuna": "3.6.1", "lightgbm": "4.3.0"}
for d in sorted(m.distributions(), key=lambda d: d.metadata["Name"].lower()):
    n = d.metadata["Name"]
    print(f"  {n}=={d.version}")
    if n.lower() in want:
        assert d.version == want.pop(n.lower()), (n, d.version)
assert not want, f"not installed: {want}"
"""], capture_output=True, text=True)
print(probe.stdout, probe.stderr)
assert probe.returncode == 0, "the venv is not Python 3.12 with the exact pins"
```

Every later cell calls `/kaggle/working/venv/bin/python` (never a bare `python`).
Save Version runs every cell from the top, so Cell 1 rebuilds the venv each time; in an interactive session, rerun Cell 1 after a restart.

Cell 2 (bash): the smoke test, run the way the real sessions run (no `--allow-dirty`). It fits the four final models but skips the JARVIS scoring
unit (the JARVIS CSV is not in the clone).

```bash
%%bash
cd /kaggle/working/te-ml-pipeline
DS=$(find /kaggle/input -name 'featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv' | head -1)
echo "dataset: $DS"
/kaggle/working/venv/bin/python thesis_paper/scripts/kaggle/smoke_test_kaggle.py --dataset "$DS" --work-dir /kaggle/working/smoke \
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
# the xgboost 2.0.3 wheel must have CUDA support: a tiny GPU fit on each device, before anything long starts
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g /kaggle/working/venv/bin/python -c "import numpy as n, xgboost as x; x.XGBRegressor(n_estimators=5, device='cuda', tree_method='hist').fit(n.random.rand(500, 8), n.random.rand(500)); print('GPU $g: xgboost', x.__version__, 'cuda fit ok')" || { echo "GPU $g: xgboost cuda fit FAILED"; exit 1; }
done
H=$(git rev-parse HEAD)
S=thesis_paper/scripts/kaggle/na7_random_dvd.py
CUDA_VISIBLE_DEVICES=0 /kaggle/working/venv/bin/python $S --out-dir /kaggle/working/na7_a --shard 0/2 --expect-commit $H --device cuda --time-budget-hours 10.5 > /kaggle/working/logs/na7_a.log 2>&1 &
PA=$!
CUDA_VISIBLE_DEVICES=1 /kaggle/working/venv/bin/python $S --out-dir /kaggle/working/na7_b --shard 1/2 --expect-commit $H --device cuda --time-budget-hours 10.5 > /kaggle/working/logs/na7_b.log 2>&1 &
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
/kaggle/working/venv/bin/python thesis_paper/scripts/kaggle/na7_random_dvd.py --out-dir /kaggle/working/na7 \
    --restore-from /kaggle/working/na7_a.tar.gz,/kaggle/working/na7_b.tar.gz \
    --expect-commit "$(git rev-parse HEAD)" --max-units 0 2>&1 | tee /kaggle/working/logs/na7_merge.log | tail -30
cat /kaggle/working/na7/status.json; echo
ls -la /kaggle/working/na7.tar.gz /kaggle/working/na7_a.tar.gz /kaggle/working/na7_b.tar.gz
sha256sum /kaggle/working/na7.tar.gz
```

Download `na7.tar.gz` (and `na7_a.tar.gz`, `na7_b.tar.gz` if `status.json` of the merge says `"complete": false`). To continue an incomplete run,
start a further NA7 session with the same Cell 2, adding `--restore-from <that shard's tar.gz>` to each shard's command (shard a restores `na7_a`, shard b restores
`na7_b`), then repeat Cell 3.

## 2b. Session G2 (GPU T4 x2): NA6 classifier on GPU 0, final models (no classifier) on GPU 1

Accelerator: GPU T4 x2. Internet on. Attach two datasets: the snapfix dataset and the private dataset `thesis-jarvis-featurized`. No previous output.
`thesis-jarvis-featurized` must hold exactly one file, `jarvis_dft3d_seebeck_featurized.csv` (72,098,294 bytes, SHA256
`3c23d5500fa0d3c6fa6a92313ee61e12fb90f93189697966d5de279adfc49c49`; local path `C:\Users\choha\te-ml-pipeline\data\external\jarvis\jarvis_dft3d_seebeck_featurized.csv`).
The cell finds it with `find`, and the script refuses it if the SHA256 differs.

Cell 1: the same clone-and-install cell as S0 (copy it unchanged).

Cell 2 (bash): both GPUs checked, then the two analyses run side by side and the cell waits for both.

```bash
%%bash
cd /kaggle/working/te-ml-pipeline
PYV=/kaggle/working/venv/bin/python
nvidia-smi --query-gpu=index,name,memory.total --format=csv
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g $PYV -c "import numpy as n, xgboost as x; x.XGBRegressor(n_estimators=5, device='cuda', tree_method='hist').fit(n.random.rand(500, 8), n.random.rand(500)); print('GPU $g: xgboost', x.__version__, 'cuda fit ok')" || { echo "GPU $g: xgboost cuda fit FAILED"; exit 1; }
done
H=$(git rev-parse HEAD)
J=$(find /kaggle/input -name jarvis_dft3d_seebeck_featurized.csv | head -1)
echo "jarvis csv: $J"
sha256sum "$J"
K=thesis_paper/scripts/kaggle
CUDA_VISIBLE_DEVICES=0 $PYV $K/na6_classifier.py --out-dir /kaggle/working/na6 --expect-commit $H --device cuda --time-budget-hours 10.5 > /kaggle/working/logs/na6.log 2>&1 &
PA=$!
CUDA_VISIBLE_DEVICES=1 $PYV $K/na_final_models.py --out-dir /kaggle/working/final_a --expect-commit $H --device cuda --time-budget-hours 10.5 \
    --jarvis-csv "$J" --jarvis-sha256 3c23d5500fa0d3c6fa6a92313ee61e12fb90f93189697966d5de279adfc49c49 > /kaggle/working/logs/final_a.log 2>&1 &
PB=$!
wait $PA; RA=$?
wait $PB; RB=$?
echo "exit codes: na6 $RA, final models $RB"
tail -4 /kaggle/working/logs/na6.log; tail -4 /kaggle/working/logs/final_a.log
cat /kaggle/working/na6/status.json; echo; cat /kaggle/working/final_a/status.json; echo
```

Cell 3 (bash): what to download, and the classifier's hash for the later final-models run.

```bash
%%bash
ls -la /kaggle/working/na6.tar.gz /kaggle/working/final_a.tar.gz
sha256sum /kaggle/working/na6.tar.gz /kaggle/working/final_a.tar.gz
sha256sum /kaggle/working/na6/final_classifier.json
ls -la /kaggle/working/final_a/models /kaggle/working/final_a/predictions_jarvis.csv
```

Download: `na6.tar.gz` and `final_a.tar.gz`; send me both SHA256 lines and the `final_classifier.json` SHA256 (the second final-models run needs it
as `--classifier-sha256`). Not downloaded separately: the logs in `/kaggle/working/logs/` (the scripts' own run_config and results are in the
bundles). The four fitted models (`final_a/models/*.json`) are in `final_a.tar.gz`; their size is unmeasured, and a forest of this depth may make the
bundle large (hundreds of MB), which Kaggle's output limit allows. If either status says `"complete": false`, upload that tar.gz as a dataset and
rerun the same Cell 2 with `--restore-from <that tar.gz>` added to that analysis' command only.

## 2c. Session G3 (GPU T4 x2): final models with the classifier plus the Materials Project candidates (GPU 0, then NA3 for S and kappa), NA3 for sigma and zT (GPU 1)

Code commit for this session: `33e9662838ba299a30bc8fccba0caee26d6bc435` (it contains the NA6 classifier at `thesis_paper/results/na6/20261004T102905/final_classifier.json`, a plain
tracked file, so the clone has it; the scripts are unchanged since the commit of the earlier sessions). Both GPUs are busy: GPU 0 runs the final-models script (a few minutes) and then
NA3 for S and kappa; GPU 1 runs NA3 for sigma and zT. Accelerator: GPU T4 x2. Internet on.

**Attach** (three datasets, no previous output):
1. the snapfix dataset `muhammadbehzadgull/te-ml-pipeline-canonical-dataset-a-snapfix`;
2. `thesis-jarvis-featurized` (`jarvis_dft3d_seebeck_featurized.csv`, SHA256 `3c23d5500fa0d3c6fa6a92313ee61e12fb90f93189697966d5de279adfc49c49`);
3. a new private dataset `thesis-mp-candidates` holding exactly `mp_perovskite_candidates_featurized_20261004T135055.csv` (local path
   `C:\Users\choha\te-ml-pipeline\data\external\mp\mp_perovskite_candidates_featurized_20261004T135055.csv`; 409 rows, 1,178,742 bytes, SHA256
   `6386c09781667cc86a52d23dee1ee6f86ea899c39cea4d22f4ca10e143aec95c`).

**Download afterwards**: `final_b.tar.gz`, `na3_a.tar.gz`, `na3_b.tar.gz`, plus the printed SHA256 lines. `final_b` holds `predictions_jarvis.csv` (with the classifier's sign and
`sign_overridden`), `predictions_mp.csv` (409 candidates x 300 to 800 K in 100 K steps, with E_hull) and the four fitted models.

Cell 1: the same clone-and-install cell as S0, with `COMMIT = "33e9662838ba299a30bc8fccba0caee26d6bc435"`.

Cell 2 (bash):

```bash
%%bash
cd /kaggle/working/te-ml-pipeline
PYV=/kaggle/working/venv/bin/python
nvidia-smi --query-gpu=index,name,memory.total --format=csv
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g $PYV -c "import numpy as n, xgboost as x; x.XGBRegressor(n_estimators=5, device='cuda', tree_method='hist').fit(n.random.rand(500, 8), n.random.rand(500)); print('GPU $g: xgboost', x.__version__, 'cuda fit ok')" || { echo "GPU $g: xgboost cuda fit FAILED"; exit 1; }
done
H=$(git rev-parse HEAD)
K=thesis_paper/scripts/kaggle
J=$(find /kaggle/input -name jarvis_dft3d_seebeck_featurized.csv | head -1)
M=$(find /kaggle/input -name mp_perovskite_candidates_featurized_20261004T135055.csv | head -1)
CLF=thesis_paper/results/na6/20261004T102905
echo "jarvis: $J"; echo "mp: $M"
sha256sum "$J" "$M" $CLF/final_classifier.json
[ "$(sha256sum "$J" | cut -d' ' -f1)" = "3c23d5500fa0d3c6fa6a92313ee61e12fb90f93189697966d5de279adfc49c49" ] || { echo "JARVIS CSV hash mismatch"; exit 1; }
[ "$(sha256sum "$M" | cut -d' ' -f1)" = "6386c09781667cc86a52d23dee1ee6f86ea899c39cea4d22f4ca10e143aec95c" ] || { echo "MP CSV hash mismatch"; exit 1; }
[ "$(sha256sum $CLF/final_classifier.json | cut -d' ' -f1)" = "11b20af8138e815ae6aeb04cecd3c92ed62b4ddf48aa15f84a931d0a9b0363c7" ] || { echo "classifier hash mismatch"; exit 1; }

gpu0() {
  CUDA_VISIBLE_DEVICES=0 $PYV $K/na_final_models.py --out-dir /kaggle/working/final_b --expect-commit $H --device cuda --time-budget-hours 10.5 \
      --jarvis-csv "$J" --jarvis-sha256 3c23d5500fa0d3c6fa6a92313ee61e12fb90f93189697966d5de279adfc49c49 \
      --mp-csv "$M" --mp-sha256 6386c09781667cc86a52d23dee1ee6f86ea899c39cea4d22f4ca10e143aec95c \
      --classifier-dir $CLF --classifier-sha256 11b20af8138e815ae6aeb04cecd3c92ed62b4ddf48aa15f84a931d0a9b0363c7 > /kaggle/working/logs/final_b.log 2>&1
  echo $? > /kaggle/working/logs/rc_final_b
  CUDA_VISIBLE_DEVICES=0 $PYV $K/na3_shap.py --targets S,kappa --out-dir /kaggle/working/na3_a --expect-commit $H --device cuda --time-budget-hours 10.5 > /kaggle/working/logs/na3_a.log 2>&1
  echo $? > /kaggle/working/logs/rc_na3_a
}
gpu0 &
PA=$!
CUDA_VISIBLE_DEVICES=1 $PYV $K/na3_shap.py --targets sigma,zT --out-dir /kaggle/working/na3_b --expect-commit $H --device cuda --time-budget-hours 10.5 > /kaggle/working/logs/na3_b.log 2>&1 &
PB=$!
wait $PA; wait $PB
echo "exit codes: final_b $(cat /kaggle/working/logs/rc_final_b), na3_a $(cat /kaggle/working/logs/rc_na3_a)"
for f in final_b na3_a na3_b; do echo "== $f"; tail -n 3 /kaggle/working/logs/$f.log; done
cat /kaggle/working/final_b/status.json; echo; cat /kaggle/working/na3_a/status.json; echo; cat /kaggle/working/na3_b/status.json; echo
```

Cell 3 (bash): hashes of what to download.

```bash
%%bash
ls -la /kaggle/working/final_b.tar.gz /kaggle/working/na3_a.tar.gz /kaggle/working/na3_b.tar.gz
sha256sum /kaggle/working/final_b.tar.gz /kaggle/working/na3_a.tar.gz /kaggle/working/na3_b.tar.gz
ls -la /kaggle/working/final_b/models /kaggle/working/final_b/predictions_mp.csv /kaggle/working/final_b/predictions_jarvis.csv
```

If a status says `"complete": false` (NA3 is the long part: 25 folds per target, each a fit plus TreeSHAP on 20,000 rows; per-unit time on T4 unmeasured), upload that tar.gz as a
dataset and rerun Cell 2 with `--restore-from <that tar.gz>` added to that process only (`na3_a`, `na3_b` or `final_b`). `na3_a` and `na3_b` have disjoint targets, so they are not merged.

## 3. NA2 CPU sessions: random forest with one fixed setting (four sessions, one per target) and LightGBM with its tuning

**Design change (docs/decisions.md, 2026-10-05, committed before any random-forest test-fold result existed).** The random forest is not tuned: every target uses the one set
`n_estimators` 500, `max_features` 1/3, `min_samples_leaf` 5, no depth limit, bootstrap sampling (Probst, Wright & Boulesteix 2019); the set is recorded in each run's identity
(`params.fixed_hyperparams`) and in `frozen/<target>_random_forest.json`. LightGBM keeps its tuning (20 Optuna trials per target, then the 25 rung folds). The abandoned tuned run of
session C1 (S only, 15 of 20 trials) is committed as a record; do not restore from it.

Code commit for these sessions: `7529e7cfb7f14341994d1f57123e11edc75443e4`, the same as S0 and section 3b (`na2_trees.py` and the harness are byte-identical between `9543576` and `7529e7c`, checked with `git diff`). **Run S0 first** (section 1, which is pinned to this commit): its checks include
`na2_random_forest_fixed_completes` and `na2_fixed_rf_has_no_tuning_units_and_records_the_set`; start the long sessions only after `all_passed: True`.

Accelerator: None (CPU, 4 cores) for all five sessions. Internet on. Attach the snapfix dataset only. No previous output.

| Session | Command (Cell 2, after Cell 1 below) | `--time-budget-hours` | Estimated run time | Download |
|---|---|---|---|---|
| R-S | `na2_trees.py --model random_forest --fixed-hyperparams --targets S --out-dir /kaggle/working/na2_rf_S` | 10.8 | 25 fits x 26 min = 11.0 h | `na2_rf_S.tar.gz` |
| R-sigma | same with `--targets sigma --out-dir /kaggle/working/na2_rf_sigma` | 10.5 | 25 x 21 min = 8.8 h | `na2_rf_sigma.tar.gz` |
| R-kappa | same with `--targets kappa --out-dir /kaggle/working/na2_rf_kappa` | 10.5 | 25 x 16 min = 6.5 h | `na2_rf_kappa.tar.gz` |
| R-zT | same with `--targets zT --out-dir /kaggle/working/na2_rf_zT` | 10.5 | 25 x 18 min = 7.4 h | `na2_rf_zT.tar.gz` |
| L | `na2_trees.py --model lightgbm --out-dir /kaggle/working/na2_lightgbm` (all four targets, tuning on) | 10.5 | about 3 to 4 h, anywhere from 1 to 12 h (see below) | `na2_lightgbm.tar.gz` |

The four random-forest sessions have disjoint targets, so their identities differ and they can run at the same time if the account allows several CPU sessions at once (not verified
here); otherwise one after the other. They are not merged: `na2_stacking.py --rf-dir a,b,c,d` takes the four bundles. A session whose status says `"complete": false` is continued by
rerunning the same cells with `--restore-from <its tar.gz, attached as a dataset>` added (same `--targets`).

**Where the estimates come from** (`thesis_paper/results/na2_timing/20261005T054113`): one fixed-setting fit on the first outer training fold was timed locally with 4 threads and scaled to 500
trees; the factor Kaggle / local, 1.64, comes from one real Kaggle measurement (tuning trial 12 of the first random-forest session, 3,534 s on Kaggle against 2,152 s estimated locally) and is
assumed to hold for the new fits. The estimates are therefore good to perhaps 20 to 30 percent: R-S may need `--restore-from` for its last fits (the budget of 10.8 h starts the 25th fit
at about 10.5 h). LightGBM was timed locally only (S, 4 threads, fit and predict): smallest / middle / largest search-space configuration 5.6 / 22.6 / 90.7 s on an inner fold and 8.8 / 26.4 /
102.8 s on an outer fold, uncalibrated. A tuning trial is three inner fits, so 20 trials cost between about 0.1 h and 2.5 h (x 1.64 if the factor applies) depending on the sizes TPE
samples, and the 25 rung fits 0.1 to 1.2 h; the middle configuration gives about 0.9 h for S (0.6 h tuning, 0.3 h rung), so about 3 to 4 h for the session; if TPE settles on the largest models it approaches 12 h. Rows differ per target (kappa and zT have 65 to 70 percent of the rows of S).

Cell 1 (Python): the clone-and-install cell of section 1 (its `COMMIT` is already `7529e7cfb7f14341994d1f57123e11edc75443e4`) (everything else identical: clone, checkout, assert HEAD and a clean
tree, the uv Python 3.12 venv with the pins, the version probe).

Cell 2 (bash), for session R-S; for the other sessions change the two occurrences of `S`/`na2_rf_S` as in the table and the time budget:

```bash
%%bash
cd /kaggle/working/te-ml-pipeline
nproc; free -g | head -2
/kaggle/working/venv/bin/python thesis_paper/scripts/kaggle/na2_trees.py --model random_forest --fixed-hyperparams --targets S \
    --out-dir /kaggle/working/na2_rf_S --expect-commit "$(git rev-parse HEAD)" --time-budget-hours 10.8 2>&1 | tee /kaggle/working/logs/na2_rf_S.log
cat /kaggle/working/na2_rf_S/status.json; echo
ls -la /kaggle/working/na2_rf_S.tar.gz && sha256sum /kaggle/working/na2_rf_S.tar.gz
```

Cell 2 (bash), for session L (LightGBM):

```bash
%%bash
cd /kaggle/working/te-ml-pipeline
nproc; free -g | head -2
/kaggle/working/venv/bin/python thesis_paper/scripts/kaggle/na2_trees.py --model lightgbm --out-dir /kaggle/working/na2_lightgbm \
    --expect-commit "$(git rev-parse HEAD)" --time-budget-hours 10.5 2>&1 | tee /kaggle/working/logs/na2_lightgbm.log
cat /kaggle/working/na2_lightgbm/status.json; echo
ls -la /kaggle/working/na2_lightgbm.tar.gz && sha256sum /kaggle/working/na2_lightgbm.tar.gz
```

Download each `.tar.gz` and send me the printed SHA256 lines and the `status.json` text. For the fixed-setting random forest the first fold of each target is asserted to have the same
row counts as the committed XGBoost rung; the 25 units of a target are the rung folds only (no tuning units). Memory: a 500-tree forest without a depth limit on 148,000 rows needs a few GB
(the machine reports it in the first lines of the cell); if a session dies from memory, Save Version keeps the finished units and the session is continued with `--restore-from`.
Afterwards, locally: `python thesis_paper/scripts/kaggle/na2_stacking.py --rf-dir <S>,<sigma>,<kappa>,<zT> --lgbm-dir <lightgbm>`.

## 3b. GPU sessions G4 (NA3 rows, for Figure 11) and G5 (NA1 nested CV), both T4 x2

Code commit for both: `7529e7cfb7f14341994d1f57123e11edc75443e4`, the same as S0 and section 3 (it adds `na3_rows.py` and `na1_nested_cv.py` and their smoke-test jobs). **Run S0 first** (section 1, pinned to this commit): the smoke test also runs both new scripts, and the Linux run is what proves them on Kaggle
(here they passed a CPU smoke run on Windows only; the GPU path of both reuses the calls of `na3_shap.py` and `src/nested_cv.py`, which ran on the T4).
Accelerator: GPU T4 x2 for both. Internet on. Attach the snapfix dataset only. No previous output. Cell 1 is the clone-and-install cell of section 1 (its `COMMIT` is the commit above).

### G4: NA3 rows (per-row SHAP values for the beeswarm, Figure 11)

What it does: repeat 0 of the same chemistry-cluster folds as Paper A, the same frozen hyperparameters; for a fixed seeded subsample of 1,000 test rows per fold (`default_rng(seed + fold)`, 5,000 rows
per target) it saves the SHAP value of every feature, the feature values, the row positions, the measured and the predicted value. Each unit asserts that the SHAP values plus the bias equal the
prediction, and records the fold R2 next to the committed one. 20 units in all; one fit and a TreeSHAP of 1,000 rows per unit, about 30 to 40 s on a T4 (a fit of the S model on 148,000 rows took 19 s in
the NA6 run and 24 s on all rows in `final_b`), so about 6 minutes per process. Bundles are about 16 MB per target.

Cell 2 (bash):

```bash
%%bash
cd /kaggle/working/te-ml-pipeline
PYV=/kaggle/working/venv/bin/python
nvidia-smi --query-gpu=index,name,memory.total --format=csv
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g $PYV -c "import numpy as n, xgboost as x; x.XGBRegressor(n_estimators=5, device='cuda', tree_method='hist').fit(n.random.rand(500, 8), n.random.rand(500)); print('GPU $g: xgboost', x.__version__, 'cuda fit ok')" || { echo "GPU $g: xgboost cuda fit FAILED"; exit 1; }
done
H=$(git rev-parse HEAD)
K=thesis_paper/scripts/kaggle
CUDA_VISIBLE_DEVICES=0 $PYV $K/na3_rows.py --targets S,kappa --out-dir /kaggle/working/na3r_a --expect-commit $H --device cuda --time-budget-hours 10.5 > /kaggle/working/logs/na3r_a.log 2>&1 &
PA=$!
CUDA_VISIBLE_DEVICES=1 $PYV $K/na3_rows.py --targets sigma,zT --out-dir /kaggle/working/na3r_b --expect-commit $H --device cuda --time-budget-hours 10.5 > /kaggle/working/logs/na3r_b.log 2>&1 &
PB=$!
wait $PA; echo "na3r_a exit $?"; wait $PB; echo "na3r_b exit $?"
for f in na3r_a na3r_b; do echo "== $f"; tail -n 4 /kaggle/working/logs/$f.log; cat /kaggle/working/$f/status.json; echo; done
ls -la /kaggle/working/na3r_a.tar.gz /kaggle/working/na3r_b.tar.gz
sha256sum /kaggle/working/na3r_a.tar.gz /kaggle/working/na3r_b.tar.gz
```

Download `na3r_a.tar.gz` and `na3r_b.tar.gz` and send me the printed SHA256 lines. I verify, commit them under `results/na3_rows_a` and `na3_rows_b`, and draw Figure 11 with
`python thesis_paper/scripts/make_figures_thesis.py --na3-rows-dirs <dir a>,<dir b>`.

### G5: NA1 nested grouped CV (re-tuning inside every outer training fold)

What it does: for each target and each of the 25 outer folds of the Paper A rung (5 repeats x 5 folds, the same folds, checked against the committed fold sizes), a fresh Optuna search of 20 trials
(TPE, median pruner, the project's XGBoost search space, 3 inner chemistry-cluster folds) on the outer TRAINING rows only, then a refit of the best set and a prediction of the outer test rows. Every trial
and every fold is a checkpointed unit (525 units per target). The result is each target's nested pooled R2 per repeat against the committed frozen-hyperparameter value, and the mean inner-CV R2 of the best trials.

**Estimate** (a model, not a measurement of this script). Reference: on the T4 one XGBoost fit with the frozen S set (500 trees, depth 10) on 148,000 rows took 19.2 s including its prediction (median of the 25
NA6 classifier fits). A trial is three fits on two thirds of the outer training rows. Averaging the cost over the search space (trees 100 to 600, depth 3 to 10, column and row subsampling) with the cost
of a deeper tree taken from the CPU measurements in CLAUDE.md (factor 1.18 per level), or growing linearly with depth, or not at all, gives a mean trial of 20 s, 22 s or 33 s for S (148,000
training rows), and TPE tends to move towards the larger frozen-like models; I take **about 28 s per trial for S and sigma, 18 s for kappa and 19 s for zT** (rows scale linearly), minus a few percent for pruned trials.
That is about 9.6 min per outer fold for S (6.7 to 14 min across the three cost models), 9.5 min for sigma, 6.2 min for kappa and 6.7 min for zT. For 25 outer folds per target: **S 4.0 h, sigma 4.0 h,
kappa 2.6 h, zT 2.8 h, 13.4 h in all on one T4 (about 9.5 to 19 h across the cost models)**.

Session plan: two processes, one per T4, split by targets so that each carries about the same load: process A `S,kappa` (about 6.6 h) and process B `sigma,zT` (about 6.8 h). **One session is expected to
be enough** (budget 10.5 h; the range of the estimate is about 5 to 10 h); if a process stops on the budget, `status.json` says `"complete": false` and the same cells are rerun with
`--restore-from <that tar.gz>` for that process only (a second session of a few hours). One repeat only (`--n-repeats 1`, 20 outer folds, about 2.7 h of T4 time, 1.4 h with both T4s) is an
option if the budget matters more than the five-repeat spread; it is a different analysis identity and would not be extended afterwards.

Cell 2 (bash):

```bash
%%bash
cd /kaggle/working/te-ml-pipeline
PYV=/kaggle/working/venv/bin/python
nvidia-smi --query-gpu=index,name,memory.total --format=csv
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g $PYV -c "import numpy as n, xgboost as x; x.XGBRegressor(n_estimators=5, device='cuda', tree_method='hist').fit(n.random.rand(500, 8), n.random.rand(500)); print('GPU $g: xgboost', x.__version__, 'cuda fit ok')" || { echo "GPU $g: xgboost cuda fit FAILED"; exit 1; }
done
H=$(git rev-parse HEAD)
K=thesis_paper/scripts/kaggle
CUDA_VISIBLE_DEVICES=0 $PYV $K/na1_nested_cv.py --targets S,kappa --out-dir /kaggle/working/na1_a --expect-commit $H --device cuda --time-budget-hours 10.5 > /kaggle/working/logs/na1_a.log 2>&1 &
PA=$!
CUDA_VISIBLE_DEVICES=1 $PYV $K/na1_nested_cv.py --targets sigma,zT --out-dir /kaggle/working/na1_b --expect-commit $H --device cuda --time-budget-hours 10.5 > /kaggle/working/logs/na1_b.log 2>&1 &
PB=$!
wait $PA; echo "na1_a exit $?"; wait $PB; echo "na1_b exit $?"
for f in na1_a na1_b; do echo "== $f"; tail -n 4 /kaggle/working/logs/$f.log; cat /kaggle/working/$f/status.json; echo; done
ls -la /kaggle/working/na1_a.tar.gz /kaggle/working/na1_b.tar.gz
sha256sum /kaggle/working/na1_a.tar.gz /kaggle/working/na1_b.tar.gz
```

To watch the speed while it runs, a separate cell can `tail -n 3 /kaggle/working/logs/na1_a.log`: each finished outer fold prints its nested R2, the committed frozen R2 and the tuning seconds, which gives the real
per-fold time within the first hour; compare it with 9.6 min (S) before letting the session run on. Download `na1_a.tar.gz` and `na1_b.tar.gz` and send me the printed SHA256 lines and both `status.json` texts.
The two halves have disjoint targets and therefore their own identities; they are combined at analysis time. The department V100S remains an alternative; `scripts/gpu/calibrate_gpu.py` is its calibration.

## 3c. NA2 nested stacking (CPU sessions, ten of them; the stack with a fitted meta-learner, trained on inner out-of-fold predictions only)

What it does (`na2_stacking_nested.py`, header of the file): for every outer fold of the Paper A rung (same folds, fold sizes checked), inside the outer TRAINING rows three inner chemistry-cluster folds
(`GroupKFold(3)`, as the tuning uses); XGBoost (the canonical frozen set of the target), LightGBM (the frozen tuned set of `results/na2_lightgbm`) and the fixed random forest are fitted on the inner training
rows and predict the inner validation rows; a non-negative ridge meta-learner (alpha 1) is fitted on those inner out-of-fold predictions and its weights are the unit. `assert_nested` checks, in the code, that
the outer train and test rows and clusters are disjoint, that the inner validation folds partition the outer training rows exactly once with disjoint rows and clusters against their own inner training
sets, and that the meta-learner is fitted on exactly the outer training rows. The stack's outer prediction is made locally (`scripts/na2_stacking_analysis.py`) from the committed outer predictions of the
three base models; no outer test label is used here. XGBoost runs on the CPU in this script (the Paper A outer fits ran on a GPU); only the inner predictions the weights are learned from are affected.

**Code commit: `13a0efd582fe9abae95a00d29c677b70eb8ff103`** (not 7529e7c: the two new scripts exist only from this commit on; `git diff 7529e7c 13a0efd -- src thesis_paper/scripts/kaggle` shows only the
two stacking scripts, so the base-model code is unchanged). Cell 1 is the clone-and-install cell of section 1 with `COMMIT` set to it. Accelerator: None (CPU, 4 cores). Internet on. Attach the snapfix dataset only.
The new script has not run on Kaggle Linux: Cell 2 starts with a smoke run (about 2 minutes, `--smoke --allow-dirty`, never a result) and stops if it fails.

**Estimate** (a model built on `results/na2_timing`, not a measurement of this script): the forest dominates. One fixed-setting forest fit on the full outer training rows took 26 / 21 / 16 / 18 min (S / sigma / kappa / zT)
on Kaggle by the timing note; an inner fit uses two thirds of those rows (about 0.65 of the time), three inner fits per fold, so 51 / 41 / 31 / 35 min of forest per outer fold, plus about 7 / 7 / 5 / 5 min for LightGBM and XGBoost:
**about 58 / 48 / 36 / 40 min per outer fold, 24 / 20 / 15 / 17 h per target for 25 folds, about 76 CPU hours in all.** The slices below keep each session under about 10 h.

| Session | Targets, repeats | `--time-budget-hours` | Estimate | Download |
|---|---|---|---|---|
| K-S1 | `--targets S --repeats 0,1 --out-dir /kaggle/working/na2_stk_S_1` | 10.8 | 10 folds, 9.7 h | `na2_stk_S_1.tar.gz` |
| K-S2 | `--targets S --repeats 2,3 --out-dir /kaggle/working/na2_stk_S_2` | 10.8 | 9.7 h | `na2_stk_S_2.tar.gz` |
| K-S3 | `--targets S --repeats 4 --out-dir /kaggle/working/na2_stk_S_3` | 10.5 | 4.8 h | `na2_stk_S_3.tar.gz` |
| K-sigma1, 2, 3 | `--targets sigma`, repeats `0,1` / `2,3` / `4`, `--out-dir /kaggle/working/na2_stk_sigma_{1,2,3}` | 10.5 | 8.0 / 8.0 / 4.0 h | `na2_stk_sigma_{1,2,3}.tar.gz` |
| K-kappa1, 2 | `--targets kappa`, repeats `0,1,2` / `3,4`, `--out-dir /kaggle/working/na2_stk_kappa_{1,2}` | 10.5 | 9.0 / 6.0 h | `na2_stk_kappa_{1,2}.tar.gz` |
| K-zT1, 2 | `--targets zT`, repeats `0,1,2` / `3,4`, `--out-dir /kaggle/working/na2_stk_zT_{1,2}` | 10.8 | 10.0 / 6.7 h | `na2_stk_zT_{1,2}.tar.gz` |

The ten sessions have disjoint units, so they are independent and can run at the same time if the account allows it. They are not merged here: `na2_stacking_analysis.py --stack-dirs` takes the ten bundles and
asserts that together they cover every target, repeat and fold once. A session whose `status.json` says `"complete": false` is continued by rerunning its cell with `--restore-from <its tar.gz>`.
A cheaper design is `--n-inner-folds 2` (about half the cost, about 40 CPU hours); it changes the design and I will not use it without your decision.

Cell 2 (bash) is the same in all ten notebooks except its first line, which sets four shell variables: `T` (`--targets`), `R` (`--repeats`), `N` (the out-dir and log name) and `B` (`--time-budget-hours`).
It is the per-row command of the table with those four values substituted, and nothing else, so it is equivalent to running the row's command by hand. It was written here as the template; the notebooks you
saved were not read back by me, so if one of them differs from this text (other than in the first line), tell me which and how, because the bundle's `run_config.json` records the arguments that were actually used.

```bash
%%bash
set -o pipefail
T=S; R=0,1; N=na2_stk_S_1; B=10.8      # <- the only line that differs between the notebooks (values below)
cd /kaggle/working/te-ml-pipeline
nproc; free -g | head -2
PYV=/kaggle/working/venv/bin/python
K=thesis_paper/scripts/kaggle
$PYV $K/na2_stacking_nested.py --smoke --allow-dirty --targets $T --out-dir /kaggle/working/smoke_$N 2>&1 | tail -n 3 || { echo "SMOKE FAILED"; exit 1; }
$PYV $K/na2_stacking_nested.py --targets $T --repeats $R --out-dir /kaggle/working/$N \
    --expect-commit "$(git rev-parse HEAD)" --time-budget-hours $B 2>&1 | tee /kaggle/working/logs/$N.log
cat /kaggle/working/$N/status.json; echo
ls -la /kaggle/working/$N.tar.gz && sha256sum /kaggle/working/$N.tar.gz
```

The first line of each notebook (`T`, `R`, `N`, `B`; N is also the downloaded file's name, `N.tar.gz`):

| Session | First line |
|---|---|
| K-S1 | `T=S; R=0,1; N=na2_stk_S_1; B=10.8` |
| K-S2 | `T=S; R=2,3; N=na2_stk_S_2; B=10.8` |
| K-S3 | `T=S; R=4; N=na2_stk_S_3; B=10.5` |
| K-sigma1 | `T=sigma; R=0,1; N=na2_stk_sigma_1; B=10.5` |
| K-sigma2 | `T=sigma; R=2,3; N=na2_stk_sigma_2; B=10.5` |
| K-sigma3 | `T=sigma; R=4; N=na2_stk_sigma_3; B=10.5` |
| K-kappa1 | `T=kappa; R=0,1,2; N=na2_stk_kappa_1; B=10.5` |
| K-kappa2 | `T=kappa; R=3,4; N=na2_stk_kappa_2; B=10.5` |
| K-zT1 | `T=zT; R=0,1,2; N=na2_stk_zT_1; B=10.8` |
| K-zT2 | `T=zT; R=3,4; N=na2_stk_zT_2; B=10.8` |

Each finished outer fold prints its meta-learner weights, the intercept and the inner out-of-fold R2 of each base model with the seconds; the first fold gives the real per-fold time (compare with 58 min for S before
letting the rest run). Download each `.tar.gz` and send me the printed SHA256 lines and the `status.json` texts. Afterwards, locally:
`python thesis_paper/scripts/na2_stacking_analysis.py --stack-dirs <the ten bundles>` and `python thesis_paper/scripts/model_comparison.py na2 --stacking-dir <its output>`.

## 3d. NA13 feature selection (CPU, one notebook)

What it does (`na13_feature_selection.py`, header of the file): the thesis's three-step selection (Pearson filter at |r| > 0.95, LassoCV on the survivors with 3 inner chemistry-cluster folds and 20 alphas, mutual-information
ranking on a 20,000-row subsample, keeping the top k) is performed INSIDE each outer training fold of the Paper A chemistry-cluster rung (same folds, 5 repeats x 5 folds, 100 units in all), then the target's frozen XGBoost
is fitted on the selected columns only and scored on the outer test rows. k is the number of features the thesis reports per target (S 25, sigma 44, kappa 39, zT 32; thesis values, tested here, not endorsed). The result is
the pooled per-repeat R2 of the selected-feature model against the committed 397-feature value, and how often each feature was selected.

**Code commit: `13a0efd582fe9abae95a00d29c677b70eb8ff103`** (the script is byte-identical to the one at `7529e7c`, which passed the Linux smoke test S0, `git diff 7529e7c 13a0efd -- thesis_paper/scripts/kaggle/na13_feature_selection.py
thesis_paper/scripts/kaggle/harness.py src` is empty). Cell 1 is the clone-and-install cell of section 1 with `COMMIT` set to it. **Accelerator: None (CPU, 4 cores). GPU is not used.** Internet on.
**Attach: the snapfix dataset only. No previous output. Nothing else to download or upload.**

**Estimate (measured, then scaled).** One real outer fold on the maintainer's 8-core PC, 4 threads, clean clone at `13a0efd`, the real data: kappa 115 s (39 features kept, fold R2 0.770), S 141 s (25 features, fold R2 0.755).
Taking sigma at the S cost (same rows) and zT at 120 s (rows between kappa and S), 25 folds per target make 25 x (141 + 138 + 115 + 120) s = 3.6 h on that PC. Kaggle's 4 cores were 1.26 to 1.64 times slower
than that PC in the two other measurements of this work, so **about 4.5 to 6 hours in one session** (budget 10.5 h; far from the limit). If `status.json` says `"complete": false`, rerun the same cell with
`--restore-from <its tar.gz>`.
Expect many `ConvergenceWarning` lines from LassoCV (about 13 to 15 per fold here: `max_iter` 2000 is not reached at the smallest alphas); the cell hides them from the console and keeps the raw log. The Lasso fits are the thesis
pipeline as described, not tuned until they converge; I will say in the paper that some did not.

Cell 2 (bash):

```bash
%%bash
set -o pipefail
cd /kaggle/working/te-ml-pipeline
nproc; free -g | head -2
/kaggle/working/venv/bin/python thesis_paper/scripts/kaggle/na13_feature_selection.py --out-dir /kaggle/working/na13 \
    --expect-commit "$(git rev-parse HEAD)" --time-budget-hours 10.5 2>&1 | tee /kaggle/working/logs/na13_raw.log | grep -v "ConvergenceWarning\|cd_fast\.enet" | tee /kaggle/working/logs/na13.log
grep -c ConvergenceWarning /kaggle/working/logs/na13_raw.log
cat /kaggle/working/na13/status.json; echo
ls -la /kaggle/working/na13.tar.gz && sha256sum /kaggle/working/na13.tar.gz
```

Each finished fold prints the number of selected features and the fold R2 with its seconds (first fold of S: compare with about 140 s x 1.3 to 1.6 before letting it run). Download `na13.tar.gz` (right-click, "Save link as") and send me the printed SHA256 line,
the `status.json` text and the warning count. Then I verify it, commit it under `results/na13`, and fill the paper (there is no NA13 marker yet; the thesis's Section 3.3 claim is C110).

## 4. Later sessions (the same Cell 1, then these)

| Order | Session | Accelerator | Cell 2 | Attach |
|---|---|---|---|---|
| 2 | Final models, no classifier, plus NA6, concurrently | GPU T4 x2 | process A (GPU 0): `na6_classifier.py --out-dir /kaggle/working/na6 --device cuda`; process B (GPU 1): `na_final_models.py --out-dir /kaggle/working/final_a --device cuda --jarvis-csv $J --jarvis-sha256 3c23d550...9c49` (`J=$(find /kaggle/input -name jarvis_dft3d_seebeck_featurized.csv)`), both with `--expect-commit`, `&`, `wait` as in G1 | snapfix dataset, `thesis-jarvis-featurized` |
| 3 | NA3 SHAP | GPU T4 x2 | A (GPU 0): `na3_shap.py --targets S,kappa --out-dir /kaggle/working/na3_a`; B (GPU 1): `--targets sigma,zT --out-dir /kaggle/working/na3_b` | snapfix dataset |
| C2 | NA2 LightGBM | CPU | see section 3 (session L) | snapfix dataset |
| C3 | NA13 feature selection | CPU | see section 3d (commit `13a0efd`, about 4.5 to 6 h) | snapfix dataset |
| later | Final models again, with the classifier and the MP candidates | GPU | as the final-models command plus `--classifier-dir <na6 dir> --classifier-sha256 <sha of final_classifier.json> --mp-csv ... --mp-sha256 ...` | snapfix, JARVIS, the na6 output, the MP csv |
| K | NA2 nested stacking | CPU, ten sessions | see section 3c (the earlier local `na2_stacking.py` is the interim cross-fitted version, not used in the paper) | snapfix dataset |
| G5 | NA1 nested CV | GPU T4 x2 (or the department V100S) | see section 3b | snapfix dataset |

NA6 and the first final-models run are different analyses and run side by side. NA3's two halves have disjoint targets and therefore their own
identities and results; they are not merged, and are combined at analysis time.

## 5. What the smoke test proves and does not

Proves (local Windows run; S0 repeats it on Linux): every script runs end to end; every output has a manifest whose SHA256 values match the files;
every run_config records dataset SHA256, git HEAD and tree_clean; a script refuses to start without `--expect-commit` or `--allow-dirty`; a killed
session resumes from a directory or tar.gz with nothing recomputed and results identical to an uninterrupted run; two NA7 shards merge into results
identical to a single run; a restore with different parameters or a tampered unit is refused; a zero time budget stops cleanly and is continued.

Does not prove: XGBoost `device="cuda"` behaviour on the T4 or two simultaneous processes on two GPUs; the time per unit; that tiny-subset numbers mean
anything; that the pip pins resolve on the current Kaggle image (S0 is that check).
