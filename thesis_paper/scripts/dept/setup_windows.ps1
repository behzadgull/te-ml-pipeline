<#
.SYNOPSIS
  One-time setup of the department machine (Windows 10, 12-core Xeon Gold, V100S) for the NA2 nested stacking and the Paper B gpu-pooled chain.

.DESCRIPTION
  1. creates the folders under -Root (everything the runs write stays OUTSIDE the clone, because the harness refuses to start on a dirty tree);
  2. installs uv (the official installer) if it is not on PATH;
  3. creates a uv-managed Python 3.12 venv with the same pins as Cell 1 of thesis_paper/scripts/kaggle/KAGGLE_CELLS.md (the pins the Paper A results were produced with) and asserts them;
  4. clones the repository at -Commit (LFS pointers only, then the Paper B split files), asserts HEAD and a clean tree;
  5. verifies the size and SHA256 of the snapfix CSV, which you copy to <Root>\data\ yourself;
  6. prints the versions, nvidia-smi, a tiny xgboost device="cuda" fit, the thread limits, and imports the Paper B module.
  It stops at the first failure. Safe to rerun (it reuses what is already right).

  Prerequisites it does not install: git for Windows (includes git-lfs), internet access. Run in Windows PowerShell 5.1 or PowerShell 7:
      powershell -ExecutionPolicy Bypass -File C:\te_run\dept\setup_windows.ps1
  Copy this whole folder (thesis_paper\scripts\dept) to C:\te_run\dept first: the scripts live outside the clone, because the pinned commit 13a0efd predates them.
#>
param(
    [string]$Root = "C:\te_run",
    [string]$Commit = "13a0efd582fe9abae95a00d29c677b70eb8ff103",
    [string]$RepoUrl = "https://github.com/behzadgull/te-ml-pipeline.git",
    [string]$CsvName = "featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv"
)
$ErrorActionPreference = "Stop"
$CsvSha = "d9fc1e5d942e4f5e22590df56dc73200ce40790723c490684ceadcbdc042e489"
$CsvBytes = 974854507

function Check($what) { if ($LASTEXITCODE -ne 0) { throw "FAILED: $what (exit code $LASTEXITCODE)" } }

New-Item -ItemType Directory -Force -Path $Root, "$Root\data", "$Root\out", "$Root\logs", "$Root\ckpt", "$Root\smoke", "$Root\backup" | Out-Null
Start-Transcript -Path "$Root\logs\setup_$(Get-Date -Format yyyyMMddTHHmmss).log" | Out-Null
Write-Host "== 1. tools"
git --version; Check "git is not installed (winget install --id Git.Git -e, then reopen the shell)"
git lfs version; Check "git-lfs is not installed (it ships with Git for Windows; reinstall Git with the LFS option)"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "installing uv (official installer, https://astral.sh/uv/install.ps1)"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    Check "uv installer"
    foreach ($p in @("$env:USERPROFILE\.local\bin", "$env:USERPROFILE\.cargo\bin")) { if (Test-Path $p) { $env:Path = "$p;$env:Path" } }
}
uv --version; Check "uv is not on PATH after the install (open a new shell and rerun)"

Write-Host "== 2. Python 3.12 venv with the Kaggle pins"
$Venv = "$Root\venv"
$Py = "$Venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { uv venv $Venv --python 3.12; Check "uv venv" }
uv pip install --python $Py "numpy==1.26.4" "pandas==2.2.2" scipy "scikit-learn==1.4.2" "xgboost==2.0.3" "optuna==3.6.1" "lightgbm==4.3.0"
Check "uv pip install"
$probe = @'
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
'@
$probe | & $Py -
Check "the venv is not Python 3.12 with the exact pins"

Write-Host "== 3. clone at the commit"
$env:GIT_LFS_SKIP_SMUDGE = "1"
$Repo = "$Root\repo"
if (-not (Test-Path "$Repo\.git")) { git -c core.longpaths=true clone --quiet $RepoUrl $Repo; Check "git clone" }
git -C $Repo config core.longpaths true
git -C $Repo fetch --quiet origin; Check "git fetch"
git -C $Repo checkout --quiet $Commit; Check "git checkout $Commit"
$head = (git -C $Repo rev-parse HEAD).Trim()
if ($head -ne $Commit) { throw "HEAD $head is not $Commit" }
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
Write-Host "fetching the Paper B split files (git lfs, 139 .npz)"
git -C $Repo lfs pull --include="paper_b/results/splits/*"; Check "git lfs pull"
$dirty = git -C $Repo status --porcelain
if ($dirty) { throw "working tree is not clean:`n$dirty" }
Write-Host "HEAD $head, tree clean"

Write-Host "== 4. the snapfix CSV"
$Csv = "$Root\data\$CsvName"
if (-not (Test-Path $Csv)) { throw "copy $CsvName to $Root\data\ first" }
$size = (Get-Item $Csv).Length
if ($size -ne $CsvBytes) { throw "$Csv is $size bytes, expected $CsvBytes" }
$sha = (Get-FileHash -Algorithm SHA256 $Csv).Hash.ToLower()
if ($sha -ne $CsvSha) { throw "SHA256 $sha does not match $CsvSha" }
Write-Host "CSV ok: $size bytes, sha256 $sha"

Write-Host "== 5. machine, GPU, threads"
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | Format-List
Get-CimInstance Win32_ComputerSystem | ForEach-Object { "RAM: {0:N1} GB" -f ($_.TotalPhysicalMemory / 1GB) }
Get-PSDrive -Name ($Root.Substring(0, 1)) | ForEach-Object { "Free on {0}: {1:N0} GB" -f $_.Name, ($_.Free / 1GB) }
nvidia-smi; Check "nvidia-smi"
$gpu = @'
import numpy as n, xgboost as x
m = x.XGBRegressor(n_estimators=20, device="cuda", tree_method="hist").fit(n.random.rand(2000, 8), n.random.rand(2000))
print("xgboost", x.__version__, "device=cuda fit ok")
'@
$gpu | & $Py -
Check "xgboost 2.0.3 device=cuda fit (the Windows wheel must be the CUDA build; if this fails, stop and tell me)"
$env:LOKY_MAX_CPU_COUNT = "4"; $env:OMP_NUM_THREADS = "4"
& $Py -c "import joblib, os; print('os.cpu_count', os.cpu_count(), '| joblib n_jobs=-1 resolves to', joblib.effective_n_jobs(-1), 'with LOKY_MAX_CPU_COUNT=4')"
Check "joblib"
Push-Location $Repo
& $Py -c "import paper_b.src.lofo_paperb; print('paper_b.src.lofo_paperb imports')"
Check "paper_b import"
& $Py paper_b\scripts\check_shared_dependencies.py
Check "paper_b shared dependencies"
Pop-Location
Write-Host "== setup complete. Root $Root, commit $head"
Stop-Transcript | Out-Null
