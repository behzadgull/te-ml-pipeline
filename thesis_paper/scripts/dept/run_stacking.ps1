<#
.SYNOPSIS
  NA2 nested stacking on the department machine: a smoke run, then all 100 units (4 targets x 5 repeats x 5 folds) as 3 parallel processes of 4 threads each.

.DESCRIPTION
  -Worker smoke   one tiny run (~3,000 rows, never a result) to prove the script and the environment; run it first.
  -Worker 1|2|3   one process: its jobs run one after another, each job its own out-dir under <Root>\out and its own log under <Root>\logs.
  -Worker all     starts workers 1, 2 and 3 in three minimised windows and returns.

  The jobs (kaggle/na2_stacking_nested.py with --targets/--repeats; no --time-budget-hours, so nothing stops it):
      worker 1:  stk_S_all        S      repeats 0-4
      worker 2:  stk_sigma_all    sigma  repeats 0-4,  then stk_kappa_r0  kappa repeat 0,   then stk_zT_r0  zT repeat 0
      worker 3:  stk_kappa_r1_4   kappa  repeats 1-4,  then stk_zT_r1_4   zT repeat 1-4
  Estimated busy time at the Kaggle per-fold cost: 24, 26 and 25 hours (see KAGGLE_CELLS.md section 3c for the model).

  Threads. Every process gets OMP_NUM_THREADS (xgboost, lightgbm), MKL/OPENBLAS/NUMEXPR_NUM_THREADS and LOKY_MAX_CPU_COUNT (joblib, which the
  random forest's n_jobs=-1 resolves through) all set to 4, and its CPU affinity is set to 4 distinct cores (physical cores 0-3, 4-7, 8-11 when
  hyper-threading is on: the even logical processors, so the idle siblings stay free for Paper B; contiguous blocks of 4 otherwise).
  src/nested_cv.py hard-codes n_jobs=-1, so the environment and the affinity are the only levers; the first lines of each log print what was applied.

  Resumable. A finished job (status.json complete) is skipped. A killed job (power cut, closed window) is resumed by running the same command again:
  the harness skips every unit already on disk. Before each job it runs verify_outputs.py, which moves unreadable unit files (a power cut can leave one
  empty) to <Root>\quarantine so they are recomputed.

  Run:   powershell -ExecutionPolicy Bypass -File C:\te_run\dept\run_stacking.ps1 -Worker smoke
         powershell -ExecutionPolicy Bypass -File C:\te_run\dept\run_stacking.ps1 -Worker all
#>
param(
    [Parameter(Mandatory = $true)][ValidateSet("smoke", "1", "2", "3", "all")][string]$Worker,
    [string]$Root = "C:\te_run",
    [string]$Commit = "13a0efd582fe9abae95a00d29c677b70eb8ff103",
    [string]$Repo = "",
    [string]$Python = "",
    [string]$Csv = "",
    [int]$Threads = 4
)
$ErrorActionPreference = "Stop"
if (-not $Repo) { $Repo = "$Root\repo" }
if (-not $Python) { $Python = "$Root\venv\Scripts\python.exe" }
if (-not $Csv) { $Csv = "$Root\data\featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv" }
$Self = $MyInvocation.MyCommand.Path

if ($Worker -eq "all") {
    foreach ($w in 1, 2, 3) {
        Start-Process powershell -WindowStyle Minimized -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$Self`"", "-Worker", $w, "-Root", "`"$Root`"", "-Commit", $Commit, "-Repo", "`"$Repo`"", "-Python", "`"$Python`"", "-Csv", "`"$Csv`"", "-Threads", $Threads
        Start-Sleep -Seconds 20   # stagger the three data loads (memory peak)
    }
    Write-Host "started workers 1, 2, 3; logs in $Root\logs, outputs in $Root\out"
    return
}

# --- environment of this process and its children
$env:OMP_NUM_THREADS = "$Threads"; $env:MKL_NUM_THREADS = "$Threads"; $env:OPENBLAS_NUM_THREADS = "$Threads"; $env:NUMEXPR_NUM_THREADS = "$Threads"
$env:LOKY_MAX_CPU_COUNT = "$Threads"; $env:PYTHONUNBUFFERED = "1"; $env:PYTHONDONTWRITEBYTECODE = "1"

function Set-Affinity([int]$w) {
    $cpu = Get-CimInstance Win32_Processor
    $cores = ($cpu | Measure-Object NumberOfCores -Sum).Sum
    $logical = ($cpu | Measure-Object NumberOfLogicalProcessors -Sum).Sum
    $idx = @()
    if ($logical -eq 2 * $cores -and $cores -ge 3 * $Threads) { $idx = (0..($Threads - 1)) | ForEach-Object { 2 * (($w - 1) * $Threads + $_) } }
    elseif ($logical -eq $cores -and $cores -ge 3 * $Threads) { $idx = (0..($Threads - 1)) | ForEach-Object { ($w - 1) * $Threads + $_ } }
    if ($idx.Count -eq 0) { Write-Host "affinity: not set (cores $cores, logical $logical)"; return }
    $mask = [int64]0
    foreach ($i in $idx) { $mask = $mask -bor ([int64]1 -shl $i) }
    (Get-Process -Id $PID).ProcessorAffinity = [IntPtr]$mask
    Write-Host "affinity: worker $w on logical processors $($idx -join ',') (mask 0x$('{0:X}' -f $mask)); cores $cores, logical $logical"
}

function Assert-Setup {
    foreach ($p in @($Python, $Csv, "$Repo\thesis_paper\scripts\kaggle\na2_stacking_nested.py")) { if (-not (Test-Path $p)) { throw "missing: $p (run setup_windows.ps1 first)" } }
}

function Run-Job([string]$Name, [string]$Targets, [string]$Repeats, [string]$Extra) {
    $out = "$Root\out\$Name"
    $log = "$Root\logs\$Name.log"
    if (Test-Path "$out\status.json") {
        if ((Get-Content "$out\status.json" -Raw | ConvertFrom-Json).complete) { Write-Host "[$Name] already complete, skipped"; return }
    }
    & $Python "$PSScriptRoot\verify_outputs.py" --stacking-dir $out --quarantine "$Root\quarantine" --since-hours 72
    Write-Host "[$Name] $(Get-Date -Format u) start: targets $Targets repeats $Repeats"
    Push-Location $Repo
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"   # Windows PowerShell 5.1 turns redirected native stderr lines into terminating errors under "Stop"
    try {
        $argList = @("-u", "thesis_paper\scripts\kaggle\na2_stacking_nested.py", "--targets", $Targets, "--repeats", $Repeats, "--out-dir", $out, "--dataset", $Csv, "--expect-commit", $Commit)
        if ($Extra) { $argList += $Extra.Split(" ") }
        & $Python @argList 2>&1 | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) { throw "[$Name] exited with code $LASTEXITCODE (see $log); rerun this command to resume" }
    } finally { $ErrorActionPreference = $prevEap; Pop-Location }
    Write-Host "[$Name] $(Get-Date -Format u) finished"
}

Assert-Setup
New-Item -ItemType Directory -Force -Path "$Root\out", "$Root\logs", "$Root\quarantine" | Out-Null

if ($Worker -eq "smoke") {
    $out = "$Root\smoke\stk"
    if (Test-Path $out) { Remove-Item -Recurse -Force $out }
    Set-Affinity 1
    Push-Location $Repo
    $ErrorActionPreference = "Continue"
    & $Python -u thesis_paper\scripts\kaggle\na2_stacking_nested.py --smoke --allow-dirty --targets S --out-dir $out --dataset $Csv 2>&1 | Tee-Object -FilePath "$Root\logs\smoke_stacking.log"
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    Pop-Location
    if ($code -ne 0) { throw "SMOKE FAILED (exit $code), see $Root\logs\smoke_stacking.log" }
    Write-Host "SMOKE OK: $(Get-Content "$out\status.json" -Raw)"
    return
}

Set-Affinity ([int]$Worker)
switch ($Worker) {
    "1" { Run-Job "stk_S_all" "S" "0,1,2,3,4" "" }
    "2" { Run-Job "stk_sigma_all" "sigma" "0,1,2,3,4" ""; Run-Job "stk_kappa_r0" "kappa" "0" ""; Run-Job "stk_zT_r0" "zT" "0" "" }
    "3" { Run-Job "stk_kappa_r1_4" "kappa" "1,2,3,4" ""; Run-Job "stk_zT_r1_4" "zT" "1,2,3,4" "" }
}
Write-Host "worker ${Worker}: all of its jobs are complete"
