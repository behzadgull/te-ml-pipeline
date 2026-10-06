<#
.SYNOPSIS
  Paper B on the V100S while the stacking runs on the CPU: GPU smoke, the gpu-pooled chain, and (later, after the stacking is done) the specialists.

.DESCRIPTION
  -Mode smoke       a minutes-long GPU smoke (zT, manganite and i_v_vi2, 1 repeat, 2 folds, 2 tuning trials, capped search space; the checkpoint dir name contains "smoke", as the code requires).
  -Mode pooled      the gpu-pooled role: tuning_once + C0-C3 for every (level, target, unit), 1 worker on GPU 0, no time budget, checkpoints in <Root>\ckpt\ckpt_gpu_pooled.
  -Mode specialist  the cpu-specialist role, 4 workers (methodology doc 8.7: 1 thread each); START IT ONLY WHEN THE STACKING IS DONE (it needs the CPU cores), checkpoints in <Root>\ckpt\ckpt_cpu_specialist.
  -Mode control     the gpu-specialist-control role (the 3 zT device-control pairs on the GPU), checkpoints in <Root>\ckpt\ckpt_gpu_control; run it after the pooled chain, not at the same time as it.

  Each role has its OWN checkpoint directory (the roles must never share one: methodology doc section 8.1). Everything is outside the clone, so the tree stays clean.
  Resumable: after a power cut or a closed window, run the same command again; every checkpoint file is written atomically and every finished unit is skipped. verify_outputs.py runs first
  and moves any unreadable file to <Root>\quarantine. Paper B does not need --restore-from for this (same machine, same directory); that flag is for moving a checkpoint to another machine.

  Threads: the pooled process gets OMP_NUM_THREADS 2 (ridge and the data handling on the CPU; xgboost trains on the GPU), no affinity, so it shares the machine with the three 4-thread stacking
  processes (12 busy threads on 12 cores): expect it, and the stacking, to run a few percent slower than alone. The specialist role is started with 4 workers after the stacking is done.

  Run:  powershell -ExecutionPolicy Bypass -File C:\te_run\dept\run_paper_b.ps1 -Mode smoke
        powershell -ExecutionPolicy Bypass -File C:\te_run\dept\run_paper_b.ps1 -Mode pooled
#>
param(
    [Parameter(Mandatory = $true)][ValidateSet("smoke", "pooled", "specialist", "control")][string]$Mode,
    [string]$Root = "C:\te_run",
    [string]$Repo = "",
    [string]$Python = "",
    [string]$Csv = "",
    [string]$LabelsRun = "paper_b\reports\family_labels\20260926T181709",
    [string]$SplitsDir = "paper_b\results\splits\20260929T113055",
    [int]$GpuIndex = 0
)
$ErrorActionPreference = "Stop"
if (-not $Repo) { $Repo = "$Root\repo" }
if (-not $Python) { $Python = "$Root\venv\Scripts\python.exe" }
if (-not $Csv) { $Csv = "$Root\data\featurized_ThermoelectricMaterials_2026-08-22-snapfix.csv" }
foreach ($p in @($Python, $Csv, "$Repo\paper_b\src\lofo_paperb.py", "$Repo\$SplitsDir\manifest.json")) { if (-not (Test-Path $p)) { throw "missing: $p (run setup_windows.ps1 first)" } }
New-Item -ItemType Directory -Force -Path "$Root\ckpt", "$Root\logs", "$Root\quarantine" | Out-Null

$env:PYTHONUNBUFFERED = "1"; $env:PYTHONDONTWRITEBYTECODE = "1"
$env:OMP_NUM_THREADS = "2"; $env:MKL_NUM_THREADS = "2"; $env:OPENBLAS_NUM_THREADS = "2"; $env:LOKY_MAX_CPU_COUNT = "2"
$env:CUDA_VISIBLE_DEVICES = "$GpuIndex"   # the V100S is the only GPU; with this set, --gpu-index is 0 for the code
$common = @("-m", "paper_b.src.lofo_paperb", "--csv", $Csv, "--labels-run", $LabelsRun, "--splits-dir", $SplitsDir)

switch ($Mode) {
    "smoke" {
        $ck = "$Root\ckpt\smoke_gpu_pooled"
        if (Test-Path $ck) { Remove-Item -Recurse -Force $ck }
        $argList = $common + @("--role", "gpu-pooled", "--workers", "1", "--gpu-index", "0", "--targets", "zT", "--units", "manganite,i_v_vi2", "--repeats", "1", "--folds", "0,1",
            "--tuning-trials", "2", "--smoke-search-space-cap", "--checkpoint-dir", $ck)
        $log = "$Root\logs\paper_b_smoke.log"
    }
    "pooled" {
        $ck = "$Root\ckpt\ckpt_gpu_pooled"
        $argList = $common + @("--role", "gpu-pooled", "--workers", "1", "--gpu-index", "0", "--checkpoint-dir", $ck)
        $log = "$Root\logs\paper_b_pooled.log"
    }
    "specialist" {
        $env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"; $env:LOKY_MAX_CPU_COUNT = "1"   # 4 workers, 1 thread each (doc 8.7)
        $ck = "$Root\ckpt\ckpt_cpu_specialist"
        $argList = $common + @("--role", "cpu-specialist", "--workers", "4", "--checkpoint-dir", $ck)
        $log = "$Root\logs\paper_b_specialist.log"
    }
    "control" {
        $ck = "$Root\ckpt\ckpt_gpu_control"
        $argList = $common + @("--role", "gpu-specialist-control", "--workers", "1", "--gpu-index", "0", "--checkpoint-dir", $ck)
        $log = "$Root\logs\paper_b_control.log"
    }
}
& $Python "$PSScriptRoot\verify_outputs.py" --paperb-dir $ck --quarantine "$Root\quarantine" --since-hours 72
Write-Host "$(Get-Date -Format u) paper_b $Mode : $($argList -join ' ')"
Push-Location $Repo
$ErrorActionPreference = "Continue"   # Windows PowerShell 5.1 turns redirected native stderr lines into terminating errors under "Stop"
& $Python -u @argList 2>&1 | Tee-Object -FilePath $log -Append
$code = $LASTEXITCODE
$ErrorActionPreference = "Stop"
Pop-Location
if ($code -ne 0) { throw "paper_b $Mode exited with code $code (see $log); rerun this command to resume" }
Write-Host "$(Get-Date -Format u) paper_b $Mode finished; progress files: $ck\status_worker*.json"
