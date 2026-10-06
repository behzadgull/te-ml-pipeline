<#
.SYNOPSIS
  Zip the run outputs (stacking out-dirs, Paper B checkpoint dirs, logs) to a second location; keep the newest -Keep archives.

.DESCRIPTION
  Uses Windows' built-in tar.exe (bsdtar, `-a` picks zip from the .zip extension), which has no 2 GB limit (Compress-Archive in Windows PowerShell 5.1 does). The runs keep writing while it
  zips, so an archive can hold a file mid-write; that is harmless (the harnesses write atomically and verify_outputs.py quarantines anything unreadable after a restore).
  Each archive is listed back after writing; a failure leaves the older archives alone.

  Run once:      powershell -ExecutionPolicy Bypass -File C:\te_run\dept\backup.ps1 -BackupDir E:\te_backup
  Every day at 03:00 (run once, in an elevated or normal prompt; it runs as you, only while you are logged in):
      schtasks /Create /TN te_backup /SC DAILY /ST 03:00 /TR "powershell -NoProfile -ExecutionPolicy Bypass -File C:\te_run\dept\backup.ps1 -BackupDir E:\te_backup"
#>
param(
    [Parameter(Mandatory = $true)][string]$BackupDir,
    [string]$Root = "C:\te_run",
    [int]$Keep = 7
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$zip = Join-Path $BackupDir "te_run_$stamp.zip"
$parts = @("out", "ckpt", "logs") | Where-Object { Test-Path "$Root\$_" }
if ($parts.Count -eq 0) { throw "nothing to back up under $Root" }
$prev = $ErrorActionPreference
$ErrorActionPreference = "Continue"
tar.exe -a -c -f $zip -C $Root @parts
$code = $LASTEXITCODE
$ErrorActionPreference = $prev
if ($code -ne 0) { Remove-Item -Force -ErrorAction SilentlyContinue $zip; throw "tar failed with exit code $code; no archive written" }
$n = (tar.exe -tf $zip | Measure-Object).Count
if ($n -lt 1) { Remove-Item -Force $zip; throw "the archive is empty" }
Write-Host ("{0} written: {1:N0} MB, {2} entries" -f $zip, ((Get-Item $zip).Length / 1MB), $n)
Get-ChildItem $BackupDir -Filter "te_run_*.zip" | Sort-Object Name -Descending | Select-Object -Skip $Keep | ForEach-Object { Write-Host "removing old archive $($_.Name)"; Remove-Item -Force $_.FullName }
