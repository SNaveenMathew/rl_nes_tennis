param(
    [string]$Config = "configs/scenario.json",
    [string]$Rom = "",
    [string]$Executable = ""
)
$ErrorActionPreference = "Stop"
$configObj = Get-Content $Config -Raw | ConvertFrom-Json
if ($Executable) { $configObj.mesen.executable = $Executable }
if ($Rom) { $configObj.mesen.rom = $Rom }
$exe = [System.IO.Path]::GetFullPath($configObj.mesen.executable)
$romPath = [System.IO.Path]::GetFullPath($configObj.mesen.rom)
$lua = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $configObj.mesen.lua_script))
foreach ($entry in @(@("Mesen executable", $exe), @("NES ROM", $romPath), @("Lua bridge", $lua))) {
    if (!(Test-Path $entry[1] -PathType Leaf)) { throw "$($entry[0]) not found: $($entry[1])" }
    Write-Host "$($entry[0]): $($entry[1])"
}
Write-Host "Starting one-worker bridge smoke test; this launches Mesen and checks actual IPC." -ForegroundColor Cyan
# Keep the authoritative config intact by writing a temp config with any overrides.
$temp = Join-Path $env:TEMP ("tennis-mesen-smoke-{0}.json" -f $PID)
$configObj.mesen.workers = 1
$json = $configObj | ConvertTo-Json -Depth 40
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($temp, $json, $utf8NoBom)
try {
    python -m tennis_rl.train --config $temp --algo hierarchical --workers 1 --smoke-test
    if ($LASTEXITCODE -ne 0) { throw "Training smoke test failed with exit code $LASTEXITCODE" }
} finally {
    Remove-Item $temp -Force -ErrorAction SilentlyContinue
}
Write-Host "Mesen bridge smoke test completed successfully." -ForegroundColor Green
