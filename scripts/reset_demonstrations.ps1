param(
    [switch]$IncludeScenarioStates
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Targets = @(
    (Join-Path $RepoRoot "demonstrations"),
    (Join-Path $RepoRoot "contact_sheets"),
    (Join-Path $RepoRoot "scenario_catalog.json")
)
if ($IncludeScenarioStates) {
    $Targets += (Join-Path $RepoRoot "scenario_states")
}

Write-Host "This operation permanently removes the following generated paths:" -ForegroundColor Yellow
foreach ($Target in $Targets) {
    Write-Host "  $Target"
}
Write-Host "It does not remove code, ROMs, Mesen, or checkpoint directories."
$Confirmation = Read-Host 'Type exactly RESET DEMONSTRATIONS to continue'
if ($Confirmation -cne "RESET DEMONSTRATIONS") {
    Write-Host "Cancelled; nothing was removed."
    exit 1
}

foreach ($Target in $Targets) {
    if (Test-Path -LiteralPath $Target) {
        $Resolved = (Resolve-Path -LiteralPath $Target).Path
        if (-not $Resolved.StartsWith($RepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove a path outside the repository: $Resolved"
        }
        Remove-Item -LiteralPath $Resolved -Recurse -Force
        Write-Host "Removed $Resolved"
    }
}
Write-Host "Reset complete. The recorder will create a fresh demonstrations directory when the first point is saved."
