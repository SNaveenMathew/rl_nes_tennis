param(
    [ValidateSet("game", "point")]
    [string]$Sampling = "game",
    [int]$Points = 1,
    [ValidateSet("serve-first", "receive-first")]
    [string]$Serve = "serve-first",
    [ValidateSet("deuce", "ad", "advantage")]
    [string]$Side = "deuce",
    [int]$Level = 1,
    [string]$Output = "demonstrations",
    [string]$Config = "configs/scenario.json",
    [int]$FrameSkip = 0,
    [string]$ScenarioTag = ""
)

$ErrorActionPreference = "Stop"
if ($FrameSkip -lt 0) { throw "FrameSkip must be 0 (use config) or a positive integer." }
if ($Sampling -eq "game" -and ($Side -eq "ad" -or $Side -eq "advantage")) {
    throw "--side (ad/advantage) is only supported when -Sampling is 'point'; a regular game always starts on the deuce side."
}
# Equivalent CLI form: python -m tennis_rl.record --config $Config --level $Level --output $Output --serve $Serve --side $Side --sampling $Sampling
$RecordArgs = @("-m", "tennis_rl.record", "--config", $Config, "--level", $Level, "--output", $Output, "--serve", $Serve, "--side", $Side, "--sampling", $Sampling)
if ($Sampling -eq "point" -and $Points -gt 0) {
    $RecordArgs += @("--points", $Points)
}
if ($FrameSkip -gt 0) {
    $RecordArgs += @("--frame-skip", $FrameSkip)
}
if (-not [string]::IsNullOrWhiteSpace($ScenarioTag)) {
    $RecordArgs += @("--scenario-tag", $ScenarioTag)
}
python @RecordArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
