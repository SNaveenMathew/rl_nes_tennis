param(
    [int]$Level = 1,
    [string]$Output = "demonstrations",
    [ValidateSet("serve-first", "receive-first")]
    [string]$Serve = "serve-first",
    [string]$Config = "configs/scenario.json",
    [int]$FrameSkip = 0,
    [string]$ScenarioTag = ""
)

$ErrorActionPreference = "Stop"
if ($FrameSkip -lt 0) { throw "FrameSkip must be 0 (use config) or a positive integer." }
# Equivalent CLI form: python -m tennis_rl.record --config $Config --level $Level --output $Output --serve $Serve
$RecordArgs = @("-m", "tennis_rl.record", "--config", $Config, "--level", $Level, "--output", $Output, "--serve", $Serve)
if ($FrameSkip -gt 0) {
    $RecordArgs += @("--frame-skip", $FrameSkip)
}
if (-not [string]::IsNullOrWhiteSpace($ScenarioTag)) {
    $RecordArgs += @("--scenario-tag", $ScenarioTag)
}
python @RecordArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
