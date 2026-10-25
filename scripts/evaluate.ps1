param(
  [ValidateSet("", "game", "point")]
  [string]$Sampling = "",
  [ValidateSet("", "serve-first", "receive-first")]
  [string]$Serve = "",
  [int]$Level = 0,
  [int]$Episodes = 20,
  [string]$Config = "configs/scenario.json",
  [string]$Checkpoint = "checkpoints_hierarchical_native256x240_frame1/hierarchical_latest.pt"
)

$ErrorActionPreference = "Stop"

$EvalArgs = @("-m", "tennis_rl.evaluate", "--config", $Config, "--checkpoint", $Checkpoint, "--episodes", $Episodes)
if ($Sampling) {
  $EvalArgs += @("--sampling", $Sampling)
}
if ($Serve) {
  $EvalArgs += @("--serve", $Serve)
}
if ($Level -ge 1 -and $Level -le 5) {
  $EvalArgs += @("--level", $Level)
}

python @EvalArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
