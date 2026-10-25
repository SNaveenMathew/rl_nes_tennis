param(
  [ValidateSet("", "ppo", "hierarchical", "dqfd")]
  [string]$Algo = "ppo",
  [ValidateSet("", "point", "game")]
  [string]$Sampling = "",
  [ValidateSet("", "serve-first", "receive-first")]
  [string]$Serve = "",
  [int]$Level = 0,
  [string]$Config = "configs/scenario.json",
  [string]$Resume = "",
  [string]$InitFrom = "",
  [int]$Workers = 0,
  [int]$MaxUpdates = 0
)

$ErrorActionPreference = "Stop"

$TrainArgs = @("-m", "tennis_rl.train", "--config", $Config)
if ($Algo) {
  $TrainArgs += @("--algo", $Algo)
}
if ($Sampling) {
  $TrainArgs += @("--sampling", $Sampling)
}
if ($Serve) {
  $TrainArgs += @("--serve", $Serve)
}
if ($Level -ge 1 -and $Level -le 5) {
  $TrainArgs += @("--level", $Level)
}
if ($Resume) {
  $TrainArgs += @("--resume", $Resume)
}
if ($InitFrom) {
  $TrainArgs += @("--init-from", $InitFrom)
}
if ($Workers -gt 0) {
  $TrainArgs += @("--workers", $Workers)
}
if ($MaxUpdates -gt 0) {
  $TrainArgs += @("--max-updates", $MaxUpdates)
}

python @TrainArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
