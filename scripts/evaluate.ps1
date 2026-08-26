param(
  [string]$Config = "configs/base.json",
  [string]$Checkpoint = "checkpoints/tennis_latest.pt",
  [int]$Episodes = 20
)

$ErrorActionPreference = "Stop"
python -m tennis_rl.evaluate --config $Config --checkpoint $Checkpoint --episodes $Episodes
