param(
  [string]$Config = "configs/scenario.json",
  [string]$Checkpoint = "checkpoints_hierarchical_native256x240_frame1/hierarchical_latest.pt",
  [int]$Episodes = 20
)

$ErrorActionPreference = "Stop"
python -m tennis_rl.evaluate --config $Config --checkpoint $Checkpoint --episodes $Episodes
