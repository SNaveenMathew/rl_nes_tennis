param(
  [string]$Config = "configs/scenario.json",
  [string]$Resume = ""
)

$ErrorActionPreference = "Stop"

if ($Resume) {
  python -m tennis_rl.train --config $Config --resume $Resume
} else {
  python -m tennis_rl.train --config $Config
}
