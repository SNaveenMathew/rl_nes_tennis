param(
  [string]$Config = "configs/scenario.json"
)

$ErrorActionPreference = "Stop"
python -m tennis_rl.train --config $Config --smoke-test
