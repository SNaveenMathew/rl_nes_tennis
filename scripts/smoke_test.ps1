param(
  [string]$Config = "configs/base.json"
)

$ErrorActionPreference = "Stop"
python -m tennis_rl.train --config $Config --smoke-test
