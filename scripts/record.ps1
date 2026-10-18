param(
    [int]$Level = 1,
    [string]$Output = "demonstrations",
    [ValidateSet("serve-first", "receive-first")]
    [string]$Serve = "serve-first"
)

python -m tennis_rl.record --config configs/base.json --level $Level --output $Output --serve $Serve
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
