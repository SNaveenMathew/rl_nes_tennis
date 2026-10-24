param(
    [string]$Config = "configs/scenario.json",
    [string]$DemoRoot = "demonstrations",
    [string]$Foundation = "checkpoints_hierarchical_native256x240_frame1/stages/foundation.pt",
    [string]$OutDir = "checkpoints_hierarchical_native256x240_frame1/stages",
    [int]$Updates = 1000,
    [int]$BatchSize = 64,
    [int]$EvalEvery = 50,
    [string[]]$Tasks = @(
        "serve_timing", "serve_placement_power", "serve_recovery",
        "receive_position", "return_stroke_timing", "intercept_track",
        "hold_cover", "recover_reposition", "approach_net", "retreat_lob",
        "stroke_selector", "stroke_timing", "transition_wait_reposition"
    )
)
$ErrorActionPreference = "Stop"
if (!(Test-Path $Foundation)) { throw "Foundation checkpoint not found: $Foundation" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
foreach ($task in $Tasks) {
    $out = Join-Path $OutDir ("leaf_{0}.pt" -f $task)
    Write-Host "`n=== Training leaf: $task ===" -ForegroundColor Cyan
    python -m tennis_rl.staged_train `
        --config $Config --demo-root $DemoRoot --stage leaf --skill $task `
        --init-from $Foundation --split train --updates $Updates `
        --batch-size $BatchSize --eval-every $EvalEvery --output $out --device cuda
    if ($LASTEXITCODE -ne 0) {
        throw "Leaf stage '$task' failed with exit code $LASTEXITCODE. Check exact task labels and train-split counts with: python -m tennis_rl.dataset stats --demo-root $DemoRoot"
    }
}
Write-Host "Finished training selected leaf experts. Assemble the desired experts with tennis_rl.assemble_hierarchy, then train --stage gates." -ForegroundColor Green
