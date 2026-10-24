# Fresh demonstration collection and training guide

This guide defines the recommended end-to-end data contract for this repository. It assumes the supplied `configs/scenario.json`: **256×240 grayscale policy input** (full spatial resolution), original **256×240 RGB** frames stored separately for annotation, and **`frame_skip=1`** (one decision per raw NES frame, about 60.1 decisions per second). Recording, training, evaluation, and deployment must use the same observation dimensions and frame skip.

## 1. Clean start

The new policy input shape changes model layer dimensions; old checkpoints are not compatible. The new default checkpoint and run directories are versioned with `native256x240_frame1` so they do not accidentally reuse the prior run outputs. Existing 84×84 or 128×120 demonstrations cannot be restored to full spatial resolution and should not be mixed into this fresh set; the old frame-skip contract is also different.

From PowerShell at the repository root:

```powershell
.\scripts\reset_demonstrations.ps1 -IncludeScenarioStates
```

When prompted, enter exactly `RESET DEMONSTRATIONS`. This removes only known generated paths: `demonstrations/`, `contact_sheets/`, `scenario_catalog.json`, and, when the explicit switch is supplied, `scenario_states/`. It does not remove source code, the ROM, or the Mesen executable. Make a backup first if any old recorded data may be useful.

## 2. Recording contract

`configs/scenario.json` already sets:

- policy image: 256×240 grayscale, four observations stacked;
- annotation image: aligned 256×240 RGB `native_frames` saved in the same point NPZ;
- temporal spacing: one raw NES frame per saved action/observation step;
- new checkpoint/log locations that are separate from legacy runs.

Run two separate recordings so both service roles are represented:

```powershell
python -m tennis_rl.record --config configs/scenario.json --output demonstrations --level 1 --frame-skip 1 --serve serve-first --scenario-tag level1_serve_rally
python -m tennis_rl.record --config configs/scenario.json --output demonstrations --level 1 --frame-skip 1 --serve receive-first --scenario-tag level1_receive_rally
```

Play the game normally. `receive-first` uses Game 1 as a warm-up and begins recording in Game 2, when Player 1 receives. Close Mesen or press Ctrl+C to finish. Record multiple complete matches, varying rally situations and actions; press A/B intentionally so both shot inputs are represented. Use descriptive tags per level and role.

You can run `scripts/record.ps1` instead; it defaults to `configs/scenario.json` and passes the same optional frame-skip/tag flags. Use `--frame-skip 1` for every recording in this dataset and train/evaluate using the same config and one-frame button pulse durations `[1]`. The shipped frame-perfect config intentionally rejects other frame skips. Never mix legacy recordings or checkpoints with this dataset.

## 3. Audit before more recording

```powershell
python -m tennis_rl.dataset stats --demo-root demonstrations
```

Check `native_capture_coverage`, `stored_observation_resolution_counts_by_point`, `recorded_frame_skip_counts_by_point`, and `controller_action_counts`. For this configuration, new point files should report 256×240 observations, frame skip 1, native RGB capture coverage for every step, and A/B shot actions when you pressed those buttons. If native frames are missing or metadata disagrees, stop and fix the recorder before collecting more data.

## 4. Review full-resolution frames

```powershell
python -m tennis_rl.annotate_demonstrations --demo-root demonstrations --make-sheets --sample-every 1 --columns 1 --scale 4 --max-cells 4 --review-gifs --gif-scale 3
```

The command indexes all point files and creates large nearest-neighbor contact-sheet pages plus an animated GIF for each point. `--sample-every 1` means show every saved step on the review sheets; it **does not** alter recording or training timing. Native RGB is used when present. Any `LEGACY UPSCALE` indication means the original native pixels are absent; do not annotate precise coordinates from an enlarged old image.

To interactively annotate selected points, omit `--make-sheets` and use `--interactive`; enter point indexes printed at startup. See [ANNOTATION_QUICK_GUIDE.md](ANNOTATION_QUICK_GUIDE.md) for complete task labels, half-open interval format, state coordinate conventions, and optional labels.

## 5. Annotation quality rules

- Task intervals use `[start, end)`: `20,27,intercept_track` labels steps 20 through 26.
- Never overlap task labels. Leave ambiguity unlabeled.
- Label the visible decision problem, not whether the point or match was won.
- Leave `manager_duration` blank unless a deliberate skill duration can be identified independently of the annotation interval.
- State coordinates refer to original 256×240 pixels normalized to `[0,1]`, with the origin at the top-left. Click the sprite center and label only clear cases.
- An explicit state/event annotation file is not generated by pixel heuristics. Do not train specialist/perception supervision while the corresponding annotation counts are zero.

## 6. Split and extract segment metadata

After you have enough varied matches and annotated examples:

```powershell
python -m tennis_rl.dataset stats --demo-root demonstrations
python -m tennis_rl.dataset split --demo-root demonstrations --holdout-fraction 0.2
python -m tennis_rl.dataset extract --demo-root demonstrations --segment-length 16 --stride 8
python -m tennis_rl.dataset stats --demo-root demonstrations
```

The split is by whole match to prevent overlapping segments from leaking between training and validation/holdout. Segment length and stride count saved steps: at `frame_skip=1`, a 16-step segment covers 16 raw NES frames, approximately 0.27 seconds.

## 7. Train and evaluate

Run the hierarchy on the freshly recorded set:

```powershell
python -m tennis_rl.train --config configs/scenario.json --algo hierarchical
```

Or compare the flat baseline paths:

```powershell
python -m tennis_rl.train --config configs/scenario.json --algo ppo
python -m tennis_rl.train --config configs/scenario.json --algo dqfd
```

Training now fails early if the demonstration dimensions or recorded frame skip disagree with the active configuration. Use the same config for evaluation/inference, unless you intentionally create a separate config whose temporal and spatial settings match the training run. Do not point at legacy checkpoints; the updated config uses the `*_native256x240_frame1` output directories.

Unit tests can check serialization, annotations, data audits, and bridge protocol, but they cannot prove the real Mesen executable/ROM behave correctly on every system or that the learned agent will win. Verify a short visible recording, run the data audit, then evaluate several held-out full matches before treating the model as improved.
