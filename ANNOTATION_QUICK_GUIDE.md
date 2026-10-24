# NES Tennis: recording and annotation quick guide

## 1. Image resolution: what is actually stored?

The NES framebuffer is **256×240 RGB**. With `configs/scenario.json`, each *new recording* stores both:

- `native_frames`: the original 256×240 RGB frame for readable annotation.
- `observations`: a 256×240 grayscale input (four frames stacked) used by training/inference.

The active frame-perfect configuration uses full 256×240 spatial resolution with grayscale policy observations and one decision per raw NES frame (`frame_skip=1`). This costs more memory and compute than smaller inputs. Keep these settings consistent across recording, training, evaluation, and inference.

Old NPZs with 84×84 or 128×120 policy observations cannot regain discarded pixels. Since this project is starting clean, do not mix those files with the new set: use the reset script below or choose a new empty demonstration root. New native frames are captured only when recording with `mesen.capture_native_frames=true`.

Check the data on disk before doing anything:

```powershell
python -m tennis_rl.dataset stats --demo-root demonstrations
```

Look for `native_capture_coverage`, `stored_observation_resolution_counts_by_point`, `recorded_frame_skip_counts_by_point`, and `controller_action_counts`. Confirm new points have native 256×240 capture and A/B actions appear if you pressed them.

## 2. Frame sampling: frame-perfect contract

The active configuration is intentionally fixed at `mesen.frame_skip=1`, `observation_width=256`, and `observation_height=240`. Each recorded observation/action pair corresponds to one raw NES frame (about 16.6 ms). Runtime validation rejects larger frame skips or smaller policy inputs, and the hierarchical manager/action pulses are constrained to one frame. This provides frame-level action resolution; it does not guarantee a perfect learned policy.

`--sample-every` is different: it only skips images in annotation sheets. `--sample-every 1` shows every saved step and does **not** change recording or training timing. Segment length and stride count saved steps; at `frame_skip=1`, a 16-step segment spans 16 raw NES frames, about 0.27 seconds.

## 3. Reset the old data and record from scratch

The reset script removes generated demonstrations/contact sheets and, when the switch is supplied, scenario savestates. It asks you to type a confirmation phrase. It will not delete source code, ROMs, the emulator, or unrelated folders. Run this from the repository root: `.\scripts\reset_demonstrations.ps1 -IncludeScenarioStates`. Then record both service roles into the newly created empty dataset using the explicit high-resolution config:

```powershell
python -m tennis_rl.record --config configs/scenario.json --output demonstrations --frame-skip 1 --serve serve-first --scenario-tag level1_serve_rally
python -m tennis_rl.record --config configs/scenario.json --output demonstrations --frame-skip 1 --serve receive-first --scenario-tag level1_receive_rally
```

Finish the recorded game normally. `receive-first` skips Game 1 and records Game 2, when the human receives. Use a unique descriptive tag for each role/level. After one short test recording, rerun `dataset stats` and verify native coverage, action counts, stored 256×240 resolution, and frame skip 1 before making lots more data. Do not start large-scale recording until the native frame path has passed this audit.

## 4. Make large, readable review images

Choose only a few points at first. `native_frames` are used when present; they are the actual Mesen pixels, not an upscale. Use one column, 4× nearest-neighbor magnification, four frames per sheet page, and an animated replay:

```powershell
python -m tennis_rl.annotate_demonstrations --demo-root demonstrations --interactive --sample-every 1 --columns 1 --scale 4 --max-cells 4 --review-gifs --gif-scale 3
```

The command first prints point indexes. At the prompt enter indexes such as `1,8,22` (zero-based). It creates very large, scrollable PNG pages and opens a step-indexed GIF that shows all saved steps for each selected point. GIF playback is only a visual aid; its playback speed is **not** the game's real timing. Use the step number, action and pulse caption, and the frame-skip metadata to know the actual interval. For very long points, omit `--gif-scale 3` or select a shorter point to avoid a large GIF.

If a page says `LEGACY UPSCALE`, it came from old, already-downsampled data. Do not annotate precise ball coordinates from a fuzzy legacy upscale.

## 5. Task annotation: strict rules

First annotate task intervals only (do **not** add `--state-labels` in this first pass). Enter `start,end,label`; ranges are half-open: `10,15,intercept_track` labels steps 10–14. Indices are the saved observation/action steps shown on the GIF/sheet. The observation at a step is the state before its corresponding recorded action interval.

**Apply these rules to every point:**

1. Label the human's visible task/decision, never whether the point was won.
2. Use one label per step; intervals may touch but must never overlap.
3. If two behaviors coincide, prioritize the actual stroke/return press decision, then lob-specific retreat, net approach, interception movement, recovery, and finally holding coverage.
4. Use a leaf label only when the evidence is clear. Unknown means leave the step unlabeled; never guess ball position/contact/intent from the point result.
5. Leave `manager_duration` blank; do not set it equal to the annotation interval length.
6. If you enter new task rows for a point, those replace that point's prior task rows. If you press Enter without entering any rows, existing labels are preserved.

### Label lookup

| Label | Use it only when… |
|---|---|
| `serve_timing` | Human is serving and is waiting/preparing before the serve starts. Stop at serve initiation. |
| `serve_placement_power` | The human is serving and a deliberate aiming/power choice is actually identifiable from the action/state. Otherwise skip. |
| `serve_recovery` | Human's serve has started and they move to cover the expected return. |
| `receive_position` | CPU is serving; human adjusts position before trying to return it. |
| `return_stroke_timing` | CPU serve is arriving and the human's return press is imminent/occurring. |
| `intercept_track` | Live rally: human moves to reach an incoming ball; exclude the shot-press moment. |
| `hold_cover` | Human clearly stays in a useful court position and no immediate interception move is needed. Do not label mere inactivity if the intention is unclear. |
| `recover_reposition` | After the human's stroke, they move toward useful coverage while play continues. |
| `approach_net` | Human visibly advances toward the net to take an attacking position. |
| `retreat_lob` | A clearly high/floating lob prompts retreat/turning back. Ordinary backward movement is not enough. |
| `stroke_selector` | The decision is normal shot versus lob before the press, and the chosen input is known. |
| `stroke_timing` | The ball is near a stroke opportunity and the decision is press now versus wait. |
| `transition_wait_reposition` | Dead time/between points or a serve-ready transition, not a live rally. |

Broad labels `serve`, `receive`, `baseline`, and `wait` train only the high-level phase gate—not a specialized leaf. In a normal rally, use `intercept_track`/`hold_cover`/`recover_reposition` only when their visible definitions fit; otherwise leave the steps unlabeled.

## 6. Annotate states separately (optional)

Task intervals are the priority. State-coordinate/event labels are a separate, more labor-intensive task; annotate only a few clear native frames at first. The default state fields are ball position, player position, agent contact window, and high floater/lob:

```powershell
python -m tennis_rl.annotate_demonstrations --demo-root demonstrations --interactive --state-labels --state-labels-only --state-fields ball_xy,player_xy,agent_contact_window,high_floater_lob --sample-every 1 --columns 1 --scale 4 --max-cells 4 --review-gifs
```

`--state-labels-only` **does not modify task_annotations.jsonl**. At each state prompt, ball/player coordinate input is normalized `x,y` in [0,1] against the original 256×240 frame; `(0,0)` is top-left and `(1,1)` bottom-right. Click the visible sprite's center. Event `1` means confidently yes, `0` confidently no, blank means unknown. Only label what can be seen; leave ambiguous ball locations blank. For `LEGACY UPSCALE` frames, skip precise coordinates.

## 7. Audit before training

```powershell
python -m tennis_rl.dataset stats --demo-root demonstrations
python -m tennis_rl.dataset split --demo-root demonstrations --holdout-fraction 0.2
python -m tennis_rl.dataset extract --demo-root demonstrations --segment-length 16 --stride 8
python -m tennis_rl.dataset stats --demo-root demonstrations
```

Do not delete a valid point just because it was lost. Inspect unusually short/long points in this fresh set; exclude a point only if its imagery/actions show it is an incomplete or invalid capture. Do not start specialist training while task annotation counts are zero or the action histogram does not show the A/B inputs you actually used. Keep train/holdout separation at the **whole-match** level because extracted windows overlap.
