# NES Tennis Demonstration Recording Guide

An exhaustive guide for recording high-quality human demonstrations to train the reinforcement learning agent quickly, efficiently, and effectively.

---

## 1. Why Demonstration Diversity Matters

The agent uses a hybrid training architecture:
1. **Micro-Segment Replayer (`segment_replay.py`)**: Supervised behavioral cloning on 16-step micro-windows with contrastive weighting (higher weight on winning segments, lower weight on losing segments).
2. **Reverse Curriculum Scenario Training (`scenario_sampler.py`)**: Loads savestates directly into game moments (`late`, `mid`, `start`) partitioned into 36 score/court cells.
3. **Motion Auxiliary Head (`model.py`)**: Self-supervised spatial tracking of the ball and player sprites.

When certain score cells or tactical shot types are missing from demonstrations, the agent receives zero imitation signal for those game contexts and must rely purely on sparse random exploration.

---

## 2. Complete Score Grid (36 Cells) & Missing Strata

NES Tennis tracks matches by game score: `0-0`, `15-0`, `30-0`, `40-0`, `Deuce`, `Advantage`, etc. The scenario system divides these into **36 cells** across **20 strata** based on serving player (`serve` vs `return`) and court side (`deuce` / right vs `ad` / left).

### Currently Uncovered / Priority Cells to Record
Play matches deliberately to produce points starting from these score states:

| Role | Early / Opening Game | Mid-Game Tight Scores | High-Pressure / Deciding Points |
| :--- | :--- | :--- | :--- |
| **Player Serving** (`serve`) | `serve\|0-1`<br>`serve\|0-2`<br>`serve\|0-3` | `serve\|1-2`<br>`serve\|1-3`<br>`serve\|2-0`<br>`serve\|2-2`<br>`serve\|2-3` | `serve\|3-0`<br>`serve\|3-2`<br>`serve\|3-3` (Deuce)<br>`serve\|4-3` (Ad-In / Game Point)<br>`serve\|3-4` (Break Point Down) |
| **Player Receiving** (`return`) | `return\|0-2`<br>`return\|0-3`<br>`return\|1-0` | `return\|2-0`<br>`return\|2-1`<br>`return\|2-2`<br>`return\|2-3` | `return\|3-0`<br>`return\|3-1`<br>`return\|3-2`<br>`return\|3-3` (Deuce Return)<br>`return\|4-3` (Break Point Up)<br>`return\|3-4` (Ad-Out / Defending Game Point) |

---

## 3. Exhaustive Tactical Scenarios Taxonomy

Record demonstrations targeting the 6 scenario categories below:

### Category A: Service Placements & Timing
*Mode: `--serve serve-first`*

1. **Flat Center "T" Serve (Deuce Court / Right Side)**:
   - *Execution*: Position Mario near the center hash mark. Press `A` to toss. At the very peak of the toss, press `D-Pad Left + A`.
   - *Outcome*: A fast, flat trajectory driving directly down the center line. Catches the AI moving slowly to the center.
2. **Wide Slice Serve (Deuce Court / Right Side)**:
   - *Execution*: Position toward the right singles sideline. Toss with `A`, then press `D-Pad Right + A` slightly below peak height.
   - *Outcome*: Sharp outward slice that drags the AI off-court into the doubles alley.
3. **Flat Center "T" Serve (Ad Court / Left Side)**:
   - *Execution*: Position near the center hash mark on the left. Toss with `A`, strike at peak with `D-Pad Right + A`.
   - *Outcome*: Fast drive down the middle "T" line.
4. **Wide Slice Serve (Ad Court / Left Side)**:
   - *Execution*: Position wide on the left baseline. Strike with `D-Pad Left + A` to pull the AI wide to their forehand side.
5. **Safe Second Serve**:
   - *Execution*: Strike slightly late as the ball is descending.
   - *Outcome*: High-clearance, slower delivery into the center service box to prevent double faulting.

---

### Category B: Return of Serve Variations
*Mode: `--serve receive-first`*

1. **Deep Crosscourt Return (High Percentage)**:
   - *Execution*: Position behind the baseline. If opponent serves to your forehand, hit diagonally across court with `Down-Right + A`. If backhand, hit with `Down-Left + A`.
   - *Outcome*: Bounces deep near the opposite baseline, pushing the server back.
2. **Down-the-Line Passing Return**:
   - *Execution*: When the opponent delivers a wide serve, step forward and strike straight down the sideline (`Up-Right + A` or `Up-Left + A`).
   - *Outcome*: Clean winner down the open tramline before the opponent can recover to the center.
3. **Compact Block / Defensive Return**:
   - *Execution*: Against fast first serves, hit without holding D-pad forward or wide; simply press `A` or `B` with minimal movement.
   - *Outcome*: Safely resets the point into a neutral baseline rally.

---

### Category C: Baseline Rallying & Stroke Mechanics
*Modes: Both `--serve serve-first` and `--serve receive-first`*

1. **Sustained Crosscourt Rally (3–6 Shot Exchanges)**:
   - *Execution*: Trade 3–6 deep crosscourt balls with the opponent. Do not rush for an immediate winner on shot 1.
   - *Outcome*: Provides extended temporal sequences for the GRU and PPO sequence buffer (`sequence_length = 32`).
2. **Sudden Change of Direction (The "Inside-Out" Winner)**:
   - *Execution*: Establish a crosscourt pattern for 2 shots, then suddenly redirect down the line into the empty court corner.
   - *Outcome*: Teaches the policy to exploit open-court space.
3. **Short Angle Crosscourt**:
   - *Execution*: Take the ball early on the rise near the service line and angle sharply across.
   - *Outcome*: Creates sharp passing shots that pull the AI completely off screen.
4. **Active Center Recovery Steps**:
   - *Execution*: Immediately after every stroke, take 2–3 recovery steps back toward the center baseline mark.
   - *Outcome*: Trains the policy out of "standing still after hitting", which is the primary cause of unforced errors.

---

### Category D: Net Play & Volleys
*Modes: Both `--serve serve-first` and `--serve receive-first`*

1. **Serve-and-Volley**:
   - *Execution*: Serve flat down the "T", immediately sprint straight forward to the service line / net, and punch a forehand or backhand volley (`B` or `A`) into the open corner.
   - *Outcome*: High-efficiency short-horizon winning sequences.
2. **Approach Shot & Punch Volley**:
   - *Execution*: When the opponent hits a weak short ball bouncing inside your service line, step forward, hit a deep drive, continue forward to the net, and volley the next return.
   - *Outcome*: Teaches smooth offensive transition from defense/neutral to attacking.
3. **Stop / Drop Volley**:
   - *Execution*: Positioned at the net when the opponent is pinned deep behind their baseline, tap `B` softly to drop the ball just over the net.
   - *Outcome*: Teaches touch control and short-ball placement.
4. **Reflex Volley**:
   - *Execution*: Standing 1–2 player lengths from the net, block balls struck directly at player body level into open space.

---

### Category E: Defensive Scramble & Recovery
*Modes: Both `--serve serve-first` and `--serve receive-first`*

1. **High Defensive Lob**:
   - *Execution*: When pushed wide into a deep corner or when the opponent charges the net, hold `Down + B` (or lob input) to hit high over the opponent.
   - *Outcome*: Resets positioning and provides anti-rush defense.
2. **Cross-Court Scramble**:
   - *Execution*: Sprint diagonally across the court after being pulled wide to recover a ball bouncing in the opposite corner.
   - *Outcome*: Teaches extreme spatial tracking and recovery movement.
3. **Controlled Losing Points (5–10 Points Total)**:
   - *Execution*: Engage in a genuine 4–6 shot rally before losing the point on a narrow out-ball or tough winner.
   - *Outcome*: Critical for training the auxiliary point outcome head and contrastive segment replayer. Avoid points where the player simply stands still; high-quality losing points show genuine effort followed by a mistake.

---

## 4. Recording Commands & Workflows

### Command for Player Serving (`serve-first`):
Game 1 is a warm-up. Control switches to recording on Game 2 when the player serves:
```powershell
python -m tennis_rl.record `
  --config configs/base.json `
  --level 1 `
  --output demonstrations `
  --serve serve-first `
  --harvest-dir scenario_states `
  --settle-steps 4 `
  --harvest-every 15
```

### Command for Player Receiving (`receive-first`):
Game 1 is a warm-up. Control switches to recording on Game 2 when the opponent serves:
```powershell
python -m tennis_rl.record `
  --config configs/base.json `
  --level 1 `
  --output demonstrations `
  --serve receive-first `
  --harvest-dir scenario_states `
  --settle-steps 4 `
  --harvest-every 15
```

---

## 5. Post-Recording Pipeline

After recording 5–10 new matches:

### 1. Harvest Savestates into the Scenario Catalog:
```powershell
python -m tennis_rl.record --harvest-dir scenario_states
```

### 2. Inspect Catalog Coverage Across the 36 Cells:
```powershell
python -m tennis_rl.scenario_preview
```
Check how many cells out of 36 are now populated. Aim for **at least 28–32 populated cells**.

### 3. Resume Scenario-PPO Training:
```powershell
python -m tennis_rl.train `
  --config configs/scenario.json `
  --resume checkpoints/tennis_latest.pt
```
The model will automatically utilize the new self-supervised motion auxiliary head, the adaptive ZPD curriculum, and the demonstration micro-segments harvested from your play.
