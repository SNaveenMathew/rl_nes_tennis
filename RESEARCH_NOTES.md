# Demonstration-learning design notes

This repository uses human gameplay as action-labeled trajectory data. The implementation is an adaptation for the existing NES Tennis PPO stack, not a claim to reproduce any published algorithm exactly.

## Evidence used

- Hester et al., *Deep Q-learning From Demonstrations*, AAAI 2018. DQfD combines demonstration action classification with reinforcement-learning updates and reports large early-learning gains on Atari. This supports the choice to use a supervised action loss during a pretraining/warm-start phase and while RL training continues.
- Ross, Gordon & Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*, AISTATS 2011. The paper explains why sequential imitation differs from i.i.d. classification because the learned policy changes the states it encounters. This supports continuing with PPO fine-tuning instead of stopping at behavior cloning.
- Goecks et al., *Efficiently Combining Human Demonstrations and Interventions for Safe Training of Autonomous Systems in Real-Time*, AAAI 2019. This supports the general demonstration-then-RL / corrective-human-interaction pattern; it does not establish this exact NES Tennis implementation.
- Mesen2 Lua documentation: `emu.getInput(port)` returns the state of controller buttons; `inputPolled` is the documented event for input handling; `getScreenBuffer()` provides the current console framebuffer. The recorder uses `getInput()` only in record mode, so the human remains in control.

## Important data-label rule

A match-level loss is not equivalent to every action being wrong. Therefore the recorder stores the actual human action at every step and uses the final match result only to choose the `winning` or `losing` dataset bucket and to weight sampling. Losing trajectories remain useful state coverage; they are not converted into negative action labels.

## Episode definition

The smallest training unit is a tennis point: observations/actions are accumulated until the existing bridge detects a score transition, point-end flag, or match termination. Each point is saved with four stacked 256×240 grayscale observations and the existing 27-action controller representation. New recordings also retain aligned native 256×240 RGB frames for human annotation when `capture_native_frames` is enabled.

## Training

Normal training automatically loads point episodes, performs configurable behavior-cloning warm-start updates, then runs the existing recurrent PPO algorithm with an additional demonstration action-classification loss. The PPO policy still chooses the next action from the full 27-action space during environment interaction.

## Partial demonstration supervision (2026-09-14)

This revision adds short-window supervision over recorded point trajectories without inventing
unobserved ball/contact labels.

* Demonstration points can be sampled as fixed-length windows (`demo_window_length`, default 64).
* Each window keeps the recorded human action as the policy target.
* The environment rewards already present in the recorded trajectory are converted to a discounted
  return-to-go using `demo_return_gamma` (default 0.99). Positive-return portions receive higher
  imitation weight; negative-return portions are down-weighted but are not relabeled as incorrect
  human actions.
* A small auxiliary binary head predicts whether the partial point ultimately belongs to a winning
  or losing point. This uses only the recorded point outcome and is separate from the policy action label.
* PPO continues to optimize the environment objective. Demonstration supervision is an auxiliary
  update, not a replacement for PPO.

This design is intentionally conservative. The current ROM/bridge exposes score, game and point-end
signals, but the repository does not contain a validated mapping from RAM to ball position, contact,
or in/out events. Therefore this revision does **not** fabricate automatic "shot returned", "ball landed
in", or "movement correct" rewards. Adding those rewards would require a validated game-state decoder
or explicit human event annotations.

Research basis:
* Hester et al., "Deep Q-learning From Demonstrations," AAAI 2018, DOI 10.1609/aaai.v32i1.11757.
  DQfD combines temporal-difference learning with supervised demonstrator-action classification and
  uses demonstrations to improve early learning.
* Hosu & Rebedea, "Playing Atari Games with Deep Reinforcement Learning and Human Checkpoint Replay",
  2016. Human-controlled checkpoints were used to address sparse-reward exploration.
* Gelada et al., "DeepMDP: Learning Continuous Latent Space Models for Representation Learning," ICML
  2019, PMLR 97:2170-2179. Auxiliary reward/transition prediction objectives were shown to improve
  Atari representation learning.
* Lyle et al., "On the Effect of Auxiliary Tasks on Representation Dynamics," AISTATS 2021, PMLR
  130:1-9. Auxiliary tasks can shape representations, particularly in sparse-reward settings.

## Segment-Based Point RL (2026-10-17)

Training an agent over monolithic tennis points (~60-600 steps) with a single sparse terminal reward creates severe credit-assignment failure. This produces two degenerate failure modes:
1. **The "Do Not Serve" Exploit**: When the agent serves first and only receives negative rewards on losing points, not serving means the point never starts and avoids negative reward.
2. **Random Action Divergence**: Adding indiscriminate step penalties forces the policy into high-entropy random flailing, missing the precise 1-frame stroke timing.

### Architecture & Grounded Mechanisms:

1. **Sub-episode Segmentation & Reverse Curriculum**:
   - Following Salimans & Chen (OpenAI 2018) and Hosu & Rebedea (2016), points are trained via sequential segments:
     - `late` (15-40 steps): Endgame resolution. Immediate reward (+1 on win, -1 on loss).
     - `mid` (20-50 steps): In-rally positioning and ball interception.
     - `start` (15-25 steps): Service and return initiation.
   - Reverse curriculum unlocks `late` first, then `mid`, then `start`, ensuring the policy already knows how to rally and finish before learning the serve.

2. **Phase-Specific Horizons & Reward Structure**:
   - **Serve Stall Penalty**: If a `start` (serve) segment reaches its 25-step horizon without the ball being served into play or a score change, a **service stall/fault penalty** (-1.0) is assigned and the episode terminates. Inaction is guaranteed to lose points.
   - **Rally Survival Progress Reward**: If a `mid` segment completes its 50-step horizon without an unforced error / point loss, the agent receives a $+0.20$ progress reward, and the future return is bootstrapped from the critic value $V(s_{t+1})$ via truncation.
   - **Terminal Point Outcomes**: Natural point wins grant $+1.0$; natural point losses grant $-1.0$.

3. **Demonstration Micro-Segment Regularization**:
   - Demonstrations are partitioned into micro-segments (16-24 steps) stratified across `serve`, `stroke`, `late`, and `approach`.
   - **Episode-Wide Return-To-Go**: In lost demonstration points, only the terminal blunder segment receives negative RTG (-1.0). Early and mid rally segments (valid serves and exchanges) receive continuation credit (+0.15) and are imitated to learn clean stroke mechanics without negative bias.
   - Factorized BC loss across `movement_head` (9) and `shot_head` (3) is backpropagated in the unified PPO update step.

Research basis:
* Salimans & Chen, "Learning Montezuma's Revenge from a Single Demonstration", OpenAI, 2018.
* Hosu & Rebedea, "Playing Atari Games with Deep Reinforcement Learning and Human Checkpoint Replay", 2016.
* Florensa et al., "Reverse Curriculum Generation for Reinforcement Learning", CoRL 2017.
* Sutton & Barto, "Reinforcement Learning: An Introduction", 2nd ed., 2018 (Section on Truncation vs Termination and Bootstrapping).
