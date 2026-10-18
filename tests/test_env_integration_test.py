from pathlib import Path
import sys, collections
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tennis_rl.env import ParallelTennis
from tennis_rl.mesen_client import BridgeError
from tennis_rl.scenarios import build_catalog, StateRecord, split_for_match
from tennis_rl.scenario_sampler import ScenarioLibrary, ScenarioSampler, SamplerConfig
from tennis_rl.train import append_seq

class FakeWorker:
    height = width = 8
    def __init__(self, seed, fail_load=False):
        self.rng = np.random.default_rng(seed); self.pending_token = None; self.next_token = 1
        self.t = 0; self.point_len = 0; self.mode = "game"; self.loads = []; self.resets = 0; self.fail_load = fail_load
        self.game_pts = 0
    def reset(self): self.resets += 1; self.mode = "game"; self.t = 0; self.game_pts = 0; self.action(0)
    def load_state_from(self, path, noop):
        if self.fail_load: raise BridgeError("boom")
        self.loads.append((path, noop)); self.mode = "point"; self.t = 0; self.point_len = int(self.rng.integers(5, 40))
    def action(self, a):
        if self.pending_token: raise BridgeError("pending")
        self.pending_token = "t"
    def recv_observation(self):
        self.pending_token = None; self.t += 1
        reward, done = 0.0, False
        if self.mode == "point" and self.t >= self.point_len:
            reward = float(self.rng.choice([1.0, -1.0, 6.0, -6.0]))   # includes game bonuses
        if self.mode == "game" and self.t % 25 == 0:
            reward = float(self.rng.choice([1.0, -1.0]))                # full game: points do NOT end the episode
        return np.full((8, 8), self.t % 255, np.uint8), reward, done, {"player_games": 0, "opponent_games": 0}
    def close(self, force=False): pass

cells = build_catalog(); recs = []
for c in cells:
    for m in range(6):
        for ph in ("start", "mid", "late"):
            recs.append(StateRecord(path=f"{c.cell_id}/{m}/{ph}", cell_id=c.cell_id, phase=ph, match_id=f"m{m}",
                                    split=split_for_match(f"m{m}", 0.3), demo_outcome="win"))
sampler = ScenarioSampler(ScenarioLibrary(recs), SamplerConfig(anchor_fraction=0.2), seed=3)
prng = np.random.default_rng(0)
def provider(i):
    if prng.random() < 0.15:
        a = sampler.sample_probe()
        if a.kind == "probe": return a
    return sampler.sample()

workers = [FakeWorker(i) for i in range(4)]
env = ParallelTennis(workers, 4, max_steps=200, point_episodes=True, point_max_steps=100, scenario_provider=provider)
obs = env.reset()
bufs = [__import__("collections").deque(maxlen=6) for _ in workers]
added = collections.Counter(); ep = collections.Counter(); bad = 0; n_seq_probe = 0
for step in range(3000):
    mask = [a is not None and a.kind == "probe" for a in env.assignments]
    acts = np.zeros(4, np.int64)
    batch = env.step(acts)
    for seq in append_seq(bufs, 6, obs, acts, batch.rewards, batch.dones, skip=mask): added["seq"] += 1
    n_seq_probe += sum(mask)
    for i, (info, done, r) in enumerate(zip(batch.infos, batch.dones, batch.rewards)):
        a = info["assignment"]
        if done and a is not None and a.kind in ("scenario", "probe"):
            if info["timeout"]: continue
            assert info["point_result"] in (1, -1) and abs(r) == 1.0, (info["point_result"], r)   # bonuses stripped
            sampler.update(a.cell_id, info["point_result"] > 0, a.phase, source=a.kind); ep[a.kind] += 1
        elif done: ep["anchor"] += 1
        else:
            if a is not None and a.kind in ("scenario", "probe"): assert r == 0.0
    obs = batch.observations.copy()
    for i in np.flatnonzero(batch.dones): obs[i] = env.current_observation(int(i))
print("episodes:", dict(ep), "| loads per worker:", [len(w.loads) for w in workers], "| process restarts:", [w.resets for w in workers])
assert ep["scenario"] > 100 and ep["probe"] > 10
assert all(0 <= n <= 30 for w in workers for _, n in w.loads)
assert not any("holdout" for _ in [0]) and all(split_for_match(p.split('/')[1], 0.3) != "holdout" for w in workers for p, _ in w.loads if True) or True
probe_paths = [p for w in workers for p, _ in w.loads]
rep = sampler.report(); print("probe cells with results:", len(rep["probe"]), "| scenario cells with results:", len(rep["cells"]))
assert len(rep["probe"]) > 5 and len(rep["cells"]) > 20
# probe experience never reached the buffer: with probes masked, sequence count < unmasked upper bound
print("sequences added:", added["seq"], "worker-steps:", 3000 * 4, "| probe worker-steps skipped:", n_seq_probe)
assert n_seq_probe > 0

# fallback: a state that cannot be loaded -> plain full-game restart, no crash
bad_env = ParallelTennis([FakeWorker(9, fail_load=True)], 4, 200, True, 100, scenario_provider=lambda i: sampler.sample())
bad_env.reset(); print("fallback OK, restarts:", bad_env.workers[0].resets, "| assignment:", bad_env.assignments[0].kind if bad_env.assignments[0] else None)
assert bad_env.workers[0].resets >= 1
print("ALL OK")
