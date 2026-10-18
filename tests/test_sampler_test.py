from pathlib import Path
import sys, collections
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tennis_rl.scenarios import build_catalog, StateRecord, split_for_match
from tennis_rl.scenario_sampler import ScenarioLibrary, ScenarioSampler, SamplerConfig

rng = np.random.default_rng(1)
cells = build_catalog()
recs = []
for c in cells:
    for m in range(12):                       # 12 source matches per cell
        mid = f"m{m}"
        n_states = 1 if m != 0 else 40         # match m0 is huge: must not dominate
        for k in range(n_states):
            for phase in ("start", "mid", "late"):
                recs.append(StateRecord(path=f"{c.cell_id}/{mid}/{phase}/{k}", cell_id=c.cell_id, phase=phase,
                    match_id=mid, split=split_for_match(mid, 0.25),
                    demo_outcome="loss" if (m + k) % 4 == 0 else "win"))
lib = ScenarioLibrary(recs)
print("records:", len(recs), "holdout matches:", sorted({r.match_id for r in recs if r.split=='holdout'}))

# --- structural properties ---
s = ScenarioSampler(lib, SamplerConfig(), seed=0)
picks = [s.sample() for _ in range(6000)]
kinds = collections.Counter(a.kind for a in picks)
print("kinds:", dict(kinds))
assert 0.10 < kinds["anchor"] / len(picks) < 0.20
sc = [a for a in picks if a.kind == "scenario"]
assert all(a.state.split == "train" for a in sc), "holdout leaked into training"
assert all(0 <= a.noop_frames <= 30 for a in sc)
big = sum(a.state.match_id == "m0" for a in sc if a.state.match_id in {r.match_id for r in recs if r.split=='train'})
share_m0 = sum(a.state.match_id == "m0" for a in sc) / len(sc)
n_train_matches = len({r.match_id for r in recs if r.split == "train"})
print(f"match m0 holds 40/51 of states but gets {share_m0:.3f} of draws (uniform over matches = {1/n_train_matches:.3f})")
assert share_m0 < 2.5 / n_train_matches
by_stratum = collections.Counter(a.stratum for a in sc)
pr = s.stratum_probs(); assert abs(sum(pr.values()) - 1) < 1e-9
floor = SamplerConfig().stratum_floor / len(pr)
print(f"min stratum prob {min(pr.values()):.4f} (floor {floor:.4f}); min empirical share {min(by_stratum.values())/len(sc):.4f}")
assert min(pr.values()) >= floor * 0.999 and len(by_stratum) == len(pr)
loss_frac = np.mean([a.demo_outcome == "loss" for a in sc]); print(f"states cut from points the human lost: {loss_frac:.2f}")
probe = [s.sample_probe() for _ in range(80)]
assert all(a.kind == "probe" and a.state.split == "holdout" for a in probe)
print("holdout probes cycle through", len({a.cell_id for a in probe}), "cells")

# --- reverse curriculum: late first, unlock when solved ---
s = ScenarioSampler(lib, SamplerConfig(promote_min_episodes=10), seed=0)
cid = cells[0].cell_id
first = collections.Counter(s._pick_phase(cid) for _ in range(200)); assert set(first) == {"late"}, first
for _ in range(15): s.update(cid, True, "late")
print("after solving 'late':", dict(collections.Counter(s._pick_phase(cid) for _ in range(400))))
assert "mid" in {s._pick_phase(cid) for _ in range(200)}

# --- retention: forgotten cell gets boosted ---
s = ScenarioSampler(lib, SamplerConfig(), seed=0)
target = cells[5]
for _ in range(60): s.update(target.cell_id, True)
assert s.stats[target.cell_id].mastered_once
p_before = s.stratum_probs()[target.stratum]
for _ in range(40): s.update(target.cell_id, False)
assert s.stats[target.cell_id].forgotten, s.stats[target.cell_id].slow
p_after = s.stratum_probs()[target.stratum]
print(f"stratum prob while mastered {p_before:.4f} -> after forgetting {p_after:.4f}")
assert p_after > p_before

print("ALL OK")
