import sys, json, tempfile, shutil
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tennis_rl.scenarios import *
from tennis_rl.scenario_index import index_demonstrations

# ---- catalog ----
cells = build_catalog()
print("cells:", len(cells), "strata:", len(strata(cells)), "scores:", len(reachable_scores()))
assert len(cells) == 36 and len(reachable_scores()) == 18
assert sum(c.side == "deuce" for c in cells if c.role == "serve") == 8
assert sum(c.side == "ad" for c in cells if c.role == "serve") == 10
# every state reachable by play from 0-0, and every transition lands in the catalog or ends the game
seen, frontier = {(0, 0)}, [(0, 0)]
while frontier:
    p, o = frontier.pop()
    for won in (True, False):
        p2, o2, over, _ = advance_score(p, o, won)
        if not over and (p2, o2) not in seen:
            seen.add((p2, o2)); frontier.append((p2, o2))
assert seen == set(reachable_scores()), seen ^ set(reachable_scores())
# court side alternates point to point, including after deuce/advantage cycles
p, o = 3, 3
for won in (True, False, True, True):
    side0 = court_side(p, o)
    p, o, over, _ = advance_score(p, o, won)
    if over: break
    assert court_side(p, o) != side0
print("score machine OK;", len(set(c.stratum for c in cells)), "strata")

# ---- demo tagging on a synthetic game: player serves, wins 4 points to 1, loses one at 15-0 ----
tmp = Path(tempfile.mkdtemp()); root = tmp / "demos"
def write_game(match, role_mode, results, match_outcome, drop=None, zero_at=None):
    d = root / "level_1" / match_outcome / f"match_{match}"; d.mkdir(parents=True)
    for i, res in enumerate(results, 1):
        if drop == i: continue
        n = 30 + 5 * i
        rew = np.zeros(n, np.float32)
        if res == "win": rew[-1] = 1.0
        elif res == "loss": rew[-1] = -1.0
        np.savez_compressed(d / f"point_{i:04d}.npz", observations=np.zeros((n,4,8,8), np.uint8),
            actions=np.zeros(n, np.int64), rewards=rew, dones=np.zeros(n, bool),
            level=np.asarray(1, np.int16), outcome=np.asarray(match_outcome), match_id=np.asarray(match),
            point_index=np.asarray(i, np.int32))
        (d / f"point_{i:04d}.json").write_text(json.dumps({"serve_mode": role_mode,
            "provisional_point_outcome": "winning" if res == "win" else "losing"}))
write_game("g1", "serve-first", ["win", "loss", "win", "win", "win"], "winning")           # 15-0, 15-15, 30-15, 40-15, game
write_game("g2", "receive-first", ["loss", "win", "none", "loss", "loss", "loss"], "losing")       # includes a zero-reward segment
write_game("g3", "serve-first", ["win", "win", "win", "win"], "winning", drop=2)             # missing file -> inconsistent
idx = index_demonstrations(root)
by = {}
for r in idx["points"]: by.setdefault(r["match_id"], []).append(r)
g1 = by["g1"]; print("g1 start cells:", [r["cell_id"] for r in g1])
assert [r["cell_id"] for r in g1] == ["serve|0-0","serve|1-0","serve|1-1","serve|2-1","serve|3-1"]
assert all(r["consistent"] for r in g1)
g2 = by["g2"]; print("g2 start cells:", [r["cell_id"] for r in g2], [r["point_result"] for r in g2])
assert [r["cell_id"] for r in g2] == ["return|0-0","return|0-1","return|1-1","return|1-1","return|1-2","return|1-3"]
assert g2[2]["point_result"] == "none" and all(r["consistent"] for r in g2)
assert not any(r["consistent"] for r in by["g3"])
d = idx["diagnostics"]; print(d)
# g1: 3 of its points were won inside a "winning" game; point 2 was lost inside a winning game -> flagged
assert d["points_whose_match_outcome_differs_from_point_result"] >= 2
shutil.rmtree(tmp)
print("ALL OK")
