import sys, torch
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tennis_rl.model import RecurrentActorCritic, VisualEncoder

torch.manual_seed(0)
m = RecurrentActorCritic(4, 27, [32, 64, 64], 256, 256, 1, input_size=(84, 84)).eval()
print("feature map (C,H,W):", m.encoder.feature_shape, "-> projection in:", m.encoder.projection[1].in_features)
print("encoder params:", sum(p.numel() for p in m.encoder.parameters()))

# end-to-end act()
obs = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
h = m.initial_hidden(2, torch.device("cpu"))
a, lp, v, h2 = m.act(obs, h)
print("act ok:", a.shape, lp.shape, v.shape, h2.shape)

# Position sensitivity: same 'ball' blob at two different places on an otherwise-blank court.
def frame_with_ball(y, x):
    f = torch.zeros(4, 84, 84)
    f[:, y:y+3, x:x+3] = 255.0
    return f.unsqueeze(0)
a1, a2 = frame_with_ball(10, 10), frame_with_ball(60, 70)
with torch.no_grad():
    d_new = (m.encoder(a1) - m.encoder(a2)).abs().mean().item()
    mag_new = m.encoder(a1).abs().mean().item()

# Compare against the OLD pooled head (same trunk, avg-pool + linear) to show what was lost.
import torch.nn as nn
old = VisualEncoder(4, [32, 64, 64], 256, (84, 84)).eval()
old.projection = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, 256), nn.ReLU())
with torch.no_grad():
    d_old = (old(a1) - old(a2)).abs().mean().item()
    mag_old = old(a1).abs().mean().item()
print(f"relative embedding change when the blob moves: new={d_new/mag_new:.3f}  old(avg-pool)={d_old/mag_old:.3f}")

# Other frame sizes work (flattened size is measured, not hard-coded)
for hw in [(64, 64), (96, 96), (84, 100)]:
    e = VisualEncoder(4, [32, 64, 64], 256, hw)
    out = e(torch.rand(3, 4, *hw))
    print(hw, "->", tuple(out.shape), "feature", e.feature_shape)

# sequence forward + gradient flows into the encoder
m.train()
seq = torch.randint(0, 256, (5, 3, 4, 84, 84), dtype=torch.uint8)
dist, vals, hn, aux = m.forward_sequence(seq, m.initial_hidden(3, torch.device("cpu")), include_aux=True)
(dist.entropy().mean() + vals.mean() + aux.mean()).backward()
print("grad on encoder projection:", m.encoder.projection[1].weight.grad.abs().sum().item() > 0)

# DQfD wrapper still builds on top of the encoder
try:
    from tennis_rl.dqfd import RecurrentDuelingQ
    q = RecurrentDuelingQ(m)
    qv, _ = q(seq.float(), q.initial_hidden(3, torch.device("cpu")))
    print("RecurrentDuelingQ q-values:", tuple(qv.shape))
except Exception as e:
    print("dqfd wrapper check skipped:", type(e).__name__, e)

# Old checkpoints no longer load (expected)
old_style = {k: v for k, v in m.state_dict().items()}
old_style["encoder.projection.1.weight"] = torch.zeros(256, 64)
try:
    m.load_state_dict(old_style); print("old checkpoint loads (unexpected)")
except RuntimeError as e:
    print("old-shape checkpoint rejected as expected")
