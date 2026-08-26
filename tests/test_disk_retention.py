from pathlib import Path

from tennis_rl.train import _prune_step_checkpoints


def test_prune_step_checkpoints_keeps_newest(tmp_path: Path):
    paths = []
    for step in (100, 200, 300, 400, 500):
        path = tmp_path / f"tennis_step_{step}.pt"
        path.write_bytes(b"x")
        paths.append(path)
    (tmp_path / "tennis_latest.pt").write_bytes(b"latest")

    _prune_step_checkpoints(tmp_path, keep=2)

    assert (tmp_path / "tennis_latest.pt").exists()
    assert (tmp_path / "tennis_step_500.pt").exists()
    assert (tmp_path / "tennis_step_400.pt").exists()
    assert len(list(tmp_path.glob("tennis_step_*.pt"))) == 2
