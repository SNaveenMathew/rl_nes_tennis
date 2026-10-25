from pathlib import Path


def test_train_cli_separated_axes():
    text = Path("tennis_rl/train.py").read_text(encoding="utf-8")
    # 1. Algorithm axis
    assert '--algo' in text
    assert 'choices=["hierarchical", "ppo", "dqfd"]' in text
    # 2. Sampling / Episode scope axis
    assert '--sampling' in text
    assert '"point", "game"' in text
    # 3. Serve mode axis
    assert '--serve' in text
    assert '"serve-first", "receive-first"' in text
    # 4. Serve side axis
    assert '--side' in text
    assert '"deuce", "ad", "advantage"' in text
    # 5. Level axis
    assert '--level' in text


def test_record_cli_separated_axes():
    text = Path("tennis_rl/record.py").read_text(encoding="utf-8")
    # Sampling / Episode scope axis
    assert '--sampling' in text
    assert '"point", "game"' in text
    # Serve mode axis
    assert '--serve' in text
    assert '"serve-first", "receive-first"' in text
    # Serve side axis
    assert '--side' in text
    assert '"deuce", "ad", "advantage"' in text
    # Level axis
    assert '--level' in text


def test_evaluate_cli_separated_axes():
    text = Path("tennis_rl/evaluate.py").read_text(encoding="utf-8")
    assert '--sampling' in text
    assert '"point", "game"' in text
    assert '--serve' in text
    assert '"serve-first", "receive-first"' in text
    assert '--side' in text
    assert '"deuce", "ad", "advantage"' in text
    assert '--level' in text


def test_watch_cli_separated_axes():
    text = Path("tennis_rl/watch.py").read_text(encoding="utf-8")
    assert '--sampling' in text
    assert '"point", "game"' in text
    assert '--serve' in text
    assert '"serve-first", "receive-first"' in text
    assert '--side' in text
    assert '"deuce", "ad", "advantage"' in text
    assert '--level' in text


def test_powershell_scripts_separated_axes():
    train_ps1 = Path("scripts/train.ps1").read_text(encoding="utf-8")
    assert '$Algo' in train_ps1
    assert '$Sampling' in train_ps1
    assert '$Serve' in train_ps1
    assert '$Side' in train_ps1
    assert '$Level' in train_ps1

    record_ps1 = Path("scripts/record.ps1").read_text(encoding="utf-8")
    assert '$Sampling' in record_ps1
    assert '$Points' in record_ps1
    assert '$Serve' in record_ps1
    assert '$Side' in record_ps1
    assert '$Level' in record_ps1

    eval_ps1 = Path("scripts/evaluate.ps1").read_text(encoding="utf-8")
    assert '$Sampling' in eval_ps1
    assert '$Serve' in eval_ps1
    assert '$Side' in eval_ps1
    assert '$Level' in eval_ps1


def test_cli_side_validation_game_vs_point():
    import subprocess
    import sys

    # 1. record.py rejects --sampling game with --side ad
    res = subprocess.run(
        [sys.executable, "-m", "tennis_rl.record", "--sampling", "game", "--side", "ad"],
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "only supported when --sampling is 'point'" in res.stderr

    # 2. train.py rejects --sampling game with --side ad
    res = subprocess.run(
        [sys.executable, "-m", "tennis_rl.train", "--sampling", "game", "--side", "ad", "--smoke-test"],
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "only supported when --sampling is 'point'" in res.stderr

    # 3. evaluate.py rejects --sampling game with --side ad
    res = subprocess.run(
        [sys.executable, "-m", "tennis_rl.evaluate", "--sampling", "game", "--side", "ad"],
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "only supported when --sampling is 'point'" in res.stderr

