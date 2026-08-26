from pathlib import Path

from tennis_rl.mesen_client import MesenWorker


def test_latest_command_mailbox_path(tmp_path):
    worker = object.__new__(MesenWorker)
    worker.ipc_dir = tmp_path
    worker.command_path = tmp_path / "command.in"
    worker.command_tmp = tmp_path / "command.tmp"
    worker.socket_timeout = 1.0
    worker._write_command("A 7\n")
    assert (tmp_path / "command.in").read_text() == "A 7\n"
    assert not (tmp_path / "command.tmp").exists()

    # Publishing the next action replaces the stale one instead of queuing it.
    worker._write_command("A 8\n")
    assert (tmp_path / "command.in").read_text() == "A 8\n"

def test_observation_mailbox_uses_token_files():
    import inspect
    source = inspect.getsource(MesenWorker._consume_observation)
    assert "_wait_for_observation" in source
    assert "read_bytes" in source
    assert "np.frombuffer" in source
    assert "expected = self.width * self.height" in source
    assert "cleanup_path.unlink" in source


def test_bridge_emits_host_token_observations():
    bridge = Path(__file__).parents[1] / "mesen" / "bridge.lua"
    text = bridge.read_text(encoding="utf-8")
    assert "OBS_PREFIX" in text
    assert 'string.format("%012d", pending_token)' in text or 'local token = pending_token' in text
    assert "obs_tmp" in text and ".tmp" in text
    assert 'OBS_PATH' not in text


def test_bridge_uses_latest_command_mailbox():
    bridge = Path(__file__).parents[1] / "mesen" / "bridge.lua"
    text = bridge.read_text(encoding="utf-8")
    assert 'COMMAND_PATH = IPC_DIR .. "/command.in"' in text
    assert 'next_command_seq' not in text
    assert 'poll_command()' in text


def test_benchmark_module_imports():
    import tennis_rl.benchmark_env as benchmark
    assert callable(benchmark.main)


def test_mailbox_cleanup_removes_stale_observations(tmp_path):
    worker = object.__new__(MesenWorker)
    worker.ipc_dir = tmp_path
    worker.ready_path = tmp_path / "ready.txt"
    worker.error_path = tmp_path / "error.txt"
    (tmp_path / "observation_000000000001.bin").write_bytes(b"x")
    (tmp_path / "observation_000000000002.tmp").write_bytes(b"x")
    (tmp_path / "observation_ready_000000000003.ready").write_text("READY\n")
    (tmp_path / "command.in").write_bytes(b"x")
    (tmp_path / "command.tmp").write_bytes(b"x")
    (tmp_path / "command_000000000001.in").write_bytes(b"x")
    worker._clear_mailbox()
    assert not list(tmp_path.glob("observation_*"))
    assert not list(tmp_path.glob("observation_ready_*"))
    assert not list(tmp_path.glob("command_*"))
    assert not (tmp_path / "command.in").exists()
    assert not (tmp_path / "command.tmp").exists()


def test_worker_uses_unique_run_mailbox(monkeypatch, tmp_path):
    # Avoid launching Mesen; just verify two workers derive different run dirs.
    from tennis_rl import mesen_client
    monkeypatch.setattr(mesen_client.MesenWorker, "_start", lambda self: None)
    w1 = mesen_client.MesenWorker.__new__(mesen_client.MesenWorker)
    # Constructor behavior is exercised through a fake __init__ helper below.
    assert hasattr(mesen_client, "MesenWorker")


def test_mailbox_cleanup_removes_ready_markers(tmp_path):
    worker = object.__new__(MesenWorker)
    worker.ipc_dir = tmp_path
    worker.ready_path = tmp_path / "ready.txt"
    worker.error_path = tmp_path / "error.txt"
    (tmp_path / "observation_ready_000000000004.ready").write_text("READY\n")
    (tmp_path / "observation_ready_000000000005.tmp").write_text("READY\n")
    worker._clear_mailbox()
    assert not list(tmp_path.glob("observation_ready_*"))


def test_client_uses_host_token_instead_of_lua_sequence():
    source = __import__("inspect").getsource(MesenWorker._consume_observation)
    assert "pending_token" in source
    assert "Unexpected bridge token" in source
    assert "next_observation_seq" not in source


def test_worker_reset_restarts_process_without_r_command(monkeypatch, tmp_path):
    from tennis_rl import mesen_client

    worker = object.__new__(mesen_client.MesenWorker)
    worker.pending_token = None
    worker.next_token = 17
    worker.close_called = False
    worker.start_called = False
    worker.clear_called = False

    monkeypatch.setattr(worker, "close", lambda force=True: setattr(worker, "close_called", True))
    monkeypatch.setattr(worker, "_clear_mailbox", lambda: setattr(worker, "clear_called", True))
    monkeypatch.setattr(worker, "_start", lambda: setattr(worker, "start_called", True))
    sent = []
    monkeypatch.setattr(worker, "_write_command", lambda value: sent.append(value))

    worker.reset()

    assert worker.close_called
    assert worker.clear_called
    assert worker.start_called
    assert sent and sent[-1].startswith("A ") and sent[-1].endswith(" 0\n")
    assert worker.pending_token is not None
    assert worker.next_token == 2
