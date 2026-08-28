from __future__ import annotations

import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np


class BridgeError(RuntimeError):
    pass


class MesenWorker:
    """One headless Mesen process connected through an atomic file mailbox.

    This transport deliberately avoids LuaSocket. Mesen's Lua runtime only
    needs I/O/OS access, and each worker gets an isolated mailbox directory.
    Commands and observations are committed with atomic rename so Python never
    reads a partially-written frame.
    """

    def __init__(
        self,
        executable: str,
        rom: str,
        lua_script: str,
        host: str,
        port: int,
        width: int,
        height: int,
        frame_skip: int,
        startup_timeout: float,
        socket_timeout: float,
        match_games: int = 6,
        game_level: int = 1,
        point_win: float = 1.0,
        point_loss: float = -1.0,
        game_win: float = 5.0,
        ipc_dir: str | None = None,
        run_id: str | None = None,
        capture_stderr: bool = False,
        headless: bool = True,
    ) -> None:
        # host/port are retained for config backward compatibility; transport is
        # now local filesystem IPC and never opens a TCP socket.
        _ = host, port
        self.executable = str(Path(executable).resolve())
        self.rom = str(Path(rom).resolve())
        self.lua_script = str(Path(lua_script).resolve())
        self.width = width
        self.height = height
        self.frame_skip = frame_skip
        self.startup_timeout = startup_timeout
        self.socket_timeout = socket_timeout
        self.match_games = match_games
        self.game_level = game_level
        self.point_win = point_win
        self.point_loss = point_loss
        self.game_win = game_win
        self.obs_bytes = width * height
        self.next_token = 1
        self.pending_token: str | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self.stderr_file: Any | None = None
        self.seq = -1
        self.capture_stderr = bool(capture_stderr)
        self.headless = bool(headless)

        base = Path(ipc_dir or (Path("runs") / "mesen_ipc"))
        # Never reuse a mailbox directory across launches. This prevents stale
        # observation/ready markers from a previous crashed run from being
        # mistaken for frames from the current Mesen process.
        self.run_id = run_id or f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:10]}"
        self.ipc_dir = (base / self.run_id / f"worker_{port}").resolve()
        self.ipc_dir.mkdir(parents=True, exist_ok=True)
        # Controller/reset commands are a latest-value mailbox, not a queue.
        # RL actions become stale immediately once a newer action exists, so
        # keeping a durable command backlog only adds latency and disk growth.
        self.command_path = self.ipc_dir / "command.in"
        self.command_tmp = self.ipc_dir / "command.tmp"
        # Observations are immutable, host-token files. The host owns the
        # synchronization token so Mesen resets/reloads cannot invalidate it.
        # Windows sharing violations when the producer writes the next frame
        # while Python is still reading the previous one.
        self.observation_glob = "observation_*.bin"
        self.ready_path = self.ipc_dir / "ready.txt"
        self.error_path = self.ipc_dir / "error.txt"
        self._clear_mailbox()
        self._start()

    def _clear_mailbox(self) -> None:
        command_path = getattr(self, "command_path", self.ipc_dir / "command.in")
        command_tmp = getattr(self, "command_tmp", self.ipc_dir / "command.tmp")
        for path in (command_path, command_tmp):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        for pattern in (
            "command_*.in",
            "command_*.tmp",
            "observation_*.bin",
            "observation_*.tmp",
            "observation_ready_*.ready",
            "observation_ready_*.tmp",
        ):
            for path in self.ipc_dir.glob(pattern):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        for path in (self.ready_path, self.error_path, self.ipc_dir / "mesen.stderr.log"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _start(self) -> None:
        if not Path(self.executable).exists():
            raise FileNotFoundError(f"Mesen executable not found: {self.executable}")
        if not Path(self.rom).exists():
            raise FileNotFoundError(f"ROM not found: {self.rom}")
        if not Path(self.lua_script).exists():
            raise FileNotFoundError(f"Lua bridge not found: {self.lua_script}")

        env = os.environ.copy()
        env.update({
            "MESEN_RL_IPC_DIR": str(self.ipc_dir),
            "MESEN_RL_W": str(self.width),
            "MESEN_RL_H": str(self.height),
            "MESEN_RL_FRAME_SKIP": str(self.frame_skip),
            "MESEN_RL_MATCH_GAMES": str(self.match_games),
            "MESEN_RL_LEVEL": str(self.game_level),
            "MESEN_RL_POINT_WIN": str(self.point_win),
            "MESEN_RL_POINT_LOSS": str(self.point_loss),
            "MESEN_RL_GAME_WIN": str(self.game_win),
        })

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        stderr_target: Any
        if self.capture_stderr:
            stderr_path = self.ipc_dir / "mesen.stderr.log"
            # Truncate at the start of every worker run. This prevents logs
            # from accumulating across repeated training runs.
            self.stderr_file = stderr_path.open("wb", buffering=0)
            stderr_target = self.stderr_file
        else:
            self.stderr_file = None
            stderr_target = subprocess.DEVNULL

        if self.headless:
            args = [
            self.executable,
            "--testrunner",
            "--enableStdout",
            "--doNotSaveSettings",
            "--timeout=86400",
            "--debug.scriptWindow.allowIoOsAccess=true",
            self.rom,
            self.lua_script,
            ]
        else:
            args = [
            self.executable,
            "--doNotSaveSettings",
            "--debug.scriptWindow.allowIoOsAccess=true",
            self.rom,
            self.lua_script,
            ]

        try:
            self.process = subprocess.Popen(
                args,
                env=env,
                cwd=str(Path(self.executable).parent),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr_target,
                creationflags=creationflags,
            )
        except Exception:
            self._close_stderr()
            raise

        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.ready_path.exists():
                try:
                    text = self.ready_path.read_text(encoding="utf-8").strip()
                except OSError:
                    text = ""
                if text.startswith("READY "):
                    parts = text.split()
                    if len(parts) >= 4:
                        got_w, got_h, got_skip = map(int, parts[1:4])
                        if (got_w, got_h, got_skip) != (self.width, self.height, self.frame_skip):
                            self.close(force=True)
                            raise BridgeError(
                                "Mesen bridge configuration mismatch: "
                                f"got {got_w}x{got_h} skip={got_skip}, "
                                f"expected {self.width}x{self.height} skip={self.frame_skip}"
                            )
                        return
            if self.error_path.exists():
                detail = self.error_path.read_text(encoding="utf-8", errors="replace")
                code = self.process.poll() if self.process else None
                self.close(force=True)
                raise BridgeError(
                    f"Mesen worker {self.ipc_dir.name} bridge startup failed "
                    f"(process_exit_code={code}):\n{detail}"
                )
            code = self.process.poll() if self.process else None
            if code is not None:
                stderr_tail = self._read_stderr_tail()
                self.close(force=True)
                raise BridgeError(
                    f"Mesen worker exited before bridge READY (exit_code={code}).\n"
                    f"stderr={stderr_tail or '<empty>'}"
                )
            time.sleep(0.01)

        stderr_tail = self._read_stderr_tail()
        self.close(force=True)
        raise BridgeError(
            f"Mesen worker {self.ipc_dir.name} did not create READY within "
            f"{self.startup_timeout}s. stderr={stderr_tail or '<empty>'}"
        )

    def _read_stderr_tail(self, max_bytes: int = 8192) -> str:
        path = self.ipc_dir / "mesen.stderr.log"
        try:
            data = path.read_bytes()
        except OSError:
            return ""
        return data[-max_bytes:].decode("utf-8", errors="replace").strip()

    def _close_stderr(self) -> None:
        if self.stderr_file is not None:
            try:
                self.stderr_file.close()
            except OSError:
                pass
            self.stderr_file = None

    def _write_command(self, command: str) -> None:
        payload = command.encode("ascii")
        deadline = time.monotonic() + min(2.0, self.socket_timeout)
        last_exc: OSError | None = None
        while time.monotonic() < deadline:
            try:
                self.command_tmp.write_bytes(payload)
                os.replace(self.command_tmp, self.command_path)
                return
            except PermissionError as exc:
                last_exc = exc
                try:
                    self.command_tmp.unlink()
                except FileNotFoundError:
                    pass
                time.sleep(0.002)
            except OSError as exc:
                last_exc = exc
                try:
                    self.command_tmp.unlink()
                except FileNotFoundError:
                    pass
                time.sleep(0.002)
        raise BridgeError(
            f"Failed to publish command mailbox after retrying for "
            f"{min(2.0, self.socket_timeout):.2f}s: {last_exc}"
        ) from last_exc

    def _raise_if_process_exited(self, waiting_for: str) -> None:
        if self.process is None:
            return
        code = self.process.poll()
        if code is None:
            return
        error = ""
        if self.error_path.exists():
            error = self.error_path.read_text(encoding="utf-8", errors="replace")
        stderr = self._read_stderr_tail()
        raise BridgeError(
            f"Mesen exited while waiting for {waiting_for}: exit_code={code}\n"
            f"bridge_error={error or '<none>'}\n"
            f"stderr={stderr or '<empty>'}"
        )

    def _wait_for(self, path: Path, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists():
                return
            self._raise_if_process_exited(path.name)
            time.sleep(0.002)
        raise BridgeError(f"Timed out waiting for {path} after {timeout}s")

    def _new_token(self) -> str:
        token = f"{self.next_token:016d}"
        self.next_token += 1
        return token

    def _observation_paths(self, token: str) -> tuple[Path, Path]:
        return (
            self.ipc_dir / f"observation_{token}.bin",
            self.ipc_dir / f"observation_ready_{token}.ready",
        )

    def _wait_for_observation(self, token: str, timeout: float) -> tuple[Path, Path]:
        path, ready_path = self._observation_paths(token)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if ready_path.exists() and path.exists():
                return path, ready_path
            self._raise_if_process_exited(f"observation token {token}")
            time.sleep(0.002)
        try:
            mailbox = sorted(
                path.name for path in self.ipc_dir.iterdir()
                if path.name.startswith(("command", "observation", "error", "ready"))
            )
        except OSError:
            mailbox = []
        detail = ", ".join(mailbox[:20]) or "<empty>"
        raise BridgeError(
            f"Timed out waiting for observation token {token} after {timeout}s; mailbox={detail}"
        )

    def _consume_observation(self) -> tuple[np.ndarray, float, bool, dict[str, int]]:
        token = self.pending_token
        if token is None:
            raise BridgeError("No pending observation token")
        observation_path, ready_path = self._wait_for_observation(token, self.socket_timeout)
        deadline = time.monotonic() + min(2.0, self.socket_timeout)
        last_exc: OSError | None = None
        data = None
        while time.monotonic() < deadline:
            try:
                data = observation_path.read_bytes()
                break
            except PermissionError as exc:
                last_exc = exc
                time.sleep(0.005)
            except OSError as exc:
                last_exc = exc
                break
        if data is None:
            raise BridgeError(
                f"Failed to read observation mailbox {observation_path.name}: {last_exc}"
            ) from last_exc
        finally_cleanup = (observation_path, ready_path)
        for cleanup_path in finally_cleanup:
            try:
                cleanup_path.unlink()
            except FileNotFoundError:
                pass

        header_end = data.find(b"\n")
        if header_end <= 0:
            raise BridgeError("Malformed observation mailbox: missing header")
        header = data[:header_end].decode("ascii", errors="strict").strip()
        payload = data[header_end + 1 :]
        parts = header.split()
        if len(parts) < 7 or parts[0] != "O":
            raise BridgeError(f"Unexpected observation header: {header!r}")

        observed_token = parts[1]
        if observed_token != token:
            raise BridgeError(
                f"Unexpected bridge token: got {observed_token}, expected {token}"
            )
        reward = float(parts[2])
        done = bool(int(parts[3]))
        player_score = int(parts[4])
        player_games = int(parts[5])
        point_end = int(parts[6])

        expected = self.width * self.height
        if len(payload) != expected:
            raise BridgeError(
                f"Malformed observation token {token}: expected {expected} grayscale bytes, "
                f"got {len(payload)}"
            )
        image = np.frombuffer(payload, dtype=np.uint8).reshape(
            self.height, self.width
        ).copy()

        self.pending_token = None
        info = {
            "player_score": player_score,
            "player_games": player_games,
            "point_end": point_end,
            "token": token,
        }
        return image, reward, done, info

    def action(self, action: int) -> None:
        if self.pending_token is not None:
            raise BridgeError(f"Cannot send action while token {self.pending_token} is pending")
        token = self._new_token()
        self.pending_token = token
        self._write_command(f"A {token} {int(action)}\n")

    def reset(self) -> None:
        # Never call emu.reset() from a Mesen frame callback. Restart the
        # worker instead so the emulator returns to the ROM's deterministic
        # startup state without invalidating Mesen's callback loop.
        if self.pending_token is not None:
            raise BridgeError(f"Cannot reset while token {self.pending_token} is pending")

        self.close(force=True)
        self.next_token = 1
        self.pending_token = None
        self._clear_mailbox()
        self._start()

        # Request exactly one initial observation from the freshly started ROM.
        token = self._new_token()
        self.pending_token = token
        self._write_command(f"A {token} 0\n")

    def press_start(self, frames: int = 60) -> None:
        """Hold the Start button for `frames` NES frames.

        Useful in non-headless (watch) mode to dismiss the Tennis title/
        difficulty screen before handing control to the policy.  The bridge
        will override the current policy action with Start for the requested
        number of frames, then resume normal action delivery.
        """
        self._write_command(f"START {frames}\n")

    def hold_button(self, button: str, frames: int = 30) -> None:
        """Hold any NES button for `frames` NES frames.

        Valid button names: left, right, up, down, a, b, start, select.
        Multiple calls stack independently — the bridge tracks per-button
        countdowns so you can queue Right and then Start without one
        cancelling the other.
        """
        self._write_command(f"HOLD {button} {frames}\n")

    def save_state(self) -> None:
        self._write_command("SV\n")
        self._wait_for(self.ready_path.with_name("state_saved.txt"), self.socket_timeout)
        try:
            self.ready_path.with_name("state_saved.txt").unlink()
        except FileNotFoundError:
            pass

    def load_state(self) -> None:
        self._write_command("LS\n")
        self._wait_for(self.ready_path.with_name("state_loaded.txt"), self.socket_timeout)
        try:
            self.ready_path.with_name("state_loaded.txt").unlink()
        except FileNotFoundError:
            pass

    def recv_observation(self) -> tuple[np.ndarray, float, bool, dict[str, int]]:
        return self._consume_observation()

    def ping(self) -> None:
        self._write_command("PING\n")
        pong = self.ready_path.with_name("pong.txt")
        self._wait_for(pong, self.socket_timeout)
        try:
            value = pong.read_text(encoding="ascii").strip()
        finally:
            try:
                pong.unlink()
            except FileNotFoundError:
                pass
        if value != "PONG":
            raise BridgeError(f"Unexpected ping response: {value!r}")

    def close(self, force: bool = False) -> None:
        try:
            if not force and self.process is not None and self.process.poll() is None:
                try:
                    self._write_command("Q\n")
                    self.process.wait(timeout=3)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if self.process is not None and self.process.poll() is None:
                self.process.kill()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            self.process = None
            self._close_stderr()
            self._clear_mailbox()
