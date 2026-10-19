"""Helpers for embedding Mesen's Lua bridge in pytest.

On Windows, Python's ``os.environ`` and an embedded Lua runtime may be backed
by different C-runtime environment snapshots. Updating ``os.environ`` after
process start is therefore not a reliable way to configure ``os.getenv`` in
Lua. Inject the test environment into the Lua runtime explicitly so tests
exercise the mode they claim to exercise instead of silently falling back to
bridge defaults.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping


def install_lua_test_environment(lua, overrides: Mapping[str, object]) -> None:
    """Make MESEN_RL_* overrides explicit within this LuaRuntime instance.

    Unspecified MESEN_RL_* values resolve to nil (the bridge's own defaults),
    rather than leaking values set by previously collected tests. Non-project
    environment lookups continue to use Lua's original ``os.getenv``.
    """
    table = lua.table()
    for key, value in overrides.items():
        table[str(key)] = str(value)
    lua.globals()["__MESEN_RL_TEST_ENV"] = table
    lua.execute(r"""
        local __native_getenv = os.getenv
        os.getenv = function(name)
            if type(name) == "string" and name:match("^MESEN_RL_") then
                return rawget(__MESEN_RL_TEST_ENV, name)
            end
            return __native_getenv(name)
        end
    """)


def read_api_info_for_bridge(ipc_dir, bridge_path, current_bridge_path) -> str:
    """Read current-bridge diagnostics without imposing them on legacy fixtures.

    ``api_info.txt`` is a diagnostic contract of the current bridge only. The
    archived legacy fixture predates it, so an absent file is expected there.
    Conversely, the current bridge must publish it or tests cannot verify which
    lockstep mode actually ran.
    """
    ipc = Path(ipc_dir)
    metadata_path = ipc / "api_info.txt"
    is_current = Path(bridge_path).resolve() == Path(current_bridge_path).resolve()
    if not metadata_path.exists():
        if is_current:
            raise AssertionError(
                "current bridge did not write api_info.txt; cannot validate the lockstep mode"
            )
        return ""
    return metadata_path.read_text(encoding="utf-8")
