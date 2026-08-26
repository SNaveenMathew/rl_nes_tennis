# Mesen bridge

The bridge is designed for Mesen/MesenCE 2.2.1 `--testrunner` mode. It uses only Mesen Lua APIs plus standard Lua I/O; **LuaSocket is not required** and network access does not need to be enabled.

The Python worker launches Mesen with `--debug.scriptWindow.allowIoOsAccess=true` and creates a private mailbox directory for each worker. Commands and observations are exchanged with atomic temp-file + rename operations.

The ROM is intentionally not included in this repository.
