# Mesen bridge

The bridge is designed for Mesen/MesenCE 2.2.1 `--testrunner` mode. It uses only Mesen Lua APIs plus standard Lua I/O; **LuaSocket is not required** and network access does not need to be enabled.

The Python worker launches Mesen with `--debug.scriptWindow.allowIoOsAccess=true` and creates a private mailbox directory for each worker. Commands and observations are exchanged with atomic temp-file + rename operations.

The ROM is intentionally not included in this repository.


## Runtime constraints

Mesen 2.2.1 requires savestate creation/loading inside a main-CPU `exec` memory callback. The bridge arms a callback for the current PC and disarms it on a frame event after the operation completes. Do not move `emu.createSavestate()` or `emu.loadSavestate()` directly into `startFrame`/`endFrame`, and do not remove a memory callback while its own callback dispatcher is running.

Framebuffer availability can be transient at startup or after loading a state. The bridge validates `emu.getScreenBuffer()` and retries a pending observation for up to 180 frames rather than dereferencing an absent first pixel. Native annotation output is fixed at 256×240 RGB; if a build exposes 239 rows, the final output row repeats the last visible row.
