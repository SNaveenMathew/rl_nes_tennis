from pathlib import Path


def test_bridge_has_27_controller_actions():
    text = Path("mesen/bridge.lua").read_text(encoding="utf-8")
    block = text.split("local ACTIONS = {", 1)[1].split("\n}\n\nlocal function write_text", 1)[0]
    # Each physical action is a { ... } entry on the lines of the table.
    action_entries = sum(line.count("{") for line in block.splitlines())
    assert action_entries == 27
