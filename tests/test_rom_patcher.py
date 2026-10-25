from pathlib import Path
import pytest

from tennis_rl.rom_patcher import patch_tennis_rom, resolve_rom


def test_rom_patcher_generates_valid_roms(tmp_path):
    base_rom = Path("Tennis (JU).nes")
    if not base_rom.is_file():
        pytest.skip("Base Tennis (JU).nes not available")

    # Serve-first level 2
    serve_rom = resolve_rom(base_rom, serve_mode="serve-first", level=2, output_dir=tmp_path)
    assert serve_rom.is_file()
    assert serve_rom.name == "Tennis (JU) - Serve-L2.nes"
    data = serve_rom.read_bytes()
    assert len(data) == 24592

    # Level byte at $F88A (file offset 16 + (0xF88A - 0xC000) = 0x389A)
    off_level = 16 + (0xF88A - 0xC000)
    assert data[off_level] == 1  # 0-indexed level 2

    # Server byte at $C7B1 (file offset 16 + (0xC7B1 - 0xC000) = 0x7C1)
    off_server = 16 + (0xC7B1 - 0xC000)
    assert data[off_server] == 0xC1  # Player 1 serves

    # Serve positioning window at $D7E2 (file offset 16 + (0xD7E2 - 0xC000) = 0x17F2)
    # Should be 0x0A (pre-toss positioning window preserved)
    off_entrance = 16 + (0xD7E2 - 0xC000)
    assert data[off_entrance] == 0x0A

    # Receive-first level 4
    receive_rom = resolve_rom(base_rom, serve_mode="receive-first", level=4, output_dir=tmp_path)
    assert receive_rom.is_file()
    assert receive_rom.name == "Tennis (JU) - Receive-L4.nes"
    rdata = receive_rom.read_bytes()
    assert len(rdata) == 24592
    assert rdata[off_level] == 3  # 0-indexed level 4
    assert rdata[off_server] == 0xC3  # CPU serves, Player receives
    assert rdata[off_entrance] == 0x0A


def test_resolve_rom_idempotent(tmp_path):
    base_rom = Path("Tennis (JU).nes")
    if not base_rom.is_file():
        pytest.skip("Base Tennis (JU).nes not available")

    rom1 = resolve_rom(base_rom, serve_mode="serve-first", level=3, output_dir=tmp_path)
    mtime1 = rom1.stat().st_mtime_ns
    rom2 = resolve_rom(base_rom, serve_mode="serve-first", level=3, output_dir=tmp_path)
    mtime2 = rom2.stat().st_mtime_ns
    assert rom1 == rom2
    assert mtime1 == mtime2  # File not rewritten if identical
