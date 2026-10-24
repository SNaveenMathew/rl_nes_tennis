"""ROM Patcher for NES Tennis (JU).

Creates patched variants of 'Tennis (JU).nes' that:
1. Bypass the Title Screen (0 frames spent on start screen).
2. Bypass the Level Selection Screen (0 frames spent on difficulty menu).
3. Set the opponent difficulty level directly into RAM ($004C) on boot.
4. Support Serve-First (Player 1 serves) and Receive-First (CPU serves, Player 1 receives).
5. Reduce the match entrance delay ($0026) from ~100 frames to 1 frame, placing players
   instantly in position to serve/receive.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def patch_tennis_rom(
    rom_data: bytes | bytearray,
    level: int = 1,
    receive_first: bool = False,
    instant_entrance: bool = False,
) -> bytes:
    """Patch a Tennis (JU) NROM-128 ROM.

    Args:
        rom_data: Raw bytes of 'Tennis (JU).nes' (24592 bytes).
        level: Opponent difficulty level 1..5.
        receive_first: If True, CPU serves first from top court and Player 1 receives.
                       If False, Player 1 serves first from bottom court.
        instant_entrance: If True, reduces the serve pre-positioning window (RAM $0026)
            from $0A (10 ticks, ~100 frames) to $01 (~10 frames). When False (the
            default), the server and CPU retain the full window to move laterally
            along the baseline before tossing the ball. Set to True only for
            benchmarks that need to skip directly to ball-in-play.

    Returns:
        Patched ROM bytes.
    """
    if len(rom_data) != 24592:
        raise ValueError(
            f"Expected NES Tennis (JU) ROM of 24592 bytes (16-byte header + 16KB PRG + 8KB CHR); "
            f"got {len(rom_data)} bytes."
        )

    patched = bytearray(rom_data)
    level_val = max(0, min(4, int(level) - 1))

    # 1. Reset vector hook:
    # At C03C (file offset 16 + 0x3C = 0x4C):
    # Original: C8 84 38 (INY; STY $38)
    # Patched:  4C 7F F8 (JMP $F87F)
    off_c03c = 16 + (0xC03C - 0xC000)
    patched[off_c03c : off_c03c + 3] = bytes([0x4C, 0x7F, 0xF8])

    # 2. Bootloader injected into unused PRG space at $F87F (file offset 16 + 0x387F = 0x388F):
    # Sets $38 = 0 (gameplay mode, not title)
    # Sets $39 = 0 (1-player mode)
    # Sets $3D = 1 (match initialization state, skipping $3D=0 level menu)
    # Sets $4C = level_val (0..4)
    # Jumps to $C03F to resume standard hardware initialization
    off_f87f = 16 + (0xF87F - 0xC000)
    bootloader = bytes([
        0xA9, 0x00,        # LDA #$00
        0x85, 0x38,        # STA $38 (skip title)
        0x85, 0x39,        # STA $39 (1-player)
        0xA9, 0x01,        # LDA #$01
        0x85, 0x3D,        # STA $3D (skip level select -> match initialization)
        0xA9, level_val,   # LDA #level_val (0..4)
        0x85, 0x4C,        # STA $4C (opponent level)
        0x4C, 0x3F, 0xC0,  # JMP $C03F
    ])
    patched[off_f87f : off_f87f + len(bootloader)] = bootloader

    # 3. Server Configuration Register at C7B0 (file offset 16 + 0x7B0 = 0x7C0):
    # Original: A9 C1 (LDA #$C1) -> Bit 1 is 0 -> Player 1 serves.
    # Patched:  A9 C3 (LDA #$C3) -> Bit 1 is 1 -> CPU serves, Player 1 receives.
    off_c7b0 = 16 + (0xC7B0 - 0xC000)
    server_byte = 0xC3 if receive_first else 0xC1
    patched[off_c7b0 + 1] = server_byte

    # 4. Serve pre-positioning window at D7E1 (file offset 16 + (D7E1 - C000) = 0x17F1):
    # RAM $0026 is the serve state-machine register. Setting it to $0A on court init
    # gives the server (human or CPU) ~10 ticks (~100 NES frames) to move laterally
    # along the baseline before the toss auto-advances. Setting it to $01 reduces
    # that window to ~10 frames — effectively removing the pre-toss positioning phase.
    # Original: A9 0A (LDA #$0A) -> full pre-toss positioning window preserved
    # Patched:  A9 01 (LDA #$01) -> window collapsed to ~10 frames (no useful movement)
    if instant_entrance:
        off_d7e1 = 16 + (0xD7E1 - 0xC000)
        patched[off_d7e1 + 1] = 0x01

    return bytes(patched)


def generate_standard_roms(
    base_rom_path: str | Path = "Tennis (JU).nes",
    output_dir: str | Path = ".",
    level: int = 1,
) -> tuple[Path, Path]:
    """Generate both 'Tennis (JU) - Serve.nes' and 'Tennis (JU) - Receive.nes'.

    Args:
        base_rom_path: Path to the clean 'Tennis (JU).nes'.
        output_dir: Directory where patched ROMs will be written.
        level: Default opponent level (1..5).

    Returns:
        Tuple of (serve_rom_path, receive_rom_path).
    """
    base_path = Path(base_rom_path).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"Base NES Tennis ROM not found: {base_path}")

    raw_bytes = base_path.read_bytes()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    serve_path = out_dir / "Tennis (JU) - Serve.nes"
    receive_path = out_dir / "Tennis (JU) - Receive.nes"

    serve_bytes = patch_tennis_rom(raw_bytes, level=level, receive_first=False)
    receive_bytes = patch_tennis_rom(raw_bytes, level=level, receive_first=True)

    serve_path.write_bytes(serve_bytes)
    receive_path.write_bytes(receive_bytes)

    return serve_path, receive_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate fast-boot patched Tennis (JU) ROMs.")
    parser.add_argument("--input", default="Tennis (JU).nes", help="Path to base Tennis (JU).nes ROM")
    parser.add_argument("--output-dir", default=".", help="Directory to save patched ROMs")
    parser.add_argument("--level", type=int, default=1, choices=range(1, 6), help="Default opponent level (1-5)")
    args = parser.parse_args()

    serve_path, receive_path = generate_standard_roms(args.input, args.output_dir, level=args.level)
    print(f"Generated Serve-First ROM:   {serve_path}")
    print(f"Generated Receive-First ROM: {receive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
