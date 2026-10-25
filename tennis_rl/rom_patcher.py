"""ROM Patcher for NES Tennis (JU).

Creates patched variants of 'Tennis (JU).nes' that:
1. Bypass the Title Screen (0 frames spent on start screen).
2. Bypass the Level Selection Screen (0 frames spent on difficulty menu).
3. Set the opponent difficulty level directly into RAM ($004C) on boot.
4. Support Serve-First (Player 1 serves) and Receive-First (CPU serves, Player 1 receives).

Primary entry-point for all downstream scripts: ``resolve_rom()``.
It generates the correct patched ROM on demand and returns an absolute path
that can be passed directly to MesenWorker.
"""

from __future__ import annotations

import argparse
import hashlib
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



def resolve_rom(
    base_rom_path: str | Path,
    serve_mode: str = "serve-first",
    level: int = 1,
    output_dir: str | Path | None = None,
) -> Path:
    """Generate the correct patched ROM on demand and return its absolute path.

    This is the single entry-point used by all downstream scripts (train,
    record, evaluate, watch). It patches the base ROM for the requested
    ``serve_mode`` and ``level``, writes it to ``output_dir`` (same directory
    as the base ROM if not specified) and returns an absolute ``Path`` ready to
    pass directly to ``MesenWorker``.

    The output filename encodes serve_mode and level so multiple processes
    running at different levels never overwrite each other::

        Tennis (JU) - Serve-L2.nes
        Tennis (JU) - Receive-L3.nes

    The file is (re)written only when its content would change, so repeated
    calls are cheap.

    Args:
        base_rom_path: Path to the clean 'Tennis (JU).nes' (24592 bytes).
        serve_mode: ``"serve-first"`` (player serves) or ``"receive-first"``
            (CPU serves, player receives).
        level: Opponent difficulty level 1..5.  Baked into ROM at $F88A so the
            game starts at the correct level even without the Lua bridge.  The
            bridge also writes MESEN_RL_LEVEL into RAM each frame, so this is
            consistent regardless of whether the bridge is active.
        output_dir: Directory where the patched ROM is written. Defaults to
            the directory containing ``base_rom_path``.

    Returns:
        Absolute ``Path`` to the patched ROM file.

    Raises:
        ValueError: If ``serve_mode`` is not recognised.
        FileNotFoundError: If ``base_rom_path`` does not exist.
    """
    if serve_mode not in {"serve-first", "receive-first"}:
        raise ValueError(
            f"serve_mode must be 'serve-first' or 'receive-first', got {serve_mode!r}"
        )
    level = max(1, min(5, int(level)))
    base_path = Path(base_rom_path).resolve()
    if not base_path.is_file():
        # Check standard fallback locations
        candidates = [
            base_path.parent / "Tennis (JU).nes",
            Path("Tennis (JU).nes").resolve(),
            Path("C:/Users/dbznf/Documents/NES/Tennis (JU).nes").resolve(),
        ]
        found = False
        for c in candidates:
            if c.is_file():
                base_path = c
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Base NES Tennis ROM not found: {base_rom_path}")

    # If the provided path is an already patched variant, prefer the clean base ROM
    clean_candidate = base_path.parent / "Tennis (JU).nes"
    if clean_candidate.is_file():
        base_path = clean_candidate

    out_dir = Path(output_dir).resolve() if output_dir else base_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "Serve" if serve_mode == "serve-first" else "Receive"
    out_path = out_dir / f"Tennis (JU) - {mode_tag}-L{level}.nes"

    raw_bytes = base_path.read_bytes()
    receive_first = serve_mode == "receive-first"
    patched = patch_tennis_rom(raw_bytes, level=level, receive_first=receive_first)

    # Only write if content differs (avoids churn when called repeatedly).
    if not out_path.exists() or out_path.read_bytes() != patched:
        out_path.write_bytes(patched)

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate fast-boot patched Tennis (JU) ROMs.")
    parser.add_argument("--input", default="Tennis (JU).nes", help="Path to base Tennis (JU).nes ROM")
    parser.add_argument("--output-dir", default=".", help="Directory to save patched ROMs")
    parser.add_argument("--level", type=int, default=1, choices=range(1, 6), help="Default opponent level (1-5)")
    parser.add_argument("--serve", choices=["serve-first", "receive-first", "both"], default="both",
                        help="Generate specific serve mode or both")
    args = parser.parse_args()

    if args.serve in {"serve-first", "receive-first"}:
        out_path = resolve_rom(args.input, serve_mode=args.serve, level=args.level, output_dir=args.output_dir)
        print(f"Generated {args.serve} (L{args.level}) ROM: {out_path}")
    else:
        s_path = resolve_rom(args.input, serve_mode="serve-first", level=args.level, output_dir=args.output_dir)
        r_path = resolve_rom(args.input, serve_mode="receive-first", level=args.level, output_dir=args.output_dir)
        print(f"Generated Serve-First (L{args.level}) ROM:   {s_path}")
        print(f"Generated Receive-First (L{args.level}) ROM: {r_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
