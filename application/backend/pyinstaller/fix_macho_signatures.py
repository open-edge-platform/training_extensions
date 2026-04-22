"""Repair Mach-O dylibs whose embedded code signature has a truncated ``datasize``.

Some wheels (notably ``openvino``) ship arm64 dylibs where the
``LC_CODE_SIGNATURE`` load command's ``datasize`` stops short of the end of the
file. ``codesign`` then refuses to re-sign the file with::

    internal error in Code Signing subsystem

When the trailing bytes are zero padding (harmless), we can bump ``datasize``
to cover them and re-sign successfully.

Usage::

    python fix_macho_signatures.py <root_dir> [<root_dir> ...]

Walks each root, and for every ``*.dylib`` / ``*.so`` that fails ad-hoc
codesigning with the specific "internal error" / "malformed" message, tries
the repair and re-signs. Files whose trailing bytes are non-zero are left
alone (repair would corrupt them).
"""

from __future__ import annotations

import os
import struct
import subprocess  # nosec B404 — fixed-argv `codesign` invocation, no shell, no untrusted input
import sys
from collections.abc import Generator

from PyInstaller.compat import is_darwin

# Mach-O 64-bit magic number (little-endian). We only support 64-bit binaries;
# any fat/universal or 32-bit file is left untouched.
MH_MAGIC_64 = 0xFEEDFACF

# Load-command id for LC_CODE_SIGNATURE — the command that points at the
# embedded signature blob at the end of the file.
LC_CODE_SIGNATURE = 0x1D

# Mach-O 64 header layout (see <mach-o/loader.h>): magic(4) cputype(4)
# cpusubtype(4) filetype(4) ncmds(4) sizeofcmds(4) ...
# We only need `magic` (offset 0) and `ncmds` (offset 16); load commands
# start right after the 32-byte header.
MAGIC_OFFSET = 0
NCMDS_OFFSET = 16
LOAD_COMMANDS_OFFSET = 32

# Each load command starts with `cmd(4) cmdsize(4)`. For LC_CODE_SIGNATURE the
# payload is `dataoff(4) datasize(4)`, so `datasize` sits at +12 from the
# start of the command.
LC_DATAOFF_OFFSET = 8
LC_DATASIZE_OFFSET = 12


# Minimum size of a load-command header (cmd + cmdsize). A `cmdsize` smaller
# than this would mean a corrupt binary that could loop forever; everything
# else is bounds-checked implicitly via `struct.unpack_from`.
LC_HEADER_SIZE = 8


def _find_code_signature(data: bytearray) -> tuple[int, int, int] | None:
    """Locate the LC_CODE_SIGNATURE load command in a Mach-O 64 binary.

    Returns ``(command_offset, dataoff, datasize)`` or ``None`` if the file
    is not a Mach-O 64 binary, has no code signature, or is truncated /
    malformed in any way that prevents safe parsing.
    """
    # `struct.unpack_from` raises `struct.error` when reading past EOF, which
    # acts as our bounds check for free. The only thing we still need to guard
    # explicitly is `cmd_size < LC_HEADER_SIZE` — that one would loop forever.
    try:
        (magic,) = struct.unpack_from("<I", data, MAGIC_OFFSET)
        if magic != MH_MAGIC_64:
            return None
        (ncmds,) = struct.unpack_from("<I", data, NCMDS_OFFSET)
        cmd_offset = LOAD_COMMANDS_OFFSET
        for _ in range(ncmds):
            cmd, cmd_size = struct.unpack_from("<II", data, cmd_offset)
            if cmd_size < LC_HEADER_SIZE:
                return None
            if cmd == LC_CODE_SIGNATURE:
                dataoff, datasize = struct.unpack_from("<II", data, cmd_offset + LC_DATAOFF_OFFSET)
                return cmd_offset, dataoff, datasize
            cmd_offset += cmd_size
    except struct.error:
        return None
    return None


def repair(path: str) -> bool:
    """Extend LC_CODE_SIGNATURE.datasize to cover trailing zero padding.

    Returns ``True`` if the file was modified, ``False`` if there was nothing
    safe to do (not a Mach-O 64, no signature, or trailing bytes are non-zero).
    """
    with open(path, "rb") as f:
        data = bytearray(f.read())

    located = _find_code_signature(data)
    if located is None:
        return False
    cmd_offset, dataoff, datasize = located

    signature_end = dataoff + datasize
    trailing_bytes = data[signature_end:]
    if not trailing_bytes or any(trailing_bytes):
        # Nothing to cover, or trailing bytes are real data we mustn't
        # silently absorb into the signature range.
        return False

    new_datasize = datasize + len(trailing_bytes)
    struct.pack_into("<I", data, cmd_offset + LC_DATASIZE_OFFSET, new_datasize)
    with open(path, "wb") as f:
        f.write(data)
    return True


# Absolute path to `codesign`. Using the absolute path keeps argv free of
# `$PATH` lookups (silences S607) and matches the system binary that ships
# with macOS in every supported version.
CODESIGN_BIN = "/usr/bin/codesign"


def codesign(path: str) -> subprocess.CompletedProcess[str]:
    """Run an ad-hoc ``codesign --force --sign -`` on the given file."""
    # Fixed-argv invocation of a system binary; `path` comes from our own
    # `os.walk` of caller-supplied roots, so there is no injection surface.
    return subprocess.run(  # noqa: S603
        [CODESIGN_BIN, "--force", "--sign", "-", path],
        capture_output=True,
        text=True,
        check=False,
    )


def _is_repairable_codesign_error(stderr: str) -> bool:
    """Only attempt repair for the specific errors we know how to fix."""
    return "internal error" in stderr or "malformed" in stderr


def _iter_mach_o_files(roots: list[str]):
    """Yield every ``*.dylib`` / ``*.so`` under each existing root directory."""
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for name in files:
                if name.endswith((".dylib", ".so")):
                    yield os.path.join(dirpath, name)


def _fix_files(paths: Generator[str]) -> tuple[int, int]:
    """Attempt to codesign each path; repair and retry on known errors.

    Returns ``(fixed, failed)`` counts.
    """
    fixed = 0
    failed = 0
    for path in paths:
        if not os.path.isfile(path):
            continue
        result = codesign(path)
        if result.returncode == 0:
            continue
        if not _is_repairable_codesign_error(result.stderr):
            failed += 1
            print(f"FAILED: {path}\n  {result.stderr.strip()}", file=sys.stderr)
            continue
        if repair(path) and codesign(path).returncode == 0:
            fixed += 1
            print(f"FIXED: {path}")
        else:
            failed += 1
            print(f"FAILED: {path}\n  {result.stderr.strip()}", file=sys.stderr)
    return fixed, failed


def fix_macho_signatures(paths: list[str]) -> None:
    """Fix Mach-O signatures for a list of file paths (e.g. from PyInstaller Analysis).

    Intended to be called from a ``.spec`` file on macOS before ``COLLECT``.
    Silently returns on non-macOS platforms.
    """
    if not is_darwin:
        return
    fixed, failed = _fix_files(p for p in paths if p.endswith((".dylib", ".so")))
    print(f"fix_macho_signatures: fixed={fixed} failed={failed}")
    if failed:
        raise SystemError(f"Failed to fix {failed} Mach-O signature(s)")


def main(roots: list[str]) -> int:
    """CLI entry point: walk *roots* directories and fix every Mach-O inside.

    Returns ``0`` on success, ``1`` if any file could not be repaired.
    """
    if not is_darwin:
        print("Not macOS; nothing to do.")
        return 0

    fixed, failed = _fix_files(_iter_mach_o_files(roots))
    print(f"\nfixed={fixed} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
