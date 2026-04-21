"""Fix malformed Mach-O dylibs that fail ad-hoc codesigning on macOS.

Some wheels (notably ``openvino``) ship arm64 dylibs whose ``LC_CODE_SIGNATURE``
load command reports a ``datasize`` that does not extend to the end of the
file. ``codesign`` then refuses to re-sign with::

    internal error in Code Signing subsystem

(and the underlying log shows ``malformed mach-o file, LC_CODE_SIGNATURE does
not point to end of file``). When the trailing bytes are zero padding, the file
is harmless and we can simply bump ``datasize`` to cover them, after which
``codesign --force --sign -`` succeeds.

Usage:
    python fix_macho_signatures.py <root_dir> [<root_dir> ...]

The script walks each root, finds ``*.dylib`` / ``*.so`` files, attempts an
ad-hoc codesign, and only tries to repair files where codesign fails with
``internal error`` or ``malformed``. Files are only modified when the trailing
bytes after the signature are all zero.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys

LC_CODE_SIGNATURE = 0x1D
MH_MAGIC_64 = 0xFEEDFACF


def repair(path: str) -> bool:
    """Bump LC_CODE_SIGNATURE.datasize to cover trailing zero padding.

    Returns True if the file was modified.
    """
    with open(path, "rb") as f:
        data = bytearray(f.read())

    if len(data) < 32 or struct.unpack_from("<I", data, 0)[0] != MH_MAGIC_64:
        return False

    ncmds = struct.unpack_from("<I", data, 16)[0]
    off = 32
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", data, off)
        if cmd == LC_CODE_SIGNATURE:
            dataoff, datasize = struct.unpack_from("<II", data, off + 8)
            sig_end = dataoff + datasize
            trailing = len(data) - sig_end
            if trailing <= 0 or any(data[sig_end:sig_end + trailing]):
                return False
            struct.pack_into("<I", data, off + 12, datasize + trailing)
            with open(path, "wb") as f:
                f.write(data)
            return True
        off += cmdsize
    return False


def codesign(path: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["codesign", "--force", "--sign", "-", path],
        capture_output=True,
        text=True,
    )


def main(roots: list[str]) -> int:
    if sys.platform != "darwin":
        print("Not macOS; nothing to do.")
        return 0

    fixed = failed = 0
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for name in files:
                if not (name.endswith(".dylib") or name.endswith(".so")):
                    continue
                path = os.path.join(dirpath, name)
                result = codesign(path)
                if result.returncode == 0:
                    continue
                err = result.stderr
                if "internal error" not in err and "malformed" not in err:
                    continue
                if repair(path) and codesign(path).returncode == 0:
                    fixed += 1
                    print(f"FIXED: {path}")
                else:
                    failed += 1
                    print(f"FAILED: {path}\n  {err.strip()}", file=sys.stderr)

    print(f"\nfixed={fixed} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
