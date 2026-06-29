# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyInstaller runtime hook: generate self-signed TLS certificates at runtime.

The MSIX-packaged backend serves HTTPS via Hypercorn but ships without any
certificate. This hook writes a self-signed certificate for ``localhost`` into the per-user data
on first launch, so that ``app.main`` can bind Hypercorn over TLS.

This hook must run after ``uwp.py`` (which sets ``DATA_DIR``) and before
``app.main`` imports settings. Certificates are only generated once and reused
on subsequent launches. Worker/child processes inherit ``DATA_DIR`` and see the
files already present, so generation is skipped.
"""

import datetime
import ipaddress
import os
from pathlib import Path

CERT_FILENAME = "localhost.pem"
KEY_FILENAME = "localhost-key.pem"


def _data_dir() -> Path:
    """
    Resolve the per-user data directory used by the backend.
    """
    data_dir = os.getenv("DATA_DIR")
    if not data_dir:
        raise OSError("DATA_DIR is not set; cannot place certificates.")
    return Path(data_dir)


def _generate_self_signed(cert_path: Path, key_path: Path) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                    x509.IPAddress(ipaddress.ip_address("::1")),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def _main() -> None:
    cert_dir = _data_dir() / "certs"
    cert_path = cert_dir / CERT_FILENAME
    key_path = cert_dir / KEY_FILENAME

    if cert_path.exists() and key_path.exists():
        print(f"Setup Hook: TLS certs already present at {cert_dir}")
        return

    cert_dir.mkdir(parents=True, exist_ok=True)
    print(f"Setup Hook: Generating self-signed TLS cert at {cert_dir}")
    try:
        _generate_self_signed(cert_path, key_path)
        print("Setup Hook: Self-signed cert generated")
    except Exception as e:  # noqa: BLE001 - hook must never crash startup
        print("Setup Hook: Failed to generate TLS cert", e)


_main()



