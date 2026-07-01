#!/usr/bin/env bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Default DATA_DIR to the backend's expected data path so the script works even
# when the variable is not provided (set -u would otherwise abort here).
DATA_DIR="${DATA_DIR:-/application/data}"
CERT_DIR="${DATA_DIR}/certs"
CERTFILE="${CERTFILE:-${CERT_DIR}/localhost.pem}"
KEYFILE="${KEYFILE:-${CERT_DIR}/localhost-key.pem}"

if [[ ! -f "$CERTFILE" || ! -f "$KEYFILE" ]]; then
    echo "TLS certs not found — generating self-signed cert (cert: ${CERTFILE}, key: ${KEYFILE})..."
    # Create the parent directories of the actual cert/key paths, which may be
    # overridden to a directory other than ${CERT_DIR}.
    mkdir -p "$(dirname "$CERTFILE")" "$(dirname "$KEYFILE")"
    openssl req -x509 -newkey rsa:2048 -nodes \
        -keyout "$KEYFILE" \
        -out "$CERTFILE" \
        -days 365 \
        -subj "/CN=localhost" \
        -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
    echo "Self-signed cert generated."
fi

exec "$@"