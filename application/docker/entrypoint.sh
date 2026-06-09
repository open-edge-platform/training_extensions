#!/usr/bin/env bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

CERT_DIR="${DATA_DIR}/certs"
CERTFILE="${CERTFILE:-${CERT_DIR}/localhost.pem}"
KEYFILE="${KEYFILE:-${CERT_DIR}/localhost-key.pem}"

if [[ ! -f "$CERTFILE" || ! -f "$KEYFILE" ]]; then
    echo "TLS certs not found — generating self-signed cert (cert: ${CERTFILE}, key: ${KEYFILE})..."
    mkdir -p "${CERT_DIR}"
    openssl req -x509 -newkey rsa:2048 -nodes \
        -keyout "$KEYFILE" \
        -out "$CERTFILE" \
        -days 365 \
        -subj "/CN=localhost" \
        -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
    echo "Self-signed cert generated."
fi

exec "$@"