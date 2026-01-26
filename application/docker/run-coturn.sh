#!/bin/bash

set -euo pipefail

# 1) Egress/LAN IP (not public) via UDP connect trick
COTURN_EXTERNAL_IP="$(uv run python - <<'PY'
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8", 80))
    print(s.getsockname()[0])
finally:
    s.close()
PY
)"

# 2) Port (default matches compose)
COTURN_PORT="${COTURN_PORT:-443}"

# 3) Build ICE_SERVERS JSON string
ICE_SERVERS="$(
  COTURN_EXTERNAL_IP="$COTURN_EXTERNAL_IP" COTURN_PORT="$COTURN_PORT" \
  uv run python - <<PY
import json, os
ip = os.environ["COTURN_EXTERNAL_IP"]
port = os.environ["COTURN_PORT"]
print(json.dumps([{
  "urls": f"turn:{ip}:{port}?transport=tcp",
  "username": "user",
  "credential": "password",
}]))
PY
)"

# 4) Start services with env vars injected
COTURN_EXTERNAL_IP="${COTURN_EXTERNAL_IP}" \
COTURN_PORT="${COTURN_PORT}" \
ICE_SERVERS="${ICE_SERVERS}" \
docker compose --profile coturn up
