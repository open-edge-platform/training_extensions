#!/bin/bash
set -e

echo "Running database migrations..."
uv run alembic -c app/alembic.ini upgrade head

echo "Starting FastAPI server..."
exec uv run fastapi run --port 7860
