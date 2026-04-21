---
name: otx-backend-dev
description: Develop and validate changes in `application/backend/` for the FastAPI `geti` service. Use when touching `application/backend/app/**`, backend tests, backend packaging, backend configuration, API routers, schemas, services, repositories, or database code, or when the UI contract depends on a backend API change. Helps with `uv` and `just` setup, targeted pytest or behave runs, OpenAPI generation, and local server workflows.
---

# OTX Backend Development

## Quick Start

- Work from `application/backend/`.
- Create or refresh the environment with `just venv --accelerator cpu` for normal local work.
- Switch to `cuda` or `xpu` only when the task depends on accelerator-specific behavior.
- Start with `just lint`, then run the narrowest relevant test target.

## Workflow

1. Keep the change inside the existing backend boundaries unless the task explicitly crosses into `library/` or `application/ui/`.
2. Keep routers thin and move business logic into services or repositories that match the existing package structure.
3. Generate a fresh OpenAPI spec when router or schema changes affect the API contract.
4. Hand off to the $otx-openapi-sync skill after backend contract changes so the UI types stay aligned.

## Architecture Reminders

- Respect the import-linter layering in `pyproject.toml`.
- Keep `app.api` above `app.services`, `app.repositories`, and `app.db`.
- Keep `app.api.routers` above `app.api.schemas`, and `app.api.schemas` above `app.api.dependencies`.
- Treat `run-server --clean` and `_clean_data` as destructive helpers.

## Verification

- Use `just lint` for Ruff, import-linter, and pyrefly checks.
- Use `just test-unit -- tests/unit/...` or `just test-unit -- -k <expr>` for routine backend changes.
- Use `just test-integration -- <pytest args>` when the change crosses service, persistence, or API boundaries.
- Use `just test-bdd -- <behave args>` when behavior is covered by BDD specs.
- Use `just gen-api-spec --output-path openapi-spec.json` after intentional API contract changes.

## Coordination Notes

- `application/backend` depends on the local editable `../../library`. Validate `library/` too when shared OTX behavior changes.
- Prefer project `just` targets over custom shell commands so local work matches CI.
