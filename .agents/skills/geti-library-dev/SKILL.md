---
name: geti-library-dev
description: Develop and validate changes in `library/` for the OTX Python package. Use when touching `library/src/**`, `library/tests/**`, `library/pyproject.toml`, recipes, or any Python API, CLI, model, training, export, or utility logic owned by the library. Helps with `uv` and `just` setup, choosing cpu, cuda, or xpu extras, and running the smallest relevant lint, unit, model, or integration checks.
---

# OTX Library Development

## Quick Start

- Work from `library/`.
- Create or refresh the environment with `just venv --device cpu` for routine work.
- Switch to `just venv --device cuda` or `just venv --device xpu` only when the task needs accelerator-specific behavior.
- Run `just lint` before wider test runs.

## Workflow

1. Confirm the change belongs in `library/`. If the task is mainly FastAPI or React work, switch to the matching backend or UI skill.
2. Inspect the nearest module and tests before editing. Keep changes inside the existing package boundaries under `src/getitune/` (legacy `src/otx/` only re-exports; prefer editing `getitune`).
3. Make the smallest change that resolves the task. Avoid lockfile churn unless dependencies changed intentionally.
4. Run the smallest relevant checks first and widen only if the changed behavior crosses package or task boundaries.

## Verification

- Use `just lint` for formatting, lint, and type issues.
- Use `just test-unit -- tests/unit/...` or `just test-unit -- -k <expr>` for normal Python behavior changes.
- Use `just test-unit-models -- <pytest args>` for model-specific code under the backend native models area.
- Use `just test-integration -- <pytest args>` only when the change affects end-to-end training, export, or integration behavior.

## Coordination Notes

- `application/backend` consumes `../../library` as a local editable dependency. If the change affects shared runtime behavior, validate the backend too.
- Update docs or examples when public library behavior changes.
- Prefer project `just` targets over ad hoc dependency-install commands so the pinned `uv` workflow stays consistent with CI.
