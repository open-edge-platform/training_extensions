# Training Extensions Agent Guide

## Role of This File

- `AGENTS.md` is the canonical repo-wide instruction file for agentic tools.
- `CLAUDE.md` imports this file for Claude Code compatibility.
- Portable task skills live in `.agents/skills/`.
- Claude-native mirrors live in `.claude/skills/`.
- Keep always-on repository rules here; keep task-specific workflows in skills.

## Repository Map

- `library/`: OTX Python package, recipes, and tests.
- `application/backend/`: FastAPI backend named `geti`; consumes `../../library` as an editable `uv` source.
- `application/ui/`: React 19 + TypeScript + RSBuild frontend.
- `library/docs/`: Sphinx-based documentation source.
- `README.md` & `CHANGELOG.md`: root-level project documentation.
- `.github/workflows/`: CI source of truth for path-based checks and required jobs.

## Data & State

- The backend stores persistent data in `application/backend/data/`.
- This includes the SQLite database (`geti.db`) and media artifacts (images, videos) under `projects/`.

## Choose the Right Workflow

- Use the `library` workflow for changes under `library/src`, `library/tests`, or OTX model, training, export, and CLI logic.
- Use the `backend` workflow for changes under `application/backend/app`, backend tests, backend packaging, or backend API schemas.
- Use the `ui` workflow for changes under `application/ui/src`, frontend tests, build config, or generated API client types.
- Use the OpenAPI sync workflow whenever backend API contracts change and the UI consumes those changes.
- Use the documentation update workflow to keep `README.md`, `CHANGELOG.md`, or `library/docs/` in sync with code changes.

## Commands: Library

- Work from `library/`.
- Create or refresh the environment with `just venv --device cpu`, `just venv --device cuda`, or `just venv --device xpu`.
- Run lint and type checks with `just lint`.
- Run unit tests with `just test-unit -- <pytest args>`.
- Run model-focused unit tests with `just test-unit-models -- <pytest args>`.
- Run integration tests with `just test-integration -- <pytest args>`.

## Commands: Backend

- Work from `application/backend/`.
- Create or refresh the environment with `just venv --accelerator cpu`, `just venv --accelerator cuda`, or `just venv --accelerator xpu`.
- Run lint and type checks with `just lint`.
- Run unit tests with `just test-unit -- <pytest args>`.
- Run integration tests with `just test-integration -- <pytest args>`.
- Run BDD tests with `just test-bdd -- <behave args>`.
- Generate an OpenAPI spec with `just gen-api-spec --output-path openapi-spec.json`.
- Start the local server with `just run-server`.
- Treat `_clean_data` and `run-server --clean` as destructive operations; use them only when the task explicitly calls for data reset.

## Commands: UI

- Work from `application/ui/`.
- Use Node `>=24.2.0` and npm `>=11.3.0`.
- Install dependencies with `npm ci`. This also fetches the core `@geti` UI packages via a `preinstall` hook.
- Build with `npm run build`.
- Run formatting checks with `npm run format:check`.
- Run lint with `npm run lint`.
- Check cyclic imports with `npm run cyclic-deps-check`.
- Run the TypeScript checker with `npm run type-check`.
- Run unit tests with `npm run test:unit` or `npm run test:unit:coverage`.
- Use `npm run test:component` and `npm run test:e2e` only when the change actually affects those layers.
- Build API typings from an existing spec with `npm run build:api`.
- Pull a spec from a running backend with `npm run update-spec` when `http://localhost:7860` is available.

## Cross-Area Rules

- Do not assume commands from one area apply to another; `library`, `application/backend`, and `application/ui` use different runtimes and toolchains.
- Backend changes can require validating `library/` because `application/backend` depends on the local editable package.
- Do not hand-edit generated UI OpenAPI typings when regeneration is possible.
- When backend request or response schemas change, regenerate the UI OpenAPI spec and TypeScript definitions in the same change set.
- Use `.github/workflows/lib-lint-and-test.yaml` and `.github/workflows/ui-lint-and-test.yaml` as the source of truth for CI expectations if local commands are ambiguous.

## Change Discipline

- Prefer minimal, area-scoped edits.
- Follow existing Ruff, import-linter, ESLint, Prettier, and TypeScript configuration instead of introducing parallel style rules.
- Update tests or docs when behavior changes.
- Run the narrowest relevant verification before finishing and state exactly what was not run.
