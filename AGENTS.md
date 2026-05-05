# AGENTS.md

## Repository structure

This is a monorepo with two independent components:

- **`library/`** -- The `getitune` Python library (PyPI: `getitune`). CV transfer-learning framework built on PyTorch Lightning and OpenVINO. This is the primary codebase. The working directory for all library commands is `library/`.
- **`application/`** -- The "Geti" desktop application. FastAPI backend + React/Tauri UI. Has its own Justfile, pyproject.toml, and package.json. Largely independent from the library.

The `otx` package under `library/src/otx/` is a **deprecated shim** that re-exports everything from `getitune`. Do not add new code there.

## Library development (`library/`)

### Setup

Requires Python >=3.11, uv ~0.10.4, and [Just](https://github.com/casey/just) as the task runner.

```bash
cd library
just venv              # CPU (default). Creates .venv, runs uv sync --frozen
just venv --device cuda  # NVIDIA GPU
just venv --device xpu   # Intel GPU
```

`just venv` enforces `uv lock --check` first -- if `uv.lock` is stale, run `just venv-lock` to regenerate it.

### Lint and type-check

```bash
just lint        # runs both: ruff + pyrefly
just ruff        # ruff check + ruff format --check
just ruff-fix    # auto-fix + format
just pyrefly     # type-check with baseline (pyrefly-baseline.json)
```

- Ruff config is in `pyproject.toml`. Line length = 120. Google-style docstrings.
- pyrefly uses a baseline file (`pyrefly-baseline.json`) -- new errors must not be added; existing ones are suppressed.
- Pre-commit hooks (via `prek`, not standard `pre-commit`) enforce `uv-lock` and `pyrefly-check` in `library/`, and `ruff` + `prettier` + `hadolint` + `markdownlint` at the repo root.

### Testing

```bash
just test-unit                           # all unit tests
just test-unit -k "test_foo"             # single test
just test-unit-models                    # model-specific unit tests only
just test-integration --task detection   # integration tests for one task
```

All commands are run from the `library/` directory. They wrap `uv run pytest`.

Integration tests require a GPU and use custom pytest options:
- `--task <task>` (e.g., `detection`, `multi_class_cls`, `semantic_segmentation`)
- `--open-subprocess` -- run CLI tests in subprocess mode
- `--run-category-only` -- restrict to the recipe category matching the task

CI matrix runs integration tests per-task with CUDA. Unit tests run on Python 3.11 and 3.14.

### Key architectural notes

- **Backends**: Three backends under `src/getitune/backend/`:
  - `lightning/` -- primary training backend (PyTorch Lightning)
  - `openvino/` -- inference-only backend for exported OpenVINO IR models
  - `native/` -- alternative training pipeline extending Lightning
- **Engine factory**: `getitune.engine.create_engine(model, data)` auto-selects a backend by calling `is_supported()` on each Engine subclass.
- **Recipes**: YAML configs in `src/getitune/recipe/<task>/` define model+data+callback configurations. They use `class_path`/`init_args` patterns consumed by `jsonargparse`.
- **CLI**: Entrypoint is `getitune.cli:main` (also aliased as `otx`). Uses `jsonargparse` with subcommands: `train`, `test`, `export`, `find`, etc.
- **Import side-effects**: Importing `getitune` overrides `HF_HUB_CACHE` to `~/.cache/torch/hub/checkpoints` and sets `ONEDNN_PRIMITIVE_CACHE_CAPACITY=10000`.

### Code conventions

- Tests skip docstrings, type annotations, and line-length rules (see `per-file-ignores` in `pyproject.toml`).
- `tests/assets/` and `tests/regression/` are excluded from ruff.
- Test fixtures are session-scoped and autouse for label info and environment setup. See `tests/conftest.py`.
- Integration test recipe lists are built dynamically onto `pytest.RECIPE_LIST` at collection time.
- **Docstrings must be short and precise.** Do not restate implementation details, internal step numbers, or explain *how* code works -- that belongs in comments. A docstring says *what* the function does in one sentence. Avoid filler phrases like "This is a fixed re-implementation of …" or "The upstream implementation (as of v1.16.0) stores …".
- **No decorative comment dividers.** Do not use ASCII art dividers or banner-style comments (e.g., `# ----------`, `# ===`, `# ***`). Use blank lines and concise inline comments to separate logical sections.
- **Always run BOTH the linter AND the type-checker after every code change and fix all errors before committing.** This is a two-step process that must NEVER be skipped:
  1. **Ruff** (lint + format): `just ruff` or `uv run ruff check <files>`. Use `just ruff-fix` for auto-fixable issues.
  2. **Pyrefly** (type-check): `just pyrefly` or `uv run pyrefly check <files>`. This catches type errors that ruff does not.
  Both must pass with zero errors before committing. Do not commit code that passes ruff but fails pyrefly, or vice versa.

## Application development (`application/`)

- **Backend**: FastAPI + SQLAlchemy + Alembic. Managed with uv. See `application/backend/pyproject.toml`.
- **UI**: React + TypeScript + Rsbuild + Tauri. See `application/ui/package.json`.
- **Docker**: `just build-image --accelerator <cpu|cuda|xpu>` and `just run-image` from `application/`.
- Separate CI workflows: `backend-lint-and-test.yaml`, `ui-lint-and-test.yaml`.

## CI

- Branch model: PRs target `develop`; release branches are `release/**`.
- Library CI (`lib-lint-and-test.yaml`): lint -> unit tests (3.11+3.14) -> integration tests (per-task, CUDA).
- Datumaro dependency is installed from git and requires the Rust toolchain to build from source.
