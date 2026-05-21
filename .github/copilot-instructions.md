## Repository layout

Monorepo with three components — different languages, toolchains, and conventions.

| Path                   | What it is                                                 | Primary stack                                                    |
| ---------------------- |------------------------------------------------------------| ---------------------------------------------------------------- |
| `library/`             | `getitune` — low-code transfer-learning CV library (PyPI). | Python 3.11+, PyTorch 2.10, OpenVINO, Lightning, Datumaro        |
| `application/backend/` | Geti™ app server (`geti` package).                         | Python 3.13, FastAPI, SQLAlchemy 2 (async), Pydantic v2, Alembic |
| `application/ui/`      | Geti™ web/desktop UI.                                      | Node 24.2+, React, TypeScript, rsbuild, Tauri                    |

The library is consumed by the backend (`getitune[cpu|xpu|cuda]` extras).

## General rules

- Do not mix code or conventions between the three sub-projects.
- Use **absolute imports** within each Python package.
- Prefer editing existing files over creating new ones; match surrounding style.
- Code must pass pre-commit checks (see `.pre-commit-config.yaml`).
- Conventional Commits: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`, `ci:`.
- **New source files require a copyright + SPDX header** (current year, `#` for Python/YAML/shell, `//` for TS/JS/Rust):

  ```
  # Copyright (C) 2026 Intel Corporation
  # SPDX-License-Identifier: Apache-2.0
  ```

- **Use `just`** for developer workflows (`application/ui/` uses `npm` scripts instead).
  Never invent ad-hoc `uv` / `docker` commands when a recipe exists.

## Python conventions (`library/` and `application/backend/`)

- Type-hint every public function, method, and module-level variable.
- Modern typing: `list[int]`, `X | None` — not `List`, `Optional`.
- Prefer `pathlib.Path` over `os.path`.
- No bare `print`.
- Google-style docstrings (`Args`, `Returns`, `Raises`).
- Logging: `library/` uses stdlib `logging`; `application/backend/` uses
  `loguru`. Do not mix them.
- Tests use `pytest`; new features require unit tests.

## Things to avoid

- Do not change Python version pins (they ensure CI reproducibility).
- Prefer stdlib or already-declared dependencies.
