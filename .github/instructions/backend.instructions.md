---
applyTo: "application/backend/**"
---

## Code conventions

- **Async everywhere** for I/O (DB, file, HTTP, WebRTC). Never call blocking
  SQLAlchemy sync APIs from request handlers.
- Routers under `app/` (`APIRouter` per resource). Keep handlers thin; business
  logic in service modules.
- Pydantic v2 models for request/response — never return raw ORM models.
- Settings via `pydantic-settings` for app config.
- Use `loguru` `logger` — not `print`, not stdlib `logging`.
- Long-running work goes in background tasks, not inline in a request.

## Database & migrations

- Schema changes **require an Alembic migration**.
- Use the async session factory — do not create engines per request.

## API contract

- After changing routes/models/status codes:
  1. `just gen-api-spec --output-path=openapi.json`
  2. In `application/ui/`: `npm run update-spec`
- Keep paths, methods, and field names stable. Breaking changes require
  coordination with the UI before merging.

## Commands

| Task                    | Recipe                                                       |
| ----------------------- | ------------------------------------------------------------ |
| Create venv             | `just venv --accelerator cpu\|xpu\|cuda`                     |
| Refresh lockfile        | `just venv-lock`                                             |
| Start server            | `just run-server`                                            |
| Regenerate OpenAPI spec | `just gen-api-spec --output-path=openapi.json`               |
| Lint + type-check       | `just lint`                                                  |
| Auto-fix lint           | `just ruff-fix`                                              |
| Type-check only         | `just pyrefly` (no baseline file — all errors must be fixed) |
| Import-graph check      | `just lint-imports`                                          |

## Testing

- `pytest` + `pytest-asyncio`. Use `testcontainers` for real-DB integration tests.
- `behave` for BDD scenarios — keep steps with feature files.
- Prefer `pytest.mark.parametrize` for multiple inputs.

## Build

- PyInstaller packages the backend (`geti.spec`). If you add a new top-level
  import, verify it appears in `geti.spec`'s `hiddenimports`.

## Do not

- Do not bypass the settings layer.
- Never `pip install` — use `uv` via `just` recipes.
