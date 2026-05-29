# Database & Storage Migration Guidelines

> **Audience:** Developers working on the Geti backend.
> **Context:** After the Geti 3.0 release the SQLite database (`geti.db`) and the file-system storage layout under `data/` are considered **stable surfaces**. Every schema or layout change must be accompanied by a migration script so that existing user data survives an upgrade.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Golden Rules](#golden-rules)
3. [Working with Alembic](#working-with-alembic)
4. [Examples](#examples)
   - [Add a New Field to an Existing Table](#1-add-a-new-field-to-an-existing-table)
   - [Remove a Field from an Existing Table](#2-remove-a-field-from-an-existing-table)
   - [Split a Table](#3-split-a-table)
   - [Merge Two Tables into One](#4-merge-two-tables-into-one)
   - [Add a New Table](#5-add-a-new-table)
   - [Remove a Table](#6-remove-a-table)
5. [File-Storage Migrations](#file-storage-migrations)
6. [Testing Migrations](#testing-migrations)
7. [Review Checklist](#review-checklist)

---

## Architecture Overview

| Component         | Technology                                     | Location                       |
| ----------------- | ---------------------------------------------- | ------------------------------ |
| Database          | SQLite (via SQLAlchemy 2.x)                    | `data/geti.db`                 |
| Schema models     | `app/db/schema.py` (`Base` declarative models) | `app/db/schema.py`             |
| Migration engine  | Alembic                                        | `app/alembic/`                 |
| Migration scripts | Python (auto-generated + hand-edited)          | `app/alembic/versions/`        |
| File storage      | Local filesystem                               | `data/projects/<project_id>/…` |
| Migration runner  | `app/db/migration.py` → `MigrationManager`     | Runs on app startup            |

### File-Storage Layout

```
data/
├── geti.db                                         # SQLite database
├── pretrained_weights/                             # Base model weights
├── staged_datasets/                                # Temporary import staging area
└── projects/
    └── <project_id>/
        ├── dataset/                                # Media files & thumbnails
        │   ├── <media_id>.<format>
        │   └── <media_id>-thumb.jpg
        ├── dataset_revisions/
        │   └── <revision_id>/                      # Exported Datumaro snapshots
        └── models/
            └── <model_id>/
                ├── training.log
                ├── metrics/version_0/metrics.csv
                └── variants/
                    └── <variant_id>/
                        └── model.<ext>
```

> **Key insight:** The database stores metadata; the filesystem stores binary artifacts. They must always stay in sync.

---

## Golden Rules

1. **Every schema change requires an Alembic migration script.** Never modify `schema.py` without a corresponding migration in `app/alembic/versions/`.
2. **Migrations must be idempotent-safe.** The `MigrationManager.run_migrations()` runs `alembic upgrade head` on every application startup.
3. **Always provide a `downgrade()` function.** Even if downgrades are not used in production, they are essential for development and testing.
4. **File-storage changes go inside the same Alembic migration script** (see [File-Storage Migrations](#file-storage-migrations)). This keeps the database version and the storage layout version in lockstep.
5. **Never rename the Alembic revision file** once it has been merged. Other developers or deployed instances may already reference it.
6. **Keep migrations small and focused.** One logical change per migration. If a feature needs both a new table and a storage move, two migrations in sequence are fine.
7. **Test with real data.** Always verify the migration against a database created by the _previous_ version of the code.

---

## Working with Alembic

### Generating a Migration

After modifying `app/db/schema.py`, auto-generate a migration:

```bash
cd application/backend
uv run alembic -c app/alembic.ini revision --autogenerate -m "describe_the_change"
```

This creates a new file in `app/alembic/versions/`. **Always review the generated code** — Alembic's autogenerate does not detect every change (e.g., column renames, data migrations, storage changes).

### Applying Migrations

```bash
# Upgrade to latest
uv run alembic -c app/alembic.ini upgrade head

# Upgrade one step
uv run alembic -c app/alembic.ini upgrade +1

# Downgrade one step
uv run alembic -c app/alembic.ini downgrade -1

# Show current revision
uv run alembic -c app/alembic.ini current

# Show migration history
uv run alembic -c app/alembic.ini history --verbose
```

### SQLite Limitations

SQLite does not support all DDL operations that other databases do. Key limitations to remember:

| Operation                                    | Supported?           | Workaround                                   |
| -------------------------------------------- | -------------------- | -------------------------------------------- |
| `ALTER TABLE ... ADD COLUMN`                 | Yes                  | --                                           |
| `ALTER TABLE ... DROP COLUMN`                | Yes (SQLite >= 3.35) | Use `op.batch_alter_table()` for safety      |
| `ALTER TABLE ... RENAME COLUMN`              | Yes (SQLite >= 3.25) | Use `op.batch_alter_table()` for safety      |
| `ALTER TABLE ... ALTER COLUMN` (type change) | No                   | Use `op.batch_alter_table()` (table rebuild) |
| `ALTER TABLE ... ADD CONSTRAINT`             | No                   | Use `op.batch_alter_table()` (table rebuild) |
| `CREATE INDEX` / `DROP INDEX`                | Yes                  | --                                           |

When in doubt, use **batch mode** (`op.batch_alter_table()`). Alembic will transparently rebuild the table:

```python
with op.batch_alter_table("projects") as batch_op:
    batch_op.add_column(sa.Column("description", sa.Text(), nullable=True))
    batch_op.drop_column("old_field")
```

> ⚠️ **SQLite foreign key constraint caveat:** `batch_alter_table` works by dropping and recreating the table. If _other_ tables have foreign keys pointing **to** the table being rebuilt, SQLite will raise `FOREIGN KEY constraint failed` on the `DROP TABLE` step — even if no rows would actually be violated. Wrap the batch operation with `PRAGMA foreign_keys = OFF/ON` to suppress this:
>
> ```python
> op.execute("PRAGMA foreign_keys = OFF")
> with op.batch_alter_table("my_table") as batch_op:
>     batch_op.alter_column("some_col", nullable=False)
> op.execute("PRAGMA foreign_keys = ON")
> ```
>
> This is safe because Alembic's batch mode copies all data into the new table before dropping the old one — referential integrity is preserved in practice. You only need this when other tables have FK references **pointing to** the table you are rebuilding (not when the table itself has outbound FKs).

---

## Examples

### 1. Add a New Field to an Existing Table

**Breaking?** No, if the new column is nullable or has a default value.

#### Step 1: Update the schema model

```python
# app/db/schema.py
class ProjectDB(BaseID):
    __tablename__ = "projects"
    # ...existing columns...
    description: Mapped[str | None] = mapped_column(Text, nullable=True)  # NEW
```

#### Step 2: Generate and review migration

```bash
uv run alembic -c app/alembic.ini revision --autogenerate -m "add_description_to_projects"
```

#### Step 3: Verify the generated migration

```python
# app/alembic/versions/<rev>_add_description_to_projects.py

def upgrade() -> None:
    op.add_column("projects", sa.Column("description", sa.Text(), nullable=True))

def downgrade() -> None:
    op.drop_column("projects", "description")
```

#### ⚠️ If the column is NOT NULL and has no default

You must backfill existing rows. Use a **three-step** approach:

```python
def upgrade() -> None:
    # 1. Add column as nullable
    op.add_column("projects", sa.Column("priority", sa.Integer(), nullable=True))
    # 2. Backfill existing rows
    op.execute("UPDATE projects SET priority = 0 WHERE priority IS NULL")
    # 3. Make it non-nullable via batch mode (required for SQLite).
    #    Disable FK checks because other tables reference `projects` and SQLite
    #    would refuse to DROP the table during the batch rebuild otherwise.
    op.execute("PRAGMA foreign_keys = OFF")
    with op.batch_alter_table("projects") as batch_op:
        batch_op.alter_column("priority", nullable=False)
    op.execute("PRAGMA foreign_keys = ON")
```

---

### 2. Remove a Field from an Existing Table

**Breaking?** Yes — any code or query referencing the column will break.

#### Pre-conditions

- Remove all references to the column in application code _first_.
- Verify no downstream code (API serializers, repositories, services) uses the field.

#### Migration

```python
def upgrade() -> None:
    # `projects` is referenced by other tables, so disable FK checks during
    # the batch rebuild to avoid "FOREIGN KEY constraint failed" on DROP TABLE.
    op.execute("PRAGMA foreign_keys = OFF")
    with op.batch_alter_table("projects") as batch_op:
        batch_op.drop_column("old_field")
    op.execute("PRAGMA foreign_keys = ON")

def downgrade() -> None:
    # Restore the column (data will be lost)
    op.execute("PRAGMA foreign_keys = OFF")
    with op.batch_alter_table("projects") as batch_op:
        batch_op.add_column(sa.Column("old_field", sa.String(100), nullable=True))
    op.execute("PRAGMA foreign_keys = ON")
```

> **Tip for graceful removal:** If you want a non-breaking transition, deprecate the column in one release (stop writing to it), then remove it in the next release.

---

### 3. Split a Table

**Breaking?** Yes — foreign keys, repositories, and services will need updates.

**Example:** Split `media` into `media` (metadata) and `media_files` (storage details).

```python
def upgrade() -> None:
    # 1. Create the new table
    op.create_table(
        "media_files",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("media_id", sa.Text(), nullable=False),
        sa.Column("format", sa.String(50), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.ForeignKeyConstraint(["media_id"], ["media.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # 2. Migrate data from the old table into the new one
    op.execute("""
        INSERT INTO media_files (id, media_id, format, size)
        SELECT id, id, format, size FROM media
    """)

    # 3. Drop the migrated columns from the original table.
    #    `media` is referenced by `dataset_items`, so disable FK checks during rebuild.
    op.execute("PRAGMA foreign_keys = OFF")
    with op.batch_alter_table("media") as batch_op:
        batch_op.drop_column("format")
        batch_op.drop_column("size")
    op.execute("PRAGMA foreign_keys = ON")


def downgrade() -> None:
    # 1. Re-add columns
    op.execute("PRAGMA foreign_keys = OFF")
    with op.batch_alter_table("media") as batch_op:
        batch_op.add_column(sa.Column("format", sa.String(50), nullable=True))
        batch_op.add_column(sa.Column("size", sa.Integer(), nullable=True))
    op.execute("PRAGMA foreign_keys = ON")

    # 2. Copy data back
    op.execute("""
        UPDATE media SET format = (
            SELECT format FROM media_files WHERE media_files.media_id = media.id
        ), size = (
            SELECT size FROM media_files WHERE media_files.media_id = media.id
        )
    """)

    # 3. Drop the new table
    op.drop_table("media_files")
```

---

### 4. Merge Two Tables into One

**Breaking?** Yes — both tables' repositories and FK references must be updated.

**Example:** Merge `evaluations` and `metric_scores` into a single `evaluation_results` table.

```python
def upgrade() -> None:
    # 1. Create the merged table
    op.create_table(
        "evaluation_results",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("model_revision_id", sa.Text(), nullable=False),
        sa.Column("model_variant_id", sa.Text(), nullable=False),
        sa.Column("dataset_revision_id", sa.Text(), nullable=False),
        sa.Column("subset", sa.String(20), nullable=False),
        sa.Column("metric", sa.String(255), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.ForeignKeyConstraint(["model_revision_id"], ["model_revisions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["model_variant_id"], ["model_variants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["dataset_revision_id"], ["dataset_revisions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # 2. Migrate data by joining the two source tables
    op.execute("""
        INSERT INTO evaluation_results (
            id, model_revision_id, model_variant_id, dataset_revision_id,
            subset, metric, score, created_at, updated_at
        )
        SELECT
            ms.id, e.model_revision_id, e.model_variant_id, e.dataset_revision_id,
            e.subset, ms.metric, ms.score, ms.created_at, ms.updated_at
        FROM metric_scores ms
        JOIN evaluations e ON ms.evaluation_id = e.id
    """)

    # 3. Drop old tables (order matters: child first)
    op.drop_table("metric_scores")
    op.drop_table("evaluations")


def downgrade() -> None:
    # Recreate original tables and reverse the data migration
    op.create_table(
        "evaluations",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("model_revision_id", sa.Text(), nullable=False),
        sa.Column("model_variant_id", sa.Text(), nullable=False),
        sa.Column("dataset_revision_id", sa.Text(), nullable=False),
        sa.Column("subset", sa.String(20), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "metric_scores",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("evaluation_id", sa.Text(), nullable=False),
        sa.Column("metric", sa.String(255), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.ForeignKeyConstraint(["evaluation_id"], ["evaluations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    # Reverse data migration omitted for brevity — follow the same join pattern in reverse
```

---

### 5. Add a New Table

**Breaking?** No — purely additive.

#### Step 1: Define the model in `schema.py`

```python
# app/db/schema.py
class AuditLogDB(BaseID):
    __tablename__ = "audit_logs"
    __table_args__ = (Index("idx_audit_logs_project", "project_id"),)

    project_id: Mapped[str] = mapped_column(Text, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    details: Mapped[dict] = mapped_column(JSON, nullable=True)
```

#### Step 2: Auto-generate the migration

```bash
uv run alembic -c app/alembic.ini revision --autogenerate -m "add_audit_logs_table"
```

#### Step 3: Verify

The generated migration should contain a `op.create_table(...)` call. Check that indexes and foreign keys are correct.

#### Step 4: Update import/export (if applicable)

If the new table contains project-scoped data, update `QUERY_STRATEGY_BY_TABLE` in `app/db/import_export/export_project.py` and `IMPORT_ORDER` in `app/db/import_export/import_project.py`:

```python
# export_project.py
QUERY_STRATEGY_BY_TABLE = {
    # ...existing entries...
    "audit_logs": "project_id",  # NEW
}

# import_project.py
IMPORT_ORDER = [
    # ...existing entries...
    "audit_logs",  # NEW — place after "projects" (its FK dependency)
]
```

---

### 6. Remove a Table

**Breaking?** Yes — all code referencing the table must be removed first.

#### Pre-conditions

- Remove the SQLAlchemy model from `schema.py`.
- Remove the repository, service code, and API endpoints that use the table.
- Remove the table from `QUERY_STRATEGY_BY_TABLE` and `IMPORT_ORDER`.
- Remove any foreign keys that reference this table from other tables (may require a separate prior migration).

#### Migration

```python
def upgrade() -> None:
    op.drop_table("deprecated_table")

def downgrade() -> None:
    op.create_table(
        "deprecated_table",
        sa.Column("id", sa.Text(), nullable=False),
        # ...recreate all columns...
        sa.PrimaryKeyConstraint("id"),
    )
```

> **Important:** If other tables have foreign keys pointing to this table, drop those constraints first (in a preceding migration or in the same migration before `drop_table`).

---

## File-Storage Migrations

Alembic migration scripts are regular Python code — they can do anything, including moving, renaming, or restructuring files on disk. Since `MigrationManager` runs `alembic upgrade head` on every application startup, this is the natural place to put storage layout changes.

The key benefit: **the Alembic revision number becomes the single source of truth for both the database schema version and the storage layout version.** This is already leveraged by the project import/export system, which checks `get_database_schema_version()` to ensure compatibility.

### Creating a Storage-Only Migration

Since a file-storage migration has **no schema changes**, Alembic's `--autogenerate` won't detect anything. Instead, generate an **empty** migration and fill in the logic by hand:

```bash
cd application/backend
uv run alembic -c app/alembic.ini revision -m "move_media_to_new_prefix"
```

> **Important:** Do NOT invent a revision ID yourself. The `alembic revision` command auto-generates a unique 12-character hex ID (e.g., `a3f8b1c9d2e4`) and sets `down_revision` to the current head. The placeholder values in the example below (like `abc123def456`) are for illustration only — in practice they will be generated for you.

If your migration involves **both** schema changes and storage changes, you can use `--autogenerate` instead — Alembic will generate the schema diff, and you then hand-edit the script to add the file-moving logic.

### Example Storage Migration

```python
"""move_media_to_new_prefix

Revision ID: abc123def456
Revises: 2d2b0c9a5c2c
Create Date: 2026-04-01 10:00:00.000000
"""

import shutil
from pathlib import Path

from alembic import op
from sqlalchemy import text

# revision identifiers, used by Alembic.
# These are auto-generated — do not change them manually.
revision: str = "abc123def456"
down_revision: str = "2d2b0c9a5c2c"
branch_labels = None
depends_on = None

# Resolve the data directory. This must match the application's settings.
# The Alembic config provides the database URL, from which we derive the data dir.
def _get_data_dir() -> Path:
    """Derive the data directory from the database URL in the Alembic config."""
    from alembic import context
    config = context.config
    db_url = config.get_main_option("sqlalchemy.url")  # e.g. sqlite:///data/geti.db
    db_path = Path(db_url.replace("sqlite:///", ""))
    return db_path.parent


def upgrade() -> None:
    """Move media files from dataset/ to media/ prefix."""
    data_dir = _get_data_dir()
    projects_dir = data_dir / "projects"

    if not projects_dir.exists():
        return  # Fresh install, nothing to migrate

    conn = op.get_bind()
    result = conn.execute(text("SELECT id FROM projects"))
    project_ids = [row[0] for row in result]

    for project_id in project_ids:
        old_path = projects_dir / project_id / "dataset"
        new_path = projects_dir / project_id / "media"

        if old_path.exists() and not new_path.exists():
            shutil.move(str(old_path), str(new_path))


def downgrade() -> None:
    """Move media files back from media/ to dataset/ prefix."""
    data_dir = _get_data_dir()
    projects_dir = data_dir / "projects"

    if not projects_dir.exists():
        return

    conn = op.get_bind()
    result = conn.execute(text("SELECT id FROM projects"))
    project_ids = [row[0] for row in result]

    for project_id in project_ids:
        old_path = projects_dir / project_id / "media"
        new_path = projects_dir / project_id / "dataset"

        if old_path.exists() and not new_path.exists():
            shutil.move(str(old_path), str(new_path))
```

### Best Practices for Storage Migrations

1. **Make moves idempotent.** Always check `if old_path.exists() and not new_path.exists()` before moving. The migration may be interrupted and re-run.
2. **Use `shutil.move`, not `os.rename`.** `shutil.move` works across filesystem boundaries.
3. **Derive the data directory from the Alembic config** (via the database URL). Do not import application settings directly — Alembic's `env.py` may run in a different context.
4. **Query the database to discover which projects/entities exist.** Don't blindly scan the filesystem — use `op.get_bind()` to execute SQL queries within the migration.
5. **Log progress.** For large migrations, add logging so operators can monitor the upgrade:

   ```python
   from loguru import logger

   logger.info("Migrating media for project {}", project_id)
   ```

6. **Handle errors gracefully.** If a file move fails, the migration should raise (causing a rollback of the DB transaction), leaving the system in the pre-migration state.

### When NOT to Use Alembic for Storage

If a storage change is purely operational (e.g., moving the entire `data/` directory to a different mount point), it should be handled via configuration (`DATA_DIR` environment variable), not a migration.

---

## Testing Migrations

### Unit Tests for Migrations

Every migration should be tested to verify:

1. `upgrade()` succeeds on a database at the previous revision.
2. `downgrade()` restores the schema to the previous state.
3. Data is preserved through the upgrade (for data-migrating scripts).

### Test Pattern

```python
import pytest
from alembic import command
from alembic.config import Config


@pytest.fixture
def alembic_config(tmp_path):
    """Create a test Alembic config pointing to a temporary database."""
    cfg = Config("app/alembic.ini")
    cfg.set_main_option("script_location", "app/alembic")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{tmp_path / 'test.db'}")
    return cfg


def test_upgrade_to_head(alembic_config):
    """Verify all migrations apply cleanly from scratch."""
    command.upgrade(alembic_config, "head")


def test_upgrade_downgrade_cycle(alembic_config):
    """Verify upgrade then downgrade returns to base."""
    command.upgrade(alembic_config, "head")
    command.downgrade(alembic_config, "base")


def test_data_preserved_after_upgrade(alembic_config, tmp_path):
    """Verify data migration preserves existing rows."""
    # 1. Upgrade to the revision BEFORE the migration under test
    command.upgrade(alembic_config, "<previous_revision>")

    # 2. Seed test data
    from sqlalchemy import create_engine, text
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    with engine.connect() as conn:
        conn.execute(text("INSERT INTO projects (id, name, task_type, exclusive_labels) VALUES ('abc', 'Test', 'detection', 0)"))
        conn.commit()

    # 3. Upgrade to the migration under test
    command.upgrade(alembic_config, "<new_revision>")

    # 4. Assert data is correct
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM projects WHERE id = 'abc'"))
        row = result.fetchone()
        assert row is not None
        # ... verify migrated/backfilled values
```

### Testing Storage Migrations

For migrations that move files, seed the filesystem in the test fixture:

```python
def test_storage_migration(alembic_config, tmp_path):
    # Set up old layout
    old_dir = tmp_path / "projects" / "proj-1" / "dataset"
    old_dir.mkdir(parents=True)
    (old_dir / "image.jpg").write_bytes(b"fake image data")

    command.upgrade(alembic_config, "head")

    # Assert new layout
    new_dir = tmp_path / "projects" / "proj-1" / "media"
    assert new_dir.exists()
    assert (new_dir / "image.jpg").read_bytes() == b"fake image data"
    assert not old_dir.exists()
```

---

## Review Checklist

Use this checklist when reviewing a PR that includes a migration:

- [ ] **`schema.py` updated** — The declarative model matches the migration.
- [ ] **Migration auto-generated and reviewed** — Hand-edit if Alembic missed changes (e.g., renames, data backfills).
- [ ] **`upgrade()` is correct** — Handles existing data; uses `batch_alter_table` where needed for SQLite.
- [ ] **`downgrade()` is correct** — Reverses the upgrade; data loss is acceptable but documented.
- [ ] **Import/export updated** — `QUERY_STRATEGY_BY_TABLE` and `IMPORT_ORDER` reflect new/removed/renamed tables.
- [ ] **Storage migrations are idempotent** — Check for existing paths before moving.
- [ ] **No hardcoded paths** — Data directory is derived from config, not hardcoded.
- [ ] **Tests exist** — Upgrade, downgrade, and data-preservation tests are present.
- [ ] **`down_revision` is correct** — Points to the actual current head, not a stale value (watch for merge conflicts!).
- [ ] **One logical change per migration** — Complex features should be split into multiple sequential migrations.
