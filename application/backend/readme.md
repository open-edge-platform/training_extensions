# Geti Tune

Geti Tune is a full-stack application for efficiently fine-tuning state-of-the-art computer vision models for tasks like classification, detection, and segmentation.

## Quick Start

### Basic Usage

```bash
# Start the server (development mode)
./run.sh
```

### E2E Testing Setup

```bash
# Full E2E setup with database seeding and test file downloads
DATABASE_FILE=geti_tune_e2e.db SEED_DB=true DOWNLOAD_FILES=true ./run.sh

# Or use the convenience script
./start_e2e.sh
```

## Configuration

### What `run.sh` Does

1. **Loads configuration** from environment variables
2. **Seeds database** (if `SEED_DB=true`):
   - Creates a test project with labels
   - Sets up pipeline with video source and model
3. **Downloads test files** (if `DOWNLOAD_FILES=true`):
   - Test video: `data/media/video.mp4`
   - Model files: `data/projects/.../model.xml` and `model.bin`
4. **Starts the FastAPI server** on `http://localhost:7860`

### Test Assets

By default, test assets are downloaded from a public URL. In CI/GitHub Actions, a repository variable `E2E_ASSETS_S3_URL` can be set to use a private asset location.

## Docker

The backend can also run in Docker:

```bash
cd ../docker

# Build and run E2E backend
docker compose --profile e2e up backend-e2e

# Or run full E2E stack
docker compose --profile e2e up --abort-on-container-exit
```

The Docker setup uses the same `run.sh` script for consistency.

## API Documentation

Once the server is running, visit:
- http://localhost:7860/docs

## Development

### Requirements

- Python 3.13+
- `uv` CLI tool for dependency management
- SQLite (included with Python)

### Project Structure

```
backend/
├── app/
│   ├── main.py           # FastAPI application entry point
│   ├── cli.py            # CLI commands (init-db, seed)
│   ├── api/              # API endpoints
│   ├── core/             # Core functionality (scheduler, lifecycle)
│   ├── db/               # Database models and migrations
│   ├── entities/         # Business logic entities
│   ├── repositories/     # Data access layer
│   ├── schemas/          # Pydantic schemas
│   ├── services/         # Business logic services
│   ├── webrtc/           # WebRTC streaming
│   └── workers/          # Background workers
├── data/                 # Runtime data (database, media, models)
├── run.sh               # Main startup script
```

## Troubleshooting

### Port already in use

```bash
lsof -ti:7860 | xargs kill -9
```

### Database locked

```bash
rm data/geti_tune.db
./run.sh
```

### Test file download fails

Check that `E2E_ASSETS_S3_URL` is accessible or verify the public URL is working. The script will exit with an error if downloads fail.
