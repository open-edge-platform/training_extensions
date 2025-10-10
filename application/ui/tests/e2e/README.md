# E2E Testing Guide

## 🚀 Quick Start

### **Local Development (Fastest - Recommended)**

```bash
npm run test:e2e
```

- ✅ Start backend on port 7860 with seeded database
- ✅ Start frontend on port 3000
- ✅ Run E2E tests
- ✅ Stop servers when done

### **Docker (CI/Production)**

For isolated containerized testing:

```bash
npm run test:e2e:docker
```

### **Docker with Watch Mode (Development)**

Auto-rebuild on file changes:

```bash
npm run test:e2e:docker:watch
```

---

## 📋 Prerequisites

### Local Development

- Python 3.10+ with `uv` installed
- Node.js 24+
- All dependencies installed (`npm install` in `ui/`, backend dependencies via `uv`)

### Docker

- Docker Desktop or Docker Engine
- Docker Compose v2.22+ (for `watch` support)

---

## 🛠️ Advanced Usage

### Run Specific Tests

```bash
# Local
npx playwright test --project=e2e tests/e2e/main.spec.ts

# Docker (requires manual setup)
cd ../docker
docker compose --profile e2e up -d backend-e2e frontend-e2e
docker compose --profile e2e run playwright-e2e npx playwright test tests/e2e/main.spec.ts
```

### Debug Tests Locally

```bash
# UI Mode (interactive debugging)
npx playwright test --project=e2e --ui

# Headed mode (see browser)
npx playwright test --project=e2e --headed

# Debug specific test
npx playwright test --project=e2e --debug tests/e2e/main.spec.ts
```

### View Test Reports

```bash
npx playwright show-report playwright-report
# Or open: ./playwright-report/index.html
```

---

## 🔧 How It Works

### Local Mode (No Docker)

1. Playwright config detects no `BASE_URL` env var
2. Starts `webServer[0]`: Backend via `./run.sh` with E2E configuration:
    - `DATABASE_FILE=geti_tune_e2e.db`
    - `SEED_DB=true` (initializes and seeds database)
    - `DOWNLOAD_FILES=true` (downloads test video and model files)
3. Starts `webServer[1]`: Frontend via `npm run start`
4. Waits for both to be healthy
5. Runs tests against `http://localhost:3000`
6. Stops servers after tests complete

### Docker Mode (CI)

1. `BASE_URL=http://frontend-e2e` env var is set
2. Playwright skips `webServer` (Docker handles it)
3. Docker Compose starts:
    - `backend-e2e`: Python backend using `run.sh` with E2E env vars, includes healthcheck
    - `frontend-e2e`: Nginx serving built frontend with proxy to backend
    - `playwright-e2e`: Test runner container
4. Backend automatically seeds database and downloads test files on startup
5. Tests run against frontend service DNS name
6. `--abort-on-container-exit` stops all when tests finish

### Watch Mode (Docker Development)

1. Same as Docker mode but with `watch` instead of `up`
2. Backend: `sync+restart` on `/app/app` changes
3. Frontend: `sync` on `/usr/share/nginx/html` changes
4. Tests auto-rerun on changes

---

## 🐛 Troubleshooting

### "Port already in use"

```bash
# Kill processes on ports 3000 or 7860
lsof -ti:3000 | xargs kill -9
lsof -ti:7860 | xargs kill -9
```

### "Backend not starting"

Check backend logs and verify configuration:

```bash
cd ../backend
# Check if run.sh works
DATABASE_FILE=geti_tune_e2e.db SEED_DB=true DOWNLOAD_FILES=true ./run.sh
# Should see: Database seeding, file downloads, then "Uvicorn running on http://0.0.0.0:7860"
```

Common issues:

- Missing test assets: Ensure `E2E_ASSETS_S3_URL` is accessible or using public default
- Database locked: Delete `data/geti_tune_e2e.db` and restart
- Port conflict: Kill process on port 7860 (`lsof -ti:7860 | xargs kill -9`)

### "Tests timing out"

- Increase timeout in `playwright.config.ts`: `actionTimeout: 30000`
- Or add explicit waits in tests: `await page.waitForLoadState('networkidle')`

### Docker build fails

```bash
# Clean rebuild
cd ../docker
docker compose --profile e2e down -v
docker compose --profile e2e build --no-cache
```

---

### Best Practices

- ✅ Use `test.step()` for clear test structure
- ✅ Use semantic locators: `getByText()`, `getByRole()`, `getByLabel()`
- ✅ Add explicit waits when needed: `waitForLoadState()`, `waitForURL()`
- ✅ Keep tests isolated (don't depend on other tests)
- ❌ Avoid `page.locator('.css-class')` (brittle)
- ❌ Avoid hardcoded sleeps (`await page.waitForTimeout(5000)`)

---

## 🔗 Resources

- [Playwright Docs](https://playwright.dev)
- [Docker Compose Watch](https://docs.docker.com/compose/file-watch/)
- [Our Backend API Docs](http://localhost:7860/docs)
