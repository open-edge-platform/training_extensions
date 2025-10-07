# E2E Test Assets - S3 Configuration

## Overview

E2E test assets (video and model files) are stored in S3 to avoid committing large files to git. The global setup scripts automatically download them before tests run.

## S3 Bucket Structure

```
your-bucket/e2e-assets/
├── test_video.mp4
└── models/
    ├── 977eeb18-eaac-449d-bc80-e340fbe052ad.bin     # Model binary
    └── 977eeb18-eaac-449d-bc80-e340fbe052ad.xml     # Model metadata
```

## Environment Variables

### `E2E_ASSETS_S3_URL`

- **Required**: Base URL of your S3 bucket
- **Default**: `https://your-bucket.s3.amazonaws.com/e2e-assets`
- **Example**: `E2E_ASSETS_S3_URL=https://my-bucket.s3.us-west-2.amazonaws.com/test-assets`

## Usage

### Local Development (Default)

Keep test assets in `tests/assets/` directory:

```bash
npm run test:e2e              # Uses local files
npm run test:e2e:seeded       # Uses local files
```

**Required local structure:**

```
tests/assets/
├── test_video.mp4
└── models/
    ├── 977eeb18-eaac-449d-bc80-e340fbe052ad.bin
    └── 977eeb18-eaac-449d-bc80-e340fbe052ad.xml
```

### CI/CD with S3

Set environment variables in your CI pipeline:

```bash
export E2E_ASSETS_S3_URL=https://your-bucket.s3.amazonaws.com/e2e-assets
npm run test:e2e
```

**GitHub Actions example:**

```yaml
- name: Run E2E tests
  env:
      E2E_ASSETS_S3_URL: ${{ secrets.E2E_ASSETS_S3_URL }}
  run: npm run test:e2e
```

## Setting Up S3

### 1. Upload Files to S3

```bash
# Upload video
aws s3 cp tests/assets/test_video.mp4 \
  s3://your-bucket/e2e-assets/test_video.mp4

# Upload models
aws s3 cp backend/data/models/977eeb18-eaac-449d-bc80-e340fbe052ad.bin \
  s3://your-bucket/e2e-assets/models/

aws s3 cp backend/data/models/977eeb18-eaac-449d-bc80-e340fbe052ad.xml \
  s3://your-bucket/e2e-assets/models/
```

## How It Works

### New Project Tests

- **Config**: `playwright.e2e.config.ts`
- **Setup**: `global-setup.ts`
- Downloads/copies: Video only

### Seeded Tests

- **Config**: `playwright.e2e-seeded.config.ts`
- **Setup**: `global-setup-seeded.ts`
- Downloads/copies: Video + Model files

### Flow

```
1. Test starts
2. Global setup runs
   ├─ If CI=true → Download from S3
   └─ If CI=false → Copy from local files
3. Files placed in backend/data/
4. Tests run
5. Global teardown cleans up (CI only)
```

## Troubleshooting

### Download Fails

```
✗ Failed to download from S3: Error: Failed to download [url]: 403
```

**Solutions:**

- Check S3 bucket permissions
- Verify `E2E_ASSETS_S3_URL` is correct
- Ensure files exist in S3
- Check network/firewall settings

### Files Not Found Locally

```
Error: Test video not found at [path]
```

**Solutions:**

- Run with S3: `CI=true npm run test:e2e`
- Or copy files locally to `tests/assets/`

### Wrong Model Files

```
Backend error: Model not found
```

**Solutions:**

- Verify model UUIDs match seeded database
- Check that both `.bin` and `.xml` files are present
- Update `MODEL_FILES` array in `global-setup-seeded.ts`

## Adding New Test Assets

1. **Upload to S3:**

    ```bash
    aws s3 cp dog.png s3://your-bucket --profile your-profile
    ```

2. **Update setup script:**

    ```typescript
    // In global-setup.ts or global-setup-seeded.ts
    const NEW_ASSET_URL = `${S3_BUCKET_URL}/new-asset.mp4`;
    await downloadFile(NEW_ASSET_URL, targetPath);
    ```

3. **Add to local fallback:**
    ```bash
    cp new-asset.mp4 tests/assets/
    ```
