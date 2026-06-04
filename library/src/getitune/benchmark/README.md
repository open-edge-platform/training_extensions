# GetiTune Benchmarking

This module runs repeatable model benchmarks for GetiTune with a CLI:

- dataset provisioning from `benchmark_catalog.yaml`
- experiment selection from `benchmark_manifest.yaml`
- phased execution (`train -> test/torch -> export -> test/export -> optimize -> test/optimize`)
- MLflow tracking and baseline comparison
- report generation (`report.md`, `aggregated.csv`, optional `failed_experiments.json`)

## Where things live

- CLI entry point: `library/src/getitune/benchmark/__main__.py`
- CLI implementation: `library/src/getitune/benchmark/cli.py`
- Runner/orchestration: `library/src/getitune/benchmark/runner.py`
- Dataset catalog: `library/benchmark_catalog.yaml`
- Benchmark manifest: `library/benchmark_manifest.yaml`
- Optional MLflow server compose file: `library/src/getitune/benchmark/docker-compose.yaml`

Run commands from `library/`.

## Quick start

1. Sync dependencies (if needed):

   ```bash
   just venv --device <cpu|cuda|xpu>
   ```

2. See all benchmark commands:

   ```bash
   python -m getitune.benchmark --help
   ```

3. Provision datasets referenced by the catalog/filters:

   ```bash
   python -m getitune.benchmark provision
   ```

4. Run a small benchmark slice:

   ```bash
   python -m getitune.benchmark run --task detection --model yolox_s --dataset wgisd --accelerator <cpu|cuda|xpu> --num-seeds 1 --no-tracking
   ```

## Useful run examples

Dry run only (prints what would execute):

```bash
python -m getitune.benchmark run --dry-run --task detection --size-tier small --num-seeds 1 --no-tracking
```

Run only train+torch-test phases:

```bash
python -m getitune.benchmark run --task detection --model yolox_s --dataset wgisd --eval-upto train --num-seeds 1 --no-tracking
```

Run with custom output location:

```bash
python -m getitune.benchmark run --task detection --model yolox_s --dataset wgisd --output-root results/local-smoke --num-seeds 1 --no-tracking
```

Run with ad-hoc config overrides:

```bash
python -m getitune.benchmark run --task detection --model yolox_s --dataset wgisd --override model.init_args.optimizer.init_args.lr=0.01 --max-epochs 10 --train-kwarg precision=32 --no-tracking
```

## Report and cleanup commands

Regenerate a report from tracked MLflow runs:

```bash
python -m getitune.benchmark report --mlflow-uri http://localhost:5000 --output-root results --accelerator <cpu|cuda|xpu>
```

Clean old MLflow runs (dry run first):

```bash
python -m getitune.benchmark clean --dry-run --max-age-days 90
```

## Output artifacts

By default (`--output-root results`), the benchmark writes:

- `results/report.md` - markdown summary with pass/regression/failure sections
- `results/aggregated.csv` - flattened metrics table
- `results/failed_experiments.json` - structured failure details (only when failures exist)

Per-seed work directories are created under:

- `results/<task>/<model>/<dataset>/<seed>/` (default scenario)
- `results/<task>/<model>/<dataset>/<scenario>/<seed>/` (non-default scenario)

Dataset provisioning writes readiness markers at:

- `data/<dataset>/.ready`

## Add a new dataset or model

Use this flow when extending benchmark coverage.

### Add a new dataset

1. Add a dataset provisioning script under `scripts/benchmark_datasets/` (for example, `prepare_my_dataset.py`).
2. Add an entry in `benchmark_catalog.yaml` that points `script` to that provisioning script.
3. Reference the dataset name from the relevant task in `benchmark_manifest.yaml` under `experiments.<task>.datasets`.
4. Provision the dataset.
5. Verify the new task+dataset combination appears in a dry run.

Catalog example (`benchmark_catalog.yaml`):

```yaml
datasets:
  - name: my_dataset
    script: "scripts/benchmark_datasets/prepare_my_dataset.py"
    size_tier: medium
    description: "Short description of dataset size/content."
    compatible_tasks:
      - detection
```

Manifest reference example (`benchmark_manifest.yaml`):

```yaml
experiments:
  detection:
    datasets:
      - wgisd
      - my_dataset
```

Commands (run from `library/`):

```bash
python -m getitune.benchmark provision --dataset my_dataset
python -m getitune.benchmark run --dry-run --task detection --dataset my_dataset --num-seeds 1 --no-tracking
```

Provisioning script contract:

- The script path is read from `benchmark_catalog.yaml` (`script: ...`).
- `provision` runs the script as `python <script> --output-dir <data_root> --name <dataset_name>`.
- The script should create `data/<dataset_name>/`; the runner writes `data/<dataset_name>/.ready` after success.

### Add a new model

1. Ensure the model recipe exists under `src/getitune/recipe/<task>/...`.
2. Add a model entry under `experiments.<task>.models` in `benchmark_manifest.yaml`.
3. Run a focused benchmark slice for the new model.

Manifest model example (`benchmark_manifest.yaml`):

```yaml
experiments:
  detection:
    models:
      - name: my_detector
        priority: extended
        recipe: detection/my_detector.yaml
```

Focused run command (from `library/`):

```bash
python -m getitune.benchmark run --task detection --model my_detector --dataset wgisd --num-seeds 1 --no-tracking
```

Notes:

- `datasets` in the manifest must reference names declared in `benchmark_catalog.yaml`.
- A `default` scenario is always present implicitly; optional `scenarios` can further restrict `datasets` and `models`.

## Optional: centralized MLflow server

You can run the included MLflow+PostgreSQL stack:

```bash
docker compose -f src/getitune/benchmark/docker-compose.yaml up -d
```

Then point benchmark commands at it:

```bash
python -m getitune.benchmark run --mlflow-uri http://localhost:5000 --task detection --model yolox_s --dataset wgisd --num-seeds 1
```

If you prefer local file-based tracking (the CLI default `./mlruns`), set:

```bash
export MLFLOW_ALLOW_FILE_STORE=true
```

without that variable, newer MLflow versions can reject file-store tracking backends.
