# OTX Benchmark Dashboard

This is an interactive Plotly Dash application for visualizing OTX benchmark results across different versions and tasks.

## Features

- **Task-level tabs**: Navigate between different OTX tasks (ANOMALY, DETECTION, SEGMENTATION, etc.)
- **Dataset-level sub-tabs**: Within each task, browse different datasets
- **Interactive plots**: Compare metrics (accuracy, latency) across OTX versions
- **Multiple visualization types**:
  - Accuracy comparison bar charts
  - Latency comparison bar charts
  - Training time bar charts
  - Epoch comparison bar charts
  - Iteration time bar charts
  - Accuracy vs Latency scatter plots

## Installation

1. Install the required dependencies:

```bash
uv pip install -e ".[ci_benchmark]"
```

## Usage

Run the dashboard with default settings:

```bash
python tests/perf_v2/plot_benchmark.py
```

This will:

- Use `tests/perf_v2/perf_history` as the benchmark directory
- Start the server on port 8050
- Open the dashboard at `http://localhost:8050`

### Advanced Usage

Customize the benchmark directory and port:

```bash
python tests/perf_v2/plot_benchmark.py --benchmark_dir /path/to/your/benchmark/data --port 8080
```

### Command Line Options

- `--benchmark_dir`: Path to the benchmark directory (default: `tests/perf_v2/perf_history`)
- `--port`: Port to run the dashboard on (default: 8050)

## Dashboard Structure

### Task Tabs (Top Level)

The dashboard includes tabs for the following tasks:

- **Anomaly**: Anomaly detection tasks
- **Multi Class Cls**: Multi-class classification
- **Multi Label Cls**: Multi-label classification
- **H Label Cls**: Hierarchical label classification
- **Detection**: Object detection
- **Keypoint Detection**: Keypoint detection
- **Instance Segmentation**: Instance segmentation
- **Semantic Segmentation**: Semantic segmentation

### Dataset Tabs (Second Level)

Within each task tab, you'll find sub-tabs for different benchmark datasets that are available for that task, plus an "Average" tab that aggregates metrics across all datasets for cross-dataset regression analysis.

### Visualization Options

For each dataset, you can choose from:

1. **Accuracy Comparison**: Multiple bar charts comparing accuracy metrics across OTX versions for different model stages:

   - Torch Model Accuracy (`torch:test/{metric}_mean`)
   - Exported Model Accuracy (`export:test/{metric}_mean`)
   - Optimized Model Accuracy (`optimize:test/{metric}_mean`)

2. **Latency Comparison**: Multiple bar charts comparing latency metrics across OTX versions for different model stages:

   - Torch Model Latency (`torch:test/latency_mean`)
   - Exported Model Latency (`export:test/latency_mean`)
   - Optimized Model Latency (`optimize:test/latency_mean`)

3. **Training Time Comparison**: Bar chart comparing training time across OTX versions

   - Training End-to-End Time (`training:e2e_time_mean`)

4. **Epoch Comparison**: Bar chart comparing number of epochs across OTX versions

   - Training Epochs (`training:epoch_mean`)

5. **Iteration Time Comparison**: Bar chart comparing iteration time across OTX versions

   - Training Iteration Time (`training:train/iter_time_mean`)

6. **Accuracy vs Latency**: Scatter plot showing the relationship between accuracy and latency

## Data Structure Requirements

The dashboard expects the following directory structure:

```
benchmark_directory/
├── v2.4.2/
│   ├── semantic_segmentation/
│   │   └── SEMANTIC_SEGMENTATION-aggregated.xlsx
│   ├── detection/
│   │   └── DETECTION-aggregated.xlsx
│   └── ...
├── v2.5/
│   ├── semantic_segmentation/
│   │   └── SEMANTIC_SEGMENTATION-aggregated.xlsx
│   └── ...
└── ...
```

Each aggregated Excel file should contain sheets named after datasets, with columns including:

- `model`: Model name
- `torch:test/{metric}_mean`: Where {metric} is the task-specific accuracy metric
- `export:test/{metric}_mean`: Exported model accuracy
- `optimize:test/{metric}_mean`: Optimized model accuracy
- `torch:test/latency_mean`: Torch model latency measurements
- `export:test/latency_mean`: Exported model latency measurements
- `optimize:test/latency_mean`: Optimized model latency measurements
- `training:e2e_time_mean`: Training end-to-end time
- `training:epoch_mean`: Number of training epochs
- `training:train/iter_time_mean`: Training iteration time
- `otx_version`: OTX version information

## Troubleshooting

### No Data Available

- Ensure your benchmark directory exists and contains the expected structure
- Check that aggregated Excel files are present for your tasks/versions
- Click the "Refresh Data" button to reload data

### Missing Tasks or Datasets

- Verify that your Excel files follow the expected naming convention
- Check the console logs for any loading errors
- Ensure your task directories match the expected case-sensitive naming

### Performance Issues

- Consider filtering to fewer versions if you have many benchmark versions
- The dashboard loads all data into memory, so large datasets may require more RAM

## Development

To modify or extend the dashboard:

1. The main class is `BenchmarkDashboard` in `plot_benchmark.py`
2. Add new visualization types by creating methods like `create_metric_comparison_plot`
3. Modify the layout in `setup_layout()` method
4. Add new callbacks in `setup_callbacks()` method

For debugging, run with the `--debug` flag to enable hot reloading and detailed error messages.
