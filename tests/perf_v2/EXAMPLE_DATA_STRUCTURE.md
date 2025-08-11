# Example Data Structure for OTX Benchmark Dashboard

This document shows the expected data structure for the OTX Benchmark Dashboard to work correctly.

## Directory Structure

```
perf_history/
├── v2.4.2/
│   ├── semantic_segmentation/
│   │   └── SEMANTIC_SEGMENTATION-aggregated.xlsx
│   ├── detection/
│   │   └── DETECTION-aggregated.xlsx
│   ├── instance_segmentation/
│   │   └── INSTANCE_SEGMENTATION-aggregated.xlsx
│   └── ...
├── v2.5/
│   ├── semantic_segmentation/
│   │   └── SEMANTIC_SEGMENTATION-aggregated.xlsx
│   ├── detection/
│   │   └── DETECTION-aggregated.xlsx
│   └── ...
└── ...
```

## Excel File Structure

Each `{TASK_NAME}-aggregated.xlsx` file should contain multiple sheets, where each sheet name corresponds to a dataset name (e.g., `tiny_cell_labels`, `medium_kitti`, etc.).

### Required Columns

Each sheet should contain the following columns:

#### Essential Columns

- `model`: Model name (e.g., "litehrnet_18", "segnext_b")
- `otx_version`: OTX version (e.g., "2.4.2", "2.5.0")

#### Accuracy Metrics (task-specific)

- `torch:test/{metric}_mean`: Accuracy for torch models
- `export:test/{metric}_mean`: Accuracy for exported models
- `optimize:test/{metric}_mean`: Accuracy for optimized models

Where `{metric}` is the task-specific metric:

- Semantic Segmentation: `Dice`
- Detection: `f1-score`
- Instance Segmentation: `f1-score`
- Classification: `accuracy`
- Anomaly: `image_F1Score`
- Keypoint Detection: `PCK`

#### Latency Metrics

- `torch:test/latency_mean`: Latency for torch models
- `export:test/latency_mean`: Latency for exported models
- `optimize:test/latency_mean`: Latency for optimized models

#### Training Metrics

- `training:e2e_time_mean`: End-to-end training time
- `training:epoch_mean`: Number of training epochs
- `training:train/iter_time_mean`: Training iteration time
- `training:gpu_mem_mean`: Training GPU memory usage

### Example Sheet Content

For a semantic segmentation task with dataset `tiny_cell_labels`:

| model        | otx_version | torch:test/Dice_mean | export:test/Dice_mean | optimize:test/Dice_mean | torch:test/latency_mean | export:test/latency_mean | optimize:test/latency_mean | training:e2e_time_mean | training:epoch_mean | training:train/iter_time_mean | training:gpu_mem_mean |
| ------------ | ----------- | -------------------- | --------------------- | ----------------------- | ----------------------- | ------------------------ | -------------------------- | ---------------------- | ------------------- | ----------------------------- | --------------------- |
| litehrnet_18 | 2.4.2       | 0.850                | 0.848                 | 0.845                   | 45.2                    | 28.1                     | 15.3                       | 1850.5                 | 50                  | 0.85                          | 8.5                   |
| litehrnet_s  | 2.4.2       | 0.820                | 0.818                 | 0.815                   | 32.1                    | 20.5                     | 12.8                       | 1420.2                 | 45                  | 0.62                          | 6.2                   |
| segnext_b    | 2.4.2       | 0.875                | 0.870                 | 0.865                   | 78.5                    | 45.2                     | 25.1                       | 2150.8                 | 60                  | 1.15                          | 12.8                  |

## Task-Specific Metrics

### Semantic Segmentation

- Primary metric: `Dice`
- Example columns: `torch:test/Dice_mean`, `export:test/Dice_mean`, `optimize:test/Dice_mean`

### Detection

- Primary metric: `f1-score`
- Example columns: `torch:test/f1-score_mean`, `export:test/f1-score_mean`, `optimize:test/f1-score_mean`

### Instance Segmentation

- Primary metric: `f1-score`
- Example columns: `torch:test/f1-score_mean`, `export:test/f1-score_mean`, `optimize:test/f1-score_mean`

### Classification (Multi-class, Multi-label, H-label)

- Primary metric: `accuracy`
- Example columns: `torch:test/accuracy_mean`, `export:test/accuracy_mean`, `optimize:test/accuracy_mean`

### Anomaly Detection

- Primary metric: `image_F1Score`
- Example columns: `torch:test/image_F1Score_mean`, `export:test/image_F1Score_mean`, `optimize:test/image_F1Score_mean`

### Keypoint Detection

- Primary metric: `PCK`
- Example columns: `torch:test/PCK_mean`, `export:test/PCK_mean`, `optimize:test/PCK_mean`

## Data Generation

If you're generating this data from OTX benchmark runs, ensure that:

1. The aggregated files are properly formatted as Excel files
2. Sheet names match the dataset names defined in your task modules
3. All required columns are present with the correct naming convention
4. Missing data is handled appropriately (empty cells or NaN values)

## Testing Your Data

Use the provided test script to verify your data structure:

```bash
python test_dashboard.py
```

This will check if your data can be loaded and processed correctly by the dashboard.

## Troubleshooting

### Common Issues

1. **Missing files**: Ensure all expected aggregated files exist in the correct directories
2. **Wrong column names**: Check that metric columns follow the exact naming convention
3. **Missing sheets**: Verify that all expected dataset sheets exist in each Excel file
4. **Data types**: Ensure numeric columns contain valid numbers (not strings)
5. **Version consistency**: Make sure `otx_version` column values are consistent within each file

### Debug Tips

1. Check the console output when running the dashboard - it will show loading progress and errors
2. Use the "Refresh Data" button in the dashboard to reload data after making changes
3. The test script provides detailed information about what data was loaded successfully
