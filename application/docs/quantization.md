# Model Quantization

Post-training quantization (PTQ) is a model optimization technique that reduces the precision of model weights
and activations from floating-point (FP32/FP16) to lower bit-widths (INT8), resulting in smaller model sizes
and faster inference on compatible hardware. This document describes the architectural design for the model
quantization feature in Geti Tune.

## Overview

The quantization feature allows users to create optimized, lower-precision versions of their trained models.
Quantized models retain most of the original model's accuracy while offering significant performance benefits:

- **Reduced memory footprint**: INT8 models are typically 2-4x smaller than FP16/FP32 models
- **Faster inference**: Lower precision operations execute faster on Intel hardware with VNNI/AMX support
- **Lower power consumption**: Reduced computational requirements translate to energy savings

## Core concepts

- **Quantized model**: A model variant that has been converted from higher precision (FP16/FP32) to INT8
  using post-training quantization. Each quantized model is linked to its source (parent) model revision.

- **Calibration dataset**: A representative subset of training data used during quantization to calibrate
  the quantization parameters. The quality of the calibration dataset directly impacts the quantized model's
  accuracy.

- **Quantization configuration**: Parameters that control the quantization process, such as the calibration
  subset size, quantization preset, and advanced NNCF options.

## Integration with OTX

The quantization feature leverages the existing optimization infrastructure in OTX (OpenVINO Training Extensions).
Specifically, it uses:

- **NNCF (Neural Network Compression Framework)**: The underlying library that performs the actual quantization
- **OTX OpenVINO Engine**: The `optimize` method in `OVEngine` that orchestrates the PTQ process
- **OTX OpenVINO Model**: The `optimize` method in `OVModel` that applies NNCF quantization to OpenVINO IR models

### NNCF Quantization Methods

NNCF provides two primary quantization functions:

#### `nncf.quantize()`

Standard post-training quantization that applies INT8 quantization to weights and activations.

```python
quantized_model = nncf.quantize(
    model,
    calibration_dataset,
    subset_size=100,
    preset=nncf.QuantizationPreset.MIXED,
)
```

**Parameters:**

- `model`: OpenVINO IR model to quantize
- `calibration_dataset`: Representative dataset for calibration
- `subset_size`: Number of samples to use for calibration (default: 100)
- `preset`: Quantization preset (`PERFORMANCE` or `MIXED`)

#### `nncf.quantize_with_accuracy_control()`

Post-training quantization with automatic accuracy recovery. This method monitors accuracy during quantization
and selectively reverts layers to higher precision if accuracy drops below a threshold.

```python
quantized_model = nncf.quantize_with_accuracy_control(
    model,
    calibration_dataset,
    validation_dataset,
    validation_fn,
    max_drop=0.01,      # Maximum allowed accuracy drop (1%)
    drop_type=nncf.DropType.ABSOLUTE,
    subset_size=100,
)
```

**Additional Parameters:**

- `validation_dataset`: Dataset for accuracy validation
- `validation_fn`: Function that computes accuracy metric (returns `(metric_value, per_sample_metrics)`)
- `max_drop`: Maximum acceptable accuracy drop (e.g., 0.01 for 1%)
- `drop_type`: How to measure accuracy drop (`ABSOLUTE` or `RELATIVE`)

**Trade-offs:**

- `nncf.quantize()`: Faster, but may have larger accuracy drop
- `nncf.quantize_with_accuracy_control()`: Very small overhead, but maintains accuracy within specified threshold

#### Method selection logic

The quantization method is selected based on the presence of the optional `max_drop` parameter in the API request:

- **`max_drop` not provided** → `nncf.quantize()` is used (standard PTQ, faster)
- **`max_drop` provided** → `nncf.quantize_with_accuracy_control()` is used (accuracy-aware PTQ, slower but preserves accuracy within the specified threshold)

This design keeps the API simple: users who want fast quantization simply omit `max_drop`, while users who
need accuracy guarantees provide a threshold value (e.g., `max_drop=0.01` for a maximum 1% accuracy drop).

The quantization process in OTX:

1. Loads the OpenVINO IR model (XML/BIN files)
2. Creates an NNCF calibration dataset from the training data
3. If `max_drop` is provided: applies `nncf.quantize_with_accuracy_control()` with the specified threshold
4. If `max_drop` is not provided: applies `nncf.quantize()` for standard PTQ
5. Saves the optimized model to the specified output directory

## Implementation approach

There are three possible approaches to implementing the quantization feature:

### Option 1: At the end of the training job

Quantization is performed automatically after training and model export.

| Pros                                                 | Cons                                      |
| ---------------------------------------------------- | ----------------------------------------- |
| Easiest to implement                                 | Increased duration of the training job    |
| Model and calibration data are immediately available | Increased size of training artifacts      |
| No additional user interaction required              | Quantization performed even if not needed |
| Single job to track                                  | Cannot re-quantize without re-training    |

### Option 2: As a separate job

Quantization is submitted as a separate, on-demand job.

| Pros                                       | Cons                                             |
| ------------------------------------------ | ------------------------------------------------ |
| Optimization performed only when requested | More development effort to implement another job |
| User has control over when to quantize     | Need to reload calibration data                  |
| Can re-quantize with different parameters  | Additional job management complexity             |
| Clear separation of concerns               |                                                  |

### Option 3: Direct API call (synchronous)

Quantization is performed synchronously during an API request.

| Pros                                       | Cons                                          |
| ------------------------------------------ | --------------------------------------------- |
| Optimization performed only when requested | May be unfeasible if operation takes too long |
| Simpler implementation than a job          | No progress reporting                         |
| Immediate result                           | Risk of HTTP timeouts                         |
|                                            | Blocks the client during quantization         |

### Benchmarking

To determine the best approach, a benchmark script was used to measure quantization time across different
model architectures, calibration dataset sizes, and quantization methods:

The benchmark tests both NNCF quantization methods:

- `nncf.quantize()`: Standard PTQ
- `nncf.quantize_with_accuracy_control()`: PTQ with accuracy recovery

#### Benchmark results

The following benchmark was performed on a CPU-only system with the available trained models:

| Architecture                         | Task                  | Method      | Input Size | Samples | Time (s) | Original (MB) | Quantized (MB) | Reduction | Status  |
| ------------------------------------ | --------------------- | ----------- | ---------- | ------- | -------- | ------------- | -------------- | --------- | ------- |
| instance-segmentation-rtmdet-tiny    | INSTANCE_SEGMENTATION | quantize    | 640x640    | 50      | 26.95    | 12.98         | 7.26           | 44.0%     | success |
| instance-segmentation-rtmdet-tiny    | INSTANCE_SEGMENTATION | quantize    | 640x640    | 100     | 52.21    | 12.98         | 7.26           | 44.0%     | success |
| instance-segmentation-rtmdet-tiny    | INSTANCE_SEGMENTATION | quantize    | 640x640    | 200     | 116.46   | 12.98         | 7.26           | 44.0%     | success |
| instance-segmentation-rtmdet-tiny    | INSTANCE_SEGMENTATION | quantize    | 640x640    | 300     | 209.08   | 12.98         | 7.26           | 44.0%     | success |
| instance-segmentation-rtmdet-tiny    | INSTANCE_SEGMENTATION | acc_control | 640x640    | 50      | 34.10    | 12.98         | 7.27           | 44.0%     | success |
| instance-segmentation-rtmdet-tiny    | INSTANCE_SEGMENTATION | acc_control | 640x640    | 100     | 69.66    | 12.98         | 7.27           | 44.0%     | success |
| instance-segmentation-rtmdet-tiny    | INSTANCE_SEGMENTATION | acc_control | 640x640    | 200     | 113.25   | 12.98         | 7.27           | 44.0%     | success |
| instance-segmentation-rtmdet-tiny    | INSTANCE_SEGMENTATION | acc_control | 640x640    | 300     | 180.40   | 12.98         | 7.27           | 44.0%     | success |
| image-classification-efficientnet-b0 | MULTI_CLASS_CLS       | quantize    | 224x224    | 50      | 5.99     | 7.94          | 5.18           | 34.8%     | success |
| image-classification-efficientnet-b0 | MULTI_CLASS_CLS       | quantize    | 224x224    | 100     | 8.45     | 7.94          | 5.18           | 34.8%     | success |
| image-classification-efficientnet-b0 | MULTI_CLASS_CLS       | quantize    | 224x224    | 200     | 13.51    | 7.94          | 5.18           | 34.8%     | success |
| image-classification-efficientnet-b0 | MULTI_CLASS_CLS       | quantize    | 224x224    | 300     | 19.38    | 7.94          | 5.18           | 34.8%     | success |
| image-classification-efficientnet-b0 | MULTI_CLASS_CLS       | acc_control | 224x224    | 50      | 7.03     | 7.94          | 5.18           | 34.8%     | success |
| image-classification-efficientnet-b0 | MULTI_CLASS_CLS       | acc_control | 224x224    | 100     | 9.52     | 7.94          | 5.18           | 34.8%     | success |
| image-classification-efficientnet-b0 | MULTI_CLASS_CLS       | acc_control | 224x224    | 200     | 14.02    | 7.94          | 5.18           | 34.8%     | success |
| image-classification-efficientnet-b0 | MULTI_CLASS_CLS       | acc_control | 224x224    | 300     | 17.76    | 7.94          | 5.18           | 34.8%     | success |
| object-detection-atss-mobilenet-v2   | DETECTION             | quantize    | 800x992    | 50      | 32.79    | 5.29          | 3.64           | 31.2%     | success |
| object-detection-atss-mobilenet-v2   | DETECTION             | quantize    | 800x992    | 100     | 62.47    | 5.29          | 3.64           | 31.2%     | success |
| object-detection-atss-mobilenet-v2   | DETECTION             | quantize    | 800x992    | 200     | 123.34   | 5.29          | 3.64           | 31.2%     | success |
| object-detection-atss-mobilenet-v2   | DETECTION             | quantize    | 800x992    | 300     | 191.23   | 5.29          | 3.64           | 31.2%     | success |
| object-detection-atss-mobilenet-v2   | DETECTION             | acc_control | 800x992    | 50      | 31.61    | 5.29          | 3.64           | 31.2%     | success |
| object-detection-atss-mobilenet-v2   | DETECTION             | acc_control | 800x992    | 100     | 56.31    | 5.29          | 3.64           | 31.2%     | success |
| object-detection-atss-mobilenet-v2   | DETECTION             | acc_control | 800x992    | 200     | 117.47   | 5.29          | 3.64           | 31.2%     | success |
| object-detection-atss-mobilenet-v2   | DETECTION             | acc_control | 800x992    | 300     | 207.48   | 5.29          | 3.64           | 31.2%     | success |

**Summary statistics:**

#### Method: `nncf.quantize`

| Calibration Samples | Min Time (s) | Max Time (s) | Avg Time (s) |
| ------------------- | ------------ | ------------ | ------------ |
| 50                  | 5.99         | 32.79        | 21.91        |
| 100                 | 8.45         | 62.47        | 41.04        |
| 200                 | 13.51        | 123.34       | 84.44        |
| 300                 | 19.38        | 209.08       | 139.89       |

#### Method: `nncf.quantize_with_accuracy_control`

| Calibration Samples | Min Time (s) | Max Time (s) | Avg Time (s) |
| ------------------- | ------------ | ------------ | ------------ |
| 50                  | 7.03         | 34.10        | 24.25        |
| 100                 | 9.52         | 69.66        | 45.16        |
| 200                 | 14.02        | 117.47       | 81.58        |
| 300                 | 17.76        | 207.48       | 135.22       |

### Method Comparison

- **Average time for `nncf.quantize`**: 71.82s
- **Average time for `nncf.quantize_with_accuracy_control`**: 71.55s
- **Accuracy control overhead**: -0.4%

### Model Size Reduction

- **Average reduction**: 36.7%
- **Min reduction**: 31.2%
- **Max reduction**: 44.0%

The benchmark results help inform the implementation decision based on actual quantization times for
the supported model architectures.

### Recommended approach

Based on the benchmark results showing quantization times ranging from **6 to 209 seconds**,
**Option 2 (separate job)** is recommended for the following reasons:

1. **User control**: Users can decide when to optimize their models
2. **Progress tracking**: Long-running quantization (up to ~1 minute) can be monitored like training jobs
3. **Resource management**: Quantization can be queued and scheduled appropriately
4. **Flexibility**: Re-quantization with different parameters is possible without re-training
5. **Clean architecture**: Clear separation between training and optimization workflows
6. **No HTTP timeouts**: A 60+ second operation would risk HTTP request timeouts if done synchronously

**Note**: A direct API call (Option 3) could still be viable for smaller models (classification takes ~6-10s),
but a job-based approach provides a consistent UX across all model types.

## Model lifecycle with quantization

The quantization feature extends the existing [model lifecycle](models.md#model-lifecycle) with an optional
optimization step:

```
Training → Export (FP16) → [Quantization (INT8)] → Evaluation → Deployment
                                  ↑
                                  └── Optional step
```

A model revision can have zero or one quantized variant. The quantized variant:

- Is stored alongside the original model files
- Has its own evaluation results
- Can be independently enabled in the inference pipeline
- Shares the same metadata (architecture, training configuration) as the parent model

## API

### Submit quantization job

Quantization is submitted as a job, similar to training jobs. This allows the operation to run in the background
while providing progress updates and logs.

| Method | Path        | Payload               | Return | Description                         |
| ------ | ----------- | --------------------- | ------ | ----------------------------------- |
| `POST` | `/api/jobs` | quantization job spec | job id | Submit a new model quantization job |

#### Request body

```json
{
  "job_type": "quantize",
  "project_id": "7b073838-99d3-42ff-9018-4e901eb047fc",
  "parameters": {
    "model_id": "6b7bb928-5d6f-46ea-8fd2-5ce80dd1e12b",
    "device": "cpu",
    "max_calibration_subset_size": 100,
    "max_drop": 0.01
  }
}
```

#### Parameters

| Parameter                     | Type   | Required | Default | Description                                                                                                                  |
| ----------------------------- | ------ | -------- | ------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `model_id`                    | UUID   | Yes      | -       | ID of the model revision to quantize                                                                                         |
| `device`                      | string | Yes      | -       | Device to use for calibration (e.g., 'cpu', 'xpu-0')                                                                         |
| `max_calibration_subset_size` | int    | No       | 100     | Maximum number of samples from training set used for calibration                                                             |
| `max_drop`                    | float  | No       | -       | Maximum allowed accuracy drop. If provided, uses `nncf.quantize_with_accuracy_control()`; if omitted, uses `nncf.quantize()` |

#### Response

```json
{
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "job_type": "quantize",
  "status": "pending",
  "project_id": "7b073838-99d3-42ff-9018-4e901eb047fc",
  "created_at": "2026-02-17T10:30:00Z",
  "metadata": {
    "project": {
      "id": "7b073838-99d3-42ff-9018-4e901eb047fc"
    },
    "model": {
      "id": "6b7bb928-5d6f-46ea-8fd2-5ce80dd1e12b"
    },
    "device": "cpu",
    "max_calibration_subset_size": 100,
    "max_drop": 0.01
  }
}
```

### Retrieve quantized model details

The existing model endpoints are extended to include information about quantized variants. The `quantization_info`
field is only returned for quantized models (null for non-quantized models). The id for model variants is also introduced.

| Method | Path                                   | Payload | Return     | Description                                |
| ------ | -------------------------------------- | ------- | ---------- | ------------------------------------------ |
| `GET`  | `/api/projects/<id>/models/<model_id>` | -       | model info | Get model info including quantized variant |

#### Response (extended model view)

```json
{
    "id": "6b7bb928-5d6f-46ea-8fd2-5ce80dd1e12b",
    "name": "My Detection Model",
    "architecture": "object-detection-yolox-s",
    "training_info": {...},
    "variants": [
      {
        "id": "4c576bce-5e97-408d-a0ea-cc3801e4c453",
        "format": "openvino",
        "precision": "fp16",
        "weights_size": 123456
      },
      {
        "id": "6b7bb928-5d6f-46ea-8fd2-5ce80dd1e12b",
        "format": "openvino",
        "precision": "int8",
        "weights_size": 12345,
        "evaluations": [
          {
            "subset": "testing",
            "metrics": { "mAP@0.5": 0.78, "mAP@0.5-0.95": 0.62 }
          }
        ],
        "quantization_info": {
            "type": "ptq",
            "max_drop": 0.01,
            "max_calibration_dataset_size": 100
        }
      },
      {
        "id": "d01945ae-1578-41f9-a2b3-11865032981c",
        "format": "onnx",
        "precision": "fp16",
        "weights_size": 123456
      },
      {
        "id": "0e432cc0-d30a-4e76-9da9-896147a271c0",
        "format": "pytorch",
        "precision": "fp32",
        "weights_size": 123456
      }
    ]
}
```

### Download quantized model binary

The existing model binary download endpoint supports downloading the quantized model variant.

| Method | Path                                          | Query params       | Return | Description                    |
| ------ | --------------------------------------------- | ------------------ | ------ | ------------------------------ |
| `GET`  | `/api/projects/<id>/models/<model_id>/binary` | `format=quantized` | zip    | Download quantized model files |

The `format` query parameter now accepts an additional value:

- `openvino` (default): FP16 OpenVINO IR model
- `onnx`: FP16 ONNX model
- `pytorch`: FP32 PyTorch checkpoint
- `quantized`: INT8 quantized OpenVINO IR model

### Enable quantized model in pipeline

The existing pipeline update endpoint is used to enable the quantized model variant for inference.

| Method  | Path                          | Payload         | Return        | Description                   |
| ------- | ----------------------------- | --------------- | ------------- | ----------------------------- |
| `PATCH` | `/api/projects/<id>/pipeline` | pipeline config | pipeline info | Update pipeline configuration |

#### Request body to enable quantized model

```json
{
  "model_id": "6b7bb928-5d6f-46ea-8fd2-5ce80dd1e12b"
}
```

#### Model variant options

The choice of the model variant is determined by its model id. See the `model_variants` DB schema later for more details.

The pipeline response includes the active model variant:

```json
{
  "id": "7b073838-99d3-42ff-9018-4e901eb047fc",
  "status": "idle",
  "model_id": "6b7bb928-5d6f-46ea-8fd2-5ce80dd1e12b",
  "precision": "int8",
  ...
}
```

### Delete quantized model

Quantized model variants can be deleted independently of the parent model. See the folder structure in the Storage section for more details.

| Method   | Path                                   | Query params        | Return | Description                       |
| -------- | -------------------------------------- | ------------------- | ------ | --------------------------------- |
| `DELETE` | `/api/projects/<id>/models/<model_id>` | `variant=quantized` | -      | Delete only the quantized variant |

## Quantization job structure

The quantization job follows the same pattern as the training job, implementing the `Execution` interface
with discrete steps marked by the `@step` decorator.

### Job parameters

```python
class QuantizationJobParams(JobParams):
    job_id: UUID
    project_id: UUID
    source_model_id: UUID                   # Source model revision to quantize
    device: DeviceInfo
    calibration_subset_size: int = 100      # Max samples for calibration
    max_drop: float | None = None           # If provided, uses nncf.quantize_with_accuracy_control()
                                            # If None, uses nncf.quantize()
```

### Job steps

The quantization job consists of the following steps:

1. **Validate Model** (0-5%)

   - Verify the source model exists and has completed training successfully
   - Check that OpenVINO IR files are available
   - Verify the model is not already quantized

2. **Prepare Calibration Dataset** (5-20%)

   - Load the dataset revision used for training the source model
   - Extract the validation subset
   - Limit to `max_calibration_subset_size` samples if necessary

3. **Initialize OV Engine** (20-25%)

   - Create an `OVEngine` instance with the source model's OpenVINO IR
   - Configure the data module with the calibration dataset

4. **Run Quantization** (25-80%)

   - Call `OVEngine.optimize()` which internally:
     - Creates an NNCF calibration dataset
     - If `max_drop` is provided:
       - Applies `nncf.quantize_with_accuracy_control()` with validation and the specified `max_drop` threshold
     - If `max_drop` is not provided:
       - Applies `nncf.quantize()` with standard PTQ configuration
     - Saves the optimized INT8 model

5. **Evaluate Quantized Model** (80-95%)

   - Evaluate the quantized model on the testing subset
   - Store evaluation metrics for comparison with the FP16 model

6. **Store Artifacts** (95-100%)
   - Move quantized model files to the model directory
   - Update database with quantization metadata
   - Clean up temporary files

### Job execution class

```python
class OTXQuantizer(Execution):
    """OTX-specific quantization implementation."""

    @step("Validate Model", 5)
    def validate_model(self, params: QuantizationJobParams) -> ModelRevision:
        """Verify the source model is valid for quantization."""
        ...

    @step("Prepare Calibration Dataset", 20)
    def prepare_calibration_dataset(
        self, model: ModelRevision, subset_size: int
    ) -> OTXDataModule:
        """Load and prepare the calibration dataset."""
        ...

    @step("Initialize OV Engine", 25)
    def initialize_engine(
        self, model: ModelRevision, datamodule: OTXDataModule
    ) -> OVEngine:
        """Create the OVEngine for quantization."""
        ...

    @step("Run Quantization", 80)
    def run_quantization(
        self,
        engine: OVEngine,
        subset_size: int,
        max_drop: float | None,
    ) -> Path:
        """Execute the quantization process.

        If max_drop is provided, uses nncf.quantize_with_accuracy_control()
        to ensure accuracy stays within the specified threshold.
        If max_drop is None, uses nncf.quantize() for standard PTQ.
        """
        ...

    @step("Evaluate Quantized Model", 95)
    def evaluate_quantized_model(
        self, engine: OVEngine, quantized_model_path: Path
    ) -> dict:
        """Evaluate the quantized model."""
        ...

    @step("Store Artifacts", 100)
    def store_artifacts(
        self, quantized_model_path: Path, model_id: UUID
    ) -> None:
        """Store quantized model files and update database."""
        ...
```

## Storage

### Database

The quantization metadata is stored in a new `model_variants` table:

| Column                     | Type    | Description                                               |
| -------------------------- | ------- | --------------------------------------------------------- |
| `id`                       | UUID    | Primary key                                               |
| `source_model_revision_id` | UUID    | Foreign key to `model_revisions` (parent model)           |
| `precision`                | VARCHAR | Precision type (e.g., 'int8')                             |
| `quantization_info`        | JSON    | Info such as `max_drop` and `max_calibration_subset_size` |
| `files_deleted`            | BOOLEAN | Whether quantized model files have been deleted           |

Quantization info is `None` for model variants which are not quantized.
Evaluation results for quantized models are stored in the existing `evaluations` table with a reference to the quantized model ID.

### Filesystem

Quantized model files are stored alongside the parent model's files:

```
BASE_DATA_DIR/
├─ projects/
│  ├─ <project_id>/
│  │  ├─ models/
│  │  │  ├─ <model_id>/
│  │  │  │  ├─ model.ckpt              # Original PyTorch checkpoint
│  │  │  │  ├─ model.xml               # Original OpenVINO IR (FP16)
│  │  │  │  ├─ model.bin               # Original OpenVINO weights (FP16)
│  │  │  │  ├─ model.onnx              # Original ONNX model (FP16)
│  │  │  │  ├─ quantized/              # Quantized model directory
│  │  │  │  │  ├─ model.xml            # Quantized OpenVINO IR (INT8)
│  │  │  │  │  ├─ model.bin            # Quantized OpenVINO weights (INT8)
│  │  │  │  │  ├─ quantization.log     # Quantization process log
```

## Error handling

The quantization job handles the following error conditions:

| Error condition              | Behavior                                |
| ---------------------------- | --------------------------------------- |
| Model not found              | Job fails with 404 error                |
| Model training not completed | Job fails with 409 error                |
| Model already quantized      | Job fails with 409 error                |
| OpenVINO IR files missing    | Job fails with 404 error                |
| Dataset revision not found   | Job fails with 404 error                |
| Calibration dataset empty    | Job fails with 400 error                |
| NNCF quantization fails      | Job fails with 500 error, logs captured |
| Insufficient disk space      | Job fails with 507 error                |

## Future considerations

### Additional quantization options

The following advanced options could be exposed in future versions:

- **Quantization preset**: `performance` vs `accuracy` mode
- **Ignored layers**: Specific layers to exclude from quantization
- **Custom calibration configuration**: Advanced NNCF parameters
- **Mixed precision**: Configure different precision for different layer types

### Quantization-aware training (QAT)

While this design focuses on post-training quantization, future versions could support quantization-aware
training, which simulates quantization during training for improved accuracy at the cost of longer training time.

### Weight compression

In addition to INT8 quantization, future versions could support weight-only compression techniques like
INT4/INT8 weight quantization with FP16 activations, which can provide further model size reduction with
minimal accuracy impact.
