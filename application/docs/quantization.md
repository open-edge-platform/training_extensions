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

The quantization process in OTX:

1. Loads the OpenVINO IR model (XML/BIN files)
2. Creates an NNCF calibration dataset from the training data
3. Applies `nncf.quantize()` to produce a mixed-precision INT8 model
4. Saves the optimized model to the specified output directory

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
    "calibration_subset_size": 300
  }
}
```

#### Parameters

| Parameter                 | Type   | Required | Default | Description                                                      |
| ------------------------- | ------ | -------- | ------- | ---------------------------------------------------------------- |
| `model_id`                | UUID   | Yes      | -       | ID of the model revision to quantize                             |
| `device`                  | string | Yes      | -       | Device to use for calibration (e.g., 'cpu', 'xpu-0')             |
| `calibration_subset_size` | int    | No       | 300     | Maximum number of samples from training set used for calibration |

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
    }
  }
}
```

### Retrieve quantized model details

The existing model endpoints are extended to include information about quantized variants.

| Method | Path                                   | Payload | Return     | Description                                |
| ------ | -------------------------------------- | ------- | ---------- | ------------------------------------------ |
| `GET`  | `/api/projects/<id>/models/<model_id>` | -       | model info | Get model info including quantized variant |

#### Response (extended model view)

```json
{
  "id": "6b7bb928-5d6f-46ea-8fd2-5ce80dd1e12b",
  "name": "My Detection Model",
  "architecture": "object-detection-yolox-s",
  "training_info": { ... },
  "variants": {
    "openvino": { "available": true, "precision": "fp16" },
    "onnx": { "available": true, "precision": "fp16" },
    "pytorch": { "available": true, "precision": "fp32" }
  },
  "quantized": {
    "available": true,
    "precision": "int8",
    "created_at": "2026-02-17T12:00:00Z",
    "calibration_subset_size": 300,
    "evaluations": [
      {
        "subset": "testing",
        "metrics": { "mAP@0.5": 0.78, "mAP@0.5-0.95": 0.62 }
      }
    ]
  }
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
  "model_id": "6b7bb928-5d6f-46ea-8fd2-5ce80dd1e12b",
  "model_variant": "quantized"
}
```

#### Model variant options

| Variant     | Description                                       |
| ----------- | ------------------------------------------------- |
| `default`   | Use the FP16 OpenVINO IR model (current behavior) |
| `quantized` | Use the INT8 quantized OpenVINO IR model          |

The pipeline response includes the active model variant:

```json
{
  "id": "7b073838-99d3-42ff-9018-4e901eb047fc",
  "status": "idle",
  "model_id": "6b7bb928-5d6f-46ea-8fd2-5ce80dd1e12b",
  "model_variant": "quantized",
  ...
}
```

### Delete quantized model

Quantized model variants can be deleted independently of the parent model.

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
    calibration_subset_size: int = 300      # Max samples for calibration
```

### Job steps

The quantization job consists of the following steps:

1. **Validate Model** (0-5%)

   - Verify the source model exists and has completed training successfully
   - Check that OpenVINO IR files are available
   - Verify the model is not already quantized

2. **Prepare Calibration Dataset** (5-20%)

   - Load the dataset revision used for training the source model
   - Extract the training subset
   - Limit to `calibration_subset_size` samples if necessary

3. **Initialize OV Engine** (20-25%)

   - Create an `OVEngine` instance with the source model's OpenVINO IR
   - Configure the data module with the calibration dataset

4. **Run Quantization** (25-80%)

   - Call `OVEngine.optimize()` which internally:
     - Creates an NNCF calibration dataset
     - Applies `nncf.quantize()` with PTQ configuration
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
        self, engine: OVEngine, subset_size: int
    ) -> Path:
        """Execute the quantization process."""
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

The quantization metadata is stored in a new `quantized_models` table:

| Column                     | Type     | Description                                     |
| -------------------------- | -------- | ----------------------------------------------- |
| `id`                       | UUID     | Primary key                                     |
| `source_model_revision_id` | UUID     | Foreign key to `model_revisions` (parent model) |
| `precision`                | VARCHAR  | Precision type (e.g., 'int8')                   |
| `calibration_subset_size`  | INTEGER  | Number of samples used for calibration          |
| `created_at`               | DATETIME | Timestamp when quantization completed           |
| `files_deleted`            | BOOLEAN  | Whether quantized model files have been deleted |

Evaluation results for quantized models are stored in the existing `evaluations` table with a reference
to the quantized model ID.

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
