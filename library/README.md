<div align="center">

<img src="../assets/geti-tune-header.png" alt="Geti Library">

**A low-code transfer learning framework for training, evaluating, optimizing, and deploying computer vision models**

---

[Key Features](#key-features) •
[Supported Tasks & Models](#supported-tasks--models) •
[Installation](#installation) •
[Quick Start](#quick-start) •
[Docs](https://open-edge-platform.github.io/training_extensions/latest/index.html) •
[License](#license)

[![PyPI](https://img.shields.io/pypi/v/getitune)](https://pypi.org/project/getitune)

<!-- markdownlint-disable MD042 -->

[![python](https://img.shields.io/badge/python-3.11%E2%80%933.14-green)]()
[![pytorch](https://img.shields.io/badge/pytorch-2.10-orange)]()
[![openvino](https://img.shields.io/badge/openvino-2026.1-purple)]()
[![numpy](https://img.shields.io/badge/numpy-%E2%89%A52.0-blue)]()

<!-- markdownlint-enable  MD042 -->

[![Codecov](https://codecov.io/gh/open-edge-platform/training_extensions/branch/develop/graph/badge.svg?token=9HVFNMPFGD)](https://codecov.io/gh/open-edge-platform/training_extensions)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/open-edge-platform/training_extensions/badge)](https://securityscorecards.dev/viewer/?uri=github.com/open-edge-platform/training_extensions)
[![Pre-Merge Test](https://github.com/open-edge-platform/training_extensions/actions/workflows/pre_merge.yaml/badge.svg)](https://github.com/open-edge-platform/training_extensions/actions/workflows/pre_merge.yaml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://static.pepy.tech/personalized-badge/getitune?period=total&units=international_system&left_color=grey&right_color=green&left_text=PyPI%20Downloads)](https://pepy.tech/project/getitune)

---

</div>

# Geti Library - getitune

The Geti™ library (`getitune`) is a low-code transfer learning framework for Computer Vision.
Its API and CLI let you train, evaluate, optimize, and deploy models quickly, even without deep expertise in deep learning.
It supports diverse combinations of model architectures, learning methods, and task types built on [PyTorch](https://pytorch.org) and the [OpenVINO™ toolkit](https://software.intel.com/en-us/openvino-toolkit).

Each supported task ships with curated "recipes": YAML files that bundle the model, data pipeline, and training configuration into a single one-stop entry point. Recipes are validated on standard datasets so you get a strong baseline out of the box.

### Key Features

- **Multi-task support**: classification, object detection, rotated detection, instance segmentation, semantic segmentation, and keypoint detection, see the [full model list below](#supported-tasks--models).
- **Intel Hardware acceleration**: Native Intel GPU (XPU) support.
- **Tiling** for large images across detection and segmentation tasks.
- **Multiple backends**: train with PyTorch Lightning, export and run inference with ONNX and OpenVINO™.
- [Datumaro](https://github.com/open-edge-platform/datumaro/tree/develop/src/datumaro/experimental) **data frontend** with automatic format detection (COCO, YOLO, VOC, native).
- **Distributed training** across multiple GPUs.
- **Mixed-precision training** to reduce memory and increase batch size.
- **Class-incremental learning** to extend an existing model with new classes.
- **Deployment** to OpenVINO™ IR and ONNX formats, with inference via [OpenVINO™ ModelAPI](https://github.com/open-edge-platform/model_api).

---

## Supported Tasks & Models

All recipes live under `src/getitune/recipe/<task>/`. Pass any of these YAMLs directly to the API as `model=...`. Recipes whose name ends in `_tile` enable the tiling pipeline for large images.

| Task                                                      | Recipe directory                                                                                                                                                                                               | Example recipes                                                                                                                                            |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Classification (multi-class / multi-label / hierarchical) | [multi_class_cls](src/getitune/recipe/classification/multi_class_cls/), [multi_label_cls](src/getitune/recipe/classification/multi_label_cls/), [h_label_cls](src/getitune/recipe/classification/h_label_cls/) | `dino_v2`, `vit_tiny`, `efficientnet_b0`, `efficientnet_b3`, `efficientnet_v2`, `mobilenet_v3_large`                                                       |
| Object detection                                          | [detection](src/getitune/recipe/detection/)                                                                                                                                                                    | `atss_mobilenetv2`, `ssd_mobilenetv2`, `yolox_{tiny,s,l,x}`, `rtdetr_50`, `dfine_x`, `deim_dfine_{l,m,x}`, `deimv2_{s,m,l}`, `rfdetr_{small,medium,large}` |
| Rotated detection                                         | [rotated_detection](src/getitune/recipe/rotated_detection/)                                                                                                                                                    | `maskrcnn_r50`, `maskrcnn_efficientnetb2b` (with `_tile` variants)                                                                                         |
| Instance segmentation                                     | [instance_segmentation](src/getitune/recipe/instance_segmentation/)                                                                                                                                            | `maskrcnn_{r50,swint,efficientnetb2b}`, `rtmdet_inst_tiny`, `rfdetr_seg_{small,medium,large,xlarge}`                                                       |
| Semantic segmentation                                     | [semantic_segmentation](src/getitune/recipe/semantic_segmentation/)                                                                                                                                            | `dino_v2`, `litehrnet_{s,18,x}`, `segnext_{t,s,b}` (with `_tile` variants)                                                                                 |
| Keypoint detection                                        | [keypoint_detection](src/getitune/recipe/keypoint_detection/)                                                                                                                                                  | `rtmpose_tiny`                                                                                                                                             |

Each task directory also ships an `openvino_model.yaml` recipe for running and optimizing pre-exported OpenVINO IR models via `OVEngine`.

---

## Installation

Requirements: **Python 3.11–3.14**, **PyTorch 2.10**, **OpenVINO™ 2026.1**, **NumPy ≥ 2.0**.

`getitune` is published on [PyPI](https://pypi.org/project/getitune/).

### Quick Install

```bash
# CPU-only (default, works on all platforms)
uv pip install getitune

# Or with pip
pip install getitune
```

### Advanced Installation: Specify Hardware Backend

`getitune` ships three mutually exclusive extras that select the right PyTorch wheel for your hardware:

| Extra    | PyTorch wheel                                  | Use when                             |
| -------- | ---------------------------------------------- | ------------------------------------ |
| `[cuda]` | `torch==2.10.0+cu128`                          | NVIDIA GPUs with CUDA 12.8 drivers.  |
| `[xpu]`  | `torch==2.10.0+xpu` + `triton-xpu`             | Intel discrete or integrated GPUs.   |
| `[cpu]`  | `torch==2.10.0+cpu` (Linux/Windows) or default | No GPU, or running on Apple silicon. |

Since PyTorch distributes GPU wheels separately, you must include the PyTorch index:

```bash
# NVIDIA GPU (CUDA 12.8)
uv pip install "getitune[cuda]" \
  --extra-index-url https://download.pytorch.org/whl/cu128

# Intel GPU (XPU)
uv pip install "getitune[xpu]" \
  --extra-index-url https://download.pytorch.org/whl/xpu

# CPU-only (no extra index needed)
uv pip install "getitune[cpu]"
```

> **macOS note**: PyTorch's `+cpu` wheel is only published for Linux and Windows. The `[cpu]` extra resolves this automatically and installs the default `torch==2.10.0` wheel on macOS.

### Install from Source

```bash
git clone https://github.com/open-edge-platform/training_extensions.git
cd training_extensions/library

# Recommended: use uv to honor the lockfile
uv sync --extra xpu          # or --extra cpu / --extra cuda
```

---

## Quick Start

### Training and Exporting

Getitune supports an API-based training approach. To run training with data config and lightning backend:

```python
from getitune.backend.lightning.engine import LightningEngine

# Initialize and train using the bundled test dataset
engine = LightningEngine(
    model="src/getitune/recipe/classification/multi_class_cls/efficientnet_b0.yaml", # path to recipe YAML / model class instance / model name (e.g., "efficientnet_b0")
    data_root="tests/assets/classification_cifar10", # path to dataset (any supported format, e.g., COCO, VOC, YOLO, Datumaro)
)

```

It is also possible to create an engine with model name and task type. Getitune will look for a matching recipe in the bundled recipes and use it to set up the training configuration:

```python
engine = LightningEngine(
    model="efficientnet_b0", # model name (e.g., "efficientnet_b0")
    task="MULTI_CLASS_CLS", # task type (e.g., "MULTI_CLASS_CLS")
    data_root="tests/assets/classification_cifar10", # path to dataset (any supported format, e.g., COCO, VOC, YOLO, Datumaro)
)
```
Run training, test, predict and export the model to OpenVINO IR format:
```python
if __name__ == "__main__": # to avoid issues with multiprocessing
  engine.train()
  metrics = engine.test() # validate on test set
  predictions = engine.predict() # predict on test set
  exported_ov_path = engine.export()  # writes OpenVINO IR
```
Geti Library also supports export to ONNX format, which can be enabled by passing export format when calling `export()`. To specify export precision, use the `export_precision` argument. By default, models are exported in FP32 precision.

```python
from getitune.types.export import ExportFormat, ExportPrecision

exported_onnx_path = engine.export(export_format=ExportFormat.ONNX, export_precision=ExportPrecision.FP16)  # writes ONNX model in FP16 precision
```

### Inference and Optimization

Getitune provides inference via PyTorch and OpenVINO / ONNX backends:

```python
from getitune.backend.openvino.engine import OVEngine

# PyTorch inference
engine = OVEngine(data="/path/to/dataset", model=exported_ov_path) # can also pass onnx model here
metrics = engine.test() # test OpenVINO model accuracy
predictions = engine.predict() # predict on test set
```
OpenVINO backend provides optimization capabilities like post-training quantization via [NNCF](https://github.com/openvinotoolkit/nncf). Post-training quantization is supported only for OpenVINO IR models.

```python
ov_engine = OVEngine(data="/path/to/dataset", model="path/to/exported_model.xml")
ov_engine.optimize()
ov_engine.test() # test optimized model accuracy
predictions = ov_engine.predict() # predict with optimized model
```


---

### Dataset Support

When you pass a path to `data=`, getitune uses [Datumaro](https://github.com/open-edge-platform/datumaro/tree/develop/src/datumaro/experimental) to auto-detect the dataset format. Supported formats:

| Format                | Detection method                                        |
| --------------------- | ------------------------------------------------------- |
| **COCO**              | `annotations/` directory with COCO JSON files           |
| **YOLO**              | `data.yaml` file (Ultralytics layout)                   |
| **Pascal VOC**        | `JPEGImages/`, `Annotations/`, `ImageSets/` directories |
| **Datumaro (native)** | `metadata.json` + `data.parquet` at root                |

Zip archives are also accepted, Datumaro extracts them on import.

```python
# Works the same regardless of format, just point to the dataset root
engine = create_engine(
    data="/path/to/coco_or_yolo_or_voc_dataset",
    model=model,  # a model class instance, or an exported OpenVINO/ONNX model path
)
engine.train()
```

---

### Advanced Usage

<details>
<summary><strong>Override training hyperparameters</strong></summary>

You can override engine-level training parameters directly:

```python
engine.train(
    max_epochs=50,
    seed=42,
    deterministic=True,
    precision="16-mixed",          # mixed-precision training
    gradient_clip_val=1.0,
    check_val_every_n_epoch=5,
)
```

For model-level hyperparameters like learning rate and optimizer, you can pass them when instantiating a model class directly:

```python
from torch.optim import AdamW
from getitune.models import EfficientNet

model = EfficientNet(
    label_info=datamodule.label_info, # or simply num_classes, e.g., int value
    model_name="efficientnet_b0",
    optimizer=lambda params: AdamW(params, lr=0.001, weight_decay=0.01),
)

engine = LightningEngine(data=datamodule, model=model)
engine.train(max_epochs=100)
```

Alternatively, you can set these in a custom recipe YAML (copy and modify an existing one):

```yaml
# my_recipe.yaml — custom learning rate and optimizer
task: MULTI_CLASS_CLS
model:
  class_path: getitune.backend.lightning.models.classification.multiclass_models.efficientnet.EfficientNetMulticlassCls
  init_args:
    label_info: 1000
    model_name: efficientnet_b0

    optimizer:
      class_path: torch.optim.AdamW
      init_args:
        lr: 0.001
        weight_decay: 0.01

    scheduler:
      class_path: getitune.backend.lightning.schedulers.LinearWarmupSchedulerCallable
      init_args:
        num_warmup_steps: 5
        main_scheduler_callable:
          class_path: lightning.pytorch.cli.ReduceLROnPlateau
          init_args:
            mode: max
            factor: 0.5
            patience: 3
            monitor: val/accuracy

data: src/getitune/recipe/_base_/data/classification.yaml

overrides:
  max_epochs: 100
```
</details>

<details>
<summary><strong>Override augmentations and datamodule</strong></summary>

For augmentations, override the data config. Augmentations run on CPU (`augmentations_cpu`) and GPU (`augmentations_gpu`) separately:

```yaml
# my_data.yaml — stronger augmentations
task: MULTI_CLASS_CLS
input_size: [224, 224]
train_subset:
  subset_name: train
  batch_size: 32
  num_workers: 8
  augmentations_cpu:
    - class_path: torchvision.transforms.v2.RandomResizedCrop
      init_args:
        size: [224, 224]
        scale: [0.08, 1.0]
  augmentations_gpu:
    - class_path: kornia.augmentation.RandomHorizontalFlip
      init_args:
        p: 0.5
    - class_path: kornia.augmentation.ColorJiggle
      init_args:
        brightness: 0.4
        contrast: 0.4
        saturation: 0.4
        hue: 0.1
        p: 0.8
    - class_path: kornia.augmentation.RandomGaussianBlur
      init_args:
        kernel_size: [3, 3]
        sigma: [0.1, 2.0]
        p: 0.3
    - class_path: kornia.augmentation.Normalize
      init_args:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
```

Augmentations can be overridden using API as well when creating a `DataModule` instance and passing it to the engine:

```python
from getitune.data.datamodule import DataModule
from getitune.config.data import SubsetConfig
import kornia.augmentation as K
from torchvision.transforms import v2

datamodule = DataModule(
    task = "MULTI_CLASS_CLS",
    data_root = "library/tests/assets/classification_cifar10",
    train_subset = SubsetConfig(
        augmentations_cpu=[
            v2.Resize((256, 256)),
        ],
        augmentations_gpu=[
            K.RandomErasing(p=0.5),
            K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ],
        batch_size=32,
        num_workers=8
    ),
    val_subset = SubsetConfig(
        subset_name="val",
        augmentations_cpu=[
            v2.Resize((256, 256)),
        ],
        batch_size=32,
        num_workers=8
    ),
    test_subset = SubsetConfig(
        subset_name="test",
        augmentations_cpu=[
            v2.Resize((256, 256)),
        ],
        batch_size=32,
        num_workers=8
    ),
)

model = EfficientNet(label_info=datamodule.label_info, model_name="efficientnet_b0")
engine = LightningEngine(data=datamodule, model=model)
engine.train()
```

Available model classes:

- **Detection:** `ATSS`, `SSD`, `YOLOX`, `RTDETR`, `DFine`, `DEIMDFine`, `DEIMV2`, `RFDETR`
- **Instance segmentation:** `MaskRCNN`, `MaskRCNNTV`, `RTMDetInst`, `RFDETRSeg`
- **Semantic segmentation:** `DinoV2Seg`, `LiteHRNet`, `SegNext`
- **Classification:** `EfficientNet`, `MobileNetV3`, `VisionTransformer`, `TimmModel`, `TVModel`
- **Keypoint:** `RTMPose`

</details>

---

## License

The Geti™ Library (`getitune`) is licensed under [Apache License Version 2.0](https://github.com/open-edge-platform/training_extensions/blob/develop/LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

---
