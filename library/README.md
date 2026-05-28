<div align="center">

# Geti Library - getitune

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

## Introduction

The Geti™ library (`getitune`) is a low-code transfer learning framework for Computer Vision.
Its API and CLI let you train, evaluate, optimize, and deploy models quickly, even without deep expertise in deep learning.
It supports diverse combinations of model architectures, learning methods, and task types built on [PyTorch](https://pytorch.org) and the [OpenVINO™ toolkit](https://software.intel.com/en-us/openvino-toolkit).

Each supported task ships with curated "recipes": YAML files that bundle the model, data pipeline, and training configuration into a single one-stop entry point. Recipes are validated on standard datasets so you get a strong baseline out of the box.

### Key Features

- **Multi-task support**: classification, object detection, rotated detection, instance segmentation, semantic segmentation, and keypoint detection, see the [full model list below](#supported-tasks--models).
- **Tiling** for large images across detection and segmentation tasks.
- **Multiple backends**: train with PyTorch Lightning, export and run inference with ONNX and OpenVINO™.
- **Hardware acceleration**: Intel GPU (XPU) and NVIDIA CUDA support.
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

> **Note:** `getitune` is not yet published to PyPI. Until the first release lands, install from source.

## Quick Install

```bash
# With uv (recommended)
uv pip install "getitune[cpu]"

# Or with pip
pip install "getitune[cpu]"
```

<details>
<summary><strong> Advanced Installation: Specify Hardware Backend</strong></summary>

`getitune` ships three mutually exclusive extras that select the right PyTorch wheel for your hardware:

| Extra    | PyTorch wheel                                                          | Use when                             |
| -------- | ---------------------------------------------------------------------- | ------------------------------------ |
| `[cpu]`  | `torch==2.10.0+cpu` (Linux/Windows) or default `torch==2.10.0` (macOS) | No GPU, or running on Apple silicon. |
| `[xpu]`  | `torch==2.10.0+xpu` + `triton-xpu`                                     | Intel discrete or integrated GPUs.   |
| `[cuda]` | `torch==2.10.0+cu128`                                                  | NVIDIA GPUs with CUDA 12.8 drivers.  |

```bash
pip install "getitune[cpu]"   # CPU-only
pip install "getitune[xpu]"   # Intel GPU (XPU)
pip install "getitune[cuda]"  # NVIDIA GPU (CUDA 12.8)
```

> **macOS note**: PyTorch's `+cpu` wheel is only published for Linux and Windows. The `[cpu]` extra resolves this automatically and installs the default `torch==2.10.0` wheel on macOS.

</details>

<details>
<summary><strong> Advanced Installation: Install from Source</strong></summary>

```bash
git clone https://github.com/open-edge-platform/training_extensions.git
cd training_extensions/library

# Recommended: use uv to honor the lockfile
uv sync --extra cpu          # or --extra xpu / --extra cuda

# Or with pip in a virtual environment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[cpu]"      # remove -e for a non-editable install
```

</details>

---

## Quick Start

### Training

Getitune supports an API-based training approach:

```python
from getitune.engine import create_engine

# Initialize and train using the bundled test dataset
engine = create_engine(
    data="tests/assets/classification_cifar10",
    model="src/getitune/recipe/classification/multi_class_cls/efficientnet_b0.yaml",
)
engine.train()
engine.test()
exported_path = engine.export()  # writes OpenVINO IR
```

---

### Inference

Getitune provides inference via PyTorch and OpenVINO backends:

```python
from getitune.engine import create_engine

# PyTorch inference
engine = create_engine(data="/path/to/dataset", model="path/to/recipe.yaml")
predictions = engine.predict()

# OpenVINO inference and optimization
ov_engine = create_engine(data="/path/to/dataset", model="path/to/exported_model.xml")
ov_engine.test()
ov_engine.optimize()  # post-training quantization via NNCF
```

> **Note:** For advanced inference options including OpenVINO optimization, check the [API Quick-Guide](https://open-edge-platform.github.io/training_extensions/latest/guide/get_started/api_tutorial.html).

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
    model="src/getitune/recipe/detection/yolox_s.yaml",
)
engine.train()
```

---

### Advanced Usage

<details>
<summary><strong>Build a DataModule explicitly</strong></summary>

```python
from getitune.data.module import DataModule
from getitune.types.task import TaskType

datamodule = DataModule(
    task=TaskType.DETECTION,
    data_root="/path/to/coco_dataset",
    input_size=(640, 640),
)

engine = create_engine(
    data=datamodule,
    model="src/getitune/recipe/detection/yolox_s.yaml",
)
engine.train()
```

</details>

<details>
<summary><strong>Instantiate a model class directly</strong></summary>

```python
from getitune.models import ATSS

model = ATSS(label_info=datamodule.label_info)
engine = create_engine(data=datamodule, model=model)
engine.train()
```

Available model classes:

- **Detection:** `ATSS`, `SSD`, `YOLOX`, `RTDETR`, `DFine`, `DEIMDFine`, `DEIMV2`
- **Instance segmentation:** `MaskRCNN`, `MaskRCNNTV`, `RTMDetInst`
- **Semantic segmentation:** `DinoV2Seg`, `LiteHRNet`, `SegNext`
- **Classification:** `EfficientNet`, `MobileNetV3`, `VisionTransformer`, `TimmModel`, `TVModel`
- **Keypoint:** `RTMPose`

</details>

Beyond the examples above, the documentation covers using custom models, overriding training parameters, and [per-task tutorials](https://open-edge-platform.github.io/training_extensions/latest/guide/tutorials/base/how_to_train/index.html).

---

## License

The Geti™ Library (`getitune`) is licensed under [Apache License Version 2.0](https://github.com/open-edge-platform/training_extensions/blob/develop/LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

---
