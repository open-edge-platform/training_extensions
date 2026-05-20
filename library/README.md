<div align="center">

# Geti Library - getitune

---

[Key Features](#key-features) •
[Supported Tasks & Recipes](#supported-tasks--recipes) •
[Installation](https://open-edge-platform.github.io/training_extensions/latest/guide/get_started/installation.html) •
[Documentation](https://open-edge-platform.github.io/training_extensions/latest/index.html) •
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

The Geti™ library supports the following computer vision tasks:

- **Classification**: multi-class, multi-label, and hierarchical image classification.
- **Object detection**: including tiling for large images.
- **Rotated object detection**: including tiling.
- **Instance segmentation**: including tiling.
- **Semantic segmentation**: including tiling.
- **Keypoint detection.**

Additional capabilities:

- **Multi-backend engine**: PyTorch Lightning and OpenVINO™. Selected automatically by `create_engine` based on the model and data you pass in.
- **Modern model zoo** across tasks (e.g. DEIM-DFine, DEIMv2, DFine, RF-DETR, RT-DETR, YOLOX, ATSS, SSD for detection; Mask R-CNN, RTMDet-Inst, RF-DETR-Seg for instance segmentation; SegNeXt, Lite-HRNet, DINOv2 for semantic segmentation; RTMPose for keypoint detection).
- **Native Intel GPU (XPU) support**: install the `[xpu]` extra to train and infer on Intel GPUs.
- **NVIDIA CUDA support** via the `[cuda]` extra (CUDA 12.8 wheels).
- [Datumaro](https://open-edge-platform.github.io/datumaro/stable/index.html) **data frontend**, with support for the most popular dataset formats per task.
- **Distributed training** across multiple GPUs.
- **Mixed-precision training** to reduce memory and increase batch size.
- **Class-incremental learning** to extend an existing model with new classes.
- **Deployment** to OpenVINO™ IR and ONNX formats, with inference via [OpenVINO™ ModelAPI](https://github.com/open-edge-platform/model_api).
- **Multiple backend support** to adapt third-party model implementations into the Geti™ repository.

---

## Supported Tasks & Recipes

All recipes live under `src/getitune/recipe/<task>/`. Pass any of these YAMLs directly to the API as `model=...`. Recipes whose name ends in `_tile` enable the tiling pipeline for large images.

| Task                                                      | Recipe directory                                                                                                                                                                                                                                              | Example recipes                                                                                                                                            |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Classification (multi-class / multi-label / hierarchical) | `src/getitune/recipe/classification/{`[`multi_class_cls`](src/getitune/recipe/classification/multi_class_cls/), [`multi_label_cls`](src/getitune/recipe/classification/multi_label_cls/), [`h_label_cls`](src/getitune/recipe/classification/h_label_cls/)`}` | `dino_v2`, `vit_tiny`, `efficientnet_b0`, `efficientnet_b3`, `efficientnet_v2`, `mobilenet_v3_large`                                                       |
| Object detection                                          | [`src/getitune/recipe/detection`](src/getitune/recipe/detection/)                                                                                                                                                                                             | `atss_mobilenetv2`, `ssd_mobilenetv2`, `yolox_{tiny,s,l,x}`, `rtdetr_50`, `dfine_x`, `deim_dfine_{l,m,x}`, `deimv2_{s,m,l}`, `rfdetr_{small,medium,large}` |
| Rotated detection                                         | [`src/getitune/recipe/rotated_detection`](src/getitune/recipe/rotated_detection/)                                                                                                                                                                             | `maskrcnn_r50`, `maskrcnn_efficientnetb2b` (with `_tile` variants)                                                                                         |
| Instance segmentation                                     | [`src/getitune/recipe/instance_segmentation`](src/getitune/recipe/instance_segmentation/)                                                                                                                                                                     | `maskrcnn_{r50,swint,efficientnetb2b}`, `rtmdet_inst_tiny`, `rfdetr_seg_{small,medium,large,xlarge}`                                                       |
| Semantic segmentation                                     | [`src/getitune/recipe/semantic_segmentation`](src/getitune/recipe/semantic_segmentation/)                                                                                                                                                                     | `dino_v2`, `litehrnet_{s,18,x}`, `segnext_{t,s,b}` (with `_tile` variants)                                                                                 |
| Keypoint detection                                        | [`src/getitune/recipe/keypoint_detection`](src/getitune/recipe/keypoint_detection/)                                                                                                                                                                           | `rtmpose_tiny`                                                                                                                                             |

Each task directory also ships an `openvino_model.yaml` recipe for running and optimizing pre-exported OpenVINO IR models via `OVEngine`.

---

## Installation

Requirements: **Python 3.11–3.14**, **PyTorch 2.10**, **OpenVINO™ 2026.1**, **NumPy ≥ 2.0**.

> **`getitune` is not yet published to PyPI.** Commands such as `pip install getitune[cpu]` will fail until the first release lands. Until then, use the "Install from source" path below. The PyPI block is kept for reference and will work once the package is published.

For the full guide (system prerequisites, GPU drivers, troubleshooting), see the [installation documentation](https://open-edge-platform.github.io/training_extensions/latest/guide/get_started/installation.html). If you plan to modify the library, install from source.

<details>
<summary>Install from PyPI (not available yet)</summary>

> The commands in this block are forward-looking. They will only work once `getitune` is published to PyPI. Until then, use the "Install from source" block.

`getitune` ships three mutually exclusive extras that select the right PyTorch wheel for your hardware:

| Extra    | PyTorch wheel                                                          | Use when                             |
| -------- | ---------------------------------------------------------------------- | ------------------------------------ |
| `[cpu]`  | `torch==2.10.0+cpu` (Linux/Windows) or default `torch==2.10.0` (macOS) | No GPU, or running on Apple silicon. |
| `[xpu]`  | `torch==2.10.0+xpu` + `triton-xpu`                                     | Intel discrete or integrated GPUs.   |
| `[cuda]` | `torch==2.10.0+cu128`                                                  | NVIDIA GPUs with CUDA 12.8 drivers.  |

Install with `pip`:

```bash
pip install "getitune[cpu]"   # CPU-only
pip install "getitune[xpu]"   # Intel GPU (XPU)
pip install "getitune[cuda]"  # NVIDIA GPU (CUDA 12.8)
```

Or with [`uv`](https://docs.astral.sh/uv/) (faster, used by this repo):

```bash
uv pip install "getitune[cpu]"
```

> **macOS note**: PyTorch's `+cpu` wheel is only published for Linux and Windows. The `[cpu]` extra resolves this automatically and installs the default `torch==2.10.0` wheel on macOS.

</details>

<details>
<summary>Install from source</summary>

```bash
# Clone the repository
git clone https://github.com/open-edge-platform/training_extensions.git
cd training_extensions/library

# Recommended: use uv to honor the lockfile
uv sync --extra cpu          # or --extra xpu / --extra cuda

# Or with pip in a virtual environment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[cpu]"      # add -e for editable mode
```

</details>

---

## Quick-Start

`getitune` is designed to be used primarily through its Python API. A CLI also exists (entry points `getitune` and `otx`), but it is a thin wrapper around the same API and is not the main focus of this README; for CLI details see the upstream [CLI Guide](https://open-edge-platform.github.io/training_extensions/latest/guide/get_started/cli_commands.html).

The sections below cover the most common API patterns: recipe-driven training, loading COCO and YOLO datasets, building a `DataModule` explicitly, instantiating model classes directly, and using the OpenVINO engine for inference and optimization.

<details>
<summary>API Usage</summary>

### 1. Quickest path: pass a recipe and a dataset directory

`create_engine` is the single entry point. Give it a recipe YAML and a dataset root and you get back an `Engine` with `train`, `test`, `predict`, and `export` methods.

```python
from getitune.engine import create_engine

engine = create_engine(
    data="/path/to/dataset",
    model="src/getitune/recipe/detection/atss_mobilenetv2.yaml",
)
engine.train()
engine.test()
exported_path = engine.export()  # writes OpenVINO IR
```

By default, artifacts go to `./getitune-workspace/`. Override with `work_dir="..."`:

```python
engine = create_engine(
    data="/path/to/dataset",
    model="src/getitune/recipe/detection/atss_mobilenetv2.yaml",
    work_dir="runs/atss_baseline",
)
```

### 2. Loading datasets

When you pass a path to `data=`, `getitune` calls `datumaro.experimental.import_dataset(data_root)` under the hood. Datumaro auto-detects the dataset format from the directory contents, so you do not need to pass an explicit format anywhere. The path may point to a directory or to a `.zip` archive (Datumaro extracts the archive automatically next to it, or to the `extract_dir` you configure when calling Datumaro directly).

Four formats are currently recognised by the auto-detector:

- **Datumaro (native)**: a `metadata.json` plus `data.parquet` file at the root. This is the most reliable interchange format and the one to prefer for round-trips.
- **COCO**: an `annotations/` directory containing COCO JSON files, or any JSON file with `images` and `annotations` keys at the top level.
- **YOLO**: a `data.yaml` file (Ultralytics layout), `obj.names` plus `obj.data` (traditional layout), or matching `images/` and `labels/` directories.
- **Pascal VOC**: `JPEGImages/`, `Annotations/`, and `ImageSets/` directories side by side at the root.

Reference layouts:

COCO detection:

```text
coco_dataset/
|-- annotations/
|   |-- instances_train.json
|   `-- instances_val.json
`-- images/
    |-- train/*.jpg
    `-- val/*.jpg
```

YOLO (Ultralytics) detection:

```text
yolo_dataset/
|-- data.yaml
|-- images/
|   |-- train/*.jpg
|   `-- val/*.jpg
`-- labels/
    |-- train/*.txt
    `-- val/*.txt
```

Pascal VOC:

```text
voc_dataset/
|-- Annotations/*.xml
|-- ImageSets/
|   `-- Main/{train,val}.txt
`-- JPEGImages/*.jpg
```

Datumaro native (round-trip friendly):

```text
datumaro_dataset/
|-- metadata.json
|-- data.parquet
`-- images/*.jpg
```

Once the directory layout matches one of these structures, the call into `create_engine` does not change:

```python
engine = create_engine(
    data="/path/to/yolo_dataset",
    model="src/getitune/recipe/detection/yolox_s.yaml",
)
engine.train()
```

The exact same line works for COCO, VOC, or Datumaro-native datasets. Only the directory contents differ.

Zip archives: `data=` also accepts the path to a `.zip` archive of any of the layouts above; Datumaro extracts it on import.

### 3. Build a `DataModule` explicitly

When you want finer control (custom input size, tiling, subset configs, worker counts), construct a `DataModule` yourself and pass the instance to `create_engine`. The `data=` argument accepts either a path or a `DataModule` (the `DATA` type alias is `DataModule | PathLike`).

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

If `train_subset` / `val_subset` / `test_subset` are not provided, `DataModule` falls back to sensible defaults via `DataModule.get_default_subset_configs(input_size)`.

### 4. Instantiate a model class directly

Recipes are the easiest path, but you can also import a model class from `getitune.models` and hand the instance to `create_engine`. This is useful when you want to construct the model programmatically or assemble it inside another framework.

Detection example (`ATSS`):

```python
from getitune.models import ATSS

model = ATSS(label_info=datamodule.label_info)

engine = create_engine(data=datamodule, model=model)
engine.train()
```

Keypoint detection example (`RTMPose`):

```python
from getitune.models import RTMPose

model = RTMPose(label_info=datamodule.label_info)

engine = create_engine(data=datamodule, model=model)
engine.train()
```

The exact constructor arguments differ per model class (some accept `label_info`, others additionally take `num_classes`, backbone configuration, or task-specific options). The recipe YAML for the same model is the easiest reference for what to pass; you can also inspect the class signature directly.

Other commonly used direct-instantiation classes re-exported from `getitune.models` include:

- Detection: `ATSS`, `SSD`, `YOLOX`, `RTDETR`, `DFine`, `DEIMDFine`, `DEIMV2`
- Instance segmentation: `MaskRCNN`, `MaskRCNNTV`, `RTMDetInst`
- Semantic segmentation: `DinoV2Seg`, `LiteHRNet`, `SegNext`
- Classification: `EfficientNet`, `MobileNetV3`, `VisionTransformer`, `TimmModel`, `TVModel`
- Keypoint: `RTMPose`

### 5. OpenVINO inference and optimization

When you pass an OpenVINO IR path as `model=`, `create_engine` automatically dispatches to `OVEngine`, which supports evaluation, inference, and post-training quantization via NNCF.

```python
ov_engine = create_engine(data="/path/to/dataset", model=exported_path)
ov_engine.test()
ov_engine.optimize()
```

You can also instantiate OpenVINO model wrappers directly, for example `from getitune.models import OVDetectionModel`, when you need full control over the IR loading pipeline.

For more examples, see the [API Quick-Guide](https://open-edge-platform.github.io/training_extensions/latest/guide/get_started/api_tutorial.html).

</details>

Beyond the examples above, the documentation covers using custom models, overriding training parameters, and [per-task tutorials](https://open-edge-platform.github.io/training_extensions/latest/guide/tutorials/base/how_to_train/index.html).

---

## License

The Geti™ Library (`getitune`) is licensed under [Apache License Version 2.0](https://github.com/open-edge-platform/training_extensions/blob/develop/LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

---
