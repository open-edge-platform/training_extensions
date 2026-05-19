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
[![Build Docs](https://github.com/open-edge-platform/training_extensions/actions/workflows/docs.yaml/badge.svg)](https://github.com/open-edge-platform/training_extensions/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://static.pepy.tech/personalized-badge/getitune?period=total&units=international_system&left_color=grey&right_color=green&left_text=PyPI%20Downloads)](https://pepy.tech/project/getitune)

---

</div>

## Introduction

The Geti™ library (`getitune`) is a low-code transfer learning framework for Computer Vision.
Its API and CLI let you train, evaluate, optimize, and deploy models quickly — even without deep expertise in deep learning.
It supports diverse combinations of model architectures, learning methods, and task types built on [PyTorch](https://pytorch.org) and the [OpenVINO™ toolkit](https://software.intel.com/en-us/openvino-toolkit).

Each supported task ships with curated "recipes" — YAML files that bundle the model, data pipeline, and training configuration into a single one-stop entry point. Recipes are validated on standard datasets so you get a strong baseline out of the box.

### Key Features

The Geti™ library supports the following computer vision tasks:

- **Classification** — multi-class, multi-label, and hierarchical image classification.
- **Object detection** — including tiling for large images.
- **Rotated object detection** — including tiling.
- **Instance segmentation** — including tiling.
- **Semantic segmentation** — including tiling.
- **Keypoint detection.**

Additional capabilities:

- **Multi-backend engine**: PyTorch Lightning, Native, and OpenVINO™ — selected automatically by `create_engine` based on the model and data you pass in.
- **Modern model zoo** across tasks (e.g. DEIM-DFine, DEIMv2, DFine, RF-DETR, RT-DETR, YOLOX, ATSS, SSD for detection; Mask R-CNN, RTMDet-Inst, RF-DETR-Seg for instance segmentation; SegNeXt, Lite-HRNet, DINOv2 for semantic segmentation; RTMPose for keypoint detection).
- **Native Intel GPU (XPU) support** — install the `[xpu]` extra to train and infer on Intel GPUs.
- **NVIDIA CUDA support** via the `[cuda]` extra (CUDA 12.8 wheels).
- [Datumaro](https://open-edge-platform.github.io/datumaro/stable/index.html) **data frontend**, with support for the most popular dataset formats per task.
- **Distributed training** across multiple GPUs.
- **Mixed-precision training** to reduce memory and increase batch size.
- **Class-incremental learning** to extend an existing model with new classes.
- **Deployment** to OpenVINO™ IR and ONNX formats, with inference via [OpenVINO™ ModelAPI](https://github.com/open-edge-platform/model_api).
- **Multiple backend support** to adapt third-party model implementations into the Geti™ repository.

---

## Supported Tasks & Recipes

All recipes live under `src/getitune/recipe/<task>/`. Pass any of these YAMLs directly to the API (`model=...`) or CLI (`--config ...`). Recipes whose name ends in `_tile` enable the tiling pipeline for large images.

| Task                                                      | Recipe directory                                                                   | Example recipes                                                                                                                                            |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Classification — multi-class / multi-label / hierarchical | `src/getitune/recipe/classification/{multi_class_cls,multi_label_cls,h_label_cls}` | `dino_v2`, `vit_tiny`, `efficientnet_b0`, `efficientnet_b3`, `efficientnet_v2`, `mobilenet_v3_large`                                                       |
| Object detection                                          | `src/getitune/recipe/detection`                                                    | `atss_mobilenetv2`, `ssd_mobilenetv2`, `yolox_{tiny,s,l,x}`, `rtdetr_50`, `dfine_x`, `deim_dfine_{l,m,x}`, `deimv2_{s,m,l}`, `rfdetr_{small,medium,large}` |
| Rotated detection                                         | `src/getitune/recipe/rotated_detection`                                            | `maskrcnn_r50`, `maskrcnn_efficientnetb2b` (with `_tile` variants)                                                                                         |
| Instance segmentation                                     | `src/getitune/recipe/instance_segmentation`                                        | `maskrcnn_{r50,swint,efficientnetb2b}`, `rtmdet_inst_tiny`, `rfdetr_seg_{small,medium,large,xlarge}`                                                       |
| Semantic segmentation                                     | `src/getitune/recipe/semantic_segmentation`                                        | `dino_v2`, `litehrnet_{s,18,x}`, `segnext_{t,s,b}` (with `_tile` variants)                                                                                 |
| Keypoint detection                                        | `src/getitune/recipe/keypoint_detection`                                           | `rtmpose_tiny`                                                                                                                                             |

Each task directory also ships an `openvino_model.yaml` recipe for running and optimizing pre-exported OpenVINO IR models via `OVEngine`.

To browse the full list from the CLI:

```bash
getitune find          # all recipes
getitune find --task DETECTION
```

---

## Installation

Requirements: **Python 3.11–3.14**, **PyTorch 2.10**, **OpenVINO™ 2026.1**, **NumPy ≥ 2.0**.

For the full guide (system prerequisites, GPU drivers, troubleshooting), see the [installation documentation](https://open-edge-platform.github.io/training_extensions/latest/guide/get_started/installation.html). If you plan to modify the library, install from source.

<details>
<summary>Install from PyPI</summary>

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

`getitune` supports both an API and a CLI. The API is flexible and ideal for customization; the CLI is the fastest way to use the library off-the-shelf.

The package installs two equivalent CLI entry points — `getitune` (preferred) and `otx` (alias kept for compatibility):

```bash
# Discover available subcommands
getitune --help               # or: otx --help

# Help for a single subcommand
getitune train --help
getitune train --help -v      # required parameters only
getitune train --help -vv     # all configurable parameters
```

See the [CLI Guide](https://open-edge-platform.github.io/training_extensions/latest/guide/get_started/cli_commands.html) and the [API Quick-Guide](https://open-edge-platform.github.io/training_extensions/latest/guide/get_started/api_tutorial.html) for full walkthroughs.

<details>
<summary>API Usage</summary>

`getitune.engine.create_engine` is the single entry point. It inspects your `model` and `data` arguments and dispatches to the right engine — `LightningEngine` for PyTorch training, `OVEngine` for OpenVINO IR models, or any custom `Engine` subclass that declares `is_supported`.

```python
from getitune.engine import create_engine

# Train an ATSS-MobileNetV2 detector
engine = create_engine(
    data="path/to/dataset/root",
    model="src/getitune/recipe/detection/atss_mobilenetv2.yaml",
)
engine.train()
engine.test()
exported_path = engine.export()  # → OpenVINO IR

# By default all artifacts go to ./getitune-workspace.
# Override with the work_dir argument:
engine = create_engine(
    data="path/to/dataset/root",
    model="src/getitune/recipe/detection/atss_mobilenetv2.yaml",
    work_dir="my_workdir",
)

# The exported IR can be evaluated and quantized with the OpenVINO engine.
# create_engine picks OVEngine automatically when given an IR path.
ov_engine = create_engine(data="path/to/dataset/root", model=exported_path)
ov_engine.test()
ov_engine.optimize()
```

To enumerate available recipes from Python, use the Lightning backend helper (the `getitune find` CLI command is equivalent):

```python
from getitune.backend.lightning.cli.utils import list_models

list_models(print_table=True)
```

For more examples, see the [API Quick-Guide](https://open-edge-platform.github.io/training_extensions/latest/guide/get_started/api_tutorial.html).

</details>

<details>
<summary>CLI Usage</summary>

```bash
# List all recipes
getitune find

# Train
getitune train \
  --config src/getitune/recipe/detection/atss_mobilenetv2.yaml \
  --data_root data/wgisd

# By default the working directory is ./getitune-workspace.
# Override it with --work_dir, or re-use a previous run via the .latest symlink.
getitune test \
  --config src/getitune/recipe/detection/atss_mobilenetv2.yaml \
  --data_root data/wgisd \
  --checkpoint getitune-workspace/.latest/train/best_checkpoint.ckpt

getitune export \
  --config src/getitune/recipe/detection/atss_mobilenetv2.yaml \
  --data_root data/wgisd \
  --checkpoint getitune-workspace/.latest/train/best_checkpoint.ckpt

# Re-use the previous run via work_dir
getitune test   --work_dir getitune-workspace/.latest/train
getitune export --work_dir getitune-workspace/.latest/train

# Or run directly from inside the working directory
cd getitune-workspace
getitune test
getitune export
```

For more examples, see the [CLI Guide](https://open-edge-platform.github.io/training_extensions/latest/guide/get_started/cli_commands.html).

</details>

Beyond the examples above, the documentation also covers using custom models, overriding training parameters, and [per-task tutorials](https://open-edge-platform.github.io/training_extensions/latest/guide/tutorials/base/how_to_train/index.html).

---

## License

The Geti™ Library (`getitune`) is licensed under [Apache License Version 2.0](https://github.com/open-edge-platform/training_extensions/blob/develop/LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

---
