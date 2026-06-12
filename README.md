<!-- markdownlint-disable MD013 MD033 MD041 MD042 -->
<div align="center">

<img src="assets/geti-header.png" alt="Geti™ - A framework to rapidly build and deploy computer vision AI models">

<br>

[![Daily checks](https://github.com/open-edge-platform/training_extensions/actions/workflows/daily.yml/badge.svg)](https://github.com/open-edge-platform/training_extensions/actions/workflows/daily.yml)
[![Docker build](https://github.com/open-edge-platform/training_extensions/actions/workflows/build.yaml/badge.svg)](https://github.com/open-edge-platform/training_extensions/actions/workflows/build.yaml)
[![Codecov](https://codecov.io/gh/open-edge-platform/training_extensions/branch/develop/graph/badge.svg?token=9HVFNMPFGD)](https://codecov.io/gh/open-edge-platform/training_extensions)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/open-edge-platform/training_extensions/badge)](https://securityscorecards.dev/viewer/?uri=github.com/open-edge-platform/training_extensions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

## Introduction

Geti™ is an end-to-end application that takes you from raw images to a deployed computer vision model — annotate, train,
optimize, and run inference, all in one place, all on your own hardware. Start with as few as 10-20 images and iterate
in a rapid, feedback-driven loop. Geti runs locally as a single Docker image or a native Windows app, and is optimized
for Intel® hardware with OpenVINO™ for fast inference across the full Intel® XPU portfolio.

<p align="center">
 <img src="assets/geti-model-lifecycle.jpg" width="600" alt="Geti™ - Learning Cycle"/>
</p>

> [!IMPORTANT]
> Information for `otx` users: this repo (`open-edge-platform/training_extensions`) previously hosted the OpenVINO
> Training Extensions project, namely `otx`; the development of that library now continues under the new name
> `getitune` in the [`library`](library) folder, as the training engine of the broader Geti™ application. The package is
> published on PyPI as [`getitune`](https://pypi.org/project/getitune/), while the old package `otx` is deprecated but
> still available for download.

## Key Features

- **Runs locally, on the edge**: fine-tune models and run inference directly on edge and client hardware — including
  Intel® Panther Lake and Arc™ Battlemage (B-series) GPUs — with no Kubernetes cluster or data-center GPU required.
  Minimum recommended setup: **8 CPU threads, 16 GB RAM, 40 GB free disk**.
- **Hardware acceleration**: optimized for modern Intel® hardware (Arc™ GPUs, Core™ Ultra processors). Every model is
  automatically exported with [OpenVINO™](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
  for deployment across the full Intel® XPU portfolio; NVIDIA® CUDA and CPU-only execution are also supported.
- **Iterative model improvement**: start with as few as 10-20 images and refine your model in a rapid, feedback-driven
  loop, using the current model's predictions to annotate new data faster.
- **Multiple computer vision tasks**: image classification, object detection, and instance segmentation from the no-code
  web interface, with hierarchical classification, rotated detection, semantic segmentation, and keypoint detection
  available through the Python API (`getitune`).
- **State-of-the-art models**: a curated catalog spanning RF-DETR, DINOv3 DETR, YOLO26, YOLOX, D-FINE, Mask R-CNN, and
  more — see the [full list below](#supported-tasks-and-models).
- **Smart annotations**: manual and semi-automated labeling powered by models like SAM (Segment Anything Model), plus
  bulk labeling to dramatically speed up dataset creation.
- **Dataset & model versioning**: track how datasets and models evolve, link models to a specific dataset revision, view
  exact training hyperparameters, and fine-tune from any previous version.
- **Dataset import & export**: COCO, Pascal VOC, and YOLO formats plus a Geti-optimized native format, with label
  filtering to selectively include or exclude labels on import/export.
- **Model optimization**: built-in quantization with accuracy-aware INT8 optimization to balance inference speed and
  accuracy on resource-constrained edge devices.
- **Integrated deployment & inference**: build custom pipelines (source → model → sink) to deploy models inside Geti and
  monitor real-time predictions on video streams. Sources include USB/IP cameras and video files; optional sinks include
  folder, MQTT, and webhook. Complete pipelines can be exported as OpenVINO™-optimized bundles for edge deployment.

## Supported tasks and models

Below is a list of tasks and model architectures supported by Geti™. Some tasks are available directly from the no-code
web interface, while others are accessible through the Python API (`getitune`) — both are part of the same Geti
application.
Would you like to see a specific model added to the list? Let us know by opening a [GitHub issue](https://github.com/open-edge-platform/training_extensions/issues)!

> [!TIP]
> Other projects of the Open Edge Platform enable even more tasks and models, check them:
>
> - [Anomalib (Studio)](https://github.com/open-edge-platform/anomalib) → anomaly detection
> - [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio) → robot learning, VLA (Vision-Language-Action)
> - [Instant Learn](https://github.com/open-edge-platform/instant-learn) → visual prompting

<!-- markdownlint-disable MD060 -->

| Computer Vision Task | Use Case | Model Architecture | Paper |
| -------------------- | -------- | ------------------ | ----- |
| **Classification** (multi-class, multi-label, hierarchical) | Assign one or more labels to an entire image, e.g. quality pass/fail, product categorization, content tagging. | ViT Tiny | [ViT](https://arxiv.org/abs/2010.11929) |
| | | DINOv2 Small | [DINOv2](https://arxiv.org/abs/2304.07193) |
| | | EfficientNet B0 / B3 | [EfficientNet](https://arxiv.org/abs/1905.11946) |
| | | EfficientNet V2 Small | [EfficientNetV2](https://arxiv.org/abs/2104.00298) |
| | | MobileNet V3 Large | [MobileNetV3](https://arxiv.org/abs/1905.02244) |
| **Object Detection** | Locate and classify objects with bounding boxes, e.g. counting items, defect localization, surveillance. | D-FINE M / L / X | [DEIM](https://arxiv.org/abs/2412.04234) + [D-FINE](https://arxiv.org/abs/2410.13842) |
| | | DINOv3 DETR S / M / L | [DINOv3](https://arxiv.org/abs/2508.10104) + [DEIMv2](https://arxiv.org/html/2509.20787v4) + [DETR](https://arxiv.org/abs/2005.12872) |
| | | MobileNet V2 ATSS | [MobileNetV2](https://arxiv.org/abs/1801.04381) + [ATSS](https://arxiv.org/abs/1912.02424) |
| | | MobileNet V2 SSD | [MobileNetV2](https://arxiv.org/abs/1801.04381) + [SSD](https://arxiv.org/abs/1512.02325) |
| | | RF-DETR S / M / L | [RF-DETR](https://arxiv.org/abs/2511.09554) |
| | | RT-DETR R50 | [RT-DETR](https://arxiv.org/abs/2304.08069) |
| | | YOLO26 Nano / Small / Medium | [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) |
| | | YOLOX Tiny / S / L / X | [YOLOX](https://arxiv.org/abs/2107.08430) |
| **Instance Segmentation** | Detect objects and produce pixel-precise masks per instance, e.g. measuring object area, robotics, medical imaging. | RTMDet Tiny | [RTMDet](https://arxiv.org/abs/2212.07784) |
| | | Mask-RCNN EfficientNet B2 | [EfficientNet](https://arxiv.org/abs/1905.11946) + [Mask R-CNN](https://arxiv.org/abs/1703.06870) |
| | | Mask-RCNN ResNet50 | [ResNet](https://arxiv.org/abs/1512.03385) + [Mask R-CNN](https://arxiv.org/abs/1703.06870) |
| | | Mask-RCNN Swin-T | [Swin Transformer](https://arxiv.org/abs/2103.14030) + [Mask R-CNN](https://arxiv.org/abs/1703.06870) |
| | | RF-DETR S / M / L | [RF-DETR](https://arxiv.org/abs/2511.09554) |
| | | YOLO26 Nano / Small / Medium | [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) |

<!-- markdownlint-enable MD060 -->

## Getting started

There are several ways to run Geti — choose the method that best fits your workflow:

- **Docker (recommended)** — download and run one of the pre-built Docker images, or build one yourself.
- **Native Windows app (MSIX)** — install Geti as a desktop application on Windows.
- **From source (for development)** — run the server and the UI as standalone components.
- **Python API (`getitune`)** — install the training engine from PyPI to drive Geti's capabilities programmatically.

System requirements (minimum recommended): **8 CPU threads, 16 GB RAM, 40 GB free disk space**. Smaller models train on
CPU; a GPU (Intel® XPU or NVIDIA® CUDA) is recommended for larger models.

For complete, up-to-date instructions and all runtime options, see the [application README](application/README.md).

### Run with Docker

The easiest and most portable way to run Geti is through Docker. Pre-built images are provided for Intel® XPU,
NVIDIA® CUDA, and CPU-only platforms, or you can build your own image from source.

**Prerequisites** (on the host system):

- Docker v29+ [[docs]](https://docs.docker.com/)
- (Optional, recommended) Just v1.46+ [[docs]](https://github.com/casey/just)
- (Only for Intel® XPU) the latest driver suitable for your hardware [[docs]](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-11.html)
- (Only for NVIDIA® GPU) NVIDIA driver and the NVIDIA Container Toolkit [[docs]](https://www.nvidia.com/Download/index.aspx)

> [!WARNING]
> The official Docker images for Geti 3.x have not been released yet. For now, build the image from source (see below).

```bash
docker pull ghcr.io/open-edge-platform/geti-xpu    # modern Intel® CPU/GPU (recommended)
docker pull ghcr.io/open-edge-platform/geti-cuda   # NVIDIA® CUDA platforms
docker pull ghcr.io/open-edge-platform/geti-cpu    # CPU-only (most lightweight)
```

Alternatively, build the image from source. From the `application` directory:

```bash
# Build for Intel® XPU (recommended); use --accelerator cuda or cpu for other targets
just build-image --accelerator xpu
```

Once you have the image, launch the application:

```bash
just run-image --accelerator xpu
```

After the container starts, open the Geti web application at [**http://localhost:7860**](http://localhost:7860)
(default settings).

### Install the Windows app (MSIX)

> [!WARNING]
> The MSIX app for Geti 3.x has not been released yet.

Geti 3.0 can be installed as a native Windows desktop application via a simple MSIX installer. Installation instructions
will be published with the release.

### Run from source (for development)

For development, you can run the Geti server and UI as standalone components without Docker.

**Prerequisites:** Just v1.46+, Node.js v24.2+, and the appropriate GPU driver/toolkit for Intel® XPU or NVIDIA® CUDA.

```bash
# Start the server (from the repo root)
cd application/backend
just venv --accelerator xpu     # initialize the environment (cpu, xpu, or cuda)
just run-server                 # add --setup-demo to pre-populate demo data
```

```bash
# Start the UI in a separate terminal (from the repo root)
cd application/ui
npm install
npm run build
npm run start
```

After the UI starts, open the Geti web application at [**http://localhost:3000**](http://localhost:3000)
(default settings). See the [application README](application/README.md) for all options.

### Use the Python API (`getitune`)

Geti's training engine is also available as a low-code Python package for developers who want to train, optimize, and
deploy models programmatically. It requires **Python 3.11–3.14**, **PyTorch 2.10**, **OpenVINO™ 2026.1**, and
**NumPy ≥ 2.0**.

```bash
pip install "getitune[cpu]"    # CPU-only
pip install "getitune[xpu]"    # Intel® GPU (XPU)
pip install "getitune[cuda]"   # NVIDIA® GPU (CUDA 12.8)
```

```python
from getitune.engine import create_engine

# Initialize and train using a bundled recipe and dataset
engine = create_engine(
    data="tests/assets/classification_cifar10",
    model="src/getitune/recipe/classification/multi_class_cls/efficientnet_b0.yaml",
)
engine.train()
engine.test()
exported_path = engine.export()  # writes OpenVINO IR
```

See the [library README](library/README.md) for the full list of recipes, advanced configuration, dataset support, and
inference/optimization examples.

## Migrating from Geti 2.x

Geti 3.0 introduces a simplified, dataset-based workflow. When upgrading:

- **Models** trained in 2.x must be retrained on imported datasets.
- **Projects with multiple datasets** require each dataset to be exported and imported separately.
- **Project-level import/export** has been replaced by dataset-based migration — transfer your data and retrain models
  in the new environment.

The REST API has also been redesigned, and model export and deployment have been streamlined (now using the
OpenVINO™ Model API instead of the SDK). Please follow the migration guidance in the
[documentation](#documentation).

## Documentation

| Component                  | README                                          | Documentation                                                                            |
| -------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **Geti application**       | [application/README.md](application/README.md)  | Coming soon!                                                                             |
| **Python API (getitune)**  | [library/README.md](library/README.md)          | [Docs](https://open-edge-platform.github.io/training_extensions/latest/index.html)       |

## Community

- To report a bug or submit a feature request, please open a [GitHub issue](https://github.com/open-edge-platform/training_extensions/issues).
- Ask questions via [GitHub Discussions](https://github.com/open-edge-platform/training_extensions/discussions).

For those who would like to contribute, see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

Thank you! We appreciate your support!

<a href="https://github.com/open-edge-platform/training_extensions/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=open-edge-platform/training_extensions" alt="Contributors" />
</a>

## License

Geti™ is licensed under the [Apache License Version 2.0](LICENSE). By contributing to the project, you agree to the
license and copyright terms therein and release your contribution under these terms.
