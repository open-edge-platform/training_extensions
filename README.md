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

Developing AI models from scratch is often slow, complex, and resource-intensive. Managing datasets, training pipelines,
optimization, and deployment typically requires stitching together multiple tools and workflows.

Geti™ is a single, end-to-end application for building and deploying computer vision AI models. It covers the complete
model lifecycle — from dataset preparation, to annotation, to model training and optimization, to deployment and
inference at the edge — all within one application. Optimized for Intel® hardware yet flexible enough to run on a variety
of platforms, Geti lets you manage datasets, fine-tune and quantize models, and deploy them for inference without
stitching together separate tools.

With **Geti™ 3.0**, the entire workflow runs **locally on your own hardware**. You no longer need a Kubernetes cluster or
a data-center GPU: you can fine-tune models and run inference directly on edge and client hardware — including
Intel® Panther Lake and Arc™ Battlemage (B-series) GPUs — keeping your data, training, and deployment on the same
machine. The result is a faster, simpler, and more private way to go from data to a deployed model, whether on a laptop,
a workstation, or an edge device.

Geti is delivered as a self-contained application that you can run as a **single Docker image** or install as a
**native Windows app**. Under the hood, Geti is powered by its own low-code training engine, which is also published on
PyPI as [`getitune`](https://pypi.org/project/getitune/) for developers who want to drive the same training,
optimization, and deployment capabilities programmatically from Python. The engine ships *inside* Geti — it is not a
separate product or a separate release.

<p align="center">
 <img src="assets/geti-model-lifecycle.jpg" width="600" alt="Geti™ - Learning Cycle"/>
</p>

> [!IMPORTANT]
> Information for `otx` users: this repo (`open-edge-platform/training_extensions`) previously hosted the OpenVINO
> Training Extensions project, namely `otx`; the development of that library now continues under the new name
> `getitune` in the [`library`](library) folder, as the training engine of the broader Geti™ application. The package is
> published on PyPI as [`getitune`](https://pypi.org/project/getitune/), while the old package `otx` is deprecated but
> still available for download.

## What's new in Geti™ 3.0

Geti 3.0 marks a fundamental shift: from a cloud-native, cluster-bound platform to a lightweight application that runs
locally on your own hardware.

- **Runs on the edge**: fine-tune models and run inference directly on edge and client hardware, keeping data, training,
  and deployment on the same machine.
- **Dramatically lower hardware requirements**: Geti no longer depends on Kubernetes, so the footprint is significantly
  reduced. The minimum recommended configuration is **8 CPU threads, 16 GB RAM, and 40 GB of free disk space**. Smaller
  models can be trained on CPU with modest memory usage, while a GPU is recommended for larger models.
- **A whole new installation experience**:
  - **Native Windows app** — run Geti directly on Windows via a simple MSIX installer.
  - **Single Docker image** — run Geti from one container on Linux or Windows, while remaining compatible with
    Kubernetes deployments.
- **Apache 2.0 license** — the Geti 3.0 code is now under the permissive Apache 2.0 license, simplifying adoption,
  integration, and redistribution.
- **State-of-the-art model architectures** — RF-DETR (S/M/L) on a DINOv2 backbone, DINOv3 DETR (S/M/L) built on Meta's
  DINOv3 self-supervised backbone, and an Ultralytics integration that brings YOLO26 (NMS-free, edge-optimized) across
  the full lifecycle: training, inference, quantization, and OpenVINO™ export.
- **Dataset versioning** — track how datasets evolve as images and annotations are added, explore revisions, link models
  to a specific dataset state, and train on any prior revision.
- **Model versioning & lineage** — track model lineage with links to parent revisions and weights, view exact training
  hyperparameters, and fine-tune from any previous model version.
- **Configurable training devices** — select from available hardware (CPU, GPU), including specific GPUs in multi-device
  setups.
- **16-bit grayscale media support** — train and infer on high-bit-depth images and video with configurable intensity
  range preprocessing (not supported for YOLOX-S/M/L).
- **Decoupled, edge-ready pipelines** — build flexible inference pipelines where data sources (cameras, files, streams)
  and output targets can be swapped or reconfigured without retraining models.
- **Pipeline export for edge deployment** — package complete pipelines (model, preprocessing, and configuration) into
  OpenVINO™-optimized bundles ready for consistent, efficient deployment on edge and Intel®-based hardware.
- **Integrated deployment & inference** — a new Inference page builds custom pipelines (source → model → sink) to deploy
  models inside Geti and monitor predictions in real time. Sources include USB/IP cameras and video files; optional
  sinks include MQTT, webhook, and folder.
- **Improved evaluation metrics** — expanded training-time and evaluation metrics, including mAP at multiple thresholds
  for detection and segmentation tasks.
- **Label filtering for dataset import/export** — selectively include or exclude labels during dataset export and
  import, automatically filtering out associated annotations.
- **Native dataset format (Geti)** — export and import datasets in a Geti-optimized format, designed for improved
  performance on large datasets while preserving full metadata.
- **Manual subset assignment** — explicitly assign images and video frames to training, validation, or test subsets in
  the annotator, with safeguards to prevent data leakage.
- **Real-time training logs** — monitor training logs live during job execution and access them later via the UI.
- **Demo scripts for inference** — exported ONNX and OpenVINO™ model packages include minimal example scripts to run
  inference on sample images.
- **Bulk annotation for classification** — assign one or more labels to multiple images or video frames at once.
- **Accuracy-aware quantization** — define an acceptable accuracy drop during INT8 model optimization to balance
  inference performance and accuracy.

> [!NOTE]
> Some capabilities from earlier versions are no longer included in Geti 3.0 or have been reworked. Anomaly detection is
> now a standalone product, [Anomalib Studio](https://github.com/open-edge-platform/anomalib). Task chaining,
> hierarchical classification, and rotated detection are being unified under simplified single-task and instance
> segmentation workflows. Semantic segmentation, keypoint detection, and XAI are not available in the Geti 3.0
> application. Multi-user workspaces, user management, and API keys/tokens are not part of Geti 3.0. Dedicated test
> dataset splits and project-level import/export have been replaced by the unified, dataset-based workflow. See the
> [migration notes](#migrating-from-geti-2x) below.

## Key Features

- **Hardware Acceleration**: Geti™ is optimized for modern Intel® hardware with AI capabilities, such as Intel® Arc™
  GPUs and Intel® Core™ Ultra processors. Every trained model is automatically exported with
  [OpenVINO™](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) and it can be
  deployed for inference across the full Intel® XPU portfolio. NVIDIA® CUDA and CPU-only execution are also supported.
- **Iterative Model Improvement**: Geti™ enables users to start building computer vision models with as few as 10-20
  images and iterate on those models in a rapid, feedback-driven loop. This allows you to quickly see results and make
  improvements without needing a large initial dataset; you can add more data as you go, and the predictions from the
  current model can help you annotate new data faster.
- **Multiple Computer Vision Tasks**: Geti™ supports image classification, object detection, and instance segmentation
  through its no-code web interface, while its Python API (`getitune`) unlocks even more use cases including hierarchical
  classification, rotated object detection, semantic segmentation, and keypoint detection.
- **Smart Annotations**: Geti™ includes powerful annotation tools that support both manual and semi-automated labeling
  by means of state-of-the-art AI models like SAM (Segment Anything Model). This significantly reduces the time and
  effort required to create high-quality training datasets.
- **Dataset Import & Export**: Geti™ supports importing and exporting datasets in common formats like COCO, Pascal VOC,
  and YOLO, plus a Geti-optimized native format, making it easy to integrate with other tools and workflows.
- **Model Optimization**: Geti™ provides built-in support for quantization and optimization techniques that can reduce
  model size and improve inference speed, making it easier to deploy models on resource-constrained edge devices.
- **Inference Stream**: Geti™ includes a built-in inference pipeline that enables your trained models for real-time
  inference on video streams, with support for various input sources (cameras, video files, RTSP stream, ...). The
  predictions are visualized directly in the web application, and you can also configure it to forward the results to
  different destinations (folder, MQTT, webhook, ...) for easy integration with other systems.

## Supported tasks and models

Below is a list of tasks and model architectures supported by Geti™. Some tasks are available directly from the no-code
web interface, while others are accessible through the Python API (`getitune`) — both are part of the same Geti
application.
Would you like to see a specific model added to the list? Let us know by opening a [GitHub issue](https://github.com/open-edge-platform/training_extensions/issues)!

<!-- markdownlint-disable MD060 -->

| Task                        | Web UI | Python API (`getitune`) |
| --------------------------- | ------ | ----------------------- |
| Multiclass Classification   | ✅     | ✅                      |
| Multilabel Classification   | ✅     | ✅                      |
| Hierarchical Classification | ✖️     | ✅                      |
| Object Detection            | ✅     | ✅                      |
| Instance Segmentation       | ✅     | ✅                      |
| Semantic Segmentation       | ✖️     | ✅                      |
| Rotated Detection           | ✖️     | ✅                      |
| Keypoint Detection          | ✖️     | ✅                      |

<!-- markdownlint-enable MD060 -->

> [!TIP]
> Other projects of the Open Edge Platform enable even more tasks and models, check them:
>
> - [Anomalib (Studio)](https://github.com/open-edge-platform/anomalib) → anomaly detection
> - [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio) → robot learning, VLA (Vision-Language-Action)
> - [Instant Learn](https://github.com/open-edge-platform/instant-learn) → visual prompting

### Image Classification

<details>
<summary>Show models</summary>

| Model Architecture    | Paper                                              |
| --------------------- | -------------------------------------------------- |
| ViT Tiny              | [ViT](https://arxiv.org/abs/2010.11929)            |
| DINOv2 Small          | [DINOv2](https://arxiv.org/abs/2304.07193)         |
| EfficientNet B0 / B3  | [EfficientNet](https://arxiv.org/abs/1905.11946)   |
| EfficientNet V2 Small | [EfficientNetV2](https://arxiv.org/abs/2104.00298) |
| MobileNet V3 Large    | [MobileNetV3](https://arxiv.org/abs/1905.02244)    |

</details>

### Object Detection

<details>
<summary>Show models</summary>

| Model Architecture     | Paper                                                                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| D-FINE M / L / X       | [DEIM](https://arxiv.org/abs/2412.04234) + [D-FINE](https://arxiv.org/abs/2410.13842)                                                 |
| DINOv3 DETR S / M / L  | [DINOv3](https://arxiv.org/abs/2508.10104) + [DEIMv2](https://arxiv.org/html/2509.20787v4) + [DETR](https://arxiv.org/abs/2005.12872) |
| MobileNet V2 ATSS      | [MobileNetV2](https://arxiv.org/abs/1801.04381) + [ATSS](https://arxiv.org/abs/1912.02424)                                            |
| MobileNet V2 SSD       | [MobileNetV2](https://arxiv.org/abs/1801.04381) + [SSD](https://arxiv.org/abs/1512.02325)                                             |
| RF-DETR S / M / L      | [RF-DETR](https://arxiv.org/abs/2511.09554)                                                                                           |
| RT-DETR R50            | [RT-DETR](https://arxiv.org/abs/2304.08069)                                                                                           |
| YOLO26 Nano / Small / Medium | [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)                                                                  |
| YOLOX Tiny / S / L / X | [YOLOX](https://arxiv.org/abs/2107.08430)                                                                                             |

</details>

### Instance Segmentation

<details>
<summary>Show models</summary>

| Model Architecture        | Paper                                                                                                 |
| ------------------------- | ----------------------------------------------------------------------------------------------------- |
| RTMDet Tiny               | [RTMDet](https://arxiv.org/abs/2212.07784)                                                            |
| Mask-RCNN EfficientNet B2 | [EfficientNet](https://arxiv.org/abs/1905.11946) + [Mask R-CNN](https://arxiv.org/abs/1703.06870)     |
| Mask-RCNN ResNet50        | [ResNet](https://arxiv.org/abs/1512.03385) + [Mask R-CNN](https://arxiv.org/abs/1703.06870)           |
| Mask-RCNN Swin-T          | [Swin Transformer](https://arxiv.org/abs/2103.14030) + [Mask R-CNN](https://arxiv.org/abs/1703.06870) |
| RF-DETR S / M / L         | [RF-DETR](https://arxiv.org/abs/2511.09554)                                                           |
| YOLO26 Nano / Small / Medium | [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)                                     |

</details>

### Semantic Segmentation

<details>
<summary>Show models</summary>

| Model Architecture          | Paper                                          |
| --------------------------- | ---------------------------------------------- |
| DINOv2 Small                | [DINOv2](https://arxiv.org/abs/2304.07193)     |
| Lite-HRNet 18 / S / X       | [Lite-HRNet](https://arxiv.org/abs/2104.06403) |
| SegNeXt Tiny / Small / Base | [SegNeXt](https://arxiv.org/abs/2209.08575)    |

</details>

### Rotated Detection

<details>
<summary>Show models</summary>

| Model Architecture        | Paper                                                                                             |
| ------------------------- | ------------------------------------------------------------------------------------------------- |
| Mask-RCNN EfficientNet B2 | [EfficientNet](https://arxiv.org/abs/1905.11946) + [Mask R-CNN](https://arxiv.org/abs/1703.06870) |
| Mask-RCNN ResNet50        | [ResNet](https://arxiv.org/abs/1512.03385) + [Mask R-CNN](https://arxiv.org/abs/1703.06870)       |

</details>

### Keypoint Detection

<details>
<summary>Show models</summary>

| Model Architecture | Paper                                       |
| ------------------ | ------------------------------------------- |
| RTMPose Tiny       | [RTMPose](https://arxiv.org/abs/2212.07784) |

</details>

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
