<!-- markdownlint-disable MD013 MD033 MD041 MD042 -->
<div align="center">

<img src="assets/geti-header.png" alt="Geti™ - A framework to rapidly build and deploy computer vision AI models">

**Enable anyone from domain experts to data scientists to rapidly develop production-ready AI models**

[Key Features](#key-features) •
[Supported tasks and models](#supported-tasks-and-models) •
[Quick Start](#quick-start) •
[Documentation](#documentation) •
[Contributing](#contributing)

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
 <img src="assets/model-lifecycle-infinity-light.png" width="600" alt="Geti™ - Learning Cycle"/>
</p>

> [!IMPORTANT]
> This repo previously hosted the OpenVINO Training Extensions project, namely `otx`;
> the development of that library now continues under the new name
> `getitune` in the [`library`](library) folder, as the training engine of the broader Geti™ application. The package is
> published on PyPI as [`getitune`](https://pypi.org/project/getitune/), while the old package `otx` is deprecated but
> still available for download.
>
> The development of the Geti™ application now continues in this repository in the [`application`](application) folder.
> Previous versions of Geti™ are still available in a separate [repository](https://github.com/open-edge-platform/geti_v2).

## Key Features

- **Interactive end-to-end model training**: Geti™ enables users to start building deep-learning computer vision models
  with as few as 10-20 images and take them to production in one environment — annotate, train, optimize, run
  inference, and improve accuracy in a rapid train-predict-annotate loop.
- **State-of-the-art model catalog**: train and fine-tune modern architectures such as RF-DETR, DINOv3 DETR, YOLO26,
  YOLOX, D-FINE, and Mask R-CNN — see the [full list below](#supported-tasks-and-models).
- **Multiple computer vision tasks**: image classification, object detection, and instance segmentation from the no-code
  web interface, with even more tasks available through the Python API (`getitune`).
- **Smart annotations**: manual and semi-automated labeling powered by models like SAM (Segment Anything Model), plus
  bulk labeling to dramatically speed up dataset creation.
- **Dataset & model versioning**: track how datasets and models evolve, link models to a specific dataset revision, view
  exact training hyperparameters, and fine-tune from any previous version.
- **Runs locally, on the edge**: fine-tune models and run inference directly on edge and client hardware — including
  Intel® Panther Lake and Arc™ Battlemage (B-series) GPUs — with no Kubernetes cluster or data-center GPU required.
  Minimum recommended setup: **8 CPU threads, 16 GB RAM, 40 GB free disk**.
- **Hardware acceleration**: optimized for modern Intel® hardware (Arc™ GPUs, Core™ Ultra processors). Every model is
  automatically exported with [OpenVINO™](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
  for deployment across the full Intel® XPU portfolio; NVIDIA® CUDA and CPU-only execution are also supported.
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

<!-- markdownlint-disable MD060 -->

<table>
  <thead>
    <tr>
      <th>Computer Vision Task</th>
      <th>Model Architecture</th>
      <th>Paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="8"><b>Object Detection</b><br>Locate and classify objects with bounding boxes, e.g. counting items, defect localization, surveillance.</td>
      <td>D-FINE M / L / X</td>
      <td><a href="https://arxiv.org/abs/2412.04234">DEIM</a> + <a href="https://arxiv.org/abs/2410.13842">D-FINE</a></td>
    </tr>
    <tr>
      <td>DINOv3 DETR S / M / L</td>
      <td><a href="https://arxiv.org/abs/2508.10104">DINOv3</a> + <a href="https://arxiv.org/html/2509.20787v4">DEIMv2</a> + <a href="https://arxiv.org/abs/2005.12872">DETR</a></td>
    </tr>
    <tr>
      <td>MobileNet V2 ATSS</td>
      <td><a href="https://arxiv.org/abs/1801.04381">MobileNetV2</a> + <a href="https://arxiv.org/abs/1912.02424">ATSS</a></td>
    </tr>
    <tr>
      <td>MobileNet V2 SSD</td>
      <td><a href="https://arxiv.org/abs/1801.04381">MobileNetV2</a> + <a href="https://arxiv.org/abs/1512.02325">SSD</a></td>
    </tr>
    <tr>
      <td>RF-DETR S / M / L</td>
      <td><a href="https://arxiv.org/abs/2511.09554">RF-DETR</a></td>
    </tr>
    <tr>
      <td>RT-DETR R50</td>
      <td><a href="https://arxiv.org/abs/2304.08069">RT-DETR</a></td>
    </tr>
    <tr>
      <td>YOLO26 Nano / Small / Medium</td>
      <td><a href="https://github.com/ultralytics/ultralytics">Ultralytics YOLO</a></td>
    </tr>
    <tr>
      <td>YOLOX Tiny / S / L / X</td>
      <td><a href="https://arxiv.org/abs/2107.08430">YOLOX</a></td>
    </tr>
    <tr>
      <td rowspan="5"><b>Instance Segmentation</b><br>Detect objects and produce pixel-precise masks per instance, e.g. measuring object area, robotics, medical imaging.</td>
      <td>RTMDet Tiny</td>
      <td><a href="https://arxiv.org/abs/2212.07784">RTMDet</a></td>
    </tr>
    <tr>
      <td>Mask-RCNN EfficientNet B2</td>
      <td><a href="https://arxiv.org/abs/1905.11946">EfficientNet</a> + <a href="https://arxiv.org/abs/1703.06870">Mask R-CNN</a></td>
    </tr>
    <tr>
      <td>Mask-RCNN ResNet50</td>
      <td><a href="https://arxiv.org/abs/1512.03385">ResNet</a> + <a href="https://arxiv.org/abs/1703.06870">Mask R-CNN</a></td>
    </tr>
    <tr>
      <td>Mask-RCNN Swin-T</td>
      <td><a href="https://arxiv.org/abs/2103.14030">Swin Transformer</a> + <a href="https://arxiv.org/abs/1703.06870">Mask R-CNN</a></td>
    </tr>
    <tr>
      <td>RF-DETR S / M / L</td>
      <td><a href="https://arxiv.org/abs/2511.09554">RF-DETR</a></td>
    </tr>
    <tr>
      <td rowspan="5"><b>Classification</b> (multi-class, multi-label)<br>Assign one or more labels to an entire image, e.g. quality pass/fail, product categorization, content tagging.</td>
      <td>ViT Tiny</td>
      <td><a href="https://arxiv.org/abs/2010.11929">ViT</a></td>
    </tr>
    <tr>
      <td>DINOv2 Small</td>
      <td><a href="https://arxiv.org/abs/2304.07193">DINOv2</a></td>
    </tr>
    <tr>
      <td>EfficientNet B0 / B3</td>
      <td><a href="https://arxiv.org/abs/1905.11946">EfficientNet</a></td>
    </tr>
    <tr>
      <td>EfficientNet V2 Small</td>
      <td><a href="https://arxiv.org/abs/2104.00298">EfficientNetV2</a></td>
    </tr>
    <tr>
      <td>MobileNet V3 Large</td>
      <td><a href="https://arxiv.org/abs/1905.02244">MobileNetV3</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-enable MD060 -->

<!-- markdownlint-enable MD060 -->

> [!TIP]
> Other projects of the Open Edge Platform enable even more tasks and models, check them:
>
> - [Anomalib (Studio)](https://github.com/open-edge-platform/anomalib) → anomaly detection
> - [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio) → robot learning, VLA (Vision-Language-Action)
> - [Instant Learn](https://github.com/open-edge-platform/instant-learn) → visual prompting
> - [OpenVINO™](https://github.com/openvinotoolkit/openvino) - Software toolkit for optimizing and deploying deep learning models.
> - [Model API](https://github.com/open-edge-platform/model_api) - wrapper that simplifies model loading, execution, and data processing for easy inference



## Quick Start

Get Geti running and train your first model in a few minutes. For full instructions and all options, see the
[official documentation](https://docs.geti.intel.com/) and the [application README](application/README.md).

**Minimum recommended setup:** 8 CPU threads, 16 GB RAM, 40 GB free disk. A GPU (Intel® XPU or NVIDIA® CUDA) is
recommended for larger models.

### 1. Run Geti

#### Windows Application
Run Geti as a native Windows application, with prebuilt images for Intel® XPU, NVIDIA® CUDA, and CPU-only environments.

Download the Windows Installer:
- [Download CPU-only version installer](https://storage.geti.intel.com/geti/packages/3.0.0/geti-cpu-3.0.0.msix)
- [Download Intel® XPU version installer](https://storage.geti.intel.com/geti/packages/3.0.0/geti-xpu-3.0.0.msix)
- [Download Nvidia® CUDA version installer](https://storage.geti.intel.com/geti/packages/3.0.0/geti-cuda-3.0.0.msix)

Install Geti Windows application and launch it from the Start menu

#### Docker

Pull a pre-built image for your hardware and launch it:

```bash
docker pull ghcr.io/open-edge-platform/geti-xpu    # modern Intel® CPU/GPU (recommended)
docker pull ghcr.io/open-edge-platform/geti-cuda   # NVIDIA® CUDA platforms
docker pull ghcr.io/open-edge-platform/geti-cpu    # CPU-only (most lightweight)

# Retag the pulled image as `geti-{cpu,xpu,cuda}:latest` for using with `just run-image`
docker tag ghcr.io/open-edge-platform/geti-cpu:latest geti-cpu:latest

just run-image --accelerator xpu                   # launch the application
```

Then open the Geti web application at [**http://localhost:7860**](http://localhost:7860).

For build-from-source options and advanced setup, see the [installation guide](https://docs.geti.intel.com/) and the
[application README](application/README.md).


#### Install natively with Ultralytics YOLO26 models (the latest NMS‑free, edge‑optimized models (Nano / Small / Medium) for object detection and instance segmentation. The integration covers the full model lifecycle: training, inference, quantization, and OpenVINO™ model export.

Linux, WSL (In order to run a script you need to have curl & git installed):
```bash
curl -fsSL https://github.com/open-edge-platform/training_extensions/blob/develop/install.sh | bash 
```

### 2. Train your first model

Once Geti is running, build your first model directly in the web UI:

1. **Create a project** — choose a task (object detection, instance segmentation, or classification) and define your labels.
2. **Upload media** — drag in 20–50 representative images to start.
3. **Annotate** — label your media with the built-in manual and AI-assisted tools.
4. **Train** — pick a recommended architecture and start training; watch progress in the Jobs panel.
5. **Deploy** — build an inference pipeline (source → model → sink) and run predictions in real time, or export an
   OpenVINO™-optimized bundle for the edge.

See [Training your first model](https://docs.geti.intel.com/) for the full walkthrough.

### Use the Python API (`getitune`)

Prefer to work programmatically? Geti's training engine is published on PyPI and can train, optimize, and deploy models
from Python. It requires **Python 3.11–3.14**, **PyTorch 2.10**, **OpenVINO™ 2026.1**, and **NumPy ≥ 2.0**.

```bash
pip install "getitune[cpu]"    # or [xpu] for Intel® GPU, [cuda] for NVIDIA® GPU
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

Geti 3.0 introduces a simplified dataset‑based workflow: datasets must be exported and imported individually, models from 2.x require retraining, project-level migration is replaced by dataset-level transfer, and the REST API and deployment now use the OpenVINO™ Model API — **Please follow the
[migration guidance](https://docs.geti.intel.com/) in the documentation.**

## Documentation

For complete user and developer documentation, visit [**docs.geti.intel.com**](https://docs.geti.intel.com/).

| Component                  | README                                          | Documentation                                                                            |
| -------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **Geti application**       | [application/README.md](application/README.md)  | [docs.geti.intel.com](https://docs.geti.intel.com/)                                       |
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
Stay tuned for further updates soon!

## Disclaimers

FFmpeg is an open source project licensed under LGPL and GPL. See [https://www.ffmpeg.org/legal.html](https://www.ffmpeg.org/legal.html). You are solely responsible for determining if your use of FFmpeg requires any additional licenses. Intel is not responsible for obtaining any such licenses, nor liable for any licensing fees due, in connection with your use of FFmpeg.
