<!-- markdownlint-disable MD013 MD033 MD041 MD042 -->

<div align="center">

<img src="assets/geti-header.png" alt="Geti™ - Build and deploy computer vision AI models with minimal effort and data">

Full-stack web application to build and deploy computer vision AI models, powered by the [getitune](../library) library.

[![python](https://img.shields.io/badge/python-3.13-green)]()
[![pytorch](https://img.shields.io/badge/pytorch-2.10-orange)]()
[![openvino](https://img.shields.io/badge/openvino-2026.1-purple)]()

</div>

## Quick start

There are several ways to run Geti, choose the method that best fits your workflow:

- **Docker (recommended)** [[instructions]](#run-with-docker) - download and run one of the pre-built Docker images, or build one yourself
- **MSIX App (Windows)** [[instructions]](#install-msix-app-windows) - install as a desktop application
- **Run from Source (for development)** [[instructions]](#run-from-source-for-development) - run the server and the UI as standalone components

### Run with Docker

The easiest and most portable way to run Geti is through Docker.
We provide [pre-built images](#option-1-download-the-image) for Intel® XPU and NVIDIA® CUDA platforms, or you can [build](#option-2-build-the-image) your own image from source.

**Prerequisites** (on the host system):

- Just v1.46+ [[docs]](https://github.com/casey/just)
- Docker v29+ [[docs]](https://docs.docker.com/)
- (Only for Intel® XPU) the latest driver suitable with your HW [[docs]](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-11.html)
- (Only for NVIDIA GPU) NVIDIA driver and the NVIDIA Container Toolkit [[docs]](https://www.nvidia.com/Download/index.aspx)

#### (Option 1) Download the image

> [!WARNING]
> The official Docker images for Geti 3.x have not been released yet.
> The only way to run Geti at the moment is to build the image from source (see [Option 2](#option-2-build-the-image)).

Choose the most suitable image for your system.

If you have a modern Intel® CPU or GPU, the **XPU** image is the recommended choice to fully exploit its AI capabilities.

```
docker pull ghcr.io/open-edge-platform/geti-xpu
```

If you have a **CUDA**-enabled platform, choose this image instead:

```
docker pull ghcr.io/open-edge-platform/geti-cuda
```

Even if you don't have a compatible GPU, you can still train models with the CPU-only image.
This is also the most lightweight choice:

```
docker pull ghcr.io/open-edge-platform/geti-cpu
```

#### (Option 2) Build the image

Here is how you can build a Docker image from scratch, by means of the application `Dockerfile`.

From the `application` directory:

```bash
# Build for Intel® XPU (recommended)
just build-image --accelerator xpu
```

The above command builds an image optimized for modern Intel® hardware.
If you have an Intel® GPU (discrete or integrated), this is the recommended configuration for best performance.
Alternatively, you can build with support for NVIDIA GPUs (`--accelerator cuda`) or with CPU-only support (`--accelerator cpu`).

Run `just --usage build-image` to see all build options.

#### Run the image

Once you have downloaded or built the Geti image, use the `run-image` command to launch the application.

```bash
# Run the image build with Intel® XPU support
just run-image --accelerator xpu
```

If you built the image with a different accelerator, make sure to specify the same one when running.

For a full list of runtime options, run `just --usage run-image`.

After the container starts, you can access the Geti web application at [**http://localhost:7860**](http://localhost:7860) (assuming default settings).

### Run from source (for development)

For development purposes, you can run the Geti server and UI as standalone components without Docker.

**Prerequisites:**

- Just v1.46+ [[docs]](https://github.com/casey/just)
- (Only for Intel® XPU) the latest driver suitable with your HW [[docs]](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-11.html)
- (Only for NVIDIA GPU) NVIDIA driver and the NVIDIA Container Toolkit [[docs]](https://www.nvidia.com/Download/index.aspx)
- Node.js v24.2+ [[docs]](https://nodejs.org/)

To run the server, use the `run-server` command after initializing the environment with `venv`:

```bash
# From the repo root
cd application/backend

# Initialize the environment with the appropriate accelerator support (cpu, xpu, or cuda)
just venv --accelerator xpu

# Run the server
just run-server
```

Run `just --usage run-image` for a full list of options for running the server. Notably, by passing the option
`--setup-demo`, the application will be pre-populated with demo data, including sample datasets and pre-trained models.

Then, build and launch the UI in a separate terminal:

```bash
# From the repo root
cd application/ui

# Install dependencies and build
npm install
npm run build

# Start the UI
npm run preview
```

After the UI starts, you can access the Geti web application at [**http://localhost:3000**](http://localhost:3000) (assuming default settings).

### Install MSIX App (Windows)

> [!WARNING]
> The MSIX App for Geti 3.x has not been released yet.

### Generate API spec

The OpenAPI specification for the Geti REST API can be generated with the `generate-api-spec` command:

```bash
# From the repo root
cd application/backend

# Generate the OpenAPI spec and save it to a custom location
just gen-api-spec --output-path="openapi.json"
```

## Supported models

Geti™ supports a variety of state-of-the-art model architectures for image classification, object detection and
instance segmentation tasks. Would you like to see a specific model added to the list? Let us know by opening a
[GitHub issue](https://github.com/open-edge-platform/training_extensions/issues)!

### Image Classification

| Model Architecture    | Paper                                              |
| --------------------- | -------------------------------------------------- |
| DeiT Tiny             | [DeiT](https://arxiv.org/abs/2012.12877)           |
| DINOv2 Small          | [DINOv2](https://arxiv.org/abs/2304.07193)         |
| EfficientNet B0 / B3  | [EfficientNet](https://arxiv.org/abs/1905.11946)   |
| EfficientNet V2 Small | [EfficientNetV2](https://arxiv.org/abs/2104.00298) |
| MobileNet V3 Large    | [MobileNetV3](https://arxiv.org/abs/1905.02244)    |

### Object Detection

| Model Architecture     | Paper                                                                                      |
| ---------------------- | ------------------------------------------------------------------------------------------ |
| D-FINE M / L / X       | [D-FINE](https://arxiv.org/abs/2410.13842)                                                 |
| DINOv3 DETR S / M / L  | [DINOv3](https://arxiv.org/abs/2508.10104) + [DETR](https://arxiv.org/abs/2005.12872)      |
| MobileNet V2 ATSS      | [MobileNetV2](https://arxiv.org/abs/1801.04381) + [ATSS](https://arxiv.org/abs/1912.02424) |
| MobileNet V2 SSD       | [MobileNetV2](https://arxiv.org/abs/1801.04381) + [SSD](https://arxiv.org/abs/1512.02325)  |
| RF-DETR S / M / L      | [RF-DETR](https://arxiv.org/abs/2511.09554)                                                |
| RT-DETR R50            | [RT-DETR](https://arxiv.org/abs/2304.08069)                                                |
| YOLOX Tiny / S / L / X | [YOLOX](https://arxiv.org/abs/2107.08430)                                                  |

### Instance Segmentation

| Model Architecture        | Paper                                                                                                 |
| ------------------------- | ----------------------------------------------------------------------------------------------------- |
| RTMDet Tiny               | [RTMDet](https://arxiv.org/abs/2212.07784)                                                            |
| Mask-RCNN EfficientNet B2 | [EfficientNet](https://arxiv.org/abs/1905.11946) + [Mask R-CNN](https://arxiv.org/abs/1703.06870)     |
| Mask-RCNN ResNet50        | [ResNet](https://arxiv.org/abs/1512.03385) + [Mask R-CNN](https://arxiv.org/abs/1703.06870)           |
| Mask-RCNN Swin-T          | [Swin Transformer](https://arxiv.org/abs/2103.14030) + [Mask R-CNN](https://arxiv.org/abs/1703.06870) |
| RF-DETR S / M / L         | [RF-DETR](https://arxiv.org/abs/2511.09554)                                                           |
