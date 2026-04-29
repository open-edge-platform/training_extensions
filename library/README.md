<div align="center">

# Geti Library - getitune

---

[Key Features](#key-features) •
[Installation](https://open-edge-platform.github.io/training_extensions/latest/guide/get_started/installation.html) •
[Documentation](https://open-edge-platform.github.io/training_extensions/latest/index.html) •
[License](#license)

[![PyPI](https://img.shields.io/pypi/v/getitune)](https://pypi.org/project/getitune)

<!-- markdownlint-disable MD042 -->

[![python](https://img.shields.io/badge/python-3.11%2B-green)]()
[![pytorch](https://img.shields.io/badge/pytorch-2.7%2B-orange)]()
[![openvino](https://img.shields.io/badge/openvino-2025.2-purple)]()

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
The API & CLI commands of the framework allows users to train, infer, optimize and deploy models easily and quickly even with low expertise in the deep learning field.
It offers diverse combinations of model architectures, learning methods, and task types based on [PyTorch](https://pytorch.org) and [OpenVINO™ toolkit](https://software.intel.com/en-us/openvino-toolkit).

The library provides a "recipe" for every supported task type, which consolidates necessary information to build a model.
Model templates are validated on various datasets and serve one-stop shop for obtaining the best models in general.

### Key Features

The Geti™ library supports the following computer vision tasks:

- **Classification**, including multi-class, multi-label and hierarchical image classification tasks.
- **Object detection** including rotated bounding box and tiling support
- **Semantic segmentation** including tiling algorithm support
- **Instance segmentation** including tiling algorithm support
- **Anomaly recognition** tasks including anomaly classification, detection and segmentation

Its features include:

- Native **Intel GPUs (XPU) support**. The package can be installed with XPU support to utilize Intel GPUs for training and testing.
- [Datumaro](https://open-edge-platform.github.io/datumaro/stable/index.html) data frontend, with support for the most popular dataset formats for each task. We are constantly working to extend supported formats to give more freedom of datasets format choice.
- **Distributed training** to accelerate the training process when you have multiple GPUs
- **Mixed-precision training** to save GPUs memory and use larger batch sizes
- **Class incremental learning** to add new classes to the existing model
- **Model deployment** to OpenVINO™ IR and ONNX formats and inference with [OpenVINO™ ModelAPI](https://github.com/open-edge-platform/model_api)
- **Multiple backend support** to easily adapt models from third-party implementations into the Geti™ repository.

---

## Installation

Please refer to the [installation guide](https://open-edge-platform.github.io/training_extensions/latest/guide/get_started/installation.html).
If you want to make changes to the library, then a local installation is recommended.

<details>
<summary>Install from PyPI</summary>
Installing the library with pip or uv is the easiest way to get started with getitune.

```bash
# Without GPU support (CPU only)
pip install getitune[cpu]

# With Intel GPU support (XPU)
pip install getitune[xpu]

# With NVIDIA GPU support (CUDA)
pip install getitune[cuda]
```

</details>

<details>
<summary>Install from source</summary>
To install from source, you need to clone the repository and install the library with pip or uv.
It is recommended to use a virtual environment to avoid conflicts with other packages.

```bash
# Clone the repository
git clone https://github.com/open-edge-platform/training_extensions.git
cd training_extensions

# Install (optional: pass the '-e' flag for editable mode
# If you have an Intel GPU, use 'xpu' to enable support.
# If you have an NVIDIA GPU, use 'cuda' to enable support.
pip install -e .[cpu]
```

</details>

---

## Quick-Start

The Geti™ library supports both API and CLI-based training. The API is more flexible and allows for more customization, while the CLI training utilizes command line interfaces, and might be easier for those who would like to use `getitune` off-the-shelf.

For the CLI, the commands below provide subcommands, how to use each subcommand, and more:

```bash
# See available subcommands
getitune --help

# Print help messages from the train subcommand
getitune train --help

# Print help messages for more details
getitune train --help -v   # Print required parameters
getitune train --help -vv  # Print all configurable parameters
```

You can find details with examples in the [CLI Guide](https://open-edge-platform.github.io/training_extensions/latest/guide/get_started/cli_commands.html). and [API Quick-Guide](https://open-edge-platform.github.io/training_extensions/latest/guide/get_started/api_tutorial.html).

Below is how to train with auto-configuration, which is provided to users with datasets and tasks:

<details>
<summary>API Usage</summary>

```python
from getitune.engine import create_engine

# get all the available recipes for all tasks
from getitune.backend.lightning.cli.utils import list_models
model_lists = list_models(print_table=True)

# instantiate native getitune engine with atss model for object detection
engine = create_engine(data="path/to/dataset/root", model="src/getitune/recipe/detection/atss_mobilenetv2.yaml")
engine.train()
engine.test()
exported_path = engine.export()

# by default all artifacts are stored in "./getitune-workspace" directory.
# working directory can be specified
engine = create_engine(data="path/to/dataset/root", model="src/getitune/recipe/detection/atss_mobilenetv2.yaml", work_dir="my_workdir")


# openvino backend is used to validate and optimize exported OpenVINO IR models
ov_engine = create_engine(data="path/to/dataset/root", model=exported_path)
ov_engine.test()
ov_engine.optimize()

```

For more examples, see documentation: [API Quick-Guide](https://open-edge-platform.github.io/training_extensions/latest/guide/get_started/api_tutorial.html)

</details>

<details>
<summary> CLI Usage </summary>

```bash
# get all recipes list
getitune find

# getitune train
getitune train --config src/getitune/recipe/detection/atss_mobilenetv2.yaml --data_root data/wgisd

# by default, working directory is "./getitune-workspace". It can be specified with "--work_dir" parameter
getitune test --config src/getitune/recipe/detection/atss_mobilenetv2.yaml --data_root data/wgisd --checkpoint getitune-workspace/.latest/train/best_checkpoint.ckpt
getitune export --config src/getitune/recipe/detection/atss_mobilenetv2.yaml --data_root data/wgisd --checkpoint getitune-workspace/.latest/train/best_checkpoint.ckpt

# or using work_dir
getitune test --work_dir getitune-workspace/.latest/train
getitune export --work_dir getitune-workspace/.latest/train

# directly from working directory
cd getitune-workspace
getitune test
getitune export

```

For more examples, see documentation: [CLI Guide](https://open-edge-platform.github.io/training_extensions/latest/guide/get_started/cli_commands.html)

</details>

In addition to the examples above, please refer to the documentation for tutorials on using custom models, training parameter overrides, and [tutorial per task types](https://open-edge-platform.github.io/training_extensions/latest/guide/tutorials/base/how_to_train/index.html), etc.

---

## License

The Geti™ Library (`getitune`) is licensed under [Apache License Version 2.0](https://github.com/open-edge-platform/training_extensions/blob/develop/LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

---
