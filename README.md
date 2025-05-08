<div align="center">

# OpenVINO™ Training Extensions

---

[Key Features](#key-features) •
[Installation](https://open-edge-platform.github.io/training_extensions/latest/guide/get_started/installation.html) •
[Documentation](https://open-edge-platform.github.io/training_extensions/latest/index.html) •
[License](#license)

[![PyPI](https://img.shields.io/pypi/v/otx)](https://pypi.org/project/otx)

<!-- markdownlint-disable MD042 -->

[![python](https://img.shields.io/badge/python-3.10%2B-green)]()
[![pytorch](https://img.shields.io/badge/pytorch-2.7%2B-orange)]()
[![openvino](https://img.shields.io/badge/openvino-2025.1-purple)]()

<!-- markdownlint-enable  MD042 -->

[![Codecov](https://codecov.io/gh/open-edge-platform/training_extensions/branch/develop/graph/badge.svg?token=9HVFNMPFGD)](https://codecov.io/gh/open-edge-platform/training_extensions)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/open-edge-platform/training_extensions/badge)](https://securityscorecards.dev/viewer/?uri=github.com/open-edge-platform/training_extensions)
[![Pre-Merge Test](https://github.com/open-edge-platform/training_extensions/actions/workflows/pre_merge.yaml/badge.svg)](https://github.com/open-edge-platform/training_extensions/actions/workflows/pre_merge.yaml)
[![Build Docs](https://github.com/open-edge-platform/training_extensions/actions/workflows/docs.yaml/badge.svg)](https://github.com/open-edge-platform/training_extensions/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://static.pepy.tech/personalized-badge/otx?period=total&units=international_system&left_color=grey&right_color=green&left_text=PyPI%20Downloads)](https://pepy.tech/project/otx)

---

</div>

## Introduction

OpenVINO™ Training Extensions is a low-code transfer learning framework for Computer Vision.
The API & CLI commands of the framework allows users to train, infer, optimize and deploy models easily and quickly even with low expertise in the deep learning field.
OpenVINO™ Training Extensions offers diverse combinations of model architectures, learning methods, and task types based on [PyTorch](https://pytorch.org) and [OpenVINO™ toolkit](https://software.intel.com/en-us/openvino-toolkit).

OpenVINO™ Training Extensions provides a "recipe" for every supported task type, which consolidates necessary information to build a model.
Model templates are validated on various datasets and serve one-stop shop for obtaining the best models in general.

Furthermore, OpenVINO™ Training Extensions provides automatic configuration for ease of use.
The framework will analyze your dataset and identify the most suitable model and figure out the best input size setting and other hyper-parameters.
The development team is continuously extending this [Auto-configuration](https://open-edge-platform.github.io/training_extensions/latest/guide/explanation/additional_features/auto_configuration.html) functionalities to make training as simple as possible so that single CLI command can obtain accurate, efficient and robust models ready to be integrated into your project.

### Key Features

OpenVINO™ Training Extensions supports the following computer vision tasks:

- **Classification**, including multi-class, multi-label and hierarchical image classification tasks.
- **Object detection** including rotated bounding box and tiling support
- **Semantic segmentation** including tiling algorithm support
- **Instance segmentation** including tiling algorithm support
- **Anomaly recognition** tasks including anomaly classification, detection and segmentation

OpenVINO™ Training Extensions provides the following usability features:

- Native **Intel GPUs (XPU) support**. OpenVINO™ Training Extensions can be installed with XPU support to utilize Intel GPUs for training and testing.
- [Auto-configuration](https://open-edge-platform.github.io/training_extensions/latest/guide/explanation/additional_features/auto_configuration.html). OpenVINO™ Training Extensions analyzes provided dataset and selects the proper task and model to provide the best accuracy/speed trade-off.
- [Datumaro](https://open-edge-platform.github.io/datumaro/stable/index.html) data frontend: OpenVINO™ Training Extensions supports the most common academic field dataset formats for each task. We are constantly working to extend supported formats to give more freedom of datasets format choice.
- **Distributed training** to accelerate the training process when you have multiple GPUs
- **Mixed-precision training** to save GPUs memory and use larger batch sizes
- **Class incremental learning** to add new classes to the existing model
- **Model deployment** to OpenVINO™ IR and ONNX formats and inference with [OpenVINO™ ModelAPI](https://github.com/open-edge-platform/model_api)

---

## Installation

Please refer to the [installation guide](https://open-edge-platform.github.io/training_extensions/latest/guide/get_started/installation.html).
If you want to make changes to the library, then a local installation is recommended.

<details>
<summary>Install from PyPI</summary>
Installing the library with pip is the easiest way to get started with otx.

```bash
pip install otx[base]
```

For Intel GPUs users:

```bash
pip install otx[base] --extra-index-url https://download.pytorch.org/whl/test/xpu
```

</details>

<details>
<summary>Install from source</summary>
To install from source, you need to clone the repository and install the library using pip via editable mode.

```bash
# Use of virtual environment is highy recommended
# Using conda
yes | conda create -n otx_env python=3.10
conda activate otx_env

# Or using your favorite virtual environment
# ...

# Clone the repository and install in editable mode
git clone https://github.com/open-edge-platform/training_extensions.git
cd training_extensions
pip install -e .[base]  # to run otx on Intel GPUs: pip install -e .[base] --extra-index-url https://download.pytorch.org/whl/test/xpu
```

</details>

---

## Quick-Start

OpenVINO™ Training Extensions supports both API and CLI-based training. The API is more flexible and allows for more customization, while the CLI training utilizes command line interfaces, and might be easier for those who would like to use OpenVINO™ Training Extensions off-the-shelf.

For the CLI, the commands below provide subcommands, how to use each subcommand, and more:

```bash
# See available subcommands
otx --help

# Print help messages from the train subcommand
otx train --help

# Print help messages for more details
otx train --help -v   # Print required parameters
otx train --help -vv  # Print all configurable parameters
```

You can find details with examples in the [CLI Guide](https://open-edge-platform.github.io/training_extensions/latest/guide/get_started/cli_commands.html). and [API Quick-Guide](https://open-edge-platform.github.io/training_extensions/latest/guide/get_started/api_tutorial.html).

Below is how to train with auto-configuration, which is provided to users with datasets and tasks:

<details>
<summary>Training via API</summary>

```python
# Training with Auto-Configuration via Engine
from otx.engine import Engine

engine = Engine(data_root="data/wgisd", task="DETECTION")
engine.train()
```

For more examples, see documentation: [API Quick-Guide](https://open-edge-platform.github.io/training_extensions/latest/guide/get_started/api_tutorial.html)

</details>

<details>
<summary>Training via CLI</summary>

```bash
otx train --data_root data/wgisd --task DETECTION
```

For more examples, see documentation: [CLI Guide](https://open-edge-platform.github.io/training_extensions/latest/guide/get_started/cli_commands.html)

</details>

In addition to the examples above, please refer to the documentation for tutorials on using custom models, training parameter overrides, and [tutorial per task types](https://open-edge-platform.github.io/training_extensions/latest/guide/tutorials/base/how_to_train/index.html), etc.

---

### Release History

Please refer to the [CHANGELOG.md](CHANGELOG.md)

---

## License

OpenVINO™ Toolkit is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

---

## Issues / Discussions

Please use [Issues](https://github.com/open-edge-platform/training_extensions/issues/new/choose) tab for your bug reporting, feature requesting, or any questions.

---

## Disclaimer

Intel is committed to respecting human rights and avoiding complicity in human rights abuses.
See Intel's [Global Human Rights Principles](https://www.intel.com/content/www/us/en/policy/policy-human-rights.html).
Intel's products and software are intended only to be used in applications that do not cause or contribute to a violation of an internationally recognized human right.

---

## Contributing

For those who would like to contribute to the library, see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

Thank you! we appreciate your support!

<a href="https://github.com/open-edge-platform/training_extensions/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=open-edge-platform/training_extensions" />
</a>

---
