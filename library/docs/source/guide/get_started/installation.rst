:octicon:`package` Installation
====================================

**************
Prerequisites
**************

The current version of OpenVINO™ Training Extensions was tested in the following environment:

- Ubuntu 20.04
- Python >= 3.11
- [`uv`](https://github.com/astral-sh/uv) for dependency and environment management

.. note::

    To enable efficient execution of multiple models, we increase the ONEDNN_PRIMITIVE_CACHE_CAPACITY environment variable from its default value to 10000.
    For more information, refer to the `Primitive cache <https://www.intel.com/content/www/us/en/docs/onednn/developer-guide-reference/2024-1/primitive-cache-002.html>`_.

*****************
Install from pypi
*****************

The easiest way to install OpenVINO™ Training Extensions is via PyPI (`otx <https://pypi.org/project/otx>`_ package).

Select the extra dependencies according to your hardware:

.. tab-set::

    .. tab-item:: CPU only

        .. code-block:: shell

            pip install otx[cpu]

    .. tab-item:: Intel GPU (XPU)

        .. code-block:: shell

            pip install otx[xpu]

    .. tab-item:: Nvidia GPU (CUDA)

        .. code-block:: shell

            pip install otx[cuda]

.. note::

    It is always recommended to use a virtual environment to avoid conflicts with other packages.

After the installation, you can verify it works with:

.. code-block:: shell

    otx --help


*******************
Install from source
*******************

If you want to install OpenVINO™ Training Extensions from source, you have to first clone the repository,
then install the package.

.. tab-set::

    .. tab-item:: CPU only

        .. code-block:: shell

            git clone https://github.com/open-edge-platform/training_extensions.git
            cd training_extensions/library
            pip install .[cpu]

    .. tab-item:: Intel GPU (XPU)

        .. code-block:: shell

            git clone https://github.com/open-edge-platform/training_extensions.git
            cd training_extensions/library
            pip install .[xpu]

    .. tab-item:: Nvidia GPU (CUDA)

        .. code-block:: shell

            git clone https://github.com/open-edge-platform/training_extensions.git
            cd training_extensions/library
            pip install .[cuda]


**************
For developers
**************

For developers, it is recommended to use `uv` to automatically manage virtual environments and dependencies.

.. code-block::

    # Clone the repo
    git clone https://github.com/open-edge-platform/training_extensions.git
    cd training_extensions/library

    # Create a virtual environment with uv and sync dependencies
    uv sync --extra cpu  # or 'xpu' or 'cuda', depending on your hardware

If you plan to edit the documentation, also install the `docs` extra.


*****************************************************
Install OpenVINO™ Training Extensions by using Docker
*****************************************************

1. By executing the following commands, it will build two
Docker images: ``otx:${OTX_VERSION}-cuda`` and ``otx:${OTX_VERSION}-cuda-pretrained-ready``.

.. code-block:: shell

    git clone https://github.com/open-edge-platform/training_extensions.git
    cd docker
    ./build.sh

2. After that, you can check whether the
images are built correctly such as

.. code-block:: shell

    docker image ls | grep otx

Example output:

.. code-block:: shell

    otx                                           2.0.0-cuda-pretrained-ready   4f3b5f98f97c   3 minutes ago   14.5GB
    otx                                           2.0.0-cuda                    8d14caccb29a   8 minutes ago   10.4GB

``otx:${OTX_VERSION}-cuda`` is a minimal Docker image with CUDA support.
``otx:${OTX_VERSION}-cuda-pretrained-ready`` includes pre-trained models on top of the base image.

*********
Run tests
*********

To run tests locally, just call `pytest`. `uv` will automatically sync all necessary dependencies.

.. code-block:: shell

    uv run pytest tests

To run integration tests:

.. code-block:: shell

    uv run pytest tests/integration

.. note::

    Integration tests may be slow because they involve downloading datasets and simulating training rounds.
    It is recommended to run them on a platform with GPU/XPU support.

***************
Troubleshooting
***************

1. If you encounter issues with `uv`, update it:

.. code-block:: shell

    uv self update

2. If you're having issues installing `torch` or `mmcv`, check CUDA compatibility with your PyTorch version.
Update your CUDA toolkit and drivers if needed. See `CUDA 11.8 Installer <https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local>`_.

3. If you're behind a proxy server, set your proxy environment variable:

.. code-block:: shell

    export HTTP_PROXY=http://<user>:<password>@<proxy>:<port>

4. For CLI-related issues, check the help message:

.. code-block:: shell

    otx --help

To see additional messages from `jsonargparse`, enable debug output:

.. code-block:: shell

    export JSONARGPARSE_DEBUG=1  # 0: Off, 1: On
