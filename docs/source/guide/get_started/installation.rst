:octicon:`package` Installation
====================================

**************
Prerequisites
**************

The current version of OpenVINO™ Training Extensions was tested in the following environment:

- Ubuntu 20.04
- Python >= 3.10
- [`uv`](https://github.com/astral-sh/uv) for dependency and environment management

**********************************************************
Install OpenVINO™ Training Extensions for users (CUDA/CPU)
**********************************************************

1. Install OpenVINO™ Training Extensions package:

* A local source in development mode

.. tab-set::

    .. tab-item:: PyPI

        .. code-block:: shell

            uv pip install otx

    .. tab-item:: Source

        .. code-block:: shell

            # Clone the training_extensions repository:
            git clone https://github.com/open-edge-platform/training_extensions.git
            cd training_extensions

            # Create a virtual environment with uv
            uv venv .otx
            source .otx/bin/activate

            # Install the package in editable mode with base dependencies
            uv pip install -e .[base]

2. Once the package is installed in the virtual environment, you can use the full
OpenVINO™ Training Extensions command line functionality.

.. code-block:: shell

    otx --help

*************************************************************
Install OpenVINO™ Training Extensions for users (Intel GPUs)
*************************************************************

1. Install OpenVINO™ Training Extensions from source to use Intel XPU functionality:

.. code-block:: shell

    git clone https://github.com/open-edge-platform/training_extensions.git
    cd training_extensions

    uv venv .otx
    source .otx/bin/activate

    uv pip install -e '.[base]' --extra-index-url https://download.pytorch.org/whl/test/xpu

.. note::

    Please refer to the `PyTorch XPU installation guide <https://pytorch.org/docs/stable/notes/get_start_xpu.html>`_
    to install prerequisites and resolve any potential issues.

2. Once installed, use the command-line interface:

.. code-block:: shell

    otx --help

****************************************************
Install OpenVINO™ Training Extensions for developers
****************************************************

1. Create and activate a development virtual environment:

.. code-block:: shell

    # Create a virtual environment using uv
    uv venv venv/otx
    source venv/otx/bin/activate

2. Install tox and generate the development environment:

.. code-block:: shell

    uv pip install tox
    # Replace '310' with another Python version if needed
    tox devenv venv/otx -e unit-test-py310
    source venv/otx/bin/activate

Now you can make code changes, and all modifications will be reflected live in the editable install.

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

To run tests locally, install development dependencies:

.. code-block:: shell

    uv pip install -e '.[dev]'
    pytest tests/

To run integration tests using `tox`:

.. code-block:: shell

    uv pip install tox
    tox -e integration-test-all

.. note::

    The first time `tox` is run, it will create virtual environments and install all required dependencies.
    This may take several minutes before the actual tests begin.

***************
Troubleshooting
***************

1. If you encounter issues with `uv pip`, update uv:

.. code-block:: shell

    pip install --upgrade uv

2. If you're having issues installing `torch` or `mmcv`, check CUDA compatibility with your PyTorch version.
Update your CUDA toolkit and drivers if needed. See `CUDA 11.8 Installer <https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local>`_.

3. If you're behind a proxy server, set your proxy environment variable:

.. code-block:: shell

    export HTTP_PROXY=http://<user>:<password>@<proxy>:<port>
    uv pip install <package>

4. For CLI-related issues, check the help message:

.. code-block:: shell

    otx --help

To see additional messages from `jsonargparse`, enable debug output:

.. code-block:: shell

    export JSONARGPARSE_DEBUG=1  # 0: Off, 1: On
