[BETA] Enable torch.compile
============================

.. warning::
    The support for `torch.compile` is in beta. Not all models are compatible with this feature.

Overview
--------

Geti Library now integrates the `torch.compile` feature from PyTorch, allowing users to optimize their models for better performance.
This feature compiles the model's operations into optimized lower-level code, which can significantly improve execution speed and reduce memory usage.

Benefits of torch.compile
-------------------------

- **Performance Optimization**: Compiled models run faster by executing optimized low-level operations.
- **Reduced Memory Footprint**: Optimized models can use less memory, which is beneficial for deploying models on resource-constrained devices.
For more information on the benefits of `torch.compile`, refer to the official `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.compile.html>`_.

How to Use torch.compile in Geti Library
----------------------------------------------------------

**Prepare Model**: Ensure that model is compatible with `torch.compile`. When building the model, give the `torch_compile` option `True`.

.. tab-set::

    .. tab-item:: API

        .. code-block:: python

            from getitune.backend.lightning.models import VisionTransformerMulticlassCls

            model = VisionTransformerMulticlassCls(..., torch_compile=True)

    .. tab-item:: CLI

        .. code-block:: bash

            (getitune) ...$ getitune train ... --model.torch_compile True
