.. raw:: html

    <div style="margin-bottom:30px;">
    <img src="../../_static/logos/otx-logo-black.png" alt="Logo" width="900" style="display:block;margin:auto;background-color:white;">
    </div>

Introduction
============

**OpenVINO™ Training Extensions** is a low-code transfer learning framework for Computer Vision.

The framework's CLI commands and API allow users to easily train, infer, optimize and export models, even with limited deep learning expertise. OpenVINO™ Training Extensions offers diverse combinations of model architectures, learning methods, and task types based on `PyTorch <https://pytorch.org/>`_ , `Lightning <https://lightning.ai/>`_ and `OpenVINO™ toolkit <https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html>`_.

OpenVINO™ Training Extensions provide `recipe <https://github.com/open-edge-platform/training_extensions/tree/develop/src/otx/recipe>`_ for every supported task type, which consolidates necessary information to build a model. Model configs are validated on various datasets and serve one-stop shop for obtaining the best models in general.

The development team is continuously expanding functionality to simplify the training process — aiming for a workflow where a single CLI command or a short API call is enough to produce accurate, efficient, and robust models ready for integration into your project.

Starting with OTX v2.4.5, we introduced a new repository structure and a more flexible backend concept. We're excited to present support for multiple backends — beginning with the OpenVINO™ backend, while all previous OTX functionality is now organized under the "native" backend.

In the future, we plan to integrate popular third-party libraries such as `Anomalib <https://github.com/open-edge-platform/anomalib>_`, `Transformers <https://huggingface.co/docs/transformers/index>_`, and more — seamlessly integrated into the repository.
This will enable users to train, test, export, and optimize a wide variety of models from different backends using the same CLI commands and unified API, without the need for reimplementation.

|

.. figure:: ../../../utils/images/diagram_otx.png
   :align: center
   :width: 100%

|

************
Key Features
************

OpenVINO™ Training Extensions supports the following computer vision tasks:

- **Classification**, including multi-class, multi-label and hierarchical image classification tasks.
- **Object detection** including rotated bounding box and tiling support
- **Semantic segmentation** including tiling algorithm support
- **Instance segmentation** including tiling algorithm support
- **Anomaly recognition** tasks including anomaly classification, detection and segmentation

OpenVINO™ Training Extensions provide the :doc:`following features <../explanation/additional_features/index>`:

- Native **Intel GPUs (XPU) support**. OpenVINO™ Training Extensions can be installed with XPU support to utilize Intel GPUs for training and testing.
- **Distributed training** to accelerate the training process when using multiple GPUs.
- **Half-precision (FP16) training** to reduce GPU memory usage and allow for larger batch sizes.
- **Class-incremental learning** to add new classes to an existing model without retraining from scratch.
- OpenVINO™ Training Extensions use `Datumaro <https://open-edge-platform.github.io/datumaro/stable/index.html>`_ as the backend for dataset handling. This allows support for many common academic dataset formats per task. More formats will be supported in the future, providing additional flexibility.
- **Multiple backend support** to easily adapt models from third-party implementations into the OpenVINO™ Training Extensions repository.


*********************
Documentation content
*********************

1. :octicon:`light-bulb` **Quick start guide**:

.. grid::
    :gutter: 1

    .. grid-item-card:: :octicon:`package` Installation Guide
        :link: installation
        :link-type: doc
        :text-align: center

        Learn more about how to install OpenVINO™ Training Extensions

    .. grid-item-card:: :octicon:`code-square` API Quick-Guide
        :link: api_tutorial
        :link-type: doc
        :text-align: center

        Learn more about how to use OpenVINO™ Training Extensions Python API.

    .. grid-item-card:: :octicon:`terminal` CLI Guide
        :link: cli_commands
        :link-type: doc
        :text-align: center

        Learn more about how to use OpenVINO™ Training Extensions CLI commands

2. :octicon:`book` **Tutorials**:

.. grid:: 1 2 2 3
    :margin: 1 1 0 0
    :gutter: 1

    .. grid-item-card:: Classification
        :link: ../tutorials/base/how_to_train/classification
        :link-type: doc
        :text-align: center

        Learn how to train a classification model

    .. grid-item-card:: Detection
        :link: ../tutorials/base/how_to_train/detection
        :link-type: doc
        :text-align: center

        Learn how to train a detection model.

    .. grid-item-card:: Instance Segmentation
        :link: ../tutorials/base/how_to_train/instance_segmentation
        :link-type: doc
        :text-align: center

        Learn how to train an instance segmentation model

    .. grid-item-card:: Semantic Segmentation
        :link: ../tutorials/base/how_to_train/semantic_segmentation
        :link-type: doc
        :text-align: center

        Learn how to train a semantic segmentation model

    .. grid-item-card:: Anomaly Task
        :link: ../tutorials/base/how_to_train/anomaly_detection
        :link-type: doc
        :text-align: center

        Learn how to train an anomaly detection model

    .. grid-item-card:: Advanced
        :link: ../tutorials/advanced/index
        :link-type: doc
        :text-align: center

        Learn how to use advanced features of OpenVINO™ Training Extensions

3. **Explanation section**:

This section consists of an algorithms explanation and describes additional features that are supported by OpenVINO™ Training Extensions.
:ref:`Algorithms <algo_section_ref>` section includes a description of all supported algorithms:

   1. Explanation of the task and main supervised training pipeline.
   2. Description of the supported datasets formats for each task.
   3. Available recipes and models.

:ref:`Additional Features <features_section_ref>` section consists of:

   1. Overview of model optimization algorithms.
   2. Auto-configuration algorithm to select the most appropriate training pipeline for a given dataset.
   3. Tiling algorithm to detect small objects in large images.
   4. explainable AI algorithms to visualize the model's decision-making process.
   5. Additional useful features like configurable input size, class incremental learning, and adaptive training.

4. **Reference**:

This section gives an overview of the OpenVINO™ Training Extensions code base, where source code for Entities, classes and functions can be found.

5. **Release Notes**:

This section contains descriptions of current and previous releases.
