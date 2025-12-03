Object Detection
================

Object detection is a computer vision task where it's needed to locate objects, finding their bounding boxes coordinates together with defining class.
The input is an image, and the output is a pair of coordinates for bounding box corners and a class number for each detected object.

The common approach to building object detection architecture is to take a feature extractor (backbone), that can be inherited from the classification task.
Then goes a head that calculates coordinates and class probabilities based on aggregated information from the image.
Additionally, some architectures use `Feature Pyramid Network (FPN) <https://arxiv.org/abs/1612.03144>`_ to transfer and process feature maps from backbone to head and called neck.

*******************
Training Pipeline
*******************

OTX supports various training configurations that can be customized per model. The default settings vary by model architecture
and are defined in the respective recipe files. To see the exact configuration for a specific model, run:

.. code-block:: shell

    (otx) ...$ otx train --config <recipe_path> --print_config

.. _od_supervised_pipeline:

Common training components include:

- ``Augmentations``: Data augmentation strategies vary by model. Common techniques include random crop, rotation, affine transformations, color/brightness distortions, and advanced techniques like Mosaic and MixUp.

- ``Optimizer``: Model-specific optimizers are used:
    - **AdamW**: Used by transformer-based models (RT-DETR, D-FINE, DEIMv2) with learning rates typically in the range of 1e-4 to 5e-4.
    - **SGD**: Used by CNN-based models (YOLOX, SSD, ATSS) with momentum ~0.9 and weight decay ~1e-4.

- ``Learning rate schedule``: `ReduceLROnPlateau <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html>`_ is commonly used for dataset-agnostic training. It reduces the learning rate when the validation metric plateaus. Many models also use warmup periods at the start of training.

- ``Loss function``: Loss functions are architecture-specific:
    - **Traditional detectors** (SSD, ATSS): `Generalized IoU Loss <https://giou.stanford.edu/>`_ for localization and `FocalLoss <https://arxiv.org/abs/1708.02002>`_ for classification.
    - **DETR-based models** (RT-DETR, D-FINE, DEIMv2): Hungarian matching with combined classification, L1 box, and GIoU losses.

- ``Additional training techniques``:
    - ``Early stopping``: Prevents overfitting by stopping training when validation metrics stop improving.
    - ``Backbone pretraining``: Most models use pretrained backbones (ImageNet, DINOv2/DINOv3) for better feature extraction.
    - ``Multi-scale training``: Optional technique to improve robustness to different object sizes.

.. note::

    Training configurations are fully customizable. Override any setting via command line or by creating a custom recipe file.
    See the :doc:`configuration guide <../../../tutorials/base/how_to_train/detection>` for details.


**************
Dataset Format
**************

At the current point we support `COCO <https://cocodataset.org/#format-data>`_ and
`Pascal-VOC <https://open-edge-platform.github.io/datumaro/stable/docs/data-formats/formats/pascal_voc.html>`_ dataset formats.
Learn more about the formats by following the links above. Here is an example of expected format for COCO dataset:

.. code::

  ├── annotations/
      ├── instances_train.json
      ├── instances_val.json
      └── instances_test.json
  ├──images/
      (Split is optional)
      ├── train
      ├── val
      └── test

.. note::

    Please, refer to our :doc:`dedicated tutorial <../../../tutorials/base/how_to_train/detection>` for more information how to train, validate and optimize detection models.

******
Models
******

We support the following ready-to-use model recipes:

.. note::

    For the most up-to-date list of available models, run ``otx find --task DETECTION``.

Transformer-based Models (DETR Family)
--------------------------------------

These models use the Detection Transformer (DETR) paradigm with end-to-end object detection. They eliminate the need for hand-designed components like anchor boxes and non-maximum suppression.

+-------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| Recipe                                                                                                                        | Name                | Complexity (GFLOPs) | Model size (MB) |
+===============================================================================================================================+=====================+=====================+=================+
| `deimv2_s <https://github.com/open-edge-platform/training_extensions/blob/develop/library/src/otx/recipe/detection/deimv2_s.yaml>`_   | DEIMv2-S            | ~15                 | ~25             |
+-------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| `deimv2_m <https://github.com/open-edge-platform/training_extensions/blob/develop/library/src/otx/recipe/detection/deimv2_m.yaml>`_   | DEIMv2-M            | ~25                 | ~35             |
+-------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| `deimv2_l <https://github.com/open-edge-platform/training_extensions/blob/develop/library/src/otx/recipe/detection/deimv2_l.yaml>`_   | DEIMv2-L            | ~50                 | ~60             |
+-------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| `deimv2_x <https://github.com/open-edge-platform/training_extensions/blob/develop/library/src/otx/recipe/detection/deimv2_x.yaml>`_   | DEIMv2-X            | ~80                 | ~90             |
+-------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| `deim_dfine_m <https://github.com/open-edge-platform/training_extensions/blob/develop/library/src/otx/recipe/detection/deim_dfine_m.yaml>`_ | DEIM-DFine-M   | ~34                 | ~52             |
+-------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| `deim_dfine_l <https://github.com/open-edge-platform/training_extensions/blob/develop/library/src/otx/recipe/detection/deim_dfine_l.yaml>`_ | DEIM-DFine-L   | ~91                 | ~124            |
+-------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| `deim_dfine_x <https://github.com/open-edge-platform/training_extensions/blob/develop/library/src/otx/recipe/detection/deim_dfine_x.yaml>`_ | DEIM-DFine-X   | ~202                | ~240            |
+-------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| `dfine_x <https://github.com/open-edge-platform/training_extensions/blob/develop/library/src/otx/recipe/detection/dfine_x.yaml>`_     | D-Fine X            | 202.5               | 240             |
+-------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| `rtdetr_18 <https://github.com/open-edge-platform/training_extensions/blob/develop/library/src/otx/recipe/detection/rtdetr_18.yaml>`_ | RT-DETR-18          | ~60                 | ~80             |
+-------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| `rtdetr_50 <https://github.com/open-edge-platform/training_extensions/blob/develop/library/src/otx/recipe/detection/rtdetr_50.yaml>`_ | RT-DETR-50          | ~136                | ~170            |
+-------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+

**DEIM Family Models:**

- `DEIM-DFine <https://github.com/ShihuaHuang95/DEIM>`_ (v1): Uses HGNetV2 backbone with D-FINE decoder. Available in M/L/X variants with increasing accuracy.
- `DEIMv2 <https://github.com/ShihuaHuang95/DEIM>`_: An improved version that combines DINOv3/ViT backbones with an efficient DETR decoder:
    - **DEIMv2-S/M**: Use lightweight ViT-Tiny backbones, ideal for edge deployment.
    - **DEIMv2-L/X**: Use DINOv3 (ViT-S) backbones with self-supervised pretraining for higher accuracy.

CNN-based Models
----------------

Traditional CNN-based detectors with anchor-based or anchor-free designs.

+-------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| Recipe                                                                                                                        | Name                | Complexity (GFLOPs) | Model size (MB) |
+===============================================================================================================================+=====================+=====================+=================+
| `yolox_tiny <https://github.com/open-edge-platform/training_extensions/blob/develop/library/src/otx/recipe/detection/yolox_tiny.yaml>`_       | YOLOX-TINY          | 6.5                 | 20.4            |
+-------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| `yolox_s <https://github.com/open-edge-platform/training_extensions/blob/develop/library/src/otx/recipe/detection/yolox_s.yaml>`_             | YOLOX-S             | 33.5                | 46              |
+-------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| `yolox_l <https://github.com/open-edge-platform/training_extensions/blob/develop/library/src/otx/recipe/detection/yolox_l.yaml>`_             | YOLOX-L             | 194.6               | 207             |
+-------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| `yolox_x <https://github.com/open-edge-platform/training_extensions/blob/develop/library/src/otx/recipe/detection/yolox_x.yaml>`_             | YOLOX-X             | 352.4               | 378             |
+-------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| `ssd_mobilenetv2 <https://github.com/open-edge-platform/training_extensions/blob/develop/library/src/otx/recipe/detection/ssd_mobilenetv2.yaml>`_         | SSD-MobileNetV2     | 9.4                 | 7.6             |
+-------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| `atss_mobilenetv2 <https://github.com/open-edge-platform/training_extensions/blob/develop/library/src/otx/recipe/detection/atss_mobilenetv2.yaml>`_       | ATSS-MobileNetV2    | 20.6                | 9.1             |
+-------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| `atss_resnext101 <https://github.com/open-edge-platform/training_extensions/blob/develop/library/src/otx/recipe/detection/atss_resnext101.yaml>`_         | ATSS-ResNeXt101     | 434.8               | 344             |
+-------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+
| `rtmdet_tiny <https://github.com/open-edge-platform/training_extensions/blob/develop/library/src/otx/recipe/detection/rtmdet_tiny.yaml>`_     | RTMDet-Tiny         | ~8                  | ~15             |
+-------------------------------------------------------------------------------------------------------------------------------+---------------------+---------------------+-----------------+

Model Selection Guide
---------------------

Choose a model based on your requirements:

- **Best accuracy**: DEIMv2-X, DEIM-DFine-X, D-Fine X, or YOLOX-X
- **Best speed/accuracy trade-off**: DEIMv2-S, DEIMv2-M, DEIM-DFine-M, or YOLOX-L
- **Fastest inference**: YOLOX-TINY, YOLOX-S, SSD-MobileNetV2, or ATSS-MobileNetV2

**Recommendations:**

- For **transformer-based models**, the DEIM family (DEIMv2 and DEIM-DFine) provides state-of-the-art accuracy with excellent inference speed. DEIMv2-S/M are particularly well-suited for real-time and edge deployment scenarios.
- For **CNN-based models**, `YOLOX <https://arxiv.org/abs/2107.08430>`_ offers an excellent speed-accuracy trade-off and strong performance across different benchmark datasets. 
    `MobileNetV2-ATSS <https://arxiv.org/abs/1912.02424>`_ and `SSD <https://arxiv.org/abs/1512.02325>`_ are also good choices for resource-constrained environments.
