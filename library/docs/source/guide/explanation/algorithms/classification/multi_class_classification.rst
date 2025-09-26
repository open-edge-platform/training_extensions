Multi-class Classification
==========================

Multi-class classification is the problem of classifying instances into one of two or more classes. We solve this problem in a common fashion, based on the feature extractor backbone and classifier head that predicts the distribution probability of the categories from the given corpus.
For the supervised training we use the following algorithms components:

.. _mcl_cls_supervised_pipeline:

- ``Learning rate schedule``: `ReduceLROnPlateau <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html>`_. It is a common learning rate scheduler that tends to work well on average for this task on a variety of different datasets.

- ``Loss function``: We use standard `Cross Entropy Loss <https://en.wikipedia.org/wiki/Cross_entropy>`_  to train a model. However, for the class-incremental scenario we use `Influence-Balanced Loss <https://arxiv.org/abs/2110.02444>`_. IB loss is a solution for the class imbalance, which avoids overfitting to the majority classes re-weighting the influential samples.

- ``Additional training techniques``
    - ``Early stopping``: To add adaptability to the training pipeline and prevent overfitting.
    - `Balanced Sampler <https://github.dev/open-edge-platform/training_extensions/blob/develop/lib/src/otx/algo/samplers/balanced_sampler.py#L11>`_: To create an efficient batch that consists of balanced samples over classes, reducing the iteration size as well.

**************
Dataset Format
**************

We support a commonly used format for multi-class image classification task: `ImageNet <https://www.image-net.org/>`_ class folder format.
This format has the following structure:

::

    data
    ├── train
        ├── class 0
            ├── 0.png
            ├── 1.png
            ...
            └── N.png
        ├── class 1
            ├── 0.png
            ├── 1.png
            ...
            └── N.png
        ...
        └── class N
            ├── 0.png
            ├── 1.png
            ...
            └── N.png
    └── val
        ...

.. note::

    Please, refer to our :doc:`dedicated tutorial <../../../tutorials/base/how_to_train/classification>` for more information how to train, validate and optimize classification models.

******
Models
******
.. _classification_models:

We support the following ready-to-use model recipes:

+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+-----------------+
| Model Name                                                                                                                                                                                                       | Complexity (GFLOPs) | Model params (M)|
+==================================================================================================================================================================================================================+=====================+=================+
| `MobileNet-V3-large <https://github.com/open-edge-platform/training_extensions/blob/develop/lib/src/otx/recipe/classification/multi_class_cls/mobilenet_v3_large.yaml>`_                                                | 0.86                | 2.97            |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+-----------------+
| `MobileNet-V3-small <https://github.com/open-edge-platform/training_extensions/blob/develop/lib/src/otx/recipe/classification/multi_class_cls/tv_mobilenet_v3_small.yaml>`_                                             | 0.22                | 0.93            |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+-----------------+
| `EfficinetNet-B0 <https://github.com/open-edge-platform/training_extensions/blob/develop/lib/src/otx/recipe/classification/multi_class_cls/efficientnet_b0.yaml>`_                                                      | 1.52                | 4.09            |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+-----------------+
| `EfficientNet-B3 <https://github.com/open-edge-platform/training_extensions/blob/develop/lib/src/otx/recipe/classification/multi_class_cls/tv_efficientnet_b3.yaml>`_                                                   | 3.84                | 10.70           |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+-----------------+
| `EfficientNet-V2-S <https://github.com/open-edge-platform/training_extensions/blob/develop/lib/src/otx/recipe/classification/multi_class_cls/efficientnet_v2.yaml>`_                                                    | 5.76                | 20.23           |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+-----------------+
| `EfficientNet-V2-l <https://github.com/open-edge-platform/training_extensions/blob/develop/lib/src/otx/recipe/classification/multi_class_cls/tv_efficientnet_v2_l.yaml>`_                                               | 48.92               | 117.23          |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+-----------------+
| `DeiT-Tiny <https://github.com/open-edge-platform/training_extensions/blob/develop/lib/src/otx/recipe/classification/multi_class_cls/deit_tiny.yaml>`_                                                                  | 2.51                | 22.0            |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+-----------------+
| `DINO-V2 <https://github.com/open-edge-platform/training_extensions/blob/develop/lib/src/otx/recipe/classification/multi_class_cls/dino_v2.yaml>`_                                                                      | 12.46               | 88.0            |
+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------+-----------------+

`MobileNet-V3 <https://arxiv.org/abs/1905.02244>`_ is the best choice when training time and computational cost are in priority, nevertheless, this recipe provides competitive accuracy as well.
`EfficientNet-B0/B3 <https://arxiv.org/abs/1905.11946>`_ consumes more Flops compared to MobileNet, providing better performance on large datasets.
`EfficientNet-V2 <https://arxiv.org/abs/2104.00298>`_ has more parameters and Flops and needs more time to train, meanwhile providing superior classification performance.
`DeiT-Tiny <https://arxiv.org/abs/2012.12877>`_ is a transformer-based model that provides a good trade-off between accuracy and computational cost.
`DINO-V2 <https://arxiv.org/abs/2304.07193>`_ produce high-performance visual features that can be directly employed with classifiers as simple as linear layers on a variety of computer vision tasks.

To see which models are available for the task, the following command can be executed:

.. code-block:: shell

        (otx) ...$ otx find --task MULTI_CLASS_CLS
