:octicon:`code-square;1em` API Quick-Guide
==============================================

Besides CLI functionality, The OpenVINO™ Training Extension provides APIs that help developers to integrate OpenVINO™ Training Extensions models into their projects.
This tutorial intends to show how to create a dataset, model and use all of the CLI functionality through APIs.

OTX aims to provide a unified interface for interacting with different backends, enabling seamless training and validation across both native and third-party backends. This allows users to easily adapt and integrate popular computer vision models.
OpenVINO™ Training Extensions APIs are designed to be easy to use and flexible, allowing you to create, train, and deploy deep learning models with minimal effort.

There are :doc:`dedicated tutorials <../tutorials/base/how_to_train/index>` in our documentation with life-practical examples on specific datasets for each task.

.. note::

    To start with we need to `install OpenVINO™ Training Extensions <https://github.com/open-edge-platform/training_extensions/blob/develop/QUICK_START_GUIDE.md#setup-openvino-training-extensions>`_.


*******************
Dataset preparation
*******************
For the demonstration, we will use the `WGISD dataset <https://github.com/thsant/wgisd>_` and object detection model.

1. Clone a repository
with `WGISD dataset <https://github.com/thsant/wgisd>`_.

.. code-block:: shell
    cd data
    git clone https://github.com/thsant/wgisd.git
    cd wgisd
    git checkout 6910edc5ae3aae8c20062941b1641821f0c30127

2. We need to rename annotations to
be distinguished by OpenVINO™ Training Extensions Datumaro manager:

.. code-block:: shell
    mv data images && mv coco_annotations annotations && mv annotations/train_bbox_instances.json instances_train.json && mv annotations/test_bbox_instances.json instances_val.json
Now it is all set to use this dataset inside OpenVINO™ Training Extensions

************************************
Quick Start with "create_engine"
************************************

Once the dataset is ready, we can create an ``Engine`` instance using the ``create_engine`` function, providing the dataset path and the model.
OpenVINO Training Extensions will prepare an engine based on the provided dataset and model.

.. tab-set::

    .. tab-item:: with a config and dataset path

        .. code-block:: python

            from otx.engine import create_engine

            engine = create_engine(data="data/wgisd", model="path/to/model/config")
            engine.train()

    .. tab-item:: with a datamodule and model instances

        .. code-block:: python

            from otx.engine import create_engine
            from otx.config.data import SubsetConfig
            from otx.backend.native.models import ATSS
            from otx.data.datamodule import OTXDataModule

            model = ATSS(model_name="atss_mobilenetv2",
                        label_info = {"label_names": ["Chardonnay", "Cabernet Franc", "Cabernet Sauvignon", "Sauvignon Blanc", "Syrah"],
                                     "label_id": [0, 1, 2, 3, 4],
                                     "label_groups": [["Chardonnay", "Cabernet Franc", "Cabernet Sauvignon", "Sauvignon Blanc", "Syrah"]]},
                        data_input_params = {"input_size": [800, 992],
                                            "mean": [0.0, 0.0, 0.0],
                                            "std": [255.0, 255.0, 255.0]})

            train_augmentation = v2.Compose([
                    otx.data.transform_libs.torchvision.Resize(size=(800, 992), transform_bbox=True),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.ToDtype(torch.float32),
                    v2.Normalize(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]),
                ])
            val_augmentation = v2.Compose([
                    otx.data.transform_libs.torchvision.Resize(size=(800, 992), transform_bbox=True),
                    v2.ToDtype(torch.float32),
                    v2.Normalize(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]),
                ])
            datamodule = OTXDataModule(data_root="data/wgisd", data_format="COCO", task="DETECTION",
                                       train_subset=SubsetConfig(subset_name="train",
                                                                augmentations=train_augmentation,
                                                                batch_size=8,),
                                       val_subset=SubsetConfig(subset_name="val",
                                                                augmentations=val_augmentation,
                                                                batch_size=8,),
                                       test_subset=SubsetConfig(subset_name="test",
                                                                augmentations=val_augmentation,
                                                                batch_size=8,))

            engine = create_engine(data=datamodule, model=model)
            engine.train()


**********************************
Check Available Model Recipes
**********************************

If you want to use other models offered by OpenVINO™ Training Extension, you can get a list of available models in OpenVINO™ Training Extension as shown below.

.. tab-set::

    .. tab-item:: List of available model names

        .. code-block:: python

            from otx.engine.utils.api import list_models

            model_lists = list_models(task="DETECTION")
            print(model_lists)

            '''
            [
                'yolox_s',
                'yolox_tiny_tile',
                'ssd_mobilenetv2',
                'atss_resnext101',
                'yolox_tiny',
                'rtdetr_18',
                'atss_resnext101_tile',
                'yolox_s_tile',
                'rtmdet_tiny_tile',
                'yolox_l_tile',
                'ssd_mobilenetv2_tile',
                'dfine_x',
                'rtmdet_tiny',
                'yolox_l',
                'rtdetr_50',
                'atss_mobilenetv2_tile',
                'yolox_x_tile',
                'yolox_x',
                'rtdetr_18_tile',
                'atss_mobilenetv2',
                'openvino_model',
                'rtdetr_101',
                'rtdetr_50_tile',
                'rtdetr_101_tile',
                'dfine_x_tile'
            ]
            '''

    .. tab-item:: Print available configuration information

        .. code-block:: python

            from otx.engine.utils.api import list_models

            model_lists = list_models(task="DETECTION", print_table=True)

            '''
            ┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃ Task      ┃ Model Name            ┃ Recipe Path                                                    ┃
            ┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │ DETECTION │ yolox_tiny            │ src/otx/recipe/detection/yolox_tiny.yaml                       │
            │ ...       │                       │                                                                │
            └───────────┴───────────────────────┴────────────────────────────────────────────────────────────────┘
            '''

.. note::

    If you're looking for a specific name, use the pattern argument.

    .. code-block:: python

        from otx.engine.utils.api import list_models

        model_lists = list_models(task="DETECTION", pattern="tile")
        print(model_lists)
        '''
        [
            'yolox_tiny_tile',
            'atss_mobilenetv2_tile',
            'atss_resnext101_tile',
            'yolox_x_tile',
            'yolox_s_tile',
            'rtmdet_tiny_tile',
            'rtdetr_50_tile',
            'yolox_l_tile',
            'ssd_mobilenetv2_tile',
            'rtdetr_18_tile',
            'rtdetr_101_tile',
            'dfine_x_tile'
        ]
        '''


You can also find the available model recipes in YAML form in the folder ``otx/recipe``.

*****************
OTX Native Engine
*****************

The ``otx.engine.Engine`` class serves as the base engine where all common entry points for various backends are defined.
In this section, we focus on the ``otx.backend.native.engine.OTXEngine`` class — the native engine implementation provided by OpenVINO™ Training Extensions.

1. Setting ``work_dir``

Specify ``work_dir``. This is the workspace for that ``OTXEngine``, and where output is stored.
The default value is currently ``./otx-workspace``.

.. code-block:: python

    from otx.backend.native.engine import OTXEngine
    from otx.backend.native.models import ATSS

    engine = OTXEngine(work_dir="work_dir",
                       data_root="data/wgisd",
                       model=ATSS(...))


3. Setting device

You can set the device by referencing the ``DeviceType`` in ``otx.types.device``.
The current default setting is ``auto``. Native OTX Engine supports ``cpu``, ``gpu``, and ``xpu`` devices for training.

.. note::

    **XPU** is a generic term used by Intel to refer to heterogeneous compute devices, including Intel GPUs, CPUs, and certain AI accelerators.
    However, when referring to the PyTorch device, XPU specifically denotes an **Intel GPU**.

.. code-block:: python

    from otx.types.device import DeviceType
    from otx.backend.native.engine import OTXEngine

    engine = OTXEngine(..., device=DeviceType.xpu)

.. tabs::

    .. tab:: CPU

        .. code-block:: python

            engine = OTXEngine(device=DeviceType.cpu)
            # or
            engine = OTXEngine(device="cpu")

    .. tab:: GPU

        .. code-block:: python

            engine = OTXEngine(device=DeviceType.gpu)
            # or
            engine = OTXEngine(device="gpu")

    .. tab:: XPU

        .. code-block:: python

            engine = OTXEngine(device=DeviceType.xpu)
            # or
            engine = OTXEngine(device="xpu")


In addition, the ``OTXEngine`` constructor can be associated with the Trainer's constructor arguments to control the Trainer's functionality.
Refer `lightning.Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.

4. Using the OpenVINO™ Training Extension configuration we can configure the Engine.

.. code-block:: python

    from otx.backend.native.engine import OTXEngine

    recipe = "src/otx/recipe/detection/atss_mobilenetv2.yaml"
    engine = OTXEngine.from_config(
        config_path=recipe,
        data_root="data/wgisd",
        work_dir="./otx-workspace",
    )


********
Training
********

Create an output model and start actual training:

1. Below is an example using the ``atss_mobilenetv2`` model provided by OpenVINO™ Training Extension.

.. code-block:: python

    from otx.engine import create_engine
    from otx.backend.native.models import ATSS

    model = ATSS(
                model_name="atss_mobilenetv2",
                label_info = {"label_names": ["Chardonnay", "Cabernet Franc", "Cabernet Sauvignon", "Sauvignon Blanc", "Syrah"],
                                "label_id": [0, 1, 2, 3, 4],
                                "label_groups": [["Chardonnay", "Cabernet Franc", "Cabernet Sauvignon", "Sauvignon Blanc", "Syrah"]]},
                data_input_params = {"input_size": [800, 992],
                                    "mean": [0.0, 0.0, 0.0],
                                    "std": [255.0, 255.0, 255.0]}
            )
    engine = create_engine(data="data/wgisd",
                           model=model)

    engine.train()

2. Alternatively, we can use the configuration file.

.. code-block:: python

    from otx.engine import create_engine

    config = "src/otx/recipe/detection/atss_mobilenetv2.yaml"

    engine = create_engine(data=config, data="data/wgisd")
    engine.train()

.. note::

    This can use callbacks provided by OpenVINO™ Training Extension and several training techniques.
    However, in this case, no arguments are specified for train.

3. You can also create a specific Engine instance using the class directly.

    .. code-block:: python

        from otx.backend.native.engine import OTXEngine

        engine = OTXEngine(data="data/wgisd", model=config, work_dir="otx-workspace")
        engine.train()

3. If you want to customize model or optimizer, you can do so as shown below:

The model used by the Engine is of type ``otx.model.entity.base.OTXModel``.

.. tab-set::

    .. tab-item:: Custom Model with checkpoint

        .. code-block:: python

            from otx.backend.native.models.detection.atss import ATSS
            from otx.backend.native.engine import OTXEngine

            model = ATSS(label_info = {"label_names": ["Chardonnay", "Cabernet Franc", "Cabernet Sauvignon", "Sauvignon Blanc", "Syrah"],
                                "label_id": [0, 1, 2, 3, 4],
                                "label_groups": [["Chardonnay", "Cabernet Franc", "Cabernet Sauvignon", "Sauvignon Blanc", "Syrah"]]},
                         model_name="atss_mobilenetv2",
                         data_input_params={"input_size": [800, 992],
                                            "mean": [0.0, 0.0, 0.0],
                                            "std": [255.0, 255.0, 255.0]})

            engine = OTXEngine(data="data/wgisd", model=model, checkpoint="<path/to/checkpoint>")
            engine.train()

    .. tab-item:: Custom Optimizer & Scheduler

        .. code-block:: python

            from torch.optim import SGD
            from torch.optim.lr_scheduler import CosineAnnealingLR
            from otx.backend.native.models.detection.atss import ATSS
            from otx.backend.native.engine import OTXEngine

            model = ATSS(label_info=5, model_name="atss_mobilenetv2",
                         data_input_params={"input_size": [800, 992],
                                            "mean": [0.0, 0.0, 0.0],
                                            "std": [255.0, 255.0, 255.0]})
            optimizer = SGD(model.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9)
            scheduler = CosineAnnealingLR(optimizer, T_max=10000, eta_min=0)

            engine = OTXEngine(
                ...,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            engine.train()

4. If you want to specify the datamodule, you can do so as shown below:

The datamodule used by the Engine is of type ``otx.data.module.OTXDataModule``.

.. code-block:: python

    from otx.data.module import OTXDataModule
    from otx.backend.native.engine import OTXEngine

    datamodule = OTXDataModule(data_root="data/wgisd", ...)

    engine = OTXEngine(data=datamodule, ...)
    engine.train()

.. note::

    If both ``data_root`` and ``datamodule`` enter ``Engine`` as input, ``Engine`` uses datamodule as the base.

.. tip::

    You can get DataModule more easily using AutoConfigurator.

    .. code-block:: python

        from otx.tools.auto_configuration import AutoConfigurator

        datamodule = AutoConfigurator(data_root="data/wgisd").get_datamodule()

5. You can use train-specific arguments with ``train()`` function.

.. tab-set::

    .. tab-item:: Change Max Epochs

        .. code-block:: python

            engine.train(max_epochs=10)

    .. tab-item:: Fix Training Seed & Set Deterministic

        .. code-block:: python

            engine.train(seed=1234, deterministic=True)

    .. tab-item:: Use Mixed Precision

        .. code-block:: python

            engine.train(precision="bf16")

        .. note::

            This uses lightning's precision value. You can use the values below:
            - "64", "32", "bf16",
            - 64, 32, bf16

    .. tab-item:: Change Validation Metric

        .. code-block:: python

            from otx.metrics.fmeasure import FMeasure

            metric = FMeasure(label_info=5)
            engine.train(metric=metric)

    .. tab-item:: Set Callbacks & Logger

        .. code-block:: python

            from lightning.pytorch.callbacks import EarlyStopping
            from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

            engine.train(callbacks=[EarlyStopping()], loggers=[TensorBoardLogger()])

In addition, the ``train()`` function can be associated with the Trainer's constructor arguments to control the Trainer's functionality.
Refer `lightning.Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.

For example, if you want to use the ``limit_val_batches`` feature provided by Trainer, you can use it like this:

.. code-block:: python

    # disable validation
    engine.train(limit_val_batches=0)


***********
Evaluation
***********

1. If the training is already in place,
we just need to use the code below:

.. tab-set::

    .. tab-item:: Evaluate Model

        .. code-block:: python

            engine.test()

    .. tab-item:: Evaluate Model with different checkpoint

        .. code-block:: python

            engine.test(checkpoint="<path/to/checkpoint>")

        .. note::

            The format that can enter the checkpoint is of type torch (.ckpt) or exported model (.onnx, .xml).

    .. tab-item:: Evaluate Model with different datamodule or dataloader

        .. code-block:: python

            from otx.data.module import OTXDataModule

            datamodule = OTXDataModule(data_root="data/wgisd", ...)
            engine.test(datamodule=datamodule)

    .. tab-item:: Evaluate Model with different metrics

        .. code-block:: python

            from otx.metrics.fmeasure import FMeasure

            metric = FMeasue(label_info=5)
            engine.test(metric=metric)

2. If you want to validate OpenVINO IR model, you need to use OVEngine. OVEngine can be created in the same way as OTXEngine:

.. tab-set::

    .. code-block:: python

        from otx.backend.openvino.engine import OVEngine
        from otx.engine import create_engine

        # using create_engine
        ov_engine = create_engine(
            data="data/wgisd",
            model="path/to/exported_model.xml",
        )

        # or directly with OVEngine
        ov_engine = OVEngine(model="path/to/exported_model.xml", data="data/wgisd")

        ov_engine.test()

***********
Exporting
***********

To export our model to OpenVINO™ IR format we need to create output model and run exporting task.
If the engine is trained, you can proceed with the export using the current engine's checkpoint:

The default value for ``export_format`` is ``OPENVINO``.
The default value for ``export_precision`` is ``FP32``.

.. tab-set::

    .. tab-item:: Export OpenVINO™ IR

        .. code-block:: python

            engine.export()

    .. tab-item:: Export ONNX

        .. code-block:: python

            engine.export(export_format="ONNX")

    .. tab-item:: Export with explain features

        .. code-block:: python

            engine.export(explain=True)

        .. note::

            This ensures that it is exported with a ``saliency_map`` and ``feature_vector`` that will be used in the XAI.

    .. tab-item:: Export with different checkpoint

        .. code-block:: python

            engine.export(checkpoint="<path/to/checkpoint>")

    .. tab-item:: Export with FP16

        .. code-block:: python

            engine.export(export_precision="FP16")


****
XAI
****

To run the XAI with the OTXEngine, we need to run the following code:

.. tab-set::

    .. tab-item:: Run XAI

        .. code-block:: python

            engine.explain(checkpoint="<path/to/checkpoint>")

    .. tab-item:: Evaluate Model with different datamodule or dataloader

        .. code-block:: python

            from otx.data.module import OTXDataModule

            datamodule = OTXDataModule(data_root="data/wgisd")
            engine.explain(..., datamodule=datamodule)

    .. tab-item:: Dump saliency_map

        .. code-block:: python

            engine.explain(..., dump=True)


************
Optimization
************

To run the optimization with PTQ on the OpenVINO™ IR model, we need to use OVEngine and run the optimization procedure:

.. tab-set::

    .. tab-item:: Run PTQ Optimization

        .. code-block:: python

            from otx.backend.openvino.engine import OVEngine
            ov_engine = OVEngine(model="path/to/exported_model.xml", data="data/wgisd")

            ov_engine.optimize()

    .. tab-item:: Evaluate Model with different datamodule or dataloader

        .. code-block:: python

            from otx.data.module import OTXDataModule
            from otx.backend.openvino.engine import OVEngine
            ov_engine = OVEngine(model="path/to/exported_model.xml", data="data/wgisd")

            datamodule = OTXDataModule(data_root="data/wgisd", ...)
            ov_engine.optimize(data=datamodule)


You can validate the optimized model as the usual model:

.. code-block:: python

    ov_engine.test(model="<path/to/optimized/ir/xml>")

************
Benchmarking
************

``OTXEngine`` allows to perform benchmarking of the trained model, and provide theoretical complexity information in case of torch model.
The estimated by ``Engine.benchmark()`` performance may differ from the performance of the deployed model, since the measurements are conducted
via OTX inference API, which can introduce additional burden.

.. tab-set::

    .. tab-item:: Benchmark Model

        .. code-block:: python

            engine = OTXEngine(data="data/wgisd", model="src/otx/recipe/detection/atss_mobilenetv2.yaml", work_dir="otx-workspace")
            engine.benchmark()

Conclusion
"""""""""""
That's it! Now, we can use OpenVINO™ Training Extensions APIs to create, train, and deploy deep learning models using the OpenVINO™ Training Extensions.
