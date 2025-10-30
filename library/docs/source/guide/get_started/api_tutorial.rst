:octicon:`code-square;1em` API Quick-Guide
==============================================

Besides CLI functionality, The OpenVINO™ Training Extension provides APIs that help developers to integrate OpenVINO™ Training Extensions models into their projects.
This tutorial intends to show how to create a dataset, model and use all of the CLI functionality through APIs.

For demonstration purposes we will use the Object Detection ATSS model with `WGISD <https://github.com/thsant/wgisd>`_ public dataset as we did for the :doc:`CLI tutorial <../tutorials/base/how_to_train/detection>`.

.. note::

    To start with we need to `install OpenVINO™ Training Extensions <https://github.com/open-edge-platform/training_extensions/blob/develop/QUICK_START_GUIDE.md#setup-openvino-training-extensions>`_.

*******************
Dataset preparation
*******************

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

************
Quick Start
************

Once the dataset is ready, we can immediately start training with the model and data pipeline. Simply pass the data root and model config path to the Engine class.
The following code snippet demonstrates how to do that:

.. code-block:: python

    from otx.engine import Engine

    engine = Engine(data="data/wgisd", model="src/otx/recipe/detection/atss_mobilenetv2.yaml")
    engine.train()


.. note::

    It is also possible to pass constructed data and model classes

    .. code-block:: python

        from otx.engine import Engine
        from otx.backend.native.models.detection.atss import ATSS
        from otx.data import OTXDataModule

        model = ATSS(label_info=5, model_name="mobilenetv2", data_input_params=DataInputParams(input_size=(512, 512), mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)))
        datamodule = OTXDataModule(task="DETECTION", data_root="data/wgisd")
        engine = Engine(data=datamodule, model=model)

**********************************
Check Available Model Recipes
**********************************

If you want to use other models offered by OpenVINO™ Training Extension, you can get a list of available models as shown below.

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

*********
Engine
*********

The ``otx.engine.Engine`` class is the main entry point for using OpenVINO™ Training Extensions APIs.

1. Setting ``work_dir``

Specify ``work_dir``. This is the workspace for that ``Engine``, and where output is stored.
The default value is currently ``./otx-workspace``.

.. code-block:: python

    from otx.engine import Engine

    engine = Engine(work_dir="work_dir")


2. Setting device

You can set the device by referencing the ``DeviceType`` in ``otx.types.device``.
The current default setting is ``auto``.

.. code-block:: python

    from otx.types.device import DeviceType
    from otx.engine import Engine

    engine = Engine(device=DeviceType.gpu)
    # or
    engine = Engine(device="gpu")


In addition, the ``Engine`` constructor can be associated with the Trainer's constructor arguments to control the Trainer's functionality.
Refer `lightning.Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.

3. Using the OpenVINO™ Training Extension configuration we can configure the Engine.

.. code-block:: python

    from otx.engine import Engine

    recipe = "src/otx/recipe/detection/atss_mobilenetv2.yaml"
    engine = Engine.from_config(
        config_path=recipe,
        data_root="data/wgisd",
        work_dir="./otx-workspace",
    )


*********
Training
*********

Create an output model and start actual training:

1. Below is an example using the ``atss_mobilenetv2`` model provided by OpenVINO™ Training Extension.

.. code-block:: python

    from otx.engine import Engine

    engine = Engine(data_root="data/wgisd", model="atss_mobilenetv2")
    engine.train()

2. Alternatively, we can use the configuration file.

.. code-block:: python

    from otx.engine import Engine

    config = "src/otx/recipe/detection/atss_mobilenetv2.yaml"

    engine = Engine.from_config(config_path=config, data_root="data/wgisd")
    engine.train()

.. note::

    This can use callbacks provided by OpenVINO™ Training Extension and several training techniques.
    However, in this case, no arguments are specified for train.

3. If you want to specify the model, you can do so as shown below:

The model used by the Engine is of type ``otx.model.entity.base.OTXModel``.

.. tab-set::

    .. tab-item:: Custom Model

        .. code-block:: python

            from otx.backend.native.models.detection.atss import ATSS
            from otx.engine import Engine

            model = ATSS(label_info=5, model_name="mobilenetv2", data_input_params=DataInputParams(input_size=(512, 512), mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)))

            engine = Engine(data_root="data/wgisd", model=model)
            engine.train()

    .. tab-item:: Custom Model with checkpoint

        .. code-block:: python

            from otx.backend.native.models.detection.atss import ATSS
            from otx.engine import Engine

            model = ATSS(label_info=5, model_name="mobilenetv2")

            engine = Engine(data_root="data/wgisd", model=model, checkpoint="<path/to/checkpoint>")
            engine.train()

    .. tab-item:: Custom Optimizer & Scheduler

        .. code-block:: python

            from torch.optim import SGD
            from torch.optim.lr_scheduler import CosineAnnealingLR
            from otx.backend.native.models.detection.atss import ATSS
            from otx.engine import Engine

            model = ATSS(label_info=5, model_name="mobilenetv2")
            optimizer = SGD(model.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9)
            scheduler = CosineAnnealingLR(optimizer, T_max=10000, eta_min=0)

            engine = Engine(
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
    from otx.engine import Engine

    # default data module for the task
    datamodule = OTXDataModule(data_root="data/wgisd", task="DETECTION")

    engine = Engine(data=datamodule, model=...)
    engine.train()

The command above will create a default ``DataModule``. You can modify parameters for dataset constructing using ``SubsetConfig``:

.. code-block:: python

    import torchvision.transforms.v2 as v2

    from otx.data.module import OTXDataModule
    from otx.config.data import SubsetConfig
    from otx.engine import Engine

    train_subset_config = SubsetConfig(
        batch_size=64,
        transforms=v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        num_workers=4,
    )

    datamodule = OTXDataModule(
        task="DETECTION",
        data_root="data/wgisd",
        train_subset=train_subset_config
    )

Similarly, you can modify ``val_subset_config`` and ``test_subset_config``:

.. tip::

    You can get DataModule more easily using AutoConfigurator.

    .. code-block:: python

        from otx.tools.auto_configuration import AutoConfigurator

        # specific data pipeline for the model
        datamodule = AutoConfigurator(data_root="data/wgisd", model_config="src/otx/recipe/detection/atss_mobilenetv2.yaml").get_datamodule()

        # default for the task
        datamodule = AutoConfigurator(data_root="data/wgisd").get_datamodule()

You can also create ``OTXDataset`` independently and then call  ``from_otx_datasets`` method to construct  ``OTXDataModule``:

.. code-block:: python

    import torchvision.transforms.v2 as v2

    from otx.data.module import OTXDataModule
    from otx.config.data import SubsetConfig
    from otx.data.dataset import OTXDetectionDataset

    train_subset_config = SubsetConfig(
        batch_size=64,
        num_workers=4,
    )

    val_subset_config = SubsetConfig(
        batch_size=64,
    )

    transforms=v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # setup Datumaro dataset
    dm_dataset = DmDataset.import_from(data_root="data/wgisd")
    train_dm_subset = dm_dataset.subsets()["train"]
    val_dm_subset = dm_dataset.subsets()["val"]
    test_dm_subset = dm_dataset.subsets()["test"]

    # setup OTX dataset
    train_dataset = OTXDetectionDataset(
        train_dm_subset,
        transforms
    )
    val_dataset = OTXDetectionDataset(
        val_dm_subset,
        transforms
    )
    test_dataset = OTXDetectionDataset(
        test_dm_subset,
        transforms
    )

    # create OTXDataModule
    datamodule = OTXDataModule.from_otx_datasets(
        train_dataset = train_dataset,
        val_dataset = val_dataset,
        test_dataset = test_dataset,
        train_subset = train_subset_config,
        val_subset = val_subset_config,
        test_subset = val_subset_config
    )

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

If the training is already in place, we just need to use the code below:

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

            datamodule = OTXDataModule(data_root="data/wgisd")
            engine.test(datamodule=datamodule)

    .. tab-item:: Evaluate Model with different metrics

        .. code-block:: python

            from otx.metrics.fmeasure import FMeasure

            metric = FMeasure(label_info=5)
            engine.test(metric=metric)


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

To run the XAI with the OpenVINO™ IR model, we need to create an output model and run the XAI procedure:

.. tab-set::

    .. tab-item:: Run XAI

        .. code-block:: python
            engine = OVEngine(model="exported_model.xml", ...)
            engine.predict(explain=True)

    .. tab-item:: Evaluate Model with different datamodule or dataloader

        .. code-block:: python

            from otx.data.module import OTXDataModule

            datamodule = OTXDataModule(data_root="data/wgisd")
            engine.predict(..., datamodule=datamodule, explain=True)

    .. tab-item:: Use ExplainConfig

        .. code-block:: python

            from otx.config.explain import ExplainConfig

            engine.predict(..., explain=True, explain_config=ExplainConfig(postprocess=True))


************
Optimization
************

To run the optimization with PTQ on the OpenVINO™ IR model, we need to create an output model and run the optimization procedure:

.. tab-set::

    .. tab-item:: Run PTQ Optimization

        .. code-block:: python

            engine.optimize(checkpoint="<path/to/ir/xml>")

    .. tab-item:: Evaluate Model with different datamodule or dataloader

        .. code-block:: python

            from otx.data.module import OTXDataModule

            datamodule = OTXDataModule(data_root="data/wgisd")
            engine.optimize(..., datamodule=datamodule)


You can validate the optimized model as the usual model. For example for the NNCF model it will look like this:

.. code-block:: python

    engine.test(checkpoint="<path/to/optimized/ir/xml>")

************
Benchmarking
************

``Engine`` allows to perform benchmarking of the trained model, and provide theoretical complexity information in case of torch model.
The estimated by ``Engine.benchmark()`` performance may differ from the performance of the deployed model, since the measurements are conducted
via OTX inference API, which can introduce additional burden.

.. tab-set::

    .. tab-item:: Benchmark Model

        .. code-block:: python

            engine.benchmark()

    .. tab-item:: Benchmark OpenVINO™ IR model

        .. code-block:: python

            engine.benchmark(checkpoint="<path/to/exported_model.xml>")

        .. note::

            Specifying a checkpoint only makes sense for OpenVINO™ IR models.

Conclusion
"""""""""""
That's it! Now, we can use OpenVINO™ Training Extensions APIs to create, train, and deploy deep learning models using the OpenVINO™ Training Extensions.
