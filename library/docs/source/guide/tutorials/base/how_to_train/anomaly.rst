Anomaly Detection Tutorial
================================

This tutorial demonstrates how to train, evaluate, and deploy a classification, detection, or segmentation model for anomaly task in industrial or medical applications.
Read :doc:`../../../explanation/algorithms/anomaly/index` for more information about the Anomaly tasks.

.. note::
    To learn more about managing the training process of the model including additional parameters and its modification, refer to :doc:`./detection`.

The process has been tested with the following configuration:

- Ubuntu 20.04
- NVIDIA GeForce RTX 3090
- Intel(R) Core(TM) i9-11900
- CUDA Toolkit 11.8


*****************************
Setup the Virtual environment
*****************************

1. To create a universal virtual environment for OpenVINO™ Training Extensions,
please follow the installation process in the :doc:`quick start guide <../../../get_started/installation>`.

2. Activate your virtual
environment:

.. code-block:: shell

    .otx/bin/activate
    # or by this line, if you created an environment, using tox
    . venv/otx/bin/activate

**************************
Dataset Preparation
**************************

1. For this example, we will use the `MVTec <https://www.mvtec.com/company/research/datasets/mvtec-ad>`_ dataset.
You can download the dataset from the link above. We will use the ``bottle`` category for this tutorial.

2. This is how it might look like in your
file system:

.. code-block::

    datasets/MVTec/bottle
    ├── ground_truth
    │   ├── broken_large
    │   │   ├── 000_mask.png
    │   │   ├── 001_mask.png
    │   │   ├── 002_mask.png
    │   │   ...
    │   ├── broken_small
    │   │   ├── 000_mask.png
    │   │   ├── 001_mask.png
    │   │   ...
    │   └── contamination
    │       ├── 000_mask.png
    │       ├── 001_mask.png
    │       ...
    ├── license.txt
    ├── readme.txt
    ├── test
    │   ├── broken_large
    │   │   ├── 000.png
    │   │   ├── 001.png
    │   │   ...
    │   ├── broken_small
    │   │   ├── 000.png
    │   │   ├── 001.png
    │   │   ...
    │   ├── contamination
    │   │   ├── 000.png
    │   │   ├── 001.png
    │   │   ...
    │   └── good
    │       ├── 000.png
    │       ├── 001.png
    │       ...
    └── train
        └── good
            ├── 000.png
            ├── 001.png
            ...

***************************
Training
***************************

1. For this example let's look at the
anomaly tasks

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$  otx find --task ANOMALY
            ┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃ Task              ┃ Model Name ┃ Recipe Path                                 ┃
            ┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │ ANOMALY           │ stfpm      │ src/otx/recipe/anomaly_detection/stfpm.yaml │
            │ ANOMALY           │ padim      │ src/otx/recipe/anomaly_detection/padim.yaml │
            │ ANOMALY           │ uflow      │ src/otx/recipe/anomaly_detection/uflow.yaml │
            └───────────────────┴────────────┴─────────────────────────────────────────────┘

    .. tab-item:: API

        .. code-block:: python

            from otx.backend.native.cli.utils import list_models

            model_lists = list_models(task="ANOMALY")
            print(model_lists)
            '''
            ['stfpm', 'padim', "uflow"]
            '''

You can see two anomaly models, STFPM, PADIM and UFLOW. For more detail on each model, refer to Anomalib's `STFPM <https://anomalib.readthedocs.io/en/v1.0.0/markdown/guides/reference/models/image/stfpm.html>`_, `PADIM <https://anomalib.readthedocs.io/en/v1.0.0/markdown/guides/reference/models/image/padim.html>`_  and `UFLOW <https://anomalib.readthedocs.io/en/v1.0.0/markdown/guides/reference/models/image/uflow.html>`_ documentation.

2. Let's proceed with PADIM for
this example.

.. tab-set::

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$  otx train --config src/otx/recipe/anomaly_detection/padim.yaml \
                                  --data_root datasets/MVTec/bottle
                                  --work_dir ./otx-workspace

    .. tab-item:: API (from_config)

        .. code-block:: python

            from otx.backend.native.engine import OTXEngine

            data_root = "datasets/MVTec/bottle"
            recipe = "src/otx/recipe/anomaly_detection/padim.yaml"

            engine = OTXEngine.from_config(
                      config_path=recipe,
                      data_root=data_root,
                      work_dir="otx-workspace",
                    )

            engine.train(...)

    .. tab-item:: API

        .. code-block:: python

            from otx.backend.native.engine import OTXEngine
            from otx.backend.native.models import PADIM

            data_root = "datasets/MVTec/bottle"
            model = PADIM(data_input_params = {"input_size": [446, 446],
                                                         "mean": [123.675, 116.28, 103.53],
                                                         "std": [58.395, 57.12, 57.375]})

            engine = OTXEngine(
                      model=model,
                      data=data_root,
                      work_dir="otx-workspace",
                    )

            # one more possibility to obtain the  engine by the given model/dataset
            # using "create_engine" function
            from otx.engine import create_engine
            engine = create_engine(
                      model=model,
                      data=data_root,
                    )

            engine.train(...)

3. ``(Optional)`` Additionally, we can tune training parameters such as batch size, learning rate, patience epochs.
Learn more about specific parameters using ``otx train --help -v`` or ``otx train --help -vv``.

For example, to decrease the batch size to 4, fix the number of epochs to 100, extend the command line above with the following line.

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx train ... --data.train_subset.batch_size 4 \
                                     --max_epochs 100

    .. tab-item:: API

        .. code-block:: python

            from otx.config.data import SubsetConfig
            from otx.data.module import OTXDataModule
            from otx.backend.native.engine import OTXEngine

            datamodule = OTXDataModule(..., train_subset=SubsetConfig(..., batch_size=4))

            engine = OTXEngine(..., datamodule=datamodule)

            engine.train(max_epochs=100)

4. The training result ``checkpoints/*.ckpt`` file is located in ``{work_dir}`` folder,
while training logs can be found in the ``{work_dir}/{timestamp}`` dir.

This will start training and generate artifacts for commands such as ``export`` and ``optimize``. You will notice the ``otx-workspace`` directory in your current working directory. This is where all the artifacts are stored.

**************
Evaluation
**************

Now we have trained the model, let's see how it performs on a specific dataset. In this example, we will use the same dataset to generate evaluation metrics. To perform evaluation you need to run the following commands:

.. tab-set::

    .. tab-item:: CLI (with work_dir)

        .. code-block:: shell

            (otx) ...$ otx test --work_dir otx-workspace
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃        Test metric        ┃       DataLoader 0        ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │        image_AUROC        │            0.8            │
            │       image_F1Score       │            0.8            │
            │        pixel_AUROC        │            0.8            │
            │       pixel_F1Score       │            0.8            │
            │      test/data_time       │    0.6517705321311951     │
            │      test/iter_time       │    0.6630784869194031     │
            └───────────────────────────┴───────────────────────────┘

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx test --config  src/otx/recipe/anomaly_detection/padim.yaml \
                                --data_root datasets/MVTec/bottle \
                                --checkpoint otx-workspace/20240313_042421/checkpoints/epoch_010.ckpt
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃        Test metric        ┃       DataLoader 0        ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │        image_AUROC        │            0.8            │
            │       image_F1Score       │            0.8            │
            │        pixel_AUROC        │            0.8            │
            │       pixel_F1Score       │            0.8            │
            │      test/data_time       │    0.6517705321311951     │
            │      test/iter_time       │    0.6630784869194031     │
            └───────────────────────────┴───────────────────────────┘

    .. tab-item:: API

        .. code-block:: python

            engine.test()


The primary metric here is the f-measure computed against the ground-truth bounding boxes. It is also called the local score. In addition, f-measure is also used to compute the global score. The global score is computed based on the global label of the image. That is, the image is anomalous if it contains at least one anomaly. This global score is stored as an additional metric.

.. note::

    All task types report Image-level F-measure as the primary metric. In addition, both localization tasks (anomaly detection and anomaly segmentation) also report localization performance (F-measure for anomaly detection and Dice-coefficient for anomaly segmentation).

*******************************
Segmentation and Classification
*******************************

While the above example shows Anomaly Detection, you can also train Anomaly Segmentation and Classification models.
To see what tasks are available, you can pass ``ANOMALY_SEGMENTATION`` and ``ANOMALY_CLASSIFICATION`` to ``otx find`` mentioned in the `Training`_ section. You can then use the same commands to train, evaluate, export and optimize the models.

.. note::

    The Segmentation and Detection tasks also require that the ``ground_truth`` masks be present to ensure that the localization metrics are computed correctly.
    The ``ground_truth`` masks are not required for the Classification task.
