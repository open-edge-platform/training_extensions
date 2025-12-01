Semantic Segmentation model
================================

This tutorial demonstrates how to train and optimize a semantic segmentation model using the VOC2012 dataset from the PASCAL Visual Object Classes Challenge 2012.
The trained model will be used to segment images by assigning a label to each pixel of the input image.

To learn more about Segmentation task, refer to :doc:`../../../explanation/algorithms/segmentation/semantic_segmentation`.

.. note::
  To learn more about managing the training process of the model including additional parameters and its modification, refer to :doc:`./detection`.

The process has been tested on the following configuration.

- Ubuntu 20.04
- NVIDIA GeForce RTX 3090
- Intel(R) Core(TM) i9-11900
- CUDA Toolkit 11.8

*************************
Setup virtual environment
*************************

1. You can follow the installation process from a :doc:`quick start guide <../../../get_started/installation>`
to create a universal virtual environment for OpenVINO™ Training Extensions.

2. Activate your virtual environment:

.. code-block:: shell

  source .venv/bin/activate

***************************
Dataset preparation
***************************

For the semnatic segmentation, we'll use the common_semantic_segmentation_dataset located at the tests/assets


*********
Training
*********

1. First of all, you need to choose which semantic segmentation model you want to train.
The list of supported recipes for semantic segmentation is available with the command line below.

.. note::

  The characteristics and detailed comparison of the models could be found in :doc:`Explanation section <../../../explanation/algorithms/segmentation/semantic_segmentation>`.

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

          (otx) ...$ otx find --task SEMANTIC_SEGMENTATION

          ┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
          ┃ Task                  ┃ Model Name        ┃ Recipe Path                                                                                    ┃
          ┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
          │ SEMANTIC_SEGMENTATION │ litehrnet_x_tile  │ src/otx/recipe/semantic_segmentation/litehrnet_x_tile.yaml                                     │
          │ SEMANTIC_SEGMENTATION │ openvino_model    │ src/otx/recipe/semantic_segmentation/openvino_model.yaml                                       │
          │ SEMANTIC_SEGMENTATION │ dino_v2_tile      │ src/otx/recipe/semantic_segmentation/dino_v2_tile.yaml                                         │
          │ SEMANTIC_SEGMENTATION │ litehrnet_s       │ src/otx/recipe/semantic_segmentation/litehrnet_s.yaml                                          │
          │ SEMANTIC_SEGMENTATION │ segnext_b         │ src/otx/recipe/semantic_segmentation/segnext_b.yaml                                            │
          │ SEMANTIC_SEGMENTATION │ litehrnet_18_tile │ src/otx/recipe/semantic_segmentation/litehrnet_18_tile.yaml                                    │
          │ SEMANTIC_SEGMENTATION │ dino_v2           │ src/otx/recipe/semantic_segmentation/dino_v2.yaml                                              │
          │ SEMANTIC_SEGMENTATION │ segnext_s         │ src/otx/recipe/semantic_segmentation/segnext_s.yaml                                            │
          │ SEMANTIC_SEGMENTATION │ segnext_b_tile    │ src/otx/recipe/semantic_segmentation/segnext_b_tile.yaml                                       │
          │ SEMANTIC_SEGMENTATION │ segnext_t         │ src/otx/recipe/semantic_segmentation/segnext_t.yaml                                            │
          │ SEMANTIC_SEGMENTATION │ segnext_t_tile    │ src/otx/recipe/semantic_segmentation/segnext_t_tile.yaml                                       │
          │ SEMANTIC_SEGMENTATION │ litehrnet_18      │ src/otx/recipe/semantic_segmentation/litehrnet_18.yaml                                         │
          │ SEMANTIC_SEGMENTATION │ segnext_s_tile    │ src/otx/recipe/semantic_segmentation/segnext_s_tile.yaml                                       │
          │ SEMANTIC_SEGMENTATION │ litehrnet_x       │ src/otx/recipe/semantic_segmentation/litehrnet_x.yaml                                          │
          │ SEMANTIC_SEGMENTATION │ litehrnet_s_tile  │ src/otx/recipe/semantic_segmentation/litehrnet_s_tile.yaml                                     │
          └───────────────────────┴───────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────┘

    .. tab-item:: API

        .. code-block:: python

          from otx.backend.native.cli.utils import list_models

          model_lists = list_models(task="SEMANTIC_SEGMENTATION")
          print(model_lists)
          '''
          [
           'segnext_s_tile',
           'segnext_t',
           'litehrnet_s',
           'litehrnet_s_tile',
           'segnext_t_tile',
           'dino_v2_tile',
           'litehrnet_x',
           'litehrnet_x_tile',
           'segnext_s',
           'openvino_model',
           'litehrnet_18',
           'dino_v2',
           'segnext_b',
           'segnext_b_tile',
           'litehrnet_18_tile'
          ]
          '''

1.  On this step we will configure configuration
with:

- all necessary configs for litehrnet_18
- train/validation sets, based on provided annotation.

Let's prepare an OpenVINO™ Training Extensions semantic segmentation workspace running the following command:

.. code-block:: shell

  # or its config path
  (otx) ...$ otx train --config src/otx/recipe/semantic_segmentation/litehrnet_18.yaml --data_root tests/assets/common_semantic_segmentation_dataset --print_config

  ...
  data_root: tests/assests/common_semantic_segmentation_dataset
  work_dir: otx-workspace
  callback_monitor: val/Dice
  disable_infer_num_classes: false
  engine:
    task: SEMANTIC_SEGMENTATION
    device: auto
  data:
  ...

.. note::

  If you want to get configuration as yaml file, please use ``--print_config`` parameter and ``> configs.yaml``.

  .. code-block:: shell

    (otx) ...$ otx train --config src/otx/recipe/semantic_segmentation/litehrnet_18.yaml --data_root tests/assests/common_semantic_segmentation_dataset --print_config > configs.yaml
    # Update configs.yaml & Train configs.yaml
    (otx) ...$ otx train --config configs.yaml

3. To start training we need to call ``otx train``

Here are the main outputs can expect with CLI:
- ``{work_dir}/{timestamp}/checkpoints/epoch_*.ckpt`` - a model checkpoint file.
- ``{work_dir}/{timestamp}/configs.yaml`` - The configuration file used in the training can be reused to reproduce the training.
- ``{work_dir}/.latest`` - The results of each of the most recently executed subcommands are soft-linked. This allows you to skip checkpoints and config file entry as a workspace.

.. tab-set::

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx train --config src/otx/recipe/semantic_segmentation/litehrnet_18.yaml --data_root tests/assests/common_semantic_segmentation_dataset

    .. tab-item:: API (from_config)

        .. code-block:: python

            from otx.backend.native.engine import OTXEngine

            data_root = "tests/assests/common_semantic_segmentation_dataset"
            recipe = "src/otx/recipe/semantic_segmentation/litehrnet_18.yaml"

            engine = OTXEngine.from_config(
                      config_path=recipe,
                      data_root=data_root,
                      work_dir="otx-workspace",
                    )


            # one more possibility to obtain the right engine by the given model/dataset
            # using "create_engine" function
            from otx.engine import create_engine
            engine = create_engine(
                      model=recipe,
                      data=data_root,
                    )

            engine.train(...)

    .. tab-item:: API

        .. code-block:: python

            from otx.backend.native.engine import OTXEngine
            from otx.backend.native.models import LiteHRNet

            data_root = "tests/assests/common_semantic_segmentation_dataset"
            model = LiteHRNet(
                model_name = "lite_hrnet_18",
                label_info = {"label_names": ["Background", "Rectangle"],
                              "label_id": [0, 1],
                              "label_groups": [["Background", "Rectangle"]]},
                data_input_params = {"input_size": [512, 512],
                                     "mean": [123.675, 116.28, 103.53],
                                     "std": [58.395, 57.12, 57.375]}
            )

            engine = OTXEngine(
                      model=model,
                      data_root=data_root,
                      work_dir="otx-workspace",
                    )

            # one more possibility to obtain the right engine by the given model/dataset
            # using "create_engine" function
            from otx.engine import create_engine
            engine = create_engine(
                      model=model,
                      data=data_root,
                    )

            engine.train(...)

The training time highly relies on the hardware characteristics, for example on 1 NVIDIA GeForce RTX 3090 the training took about 18 seconds with full dataset.

4. ``(Optional)`` Additionally, we can tune training parameters such as batch size, learning rate, patience epochs or warm-up iterations.
Learn more about recipe-specific parameters using ``otx train params --help``.

It can be done by manually updating parameters in the ``configs.yaml`` file in your workplace or via the command line.

For example, to decrease the batch size to 4, fix the number of epochs to 100 and disable early stopping, extend the command line above with the following line.

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

            engine = OTXEngine(..., data=datamodule)

            engine.train(max_epochs=100)


5. The training result ``checkpoints/*.ckpt`` file is located in ``{work_dir}`` folder,
while training logs can be found in the ``{work_dir}/{timestamp}`` dir.

.. note::
  We also can visualize the training using ``Tensorboard`` as these logs are located in ``{work_dir}/{timestamp}/tensorboard``.

.. code-block::

  otx-workspace
    ├── 20240403_134256/
    |   ├── csv/
    |   ├── checkpoints/
    |   |   └── epoch_*.pth
    |   ├── tensorboard/
    |   └── configs.yaml
    └── .latest
        └── train/
  ...

After that, we have the PyTorch semantic segmentation model trained with OpenVINO™ Training Extensions, which we can use for evaluation, export, optimization and deployment.

6. It is also possible to resume training from the last checkpoint.
For this, we can use the ``--resume`` parameter with the path to the checkpoint file.

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx train --config src/otx/recipe/semantic_segmentation/litehrnet_18.yaml \
                                  --data_root tests/assets/common_semantic_segmentation_dataset \
                                  --checkpoint otx-workspace/20240403_134256/checkpoints/epoch_014.ckpt \
                                  --resume True

    .. tab-item:: API

        .. code-block:: python

            engine.train(resume=True,
                         checkpoint="otx-workspace/20240403_134256/checkpoints/epoch_014.ckpt")

***********
Validation
***********

1. ``otx test`` runs evaluation of a trained
model on a specific dataset.

The test function receives test annotation information and model snapshot, trained in the previous step.

``otx test`` will output a Dice for semantic segmentation.

2. The command below will run validation on our dataset
and save performance results in ``otx-workspace``:

.. tab-set::

    .. tab-item:: CLI (with work_dir)

        .. code-block:: shell

            (otx) ...$ otx test --work_dir otx-workspace
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃        Test metric        ┃       DataLoader 0        ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │      test/Dice            │   0.1556396484375         │
            └───────────────────────────┴───────────────────────────┘

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx test --config  src/otx/recipe/semantic_segmentation/maskrcnn_r50.yaml \
                                --data_root tests/assets/common_semantic_segmentation_dataset \
                                --checkpoint otx-workspace/20240312_051135/checkpoints/epoch_059.ckpt
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃        Test metric        ┃       DataLoader 0        ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │      test/Dice            │   0.1556396484375         │
            └───────────────────────────┴───────────────────────────┘

    .. tab-item:: API

        .. code-block:: python

            engine.test()


3. The output of ``{work_dir}/{timestamp}/csv/version_0/metrics.csv`` consists of
a dict with target metric name and its value.

The next tutorial on how to export, optimize, and deploy the model is available at :doc:`../export`.
