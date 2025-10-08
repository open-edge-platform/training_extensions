Classification  model
================================

This live example shows how to easily train, validate, optimize and export classification model on the `flowers dataset <https://www.tensorflow.org/hub/tutorials/image_feature_vector#the_flowers_dataset>`_ from TensorFlow.
To learn more about Classification task, refer to :doc:`../../../explanation/algorithms/classification/index`.

.. note::

  To learn deeper how to manage training process of the model including additional parameters and its modification, refer to :doc:`./classification`.

The process has been tested on the following configuration.

- Ubuntu 20.04
- NVIDIA GeForce RTX 3090
- Intel(R) Core(TM) i9-10980XE
- CUDA Toolkit 11.8

.. note::

  While this example shows how to work with :doc:`multi-class classification <../../../explanation/algorithms/classification/multi_class_classification>`, it is easy to extend it for the :doc:`multi-label <../../../explanation/algorithms/classification/multi_label_classification>` or :doc:`hierarchical <../../../explanation/algorithms/classification/hierarhical_classification>` classification.
  Substitute the dataset with a multi-label or hierarchical one. Everything else remains the same.


*************************
Setup virtual environment
*************************

1. You can follow the installation process from a :doc:`quick start guide <../../../get_started/installation>`
to create a universal virtual environment for OpenVINO™ Training Extensions.

2. Activate your virtual
environment:

.. code-block:: shell

  .otx/bin/activate
  . venv/otx/bin/activate

***************************
Dataset preparation
***************************

Download and prepare a `flowers dataset <https://www.tensorflow.org/hub/tutorials/image_feature_vector#the_flowers_dataset>`_
with the following command:

To prepare the classification dataset, need to make the directory for the train/validation and test.
Since this is just example, we'll use the same train/val/test datasets.

.. code-block:: shell

  cd data

  # download and unzip the data
  wget http://download.tensorflow.org/example_images/flower_photos.tgz
  tar -xzvf flower_photos.tgz

  # construct the data structure to insert to the OTX
  cd flower_photos
  mkdir train
  mv daisy dandelion roses sunflowers tulips train
  cp -r train val
  cp -r train test

  # move the original directory
  cd ../..

|

.. image:: ../../../../../utils/images/flowers_example.jpg
  :width: 600

|

Then the final dataset directory likes below,
please keep the exact same name for the train/val/test folder, to identify the dataset.

.. code-block::

  flower_photos
    train
      ├── daisy
      ├── dandelion
      ├── roses
      ├── sunflowers
      ├── tulips
    val
      ├── daisy
      ├── ...
    test
      ├── daisy
      ├── ...

*********
Training
*********

1. First of all, you need to choose which classification model you want to train.
The list of supported recipes for classification is available with the command line below.

.. note::

  The characteristics and detailed comparison of the models could be found in :doc:`Explanation section <../../../explanation/algorithms/classification/multi_class_classification>`.

.. tab-set::

  .. tab-item:: CLI

    .. code-block:: shell

      (otx) ...$ otx find --task MULTI_CLASS_CLS
      ┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
      ┃ Task            ┃ Model Name            ┃ Recipe Path                                                                                                 ┃
      ┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
      │ MULTI_CLASS_CLS │ openvino_model        │ src/otx/recipe/classification/multi_class_cls/openvino_model.yaml                                           │
      │ MULTI_CLASS_CLS │ tv_efficientnet_v2_l  │ src/otx/recipe/classification/multi_class_cls/tv_efficientnet_v2_l.yaml                                     │
      │ MULTI_CLASS_CLS │ dino_v2               │ src/otx/recipe/classification/multi_class_cls/dino_v2.yaml                                                  │
      │ MULTI_CLASS_CLS │ efficientnet_v2       │ src/otx/recipe/classification/multi_class_cls/efficientnet_v2.yaml                                          │
      │ MULTI_CLASS_CLS │ tv_efficientnet_b3    │ src/otx/recipe/classification/multi_class_cls/tv_efficientnet_b3.yaml                                       │
      │ MULTI_CLASS_CLS │ deit_tiny             │ src/otx/recipe/classification/multi_class_cls/deit_tiny.yaml                                                │
      │ MULTI_CLASS_CLS │ mobilenet_v3_large    │ src/otx/recipe/classification/multi_class_cls/mobilenet_v3_large.yaml                                       │
      │ MULTI_CLASS_CLS │ efficientnet_b0       │ src/otx/recipe/classification/multi_class_cls/efficientnet_b0.yaml                                          │
      │ MULTI_CLASS_CLS │ tv_mobilenet_v3_small │ src/otx/recipe/classification/multi_class_cls/tv_mobilenet_v3_small.yaml                                    │
      └─────────────────┴───────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  .. tab-item:: API

    .. code-block:: python

      from otx.backend.native.cli.utils import list_models

      model_lists = list_models(task="MULTI_CLASS_CLS", pattern="*efficient")
      print(model_lists)
      '''
      [
        'efficientnet_b0',
        'efficientnet_v2_light',
        'efficientnet_b0_light',
        ...
      ]
      '''

1. On this step we will prepare custom configuration
with:

- all necessary configs for otx_efficientnet_b0
- train/validation sets, based on provided annotation.

It may be counterintuitive, but for ``--data_root`` we need to pass the path to the dataset folder root (in our case it's ``data/flower_photos``) instead of the folder with validation images.
This is because the function automatically detects annotations and images according to the expected folder structure we achieved above.

Let's check the multi-class classification configuration running the following command:

.. code-block:: shell

  (otx) ...$ otx train --config src/otx/recipe/classification/multi_class_cls/mobilenet_v3_large.yaml  --data_root data/flower_photos --print_config

  ...
  data_root: data/flower_photos
  work_dir: otx-workspace
  callback_monitor: val/accuracy
  disable_infer_num_classes: false
  engine:
    task: MULTI_CLASS_CLS
    device: auto
  data:
  ...

.. note::

    If you want to get configuration as yaml file, please use ``--print_config`` parameter and ``> configs.yaml``.

    .. code-block:: shell

        (otx) ...$ otx train --config  src/otx/recipe/classification/multi_class_cls/mobilenet_v3_large.yaml --data_root data/flower_photos --print_config > configs.yaml
        # Update configs.yaml & Train configs.yaml
        (otx) ...$ otx train --config configs.yaml


3. ``otx train`` trains a model (a particular model recipe)
on a dataset and results:

Here are the main outputs can expect with CLI:
- ``{work_dir}/{timestamp}/checkpoints/epoch_*.ckpt`` - a model checkpoint file.
- ``{work_dir}/{timestamp}/configs.yaml`` - The configuration file used in the training can be reused to reproduce the training.
- ``{work_dir}/.latest`` - The results of each of the most recently executed subcommands are soft-linked. This allows you to skip checkpoints and config file entry as a workspace.

.. tab-set::

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx train --config src/otx/recipe/classification/multi_class_cls/mobilenet_v3_large.yaml --data_root data/flower_photos

    .. tab-item:: API (from_config)

        .. code-block:: python

            from otx.backend.native.engine import OTXEngine

            data_root = "data/flower_photos"
            recipe = "src/otx/recipe/classification/multi_class_cls/mobilenet_v3_large.yaml"

            engine = OTXEngine.from_config(
                      config_path=recipe,
                      data_root=data_root,
                      work_dir="otx-workspace",
                    )

            # it is also possible to pass a config as a model to the Engine directly
            engine = OTXEngine(
                      model=recipe,
                      data=data_root,
                      work_dir="otx-workspace",
                    )

            # one more possibility to obtain the right engine by the given model/dataset
            # using "create_engine" function
            from otx.engine import create_engine
            engine = create_engine(
                      model=recipe,
                      data=data_root,
                    )

            engine.train()

    .. tab-item:: API

        .. code-block:: python

            from otx.backend.native.engine import OTXEngine
            from otx.backend.native.models import MobileNetV3MulticlassCls

            data_root = "data/flower_photos"
            model = MobileNetV3MulticlassCls(label_info = {"label_names": ["daisy", "dandelion", "roses", "sunflowers", "tulips"],
                                                   "label_id": [0, 1, 2, 3, 4],
                                                   "label_groups": [["daisy", "dandelion", "roses", "sunflowers", "tulips"]]},
                                     data_input_params = {"input_size": [224, 224],
                                                         "mean": [123.675, 116.28, 103.53],
                                                         "std": [58.395, 57.12, 57.375]})

            engine = OTXEngine(
                      model=model,
                      data=data_root,
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


4. ``(Optional)`` Additionally, we can tune training parameters such as batch size, learning rate, patience epochs or warm-up iterations.
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

            engine = OTXEngine(..., data=datamodule)

            engine.train(max_epochs=100)


5. The training result ``checkpoints/*.ckpt`` file is located in ``{work_dir}`` folder,
while training logs can be found in the ``{work_dir}/{timestamp}`` dir.

.. note::
    We also can visualize the training using ``Tensorboard`` as these logs are located in ``{work_dir}/{timestamp}/tensorboard``.

.. code-block::

    otx-workspace
    ├── 20240403_134256/
        ├── csv/
        ├── checkpoints/
        |   └── epoch_*.pth
        ├── tensorboard/
        └── configs.yaml
    └── .latest
        └── train/
    ...

The training time highly relies on the hardware characteristics, for example on 1 NVIDIA GeForce RTX 3090 the training took about 3 minutes.

After that, we have the PyTorch multi-class classification model trained with OpenVINO™ Training Extensions, which we can use for evaluation, export, optimization and deployment.

6. It is also possible to resume training from the last checkpoint.
For this, we can use the ``--resume`` parameter with the path to the checkpoint file.

.. tab-set::

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx train --config src/otx/recipe/classification/multi_class_cls/mobilenet_v3_large.yaml \
                                  --data_root data/flower_photos \
                                  --checkpoint otx-workspace/20240403_134256/checkpoints/epoch_014.ckpt \
                                  --resume True

    .. tab-item:: API

        .. code-block:: python

            from otx.backend.native.engine import OTXEngine
            engine = OTXEngine(model="src/otx/recipe/classification/multi_class_cls/mobilenet_v3_large.yaml", data="data/flower_photos", work_dir="otx-workspace")

            engine.train(resume=True,
                         checkpoint="otx-workspace/20240403_134256/checkpoints/epoch_014.ckpt")

***********
Evaluation
***********

1. ``otx test`` runs evaluation of a
trained model on a particular dataset.

Test function receives test annotation information and model snapshot, trained in previous step.

The default metric is accuracy measure.

2. That's how we can evaluate the snapshot in ``otx-workspace``
folder on flower_photos dataset and save results to ``otx-workspace``:

.. tab-set::

    .. tab-item:: CLI (with work_dir)

        .. code-block:: shell

            (otx) ...$ otx test --work_dir otx-workspace
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃        Test metric        ┃       DataLoader 0        ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │      test/data_time       │    0.9929155111312866     │
            │       test/map_50         │    0.0430680550634861     │
            │      test/iter_time       │    0.058606021106243134   │
            └───────────────────────────┴───────────────────────────┘

    .. tab-item:: CLI (with config)

        .. code-block:: shell

            (otx) ...$ otx test --config  src/otx/recipe/classification/multi_class_cls/mobilenet_v3_large.yaml \
                                --data_root data/flower_photos \
                                --checkpoint otx-workspace/20240312_051135/checkpoints/epoch_014.ckpt
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃        Test metric        ┃       DataLoader 0        ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │      test/data_time       │    0.9929155111312866     │
            │       test/map_50         │    0.0430680550634861     │
            │      test/iter_time       │    0.058606021106243134   │
            └───────────────────────────┴───────────────────────────┘

    .. tab-item:: API

        .. code-block:: python

            metric = engine.test()


3. The output of ``{work_dir}/{timestamp}/csv/version_0/metrics.csv`` consists of
a dict with target metric name and its value.

The next tutorial on how to export, optimize, and deploy the model is available at :doc:`../export`.
