Adaptive Training
==================

Adaptive-training focuses to adjust the number of iterations or interval for the validation to achieve the fast training.
In the small data regime, we don't need to validate the model at every epoch since there are a few iterations at a single epoch.
To handle this, we have implemented module named ``AdaptiveTrainScheduling``. This callback controls the interval of the validation to do faster training.

.. note::
    ``AdaptiveTrainScheduling`` changes the interval of the validation, evaluation and updating learning rate by checking the number of dataset.


.. tab-set::

    .. tab-item:: API

        .. code-block:: python

            from otx.backend.native.engine import OTXEngine
            from otx.backend.native.callbacks.adaptive_train_scheduling import AdaptiveTrainScheduling

            engine = OTXEngine(data_root="<path_to_data_root>", model="path/to/config/model.yaml")
            engine.train(callbacks=[AdaptiveTrainScheduling()])

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx train --config path/to/config/model.yaml --data_root  --callbacks otx.algo.callbacks.adaptive_train_scheduling.AdaptiveTrainScheduling

Auto-adapt batch size
=====================

This feature adapts a batch size based on the current hardware environment.
There are two methods available for adapting the batch size.

1. Prevent GPU Out of Memory (`Safe` mode)

The first method checks if the current batch size is compatible with the available GPU devices.
Larger batch sizes consume more GPU memory for training. Therefore, the system verifies if training is possible with the current batch size.
If it's not feasible, the batch size is decreased to reduce GPU memory usage.
However, setting the batch size too low can slow down training.
To address this, the batch size is reduced to the maximum amount that could be run safely on the current GPU resource.
The learning rate is also adjusted based on the updated batch size accordingly.

To use this feature, add the following parameter:

.. tab-set::

    .. tab-item:: API

        .. code-block:: python

            from otx.backend.native.engine import OTXEngine

            engine = OTXEngine(data_root="<path_to_data_root>", model="path/to/config/model.yaml")
            engine.train(adaptive_bs="Safe")

    .. tab-item:: CLI

        .. code-block:: bash

            (otx) ...$ otx train ...  --adaptive_bs Safe

2. Find the maximum executable batch size (`Full` mode)

The second method aims to find a possible large batch size that reduces the overall training time.
Increasing the batch size reduces the effective number of iterations required to sweep the whole dataset, thus speeds up the end-to-end training.
However, it does not search for the maximum batch size as it is not efficient and may require significantly more time without providing substantial acceleration compared to a large batch size.
Similar to the previous method, the learning rate is adjusted according to the updated batch size accordingly.

To use this feature, add the following parameter:

.. tab-set::

    .. tab-item:: API

        .. code-block:: python

            from otx.backend.native.engine import OTXEngine

            engine = OTXEngine(data_root="<path_to_data_root>", model="path/to/config/model.yaml")
            engine.train(adaptive_bs="Full")

    .. tab-item:: CLI

        .. code-block:: bash

            (otx) ...$ otx train ...  --adaptive_bs Full


.. Warning::
    When using a fixed epoch, training with larger batch sizes is generally faster than with smaller batch sizes.
    However, if early stop is enabled, training with a lower batch size can finish early.


Auto-adapt num_workers
======================

This feature adapts the ``num_workers`` parameter based on the current hardware environment.
The ``num_workers`` parameter controls the number of subprocesses used for data loading during training.
While increasing ``num_workers`` can reduce data loading time, setting it too high can consume a significant amount of CPU memory.

To simplify the process of setting ``num_workers`` manually, this feature automatically determines the optimal value based on the current hardware status.

To use this feature, add the following parameter:

.. tab-set::

    .. tab-item:: API

        .. code-block:: python

            from otx.data.module import OTXDataModule

            datamodule = OTXDataModule(..., auto_num_workers=True)

    .. tab-item:: CLI

        .. code-block:: shell

            (otx) ...$ otx train ... --data.auto_num_workers True
