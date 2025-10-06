Hierarchical Classification Metric Collection
=============================================

.. note::

   The hierarchical classification metrics are designed for structured label spaces (e.g., taxonomies in biology or medicine).
   See the method in ``otx.metrics.hier_metric_collection.hier_metric_collection_callable``.

Overview
--------

OpenVINO™ Training Extensions provides a unified **hierarchical metric collection** for classification tasks
that involve taxonomic or multi-level labels. This extends flat classification metrics (accuracy, mAP) with
hierarchy-aware evaluation.

Benefits
--------

- **Structure-Aware**: Evaluates not only flat accueacy, but also taxonomy-aware metrics.
- **Robustness**: Partial credit is given when higher-level predictions are correct, even if fine-grained labels are wrong.
- **Flexibility**: Works seamlessly across multiclass and hierarchical-label tasks.

Supported
---------

- **Label Types**: 
  - Hierarchical-label classification
- **Tasks**:
  - Taxonomy-aware hierarchical classification

How to Use Hierarchical Metric Collection
-----------------------------------------

.. tab-set::

   .. tab-item:: API

      .. code-block:: python

         from otx.metrics.hier_metric_collection import hier_metric_collection_callable
         from otx.core.types.label import HLabelInfo

         # Suppose label_info is loaded from a Datumaro dataset
         metric = hier_metric_collection_callable(label_info)

         # During training / validation
         metric.update(preds, targets)
         results = metric.compute()

   .. tab-item:: CLI

      .. code-block:: bash

         (otx) $ otx train ... --metric otx.metrics.hier_metric_collection.hier_metric_collection_callable

   .. tab-item:: YAML

      .. code-block:: yaml

         task: H_LABEL_CLS
         model:
            class_path: <your_model_info>
            init_args:
               label_info: <your_label_info>

         metric:
            class_path: otx.metrics.hier_metric_collection.hier_metric_collection_callable

How to Use the Metric Collection with the Engine
-------------------------------------------

.. tab-set::

   .. tab-item:: API

      .. code-block:: python

         from otx.engine import create_engine
         from otx.metrics.hier_metric_collection import hier_metric_collection_callable
         from otx.core.types.label import HLabelInfo

         # 1) Build or load your label_info (e.g., from a Datumaro dataset)
         #    label_info: HLabelInfo = ...

         # 2) Create your model and data objects (specific to your project)
         model = ...
         data = ...

         # 3) Create an Engine and pass the metric callable into train/test
         engine = create_engine(model, data)
         engine.train(metric=hier_metric_collection_callable)   # the Engine will construct the MetricCollection
         engine.test(metric=hier_metric_collection_callable)


   .. tab-item:: What gets computed?

      The callable returns a ``torchmetrics.MetricCollection`` with keys:

      - ``"accuracy"`` — hierarchical head accuracy
      - ``"leaf_accuracy"`` — macro accuracy on the leaf level
      - ``"full_path_accuracy"`` — exact match across all hierarchy levels
      - ``"inconsistent_path_ratio"`` — ratio of parent→child violations in predicted paths
      - ``"weighted_precision"`` — label-count–weighted macro precision across levels

