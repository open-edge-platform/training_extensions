Using Tree-Path KL Divergence for Hierarchical Classification
=============================================================

This tutorial explains how to train hierarchical classification models in
OpenVINO™ Training Extensions with **Tree-Path KL Divergence Loss**, a training-time
regularizer that encourages consistent predictions along the taxonomy path
from root to leaf. The method is implemented in:

- :class:`otx.backend.native.models.classification.losses.tree_path_kl_divergence_loss.TreePathKLDivergenceLoss`
- :class:`otx.backend.native.models.classification.classifier.h_label_classifier.KLHLabelClassifier`

The feature is currently exposed by default in
:class:`otx.backend.native.models.classification.hlabel_models.timm_model.TimmModelHLabelCls`.
Users may adapt other architectures with minimal modifications by adding the
same wrapper (``KLHLabelClassifier``) in their model’s ``_finalize_model()``.

Overview
--------

Hierarchical classification models predict multiple levels of labels
(e.g., manufacturer → family → variant). Standard cross-entropy treats each
level independently, which means models may output **inconsistent**
combinations such as:

- predicting a correct fine-grained leaf but an incompatible ancestor, or
- predicting parents and children belonging to different branches.

Tree-Path KL Divergence introduces a path-consistency objective by comparing:

- the model’s *combined* probability distribution across all levels, and
- a **tree-consistent target distribution** that places probability mass on
  each ground-truth category along the path.

This encourages smooth transitions between hierarchy levels and reduces
structurally invalid predictions.

How It Works
------------

Tree-Path KL Divergence operates on:

- a **list of logits** from each hierarchy level (root → ... → leaf), and
- a **target index** for each corresponding level.

The algorithm implemented in
:class:`TreePathKLDivergenceLoss` performs the following:

1. Concatenates all level logits and applies log-softmax.
2. Constructs a sparse target distribution that allocates equal probability to
   the correct class at each level.
3. Computes KL divergence between the model’s distribution and the path-aware
   target distribution.
4. Scales the result by ``loss_weight`` (typically ``1.0``).

In :class:`KLHLabelClassifier`, this KL term is added to the hierarchical
cross-entropy loss:

- cross-entropy is averaged across all hierarchy levels,
- KL divergence is multiplied by ``kl_weight``,
- ``kl_weight = 0`` disables the KL term completely.

Enabling Tree-Path KL Divergence
--------------------------------

The recommended entry point is the provided recipe:

.. code-block:: text

   recipe/classification/h_label_cls/efficientnet_v2_kl.yaml

This recipe uses :class:`TimmModelHLabelCls` and exposes the argument
``kl_weight`` directly in ``init_args``:

.. code-block:: yaml

   task: H_LABEL_CLS
   model:
     class_path: otx.backend.native.models.classification.hlabel_models.timm_model.TimmModelHLabelCls
     init_args:
       label_info: <LABEL-TREE-INFO>
       model_name: tf_efficientnetv2_s.in21k
       kl_weight: 1.0

Using the CLI
--------------------------------

To train a hierarchical model with Tree-Path KL Divergence, the CLI requires:

- ``--data_root``: a path to a directory containing an **``annotations/`` folder**  
  whose JSON annotation files follow **Datumaro format**.
  See the format specification here:

   https://open-edge-platform.github.io/datumaro/stable/docs/data-formats/datumaro_format.html

- ``--config``: the **path to a recipe YAML file**, such as  
  ``recipe/classification/h_label_cls/efficientnet_v2_kl.yaml``.

A full training command example:

.. code-block:: bash

   (otx) $ otx train \
       --config recipe/classification/h_label_cls/efficientnet_v2_kl.yaml \
       --data_root /path/to/dataset_with_annotations \
       --model.kl_weight 1.0

To disable Tree-Path KL Divergence and train a standard hierarchical model:

.. code-block:: bash

   (otx) $ otx train \
       --config recipe/classification/h_label_cls/efficientnet_v2_kl.yaml \
       --model.kl_weight 0.0

Extending Other Architectures
-----------------------------

Currently, Tree-Path KL Divergence is automatically supported only by
``TimmModelHLabelCls``. To integrate the feature into other architectures, add
the following logic to the model’s ``_finalize_model`` method:

1. Accept a new ``kl_weight`` argument in the model init.
2. After constructing the underlying model, wrap it as:

   .. code-block:: python

      if self.kl_weight > 0:
          model = KLHLabelClassifier(model, kl_weight=self.kl_weight)

3. Ensure that the model returns a list of logits aligned with the hierarchy.

Only a few lines are required, and this enables the same training procedure
for any backbone (ResNet, ViT, ConvNeXt, etc.).

When to Use Tree-Path KL Divergence
-----------------------------------

Tree-Path KL Divergence is most helpful when:

- the label space forms a strict taxonomy,
- incorrect parent/child combinations are undesirable,
- fine-grained classes are scarce and benefit from structural priors,
- you want improved consistency across hierarchy levels.

Practically, start with:

- ``kl_weight = 1.0`` or ``2.0`` for most datasets,
- monitor both fine-grained and coarse-level accuracy,
- adjust ``kl_weight`` based on the trade-off between accuracy and
  hierarchical consistency.

Practical Tips
--------------

- Ensure that ``label_info`` correctly describes the hierarchy.
- Excessively large ``kl_weight`` values may over-regularize the model.
- For benchmarking, compare:
  - ``kl_weight = 0`` (baseline),
  - ``kl_weight = 1–4`` (KL-enabled variants).
- Tree-Path KL acts as a *training-time* consistency constraint; it does not
  modify architecture or inference cost.

Limitations
-----------

- Supported out-of-the-box only for :class:`TimmModelHLabelCls`.
- Requires the model to output logits for **each level** of the hierarchy.
- Not applicable to flat classification tasks.


