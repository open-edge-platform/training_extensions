PEFT: Parameter-Efficient Fine-Tuning (LoRA & DoRA) for Classification
======================================================================

.. note::

    PEFT (LoRA, DoRA) is only supported for VisionTransformer models.
    See the method in otx.backend.native.models.classification.utils.peft


Overview
--------

OpenVINO™ Training Extensions supports Parameter-Efficient Fine-Tuning (PEFT) for Transformer classifiers via Low Rank Adaptation (LoRA) and Weight-Decomposed Low-Rank Adaptation (DoRA).
These methods adapt pre-trained models with a small number of additional parameters instead of fully fine-tuning all weights.

Benefits
--------

- **Efficiency**: Minimal extra parameters and faster adaptation.
- **Performance**: Competitive accuracy compared to full fine-tuning.
- **Flexibility**: Apply LoRA or DoRA selectively to model components.

Supported
---------

- **Backbones**: Vision Transformer family (e.g., DINOv2)
- **Tasks**: Multiclass, Multi-label, Hierarchical Label Classification

How to Use PEFT in OpenVINO™ Training Extensions
--------------------------------------------------

.. tab-set::

   .. tab-item:: API

      .. code-block:: python

         from training_extensions.src.otx.backend.native.models.classification.multiclass_models.vit import VisionTransformerMulticlassCls

         # Choose one: "lora" or "dora"
         model = VisionTransformerForMulticlassCls(..., peft="lora")

   .. tab-item:: CLI

      .. code-block:: bash

         (otx) $ otx train ... --model.peft dora

   .. tab-item:: YAML

      .. code-block:: yaml

         task: MULTI_CLASS_CLS
         model:
            class_path: otx.backend.native.models.classification.multiclass_models.vit.VisionTransformerMulticlassCls
            init_args:
               label_info: 1000
               model_name: "dinov2-small"
               peft: "dora"

               optimizer:
                  class_path: torch.optim.AdamW
                  init_args:
                     lr: 0.0001
                     weight_decay: 0.05

Alternative
-----------

- **Linear Fine-Tuning**: Train only the classification head while keeping all backbone frozen.
  This approach works with *all* classification backbones.

How to Use Linear Fine-Tuning
-----------------------------

.. tab-set::

   .. tab-item:: API

      .. code-block:: python

         from training_extensions.src.otx.backend.native.models.classification.multiclass_models.vit import VisionTransformerMulticlassCls

         # Linear FT = freeze_backbone=True, no PEFT
         model = VisionTransformerMulticlassCls(
             ...,
             freeze_backbone=True,
         )

   .. tab-item:: CLI

      .. code-block:: bash

         (otx) $ otx train ... --model.freeze_backbone true

   .. tab-item:: YAML

      .. code-block:: yaml

         task: MULTI_CLASS_CLS
         model:
            class_path: otx.backend.native.models.classification.multiclass_models.vit.VisionTransformerMulticlassCls
            init_args:
               label_info: 1000
               model_name: "dinov2-small"
               peft: ""
               freeze_backbone: true