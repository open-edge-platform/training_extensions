# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from otx.backend.native.models.base import DataInputParams
from otx.backend.native.models.classification.classifier import ImageClassifier
from otx.backend.native.models.classification.multiclass_models.shufflenet_v2 import ShuffleNetV2MulticlassCls
from otx.data.entity.base import OTXBatchLossEntity
from otx.data.entity.torch import OTXPredBatch


@pytest.fixture()
def fxt_multi_class_cls_model():
    return ShuffleNetV2MulticlassCls(
        model_name="shufflenetv2_large",
        label_info=10,
        data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    )


class TestShuffleNetV2MulticlassCls:
    def test_create_model(self, fxt_multi_class_cls_model):
        assert isinstance(fxt_multi_class_cls_model.model, ImageClassifier)

    def test_customize_inputs(self, fxt_multi_class_cls_model, fxt_multiclass_cls_batch_data_entity):
        outputs = fxt_multi_class_cls_model._customize_inputs(fxt_multiclass_cls_batch_data_entity)
        assert "images" in outputs
        assert "labels" in outputs
        assert "mode" in outputs

    def test_customize_outputs(self, fxt_multi_class_cls_model, fxt_multiclass_cls_batch_data_entity):
        outputs = torch.randn(2, 10)
        fxt_multi_class_cls_model.training = True
        preds = fxt_multi_class_cls_model._customize_outputs(outputs, fxt_multiclass_cls_batch_data_entity)
        assert isinstance(preds, OTXBatchLossEntity)

        fxt_multi_class_cls_model.training = False
        preds = fxt_multi_class_cls_model._customize_outputs(outputs, fxt_multiclass_cls_batch_data_entity)
        assert isinstance(preds, OTXPredBatch)

    @pytest.mark.parametrize("explain_mode", [True, False])
    def test_predict_step(self, fxt_multi_class_cls_model, fxt_multiclass_cls_batch_data_entity, explain_mode):
        fxt_multi_class_cls_model.eval()
        fxt_multi_class_cls_model.explain_mode = explain_mode
        outputs = fxt_multi_class_cls_model.predict_step(batch=fxt_multiclass_cls_batch_data_entity, batch_idx=0)

        assert isinstance(outputs, OTXPredBatch)
        assert outputs.has_xai_outputs == explain_mode

    def test_set_input_size(self):
        data_input_params = DataInputParams((300, 300), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        model = ShuffleNetV2MulticlassCls(
            model_name="shufflenetv2_large",
            label_info=10,
            data_input_params=data_input_params,
        )
        assert model.model.backbone.in_size == data_input_params.input_size[-2:]

    def test_freeze_backbone(self):
        data_input_params = DataInputParams((300, 300), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

        model = ShuffleNetV2MulticlassCls(
            model_name="shufflenetv2_large",
            label_info=10,
            data_input_params=data_input_params,
            freeze_backbone=True,
        )

        classification_layers = model._identify_classification_layers()
        assert all(param.requires_grad == (name in classification_layers) for name, param in model.named_parameters())

        model = ShuffleNetV2MulticlassCls(
            model_name="shufflenetv2_large",
            label_info=10,
            data_input_params=data_input_params,
            freeze_backbone=False,
        )

        assert all(param.requires_grad for param in model.parameters())

    def test_small_variant(self):
        model = ShuffleNetV2MulticlassCls(
            model_name="shufflenetv2_small",
            label_info=10,
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        assert isinstance(model.model, ImageClassifier)
