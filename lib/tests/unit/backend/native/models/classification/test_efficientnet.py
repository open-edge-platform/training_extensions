# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from otx.backend.native.models.base import DataInputParams
from otx.backend.native.models.classification.classifier import ImageClassifier
from otx.backend.native.models.classification.hlabel_models.efficientnet import EfficientNetHLabelCls
from otx.backend.native.models.classification.multiclass_models.efficientnet import EfficientNetMulticlassCls
from otx.backend.native.models.classification.multilabel_models.efficientnet import EfficientNetMultilabelCls
from otx.data.entity.base import OTXBatchLossEntity
from otx.data.entity.torch import OTXPredBatch


@pytest.fixture()
def fxt_multi_class_cls_model():
    return EfficientNetMulticlassCls(
        model_name="efficientnet_b0",
        label_info=10,
        data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    )


class TestEfficientNetForMulticlassCls:
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
        model = EfficientNetMulticlassCls(
            model_name="efficientnet_b0",
            label_info=10,
            data_input_params=data_input_params,
        )
        assert model.model.backbone.in_size == data_input_params.input_size[-2:]

    def test_freeze_backbone(self):
        data_input_params = DataInputParams((300, 300), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

        model = EfficientNetMulticlassCls(
            model_name="efficientnet_b0",
            label_info=10,
            data_input_params=data_input_params,
            freeze_backbone=True,
        )

        classification_layers = model._identify_classification_layers()
        assert all(param.requires_grad == (name in classification_layers) for name, param in model.named_parameters())

        model = EfficientNetMulticlassCls(
            model_name="efficientnet_b0",
            label_info=10,
            data_input_params=data_input_params,
            freeze_backbone=False,
        )

        assert all(param.requires_grad for param in model.parameters())


@pytest.fixture()
def fxt_multi_label_cls_model():
    return EfficientNetMultilabelCls(
        model_name="efficientnet_b0",
        label_info=10,
        data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    )


class TestEfficientNetForMultilabelCls:
    def test_create_model(self, fxt_multi_label_cls_model):
        assert isinstance(fxt_multi_label_cls_model.model, ImageClassifier)

    def test_customize_inputs(self, fxt_multi_label_cls_model, fxt_multilabel_cls_batch_data_entity):
        outputs = fxt_multi_label_cls_model._customize_inputs(fxt_multilabel_cls_batch_data_entity)
        assert "images" in outputs
        assert "labels" in outputs
        assert "mode" in outputs

    def test_customize_outputs(self, fxt_multi_label_cls_model, fxt_multilabel_cls_batch_data_entity):
        outputs = torch.randn(2, 10)
        fxt_multi_label_cls_model.training = True
        preds = fxt_multi_label_cls_model._customize_outputs(outputs, fxt_multilabel_cls_batch_data_entity)
        assert isinstance(preds, OTXBatchLossEntity)

        fxt_multi_label_cls_model.training = False
        preds = fxt_multi_label_cls_model._customize_outputs(outputs, fxt_multilabel_cls_batch_data_entity)
        assert isinstance(preds, OTXPredBatch)

    @pytest.mark.parametrize("explain_mode", [True, False])
    def test_predict_step(self, fxt_multi_label_cls_model, fxt_multilabel_cls_batch_data_entity, explain_mode):
        fxt_multi_label_cls_model.eval()
        fxt_multi_label_cls_model.explain_mode = explain_mode
        outputs = fxt_multi_label_cls_model.predict_step(batch=fxt_multilabel_cls_batch_data_entity, batch_idx=0)

        assert isinstance(outputs, OTXPredBatch)
        assert outputs.has_xai_outputs == explain_mode

    def test_set_input_size(self):
        data_input_params = DataInputParams((300, 300), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        model = EfficientNetMultilabelCls(
            model_name="efficientnet_b0",
            label_info=10,
            data_input_params=data_input_params,
        )
        assert model.model.backbone.in_size == data_input_params.input_size[-2:]

    def test_freeze_backbone(self):
        data_input_params = DataInputParams((300, 300), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

        model = EfficientNetMultilabelCls(
            model_name="efficientnet_b0",
            label_info=10,
            data_input_params=data_input_params,
            freeze_backbone=True,
        )

        classification_layers = model._identify_classification_layers()
        assert all(param.requires_grad == (name in classification_layers) for name, param in model.named_parameters())

        model = EfficientNetMultilabelCls(
            model_name="efficientnet_b0",
            label_info=10,
            data_input_params=data_input_params,
            freeze_backbone=False,
        )

        assert all(param.requires_grad for param in model.parameters())


@pytest.fixture()
def fxt_h_label_cls_model(fxt_hlabel_cifar):
    return EfficientNetHLabelCls(
        model_name="efficientnet_b0",
        label_info=fxt_hlabel_cifar,
        data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    )


class TestEfficientNetForHLabelCls:
    def test_create_model(self, fxt_h_label_cls_model):
        assert isinstance(fxt_h_label_cls_model.model, ImageClassifier)

    def test_customize_inputs(self, fxt_h_label_cls_model, fxt_hlabel_cls_batch_data_entity):
        outputs = fxt_h_label_cls_model._customize_inputs(fxt_hlabel_cls_batch_data_entity)
        assert "images" in outputs
        assert "labels" in outputs
        assert "mode" in outputs

    def test_customize_outputs(self, fxt_h_label_cls_model, fxt_hlabel_cls_batch_data_entity):
        outputs = torch.randn(2, 10)
        fxt_h_label_cls_model.training = True
        preds = fxt_h_label_cls_model._customize_outputs(outputs, fxt_hlabel_cls_batch_data_entity)
        assert isinstance(preds, OTXBatchLossEntity)

        fxt_h_label_cls_model.training = False
        preds = fxt_h_label_cls_model._customize_outputs(outputs, fxt_hlabel_cls_batch_data_entity)
        assert isinstance(preds, OTXPredBatch)

    @pytest.mark.parametrize("explain_mode", [True, False])
    def test_predict_step(self, fxt_h_label_cls_model, fxt_hlabel_cls_batch_data_entity, explain_mode):
        fxt_h_label_cls_model.eval()
        fxt_h_label_cls_model.explain_mode = explain_mode
        outputs = fxt_h_label_cls_model.predict_step(batch=fxt_hlabel_cls_batch_data_entity, batch_idx=0)

        assert isinstance(outputs, OTXPredBatch)
        assert outputs.has_xai_outputs == explain_mode

    def test_set_input_size(self, fxt_hlabel_data):
        data_input_params = DataInputParams((300, 300), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        model = EfficientNetHLabelCls(
            model_name="efficientnet_b0",
            label_info=fxt_hlabel_data,
            data_input_params=data_input_params,
        )
        assert model.model.backbone.in_size == data_input_params.input_size[-2:]

    def test_freeze_backbone(self, fxt_hlabel_data):
        data_input_params = DataInputParams((300, 300), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

        model = EfficientNetHLabelCls(
            model_name="efficientnet_b0",
            label_info=fxt_hlabel_data,
            data_input_params=data_input_params,
            freeze_backbone=True,
        )

        classification_layers = model._identify_classification_layers()
        assert all(param.requires_grad == (name in classification_layers) for name, param in model.named_parameters())

        model = EfficientNetHLabelCls(
            model_name="efficientnet_b0",
            label_info=fxt_hlabel_data,
            data_input_params=data_input_params,
            freeze_backbone=False,
        )

        assert all(param.requires_grad for param in model.parameters())
