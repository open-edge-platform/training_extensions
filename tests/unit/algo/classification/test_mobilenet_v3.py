# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from otx.algo.classification.classifier import ImageClassifier
from otx.algo.classification.hlabel_models.mobilenet_v3 import MobileNetV3HLabelCls
from otx.algo.classification.multiclass_models.mobilenet_v3 import MobileNetV3MulticlassCls
from otx.algo.classification.multilabel_models.mobilenet_v3 import MobileNetV3MultilabelCls
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.model.base import DataInputParams
from otx.data.torch import TorchPredBatch


@pytest.fixture()
def fxt_multi_class_cls_model():
    return MobileNetV3MulticlassCls(
        model_name="mobilenetv3_large",
        label_info=10,
        data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    )


class TestMobileNetV3MulticlassCls:
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
        assert isinstance(preds, TorchPredBatch)

    @pytest.mark.parametrize("explain_mode", [True, False])
    def test_predict_step(self, fxt_multi_class_cls_model, fxt_multiclass_cls_batch_data_entity, explain_mode):
        fxt_multi_class_cls_model.eval()
        fxt_multi_class_cls_model.explain_mode = explain_mode
        outputs = fxt_multi_class_cls_model.predict_step(batch=fxt_multiclass_cls_batch_data_entity, batch_idx=0)

        assert isinstance(outputs, TorchPredBatch)
        assert outputs.has_xai_outputs == explain_mode

    def test_set_input_size(self):
        data_input_params = DataInputParams((300, 300), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        model = MobileNetV3MulticlassCls(
            model_name="mobilenetv3_large",
            label_info=10,
            data_input_params=data_input_params,
        )
        assert model.model.backbone.in_size == data_input_params.input_size[-2:]


@pytest.fixture()
def fxt_multi_label_cls_model():
    return MobileNetV3MultilabelCls(
        model_name="mobilenetv3_large",
        label_info=10,
        data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    )


class TestMobileNetV3MultilabelCls:
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
        assert isinstance(preds, TorchPredBatch)

    @pytest.mark.parametrize("explain_mode", [True, False])
    def test_predict_step(self, fxt_multi_label_cls_model, fxt_multilabel_cls_batch_data_entity, explain_mode):
        fxt_multi_label_cls_model.eval()
        fxt_multi_label_cls_model.explain_mode = explain_mode
        outputs = fxt_multi_label_cls_model.predict_step(batch=fxt_multilabel_cls_batch_data_entity, batch_idx=0)

        assert isinstance(outputs, TorchPredBatch)
        assert outputs.has_xai_outputs == explain_mode

    def test_set_input_size(self):
        data_input_params = DataInputParams((300, 300), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        model = MobileNetV3MultilabelCls(
            model_name="mobilenetv3_large",
            label_info=10,
            data_input_params=data_input_params,
        )
        assert model.model.backbone.in_size == data_input_params.input_size[-2:]


@pytest.fixture()
def fxt_h_label_cls_model(fxt_hlabel_cifar):
    return MobileNetV3HLabelCls(
        model_name="mobilenetv3_large",
        label_info=fxt_hlabel_cifar,
        data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    )


class TestMobileNetV3HLabelCls:
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
        assert isinstance(preds, TorchPredBatch)

    @pytest.mark.parametrize("explain_mode", [True, False])
    def test_predict_step(self, fxt_h_label_cls_model, fxt_hlabel_cls_batch_data_entity, explain_mode):
        fxt_h_label_cls_model.eval()
        fxt_h_label_cls_model.explain_mode = explain_mode
        outputs = fxt_h_label_cls_model.predict_step(batch=fxt_hlabel_cls_batch_data_entity, batch_idx=0)

        assert isinstance(outputs, TorchPredBatch)
        assert outputs.has_xai_outputs == explain_mode

    def test_set_input_size(self, fxt_hlabel_data):
        data_input_params = DataInputParams((300, 300), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        model = MobileNetV3HLabelCls(
            model_name="mobilenetv3_large",
            label_info=fxt_hlabel_data,
            data_input_params=data_input_params,
        )
        assert model.model.backbone.in_size == data_input_params.input_size[-2:]
