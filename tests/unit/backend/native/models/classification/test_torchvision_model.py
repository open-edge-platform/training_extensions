# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from otx.backend.native.models.base import DataInputParams
from otx.backend.native.models.classification.classifier import ImageClassifier
from otx.backend.native.models.classification.heads import LinearClsHead
from otx.backend.native.models.classification.hlabel_models.torchvision_model import TVModelHLabelCls
from otx.backend.native.models.classification.multiclass_models.torchvision_model import TVModelMulticlassCls
from otx.backend.native.models.classification.multilabel_models.torchvision_model import TVModelMultilabelCls
from otx.data.entity.base import OTXBatchLossEntity
from otx.data.entity.torch import OTXPredBatch
from otx.types.export import TaskLevelExportParameters
from otx.types.task import OTXTaskType


@pytest.fixture()
def fxt_tv_model():
    return TVModelMulticlassCls(
        model_name="mobilenet_v3_small",
        label_info=10,
        data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    )


@pytest.fixture()
def fxt_tv_model_and_data_entity(
    request,
    fxt_multiclass_cls_batch_data_entity,
    fxt_multilabel_cls_batch_data_entity,
    fxt_hlabel_cls_batch_data_entity,
    fxt_hlabel_multilabel_info,
):
    if request.param == OTXTaskType.MULTI_CLASS_CLS:
        return TVModelMulticlassCls(
            model_name="mobilenet_v3_small",
            label_info=10,
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ), fxt_multiclass_cls_batch_data_entity
    if request.param == OTXTaskType.MULTI_LABEL_CLS:
        return TVModelMultilabelCls(
            model_name="mobilenet_v3_small",
            label_info=10,
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ), fxt_multilabel_cls_batch_data_entity
    if request.param == OTXTaskType.H_LABEL_CLS:
        return TVModelHLabelCls(
            model_name="mobilenet_v3_small",
            label_info=fxt_hlabel_multilabel_info,
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ), fxt_hlabel_cls_batch_data_entity
    return None


class TestOTXTVModel:
    def test_create_model(self, fxt_tv_model):
        assert isinstance(fxt_tv_model.model, ImageClassifier)

        model = TVModelMulticlassCls(
            model_name="mobilenet_v3_small",
            label_info=10,
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        assert isinstance(model.model.head, LinearClsHead)

    @pytest.mark.parametrize(
        "fxt_tv_model_and_data_entity",
        [OTXTaskType.MULTI_CLASS_CLS, OTXTaskType.MULTI_LABEL_CLS, OTXTaskType.H_LABEL_CLS],
        indirect=True,
    )
    def test_customize_inputs(self, fxt_tv_model_and_data_entity):
        tv_model, data_entity = fxt_tv_model_and_data_entity
        outputs = tv_model._customize_inputs(data_entity)
        assert "images" in outputs
        assert "labels" in outputs
        assert "mode" in outputs

    @pytest.mark.parametrize(
        "fxt_tv_model_and_data_entity",
        [OTXTaskType.MULTI_CLASS_CLS, OTXTaskType.MULTI_LABEL_CLS, OTXTaskType.H_LABEL_CLS],
        indirect=True,
    )
    def test_customize_outputs(self, fxt_tv_model_and_data_entity):
        tv_model, data_entity = fxt_tv_model_and_data_entity
        outputs = torch.randn(2, 10)
        tv_model.training = True
        preds = tv_model._customize_outputs(outputs, data_entity)
        assert isinstance(preds, OTXBatchLossEntity)

        tv_model.training = False
        preds = tv_model._customize_outputs(outputs, data_entity)
        assert isinstance(preds, OTXPredBatch)

    def test_export_parameters(self, fxt_tv_model):
        export_parameters = fxt_tv_model._export_parameters
        assert isinstance(export_parameters, TaskLevelExportParameters)
        assert export_parameters.model_type == "Classification"
        assert export_parameters.task_type == "classification"

    @pytest.mark.parametrize("explain_mode", [True, False])
    def test_predict_step(self, fxt_tv_model, fxt_multiclass_cls_batch_data_entity, explain_mode):
        fxt_tv_model.eval()
        fxt_tv_model.explain_mode = explain_mode
        outputs = fxt_tv_model.predict_step(batch=fxt_multiclass_cls_batch_data_entity, batch_idx=0)

        assert isinstance(outputs, OTXPredBatch)
        assert outputs.has_xai_outputs == explain_mode
        if explain_mode:
            assert outputs.feature_vector[0].ndim == 2
            assert outputs.saliency_map[0].ndim == 3
            assert outputs.saliency_map[0].shape[-2:] != torch.Size([1, 1])

    @pytest.mark.parametrize(
        ("model_cls", "label_info_param"),
        [
            (TVModelMulticlassCls, 10),
            (TVModelMultilabelCls, 10),
            (TVModelHLabelCls, "fxt_hlabel_multilabel_info"),
        ],
        ids=["multiclass", "multilabel", "hlabel"],
    )
    def test_freeze_backbone(self, model_cls, label_info_param, request):
        data_input_params = DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

        if isinstance(label_info_param, str):
            label_info = request.getfixturevalue(label_info_param)
        else:
            label_info = label_info_param

        # Test with freeze_backbone=True
        model = model_cls(
            model_name="mobilenet_v3_small",
            label_info=label_info,
            data_input_params=data_input_params,
            freeze_backbone=True,
        )

        classification_layers = model._identify_classification_layers()
        assert all(param.requires_grad == (name in classification_layers) for name, param in model.named_parameters())

        # Test with freeze_backbone=False
        model = model_cls(
            model_name="mobilenet_v3_small",
            label_info=label_info,
            data_input_params=data_input_params,
            freeze_backbone=False,
        )

        assert all(param.requires_grad for param in model.parameters())
