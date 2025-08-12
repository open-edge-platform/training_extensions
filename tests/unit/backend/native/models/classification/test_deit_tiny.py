# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest

from otx.backend.native.models.base import DataInputParams
from otx.backend.native.models.classification.hlabel_models.vit import VisionTransformerHLabelCls
from otx.backend.native.models.classification.multiclass_models.vit import VisionTransformerMulticlassCls
from otx.backend.native.models.classification.multilabel_models.vit import VisionTransformerMultilabelCls
from otx.backend.native.models.utils.support_otx_v1 import OTXv1Helper
from otx.data.entity.base import OTXBatchLossEntity


class TestDeitTiny:
    @pytest.fixture(
        params=[
            (VisionTransformerMulticlassCls, "fxt_multiclass_cls_batch_data_entity", "fxt_multiclass_labelinfo"),
            (VisionTransformerMultilabelCls, "fxt_multilabel_cls_batch_data_entity", "fxt_multilabel_labelinfo"),
            (VisionTransformerHLabelCls, "fxt_hlabel_cls_batch_data_entity", "fxt_hlabel_cifar"),
        ],
        ids=["multiclass", "multilabel", "hlabel"],
    )
    def fxt_model_and_input(self, request):
        model_cls, input_fxt_name, label_info_fxt_name = request.param
        fxt_input = request.getfixturevalue(input_fxt_name)
        fxt_label_info = request.getfixturevalue(label_info_fxt_name)

        model = model_cls(
            label_info=fxt_label_info,
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

        return model, fxt_input

    @pytest.mark.parametrize("explain_mode", [True, False])
    def test_deit_tiny(self, fxt_model_and_input, explain_mode, mocker):
        fxt_model, fxt_input = fxt_model_and_input

        fxt_model.train()
        assert isinstance(fxt_model(fxt_input), OTXBatchLossEntity)

        fxt_model.eval()
        assert not isinstance(fxt_model(fxt_input), OTXBatchLossEntity)

        fxt_model.explain_mode = explain_mode
        preds = fxt_model.predict_step(fxt_input, batch_idx=0)
        assert len(preds.labels) == fxt_input.batch_size
        assert len(preds.scores) == fxt_input.batch_size
        assert preds.has_xai_outputs == explain_mode

        mock_load_ckpt = mocker.patch.object(OTXv1Helper, "load_cls_effnet_b0_ckpt")
        fxt_model.load_from_otx_v1_ckpt({})
        mock_load_ckpt.assert_called_once_with({}, "multiclass", "model.")

    @pytest.mark.parametrize(
        ("model_cls", "label_info_fxt_name"),
        [
            (VisionTransformerMulticlassCls, "fxt_multiclass_labelinfo"),
            (VisionTransformerMultilabelCls, "fxt_multilabel_labelinfo"),
            (VisionTransformerHLabelCls, "fxt_hlabel_cifar"),
        ],
        ids=["multiclass", "multilabel", "hlabel"],
    )
    def test_freeze_backbone(self, model_cls, label_info_fxt_name, request):
        data_input_params = DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        fxt_label_info = request.getfixturevalue(label_info_fxt_name)

        # Test with freeze_backbone=True
        model = model_cls(
            label_info=fxt_label_info,
            data_input_params=data_input_params,
            freeze_backbone=True,
        )

        classification_layers = model._identify_classification_layers()
        assert all(param.requires_grad == (name in classification_layers) for name, param in model.named_parameters())

        # Test with freeze_backbone=False
        model = model_cls(
            label_info=fxt_label_info,
            data_input_params=data_input_params,
            freeze_backbone=False,
        )

        assert all(param.requires_grad for param in model.parameters())
