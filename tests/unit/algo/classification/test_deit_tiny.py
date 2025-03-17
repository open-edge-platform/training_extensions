# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import pytest

from otx.algo.classification.hlabel_models.vit import VisionTransformerHLabelCls
from otx.algo.classification.multiclass_models.vit import VisionTransformerMulticlassCls
from otx.algo.classification.multilabel_models.vit import VisionTransformerMultilabelCls
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.model.base import DataInputParams


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
