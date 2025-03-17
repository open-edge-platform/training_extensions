# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Segment Anything model for the OTX visual prompting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
from torch import Tensor, nn

from otx.algo.visual_prompting.decoders import SAMMaskDecoder
from otx.algo.visual_prompting.encoders import SAMImageEncoder, SAMPromptEncoder
from otx.algo.visual_prompting.losses.sam_loss import SAMCriterion
from otx.algo.visual_prompting.visual_prompters import SegmentAnything
from otx.core.metrics.visual_prompting import VisualPromptingMetricCallable
from otx.core.model.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.visual_prompting import OTXVisualPromptingModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import LabelInfoTypes, NullLabelInfo

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


class SAM(OTXVisualPromptingModel):
    """OTX visual prompting model class for Segment Anything Model (SAM).

    Attributes:
        pretrained_weights (dict[str, str]): Dictionary containing URLs for pretrained weights.
        input_size_multiplier (int): Multiplier for input size.

    Args:
        data_input_params (DataInputParams): Parameters for data input.
        label_info (LabelInfoTypes, optional): Information about labels. Defaults to NullLabelInfo().
        model_name (Literal["tiny_vit", "vit_b"], optional): Name of the model to use. Defaults to "tiny_vit".
        optimizer (OptimizerCallable, optional): Callable for optimizer. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Callable for learning rate scheduler.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Callable for metric. Defaults to VisualPromptingMetricCallable.
        torch_compile (bool, optional): Whether to use torch compile. Defaults to False.
        freeze_image_encoder (bool, optional): Whether to freeze the image encoder. Defaults to True.
        freeze_prompt_encoder (bool, optional): Whether to freeze the prompt encoder. Defaults to True.
        freeze_mask_decoder (bool, optional): Whether to freeze the mask decoder. Defaults to False.
        use_stability_score (bool, optional): Whether to use stability score. Defaults to False.
        return_single_mask (bool, optional): Whether to return a single mask. Defaults to True.
        return_extra_metrics (bool, optional): Whether to return extra metrics. Defaults to False.
        stability_score_offset (float, optional): Offset for stability score. Defaults to 1.0.

    """

    pretrained_weights: ClassVar[dict[str, str]] = {
        "tiny_vit": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }

    input_size_multiplier = 16

    def __init__(
        self,
        data_input_params: DataInputParams,
        label_info: LabelInfoTypes = NullLabelInfo(),
        model_name: Literal["tiny_vit", "vit_b"] = "tiny_vit",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = VisualPromptingMetricCallable,
        torch_compile: bool = False,
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        freeze_mask_decoder: bool = False,
        use_stability_score: bool = False,
        return_single_mask: bool = True,
        return_extra_metrics: bool = False,
        stability_score_offset: float = 1.0,
    ) -> None:
        if data_input_params.input_size[0] != data_input_params.input_size[1]:
            msg = f"SAM should use square image size, but got {data_input_params.input_size}"
            raise ValueError(msg)

        self.image_size = data_input_params.input_size[0]
        self.use_stability_score = use_stability_score
        self.return_single_mask = return_single_mask
        self.return_extra_metrics = return_extra_metrics
        self.stability_score_offset = stability_score_offset
        self.image_embedding_size = data_input_params.input_size[0] // self.input_size_multiplier

        super().__init__(
            label_info=label_info,
            model_name=model_name,
            data_input_params=data_input_params,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

        self.load_state_dict(checkpoint=self.pretrained_weights[model_name])
        self.freeze_networks(freeze_image_encoder, freeze_prompt_encoder, freeze_mask_decoder)

    def _create_model(self, num_classes: int | None = None) -> nn.Module:
        image_encoder = SAMImageEncoder(backbone_type=self.model_name, img_size=self.data_input_params.input_size[0])
        prompt_encoder = SAMPromptEncoder(
            image_embedding_size=(self.image_embedding_size, self.image_embedding_size),
            input_image_size=self.data_input_params.input_size,
        )
        mask_decoder = SAMMaskDecoder()
        criterion = SAMCriterion(image_size=self.data_input_params.input_size[0])
        return SegmentAnything(
            image_encoder=image_encoder,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
            criterion=criterion,
            image_size=self.data_input_params.input_size[0],
            use_stability_score=self.use_stability_score,
            return_single_mask=self.return_single_mask,
            return_extra_metrics=self.return_extra_metrics,
            stability_score_offset=self.stability_score_offset,
        )

    def load_state_dict(  # type: ignore[override]
        self,
        checkpoint: str | None = None,
        strict: bool = True,
        assign: bool = False,
    ) -> None:
        """Load checkpoint for SAM.

        This method loads a pre-trained state dictionary for the SAM model. It can load from
        a provided state dictionary or from a URL specified in the `load_from` parameter.

        Args:
            state_dict (dict[str, Any] | None, optional): The state dictionary to load.
                Defaults to None.
            strict (bool, optional): Whether to strictly enforce that the keys in state_dict
                match the keys returned by this module's state_dict() function. Defaults to True.
            assign (bool, optional): Whether to copy parameters instead of moving them.
                Defaults to False.
            load_from (str | None, optional): URL to load the checkpoint from. If provided,
                this will be used instead of the state_dict argument. Defaults to None.

        Raises:
            ValueError: If the checkpoint format is not desirable for torch.hub.load_state_dict_from_url.

        Note:
            If loading from a URL, some keys are removed from the loaded state dictionary
            and a 'model.' prefix is added to all remaining keys.
        """
        if isinstance(checkpoint, str) and checkpoint.startswith("http"):
            _state_dict: dict[str, Any] = torch.hub.load_state_dict_from_url(str(checkpoint))
            for key in [
                "image_encoder.norm_head.weight",
                "image_encoder.norm_head.bias",
                "image_encoder.head.weight",
                "image_encoder.head.bias",
            ]:
                if key in _state_dict:
                    _state_dict.pop(key)

            # add prefix 'model.' to all keys
            for key in list(_state_dict.keys()):
                _state_dict["model." + key] = _state_dict.pop(key)

            state_dict = _state_dict
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            msg = f"Invalid checkpoint type or format: {type(checkpoint)}: {checkpoint}"
            raise ValueError(msg)

        super().load_state_dict(state_dict, strict, assign)  # type: ignore[misc]

    def freeze_networks(
        self,
        freeze_image_encoder: bool,
        freeze_prompt_encoder: bool,
        freeze_mask_decoder: bool,
    ) -> None:
        """Freeze networks depending on config.

        Args:
            freeze_image_encoder (bool): Whether to freeze the image encoder.
            freeze_prompt_encoder (bool): Whether to freeze the prompt encoder.
            freeze_mask_decoder (bool): Whether to freeze the mask decoder.
        """
        for param in self.model.image_encoder.parameters():
            param.requires_grad = not freeze_image_encoder

        for param in self.model.prompt_encoder.parameters():
            param.requires_grad = not freeze_prompt_encoder

        for param in self.model.mask_decoder.parameters():
            param.requires_grad = not freeze_mask_decoder

    @torch.no_grad()
    def forward_for_tracing(
        self,
        image_embeddings: Tensor,
        point_coords: Tensor,
        point_labels: Tensor,
        mask_input: Tensor,
        has_mask_input: Tensor,
        ori_shape: Tensor,
    ) -> tuple[Tensor, ...]:
        """Forward method for SAM inference (export/deploy).

        Args:
            image_embeddings (Tensor): The image embedding with a batch index of length 1.
                If it is a zero tensor, the image embedding will be computed from the image.
            point_coords (Tensor): Coordinates of sparse input prompts,
                corresponding to both point inputs and box inputs.
                Boxes are encoded using two points, one for the top-left corner and one for the bottom-right corner.
                Coordinates must already be transformed to long-side 1024. Has a batch index of length 1.
            point_labels (Tensor): Labels for the sparse input prompts.
                0 is a negative input point, 1 is a positive input point,
                2 is a top-left box corner, 3 is a bottom-right box corner, and -1 is a padding point.
                If there is no box input, a single padding point with label -1 and
                coordinates (0.0, 0.0) should be concatenated.
            mask_input (Tensor): A mask input to the model with shape 1x1x256x256.
                This must be supplied even if there is no mask input. In this case, it can just be zeros.
            has_mask_input (Tensor): An indicator for the mask input.
                1 indicates a mask input, 0 indicates no mask input.
                This input has 1x1 shape due to supporting openvino input layout.
            ori_shape (Tensor): The size of the input image in (H,W) format, before any transformation.
                This input has 1x2 shape due to supporting openvino input layout.
        """
        return self.model.forward_for_tracing(
            image_embeddings=image_embeddings,
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            has_mask_input=has_mask_input,
            ori_shape=ori_shape,
        )
