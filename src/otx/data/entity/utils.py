# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for OTX data entities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.utils._pytree as pytree

if TYPE_CHECKING:
    from otx.data.entity import ImageInfo, OTXDataItem


def register_pytree_node(cls: type[OTXDataItem]) -> type[OTXDataItem]:
    """Decorator to register an OTX data entity with PyTorch's PyTree.

    This decorator should be applied to every OTX data entity, as TorchVision V2 transforms
    use the PyTree to flatten and unflatten the data entity during runtime.

    Example:
        `MulticlassClsDataEntity` example ::

            @register_pytree_node
            @dataclass
            class MulticlassClsDataEntity(OTXDataEntity):
                ...
    """
    flatten_fn = lambda obj: (list(obj.values()), list(obj.keys()))
    unflatten_fn = lambda values, context: cls(**dict(zip(context, values)))
    pytree.register_pytree_node(
        cls,
        flatten_fn=flatten_fn,
        unflatten_fn=unflatten_fn,
    )
    return cls


def stack_batch(
    tensor_list: list[torch.Tensor],
    img_info_list: list[ImageInfo],
    pad_size_divisor: int = 1,
    pad_value: int | float = 0,
) -> tuple[torch.Tensor, list[ImageInfo]]:
    """Stack multiple tensors to form a batch.

    Pad the tensor to the max shape use the right bottom padding mode in these images.
    If ``pad_size_divisor > 0``, add padding to ensure the shape of each dim is
    divisible by ``pad_size_divisor``.

    Args:
        tensor_list (List[Tensor]): A list of tensors with the same dim.
        img_info_list (List[Tensor]): A list of img_info to be updated.
        pad_size_divisor (int): If ``pad_size_divisor > 0``, add padding
            to ensure the shape of each dim is divisible by
            ``pad_size_divisor``. This depends on the model, and many
            models need to be divisible by 32. Defaults to 1
        pad_value (int, float): The padding value. Defaults to 0.

    Returns:
        (tuple[torch.Tensor, list[ImageInfo]]): The n dim tensor and updated a list of ImageInfo.
    """
    dim = tensor_list[0].dim()
    num_img = len(tensor_list)
    all_sizes: torch.Tensor = torch.Tensor([tensor.shape for tensor in tensor_list])
    max_sizes = torch.ceil(torch.max(all_sizes, dim=0)[0] / pad_size_divisor) * pad_size_divisor
    padded_sizes = max_sizes - all_sizes
    # The first dim normally means channel,  which should not be padded.
    padded_sizes[:, 0] = 0
    if padded_sizes.sum() == 0:
        return torch.stack(tensor_list), img_info_list
    # `pad` is the second arguments of `F.pad`. If pad is (1, 2, 3, 4),
    # it means that padding the last dim with 1(left) 2(right), padding the
    # penultimate dim to 3(top) 4(bottom). The order of `pad` is opposite of
    # the `padded_sizes`. Therefore, the `padded_sizes` needs to be reversed,
    # and only odd index of pad should be assigned to keep padding "right" and
    # "bottom".
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    batch_info = []
    for idx, (tensor, info) in enumerate(zip(tensor_list, img_info_list)):
        padded_img = torch.nn.functional.pad(tensor, tuple(pad[idx].tolist()), value=pad_value)
        # update img_info.img_shape
        info.img_shape = padded_img.shape[1:]
        # update img_info.padding
        left, top, right, bottom = info.padding
        info.padding = (left + pad[idx, 0], top + pad[idx, 2], right + pad[idx, 1], bottom + pad[idx, 3])
        # append padded img
        batch_tensor.append(padded_img)
        batch_info.append(info)

    stacked_images = torch.stack(batch_tensor)

    return stacked_images, batch_info
