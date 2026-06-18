# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Monkey-patches for rfdetr's transformer during torch.export.

rfdetr 1.4.3 lacks the export fix for dynamic batch shapes
(first released in 1.6.4, post-API refactor).  The patched functions below are
vendored from rfdetr 1.5.2's ``rfdetr/models/transformer.py`` (functionally identical
to 1.4.3 for these two functions) with the minimal edits required for tracing.

The patch is activated **only during ``export()``** via a context manager so
train/eval numerics are completely unchanged.
"""

from __future__ import annotations

import contextlib
from typing import Generator

import torch
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Patched gen_encoder_output_proposals (rfdetr 1.5.2 baseline)
# ---------------------------------------------------------------------------
# Changes from upstream 1.5.2:
#  1. Parameter ``spatial_shapes`` renamed to ``spatial_shapes_list`` and
#     iterated directly as Python-int (H,W) tuples.
#  2. ``torch.tensor([H_ for _ in range(N_)], ...)`` replaced with
#     ``torch.full((N_,), H_, …)`` to avoid baking a Python-int batch size
#     into the traced graph (necessary for torch.export / ONNX export with
#     dynamic dims).


def _patched_gen_encoder_output_proposals(
    memory: Tensor,
    memory_padding_mask: Tensor | None,
    spatial_shapes_list: list[tuple[int, int]],
    unsigmoid: bool = True,
) -> tuple[Tensor, Tensor]:
    """Generate encoder output proposals for two-stage RF-DETR.

    This is a modified version of ``gen_encoder_output_proposals`` from
    ``rfdetr.models.transformer`` (v1.5.2) that is safe for torch.export.

    Args:
        memory: Encoder memory, shape ``(bs, sum_hw, d_model)``.
        memory_padding_mask: Padding mask, shape ``(bs, sum_hw)``.
        spatial_shapes_list: List of ``(H, W)`` Python-int tuples per
            feature level.
        unsigmoid: Whether to apply inverse-sigmoid to output proposals.

    Returns:
        output_memory, output_proposals
    """
    n_batch, s_spatial, c_dim = memory.shape
    proposals = []
    _cur = 0
    for _lvl, (h_spatial, w_spatial) in enumerate(spatial_shapes_list):
        if memory_padding_mask is not None:
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + h_spatial * w_spatial)].view(
                n_batch, h_spatial, w_spatial, 1
            )
            valid_h = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_w = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
        else:
            valid_h = torch.full((n_batch,), h_spatial, dtype=torch.float32, device=memory.device)
            valid_w = torch.full((n_batch,), w_spatial, dtype=torch.float32, device=memory.device)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, h_spatial - 1, h_spatial, dtype=torch.float32, device=memory.device),
            torch.linspace(0, w_spatial - 1, w_spatial, dtype=torch.float32, device=memory.device),
            indexing="ij",
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

        scale = torch.cat([valid_w.unsqueeze(-1), valid_h.unsqueeze(-1)], 1).view(n_batch, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(n_batch, -1, -1, -1) + 0.5) / scale

        wh = torch.ones_like(grid) * 0.05 * (2.0**_lvl)

        proposal = torch.cat((grid, wh), -1).view(n_batch, -1, 4)
        proposals.append(proposal)
        _cur += h_spatial * w_spatial

    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)

    if unsigmoid:
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))
    else:
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float(0))

    output_memory = memory
    if memory_padding_mask is not None:
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory.to(memory.dtype), output_proposals.to(memory.dtype)


# ---------------------------------------------------------------------------
# Patched Transformer.forward (rfdetr 1.5.2 baseline)
# ---------------------------------------------------------------------------
# Change from upstream 1.5.2:
#   Build ``spatial_shapes_list`` (Python-int (h,w) tuples) before converting
#   to tensor, and pass the **list** into ``gen_encoder_output_proposals``
#   instead of the tensor.


def _patched_transformer_forward(
    self: nn.Module,
    srcs: list[Tensor],
    masks: list[Tensor] | None,
    pos_embeds: list[Tensor],
    refpoint_embed: Tensor,
    query_feat: Tensor,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    """Patched ``Transformer.forward`` from rfdetr 1.5.2.

    Identical to upstream 1.5.2 except ``spatial_shapes_list`` is passed
    to ``gen_encoder_output_proposals`` instead of the tensor.
    """
    src_flatten: list[Tensor] = []
    mask_flatten: list[Tensor] | None = [] if masks is not None else None
    lvl_pos_embed_flatten: list[Tensor] = []
    spatial_shapes: list[tuple[int, int]] = []
    valid_ratios: list[Tensor] | None = [] if masks is not None else None
    for _lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
        bs, c, h, w = src.shape
        spatial_shape = (h, w)
        spatial_shapes.append(spatial_shape)

        src_flat = src.flatten(2).transpose(1, 2)
        pos_embed_flat = pos_embed.flatten(2).transpose(1, 2)
        lvl_pos_embed_flatten.append(pos_embed_flat)
        src_flatten.append(src_flat)
        if masks is not None:
            if mask_flatten is None:
                raise RuntimeError  # unreachable — kept to satisfy type-checker
            mask = masks[_lvl].flatten(1)
            mask_flatten.append(mask)
    memory = torch.cat(src_flatten, 1)
    if masks is not None:
        mask_flatten = torch.cat(mask_flatten, 1)  # type: ignore[arg-type]
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)  # type: ignore[union-attr]
    lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

    # --- PATCH: keep spatial_shapes_list (Python-int tuples) for export ---
    spatial_shapes_list = spatial_shapes
    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=memory.device)  # type: ignore[assignment]
    # --------------------------------------------------------------------

    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

    if self.two_stage:  # type: ignore[union-attr]
        output_memory, output_proposals = _patched_gen_encoder_output_proposals(
            memory,
            mask_flatten,
            spatial_shapes_list,
            unsigmoid=not self.bbox_reparam,  # type: ignore[union-attr]
        )
        refpoint_embed_ts, memory_ts, boxes_ts = [], [], []
        group_detr = self.group_detr if self.training else 1  # type: ignore[union-attr]
        for g_idx in range(group_detr):
            output_memory_gidx = self.enc_output_norm[g_idx](self.enc_output[g_idx](output_memory))  # type: ignore[union-attr]

            enc_outputs_class_unselected_gidx = self.enc_out_class_embed[g_idx](output_memory_gidx)  # type: ignore[union-attr]
            if self.bbox_reparam:  # type: ignore[union-attr]
                enc_outputs_coord_delta_gidx = self.enc_out_bbox_embed[g_idx](output_memory_gidx)  # type: ignore[union-attr]
                enc_outputs_coord_cxcy_gidx = (
                    enc_outputs_coord_delta_gidx[..., :2] * output_proposals[..., 2:] + output_proposals[..., :2]
                )
                enc_outputs_coord_wh_gidx = enc_outputs_coord_delta_gidx[..., 2:].exp() * output_proposals[..., 2:]
                enc_outputs_coord_unselected_gidx = torch.concat(
                    [enc_outputs_coord_cxcy_gidx, enc_outputs_coord_wh_gidx], dim=-1
                )
            else:
                enc_outputs_coord_unselected_gidx = (
                    self.enc_out_bbox_embed[g_idx](output_memory_gidx) + output_proposals  # type: ignore[union-attr]
                )

            topk = min(self.num_queries, enc_outputs_class_unselected_gidx.shape[-2])  # type: ignore[union-attr]
            topk_proposals_gidx = torch.topk(enc_outputs_class_unselected_gidx.max(-1)[0], topk, dim=1)[1]

            refpoint_embed_gidx_undetach = torch.gather(
                enc_outputs_coord_unselected_gidx, 1, topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, 4)
            )
            refpoint_embed_gidx = refpoint_embed_gidx_undetach.detach()

            tgt_undetach_gidx = torch.gather(
                output_memory_gidx,
                1,
                topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, self.d_model),  # type: ignore[union-attr]
            )

            refpoint_embed_ts.append(refpoint_embed_gidx)
            memory_ts.append(tgt_undetach_gidx)
            boxes_ts.append(refpoint_embed_gidx_undetach)
        refpoint_embed_ts = torch.cat(refpoint_embed_ts, dim=1)
        memory_ts = torch.cat(memory_ts, dim=1)
        boxes_ts = torch.cat(boxes_ts, dim=1)

    if self.dec_layers > 0:  # type: ignore[union-attr]
        tgt = query_feat.unsqueeze(0).repeat(bs, 1, 1)
        refpoint_embed = refpoint_embed.unsqueeze(0).repeat(bs, 1, 1)
        if self.two_stage:  # type: ignore[union-attr]
            ts_len = refpoint_embed_ts.shape[-2]
            refpoint_embed_ts_subset = refpoint_embed[..., :ts_len, :]
            refpoint_embed_subset = refpoint_embed[..., ts_len:, :]

            if self.bbox_reparam:  # type: ignore[union-attr]
                refpoint_embed_cxcy = refpoint_embed_ts_subset[..., :2] * refpoint_embed_ts[..., 2:]
                refpoint_embed_cxcy = refpoint_embed_cxcy + refpoint_embed_ts[..., :2]
                refpoint_embed_wh = refpoint_embed_ts_subset[..., 2:].exp() * refpoint_embed_ts[..., 2:]
                refpoint_embed_ts_subset = torch.concat([refpoint_embed_cxcy, refpoint_embed_wh], dim=-1)
            else:
                refpoint_embed_ts_subset = refpoint_embed_ts_subset + refpoint_embed_ts

            refpoint_embed = torch.concat([refpoint_embed_ts_subset, refpoint_embed_subset], dim=-2)

        hs, references = self.decoder(  # type: ignore[union-attr]
            tgt,
            memory,
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten,
            refpoints_unsigmoid=refpoint_embed,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios.to(memory.dtype) if valid_ratios is not None else valid_ratios,
        )
    else:
        hs = None
        references = None

    if self.two_stage:  # type: ignore[union-attr]
        if self.bbox_reparam:  # type: ignore[union-attr]
            return hs, references, memory_ts, boxes_ts
        return hs, references, memory_ts, boxes_ts.sigmoid()
    return hs, references, None, None


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def patched_rfdetr_transformer_for_export() -> Generator[None, None, None]:
    """Context manager that swaps rfdetr's transformer with export-safe versions.

    Monkey-patches ``rfdetr.models.transformer.gen_encoder_output_proposals``
    and ``rfdetr.models.transformer.Transformer.forward`` for the duration
    of the context, restoring the originals on exit.

    Intended to wrap the ``super().export()`` call inside the mixin's
    ``export()`` override so the patch is active only during export tracing.
    """
    import rfdetr.models.transformer as _t

    _orig_gen = _t.gen_encoder_output_proposals
    _orig_forward = _t.Transformer.forward

    _t.gen_encoder_output_proposals = _patched_gen_encoder_output_proposals  # type: ignore[assignment]
    _t.Transformer.forward = _patched_transformer_forward  # type: ignore[assignment]

    try:
        yield
    finally:
        _t.gen_encoder_output_proposals = _orig_gen  # type: ignore[assignment]
        _t.Transformer.forward = _orig_forward  # type: ignore[assignment]
