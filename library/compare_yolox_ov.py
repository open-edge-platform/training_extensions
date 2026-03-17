"""
Compare OTX YOLOX (Torch) vs exported OpenVINO model.

Usage
-----
python compare_yolox_ov.py \
    --checkpoint  /path/to/checkpoint.ckpt \
    --model_name  yolox_s \
    --num_classes 80 \
    --ov_model    /path/to/model.xml \
    --images_dir  /path/to/images \
    [--annotations /path/to/coco_annotations.json] \
    [--input_size  640 640] \
    [--with_nms | --no_nms]   (default: --with_nms)
    [--device     cpu]
    [--score_thr  0.01]
    [--iou_thr    0.65]

Notes
-----
* Both torch and OV models receive the **same** preprocessed tensor
  (letterbox-resized, mean/std-normalized, optionally channel-swapped).
* For proper mAP, supply --annotations (COCO JSON).  Without it the
  script still runs and prints per-image box statistics.
* Supports two OV model variants:
    --with_nms : model outputs boxes [B,N,5] + labels [B,N]
    --no_nms   : model outputs bboxes[B,anchors,4] + scores[B,anchors,C]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import tv_tensors

# ── path setup ───────────────────────────────────────────────────────────────
LIB_DIR = Path(__file__).resolve().parent
if str(LIB_DIR / "src") not in sys.path:
    sys.path.insert(0, str(LIB_DIR / "src"))

import openvino as ov  # noqa: E402  (after path setup)
from otx.backend.native.models.base import DataInputParams  # noqa: E402
from otx.backend.native.models.detection.yolox import YOLOX  # noqa: E402
from otx.data.entity.base import ImageInfo  # noqa: E402
from otx.data.entity.sample import OTXSampleBatch  # noqa: E402
from otx.data.entity.utils import stack_batch  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def letterbox(
    img: np.ndarray,
    target_hw: tuple[int, int],
    pad_value: int = 114,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Letterbox-resize *img* (H,W,C uint8 BGR) to *target_hw*.

    Returns
    -------
    padded  : ndarray, shape (*target_hw, C), dtype uint8
    scale   : float   (applied to the image before padding)
    pad_tl  : (pad_top, pad_left)  pixel offsets of the resized image
    """
    th, tw = target_hw
    h, w = img.shape[:2]
    scale = min(th / h, tw / w)
    nh, nw = round(h * scale), round(w * scale)

    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Center-pad to target size
    pad_top = (th - nh) // 2
    pad_left = (tw - nw) // 2
    pad_bottom = th - nh - pad_top
    pad_right = tw - nw - pad_left

    padded = cv2.copyMakeBorder(
        img_resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )
    return padded, scale, (pad_top, pad_left)


def preprocess_image(
    img_bgr: np.ndarray,
    data_params: DataInputParams,
    swap_rgb: bool,
) -> tuple[torch.Tensor, float, tuple[int, int], tuple[int, int]]:
    """Full preprocessing pipeline: letterbox → channel-swap → normalize.

    Returns
    -------
    tensor     : float32 tensor (1, C, H, W), ready for torch/OV inference
    scale      : letterbox scale factor
    pad_tl     : (pad_top, pad_left)
    ori_hw     : original (H, W) before letterbox
    """
    ori_h, ori_w = img_bgr.shape[:2]
    target_hw = data_params.input_size  # (H, W)

    padded, scale, pad_tl = letterbox(img_bgr, target_hw, pad_value=114)

    # Channel swap: OTX loads training data as RGB;
    #   yolox_tiny → RGB (swap_rgb=False)
    #   yolox_s/l/x → RGB as well (swap_rgb=True is for ModelAPI deployment hint)
    if swap_rgb:
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    else:
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    # Normalize
    img_f = padded.astype(np.float32)
    mean = np.array(data_params.mean, dtype=np.float32)
    std = np.array(data_params.std, dtype=np.float32)
    img_f = (img_f - mean) / std

    # HWC → NCHW
    tensor = torch.from_numpy(img_f.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, scale, pad_tl, (ori_h, ori_w)


def rescale_boxes_letterbox(
    boxes: np.ndarray,
    scale: float,
    pad_tl: tuple[int, int],
    ori_hw: tuple[int, int],
) -> np.ndarray:
    """Map boxes from model-input space back to original-image space.

    boxes : ndarray of shape (N, 4+) in [x1, y1, x2, y2, ...] format
    """
    if len(boxes) == 0:
        return boxes
    pad_top, pad_left = pad_tl
    ori_h, ori_w = ori_hw
    boxes = boxes.copy()
    boxes[:, 0] = np.clip((boxes[:, 0] - pad_left) / scale, 0, ori_w)
    boxes[:, 1] = np.clip((boxes[:, 1] - pad_top) / scale, 0, ori_h)
    boxes[:, 2] = np.clip((boxes[:, 2] - pad_left) / scale, 0, ori_w)
    boxes[:, 3] = np.clip((boxes[:, 3] - pad_top) / scale, 0, ori_h)
    return boxes


# ─────────────────────────────────────────────────────────────────────────────
# Torch inference
# ─────────────────────────────────────────────────────────────────────────────

def load_torch_model(ckpt_path: str, model_name: str, num_classes: int) -> YOLOX:
    """Load OTX YOLOX from a Lightning checkpoint."""
    print(f"[Torch] Loading checkpoint: {ckpt_path}")
    model = YOLOX.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        strict=False,
        weights_only=False,
    )
    model.eval()
    return model


@torch.no_grad()
def run_torch_with_nms(
    model: YOLOX,
    tensor: torch.Tensor,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run YOLOX torch model WITH NMS via export_by_feat.

    Parameters
    ----------
    tensor : (B, C, H, W) — supports any batch size >= 1

    Returns
    -------
    dets   : ndarray (B, K, 5) [x1, y1, x2, y2, score]  (model-input coords)
    labels : ndarray (B, K)   class indices as float
    """
    tensor = tensor.to(device)
    inner = model.model  # SingleStageDetector
    inner.eval()

    shape = (int(tensor.shape[2]), int(tensor.shape[3]))
    meta_info = {
        "pad_shape": shape,
        "batch_input_shape": shape,
        "img_shape": shape,
        "scale_factor": (1.0, 1.0),
    }
    batch_img_metas = [meta_info] * tensor.shape[0]

    x = inner.extract_feat(tensor)
    outs = inner.bbox_head(x)  # (cls_scores, bbox_preds, objectnesses)

    dets_batch, labels_batch = inner.bbox_head.export_by_feat(
        *outs,
        batch_img_metas=batch_img_metas,
        rescale=False,
        with_nms=True,
    )
    # dets_batch: [B, K, 5], labels_batch: [B, K]
    return dets_batch.cpu().numpy(), labels_batch.cpu().numpy()


@torch.no_grad()
def run_torch_without_nms(
    model: YOLOX,
    tensor: torch.Tensor,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run YOLOX torch model WITHOUT NMS via export_by_feat.

    The non-NMS path returns (B, K, 5) + (B, K) — static top-K selection.

    Parameters
    ----------
    tensor : (B, C, H, W) — supports any batch size >= 1

    Returns
    -------
    dets   : ndarray (B, K, 5) [x1, y1, x2, y2, score]  (model-input coords)
    labels : ndarray (B, K)   class indices as float
    """
    tensor = tensor.to(device)
    inner = model.model
    inner.eval()

    shape = (int(tensor.shape[2]), int(tensor.shape[3]))
    meta_info = {
        "pad_shape": shape,
        "batch_input_shape": shape,
        "img_shape": shape,
        "scale_factor": (1.0, 1.0),
    }
    batch_img_metas = [meta_info] * tensor.shape[0]

    x = inner.extract_feat(tensor)
    outs = inner.bbox_head(x)

    dets_batch, labels_batch = inner.bbox_head.export_by_feat(
        *outs,
        batch_img_metas=batch_img_metas,
        rescale=False,
        with_nms=False,
    )
    # dets_batch: [B, K, 5], labels_batch: [B, K]
    return dets_batch.cpu().numpy(), labels_batch.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# OpenVINO inference
# ─────────────────────────────────────────────────────────────────────────────

def load_ov_model(xml_path: str, device: str = "CPU", with_nms: bool = True) -> ov.CompiledModel:
    """Compile an OpenVINO IR model.

    Supports CPU, GPU, and NPU devices.
    For NPU, LATENCY performance hint is set automatically (required by the NPU plugin).
    with_nms: whether the *script* is configured to use NMS outputs (i.e. --with_nms was
    passed).  Used only to produce a contextually accurate error message on NPU.
    """
    print(f"[OV] Loading model: {xml_path}")
    core = ov.Core()

    available = core.available_devices
    print(f"[OV] Available devices: {available}")

    device_upper = device.upper()
    if device_upper not in available:
        print(f"[OV] WARNING: requested device '{device_upper}' not found in {available}. "
              f"Falling back to CPU.")
        device_upper = "CPU"

    model = core.read_model(xml_path)

    # NPU requires LATENCY hint and does not support dynamic shapes —
    # reshape the model to a fixed batch=1 before compilation.
    config: dict = {}
    if device_upper == "NPU":
        # # Step 1: resolve the dynamic batch dimension first.
        # # The non-NMS export path produces outputs like [?,100,5] where ? is
        # # just the batch dim.  Reshaping the input to batch=1 propagates
        # # through the graph and makes all shapes fully static.
        # input_shape = model.inputs[0].partial_shape
        # if input_shape.is_dynamic:
        #     static_shape = [d.get_length() if not d.is_dynamic else 1
        #                     for d in input_shape]
        #     model.reshape({model.inputs[0]: static_shape})
        #     print(f"[OV] NPU: reshaped input to static {static_shape}")

        # Step 2: after resolving batch, check for truly dynamic outputs
        # (e.g. NMS-based models where the number of detections is unknown).
        # for out in model.outputs:
        #     ps = out.partial_shape
        #     if ps.is_dynamic:
        #         if with_nms:
        #             fix_hint = (
        #                 "Fix: re-export the model WITHOUT NMS, then run with\n"
        #                 "  --no_nms --ov_device NPU\n"
        #                 "NMS will be applied on CPU after NPU inference."
        #             )
        #         else:
        #             fix_hint = (
        #                 "You passed --no_nms, but the OV model you loaded was\n"
        #                 "exported WITH NMS (it has dynamic output shapes).\n"
        #                 "The --no_nms script flag only controls how this script\n"
        #                 "interprets results; it cannot change the model's graph.\n\n"
        #                 "Options:\n"
        #                 "  1. Re-export from the checkpoint WITHOUT NMS, then use\n"
        #                 "       --no_nms --ov_device NPU   (recommended)\n"
        #                 "  2. Use the current (NMS) model on CPU/GPU:\n"
        #                 "       --with_nms --ov_device CPU"
        #             )
        #         raise RuntimeError(
        #             f"\n\n[NPU] Cannot compile model on NPU: output '{out.any_name}' "
        #             f"has dynamic shape {ps}.\n"
        #             "NPU requires fully static shapes. Models with dynamic outputs\n"
        #             "(e.g. from NMS or strided-slice Focus ops) cannot be compiled\n"
        #             "by the VPUX compiler.\n\n"
        #             + fix_hint + "\n"
        #         )

        config["PERFORMANCE_HINT"] = "LATENCY"

    compiled = core.compile_model(model, device_upper, config)
    print(f"[OV] Compiled on: {device_upper}")
    print(f"[OV] Input(s):   {[inp.any_name for inp in compiled.inputs]}")
    print(f"[OV] Output(s):  {[out.any_name for out in compiled.outputs]}")
    return compiled


def run_ov_with_nms(
    compiled: ov.CompiledModel,
    tensor: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Run OV model WITH NMS.

    Expected outputs: 'boxes' [B,N,5] and 'labels' [B,N]

    Parameters
    ----------
    tensor : (B, C, H, W) — supports any batch size >= 1

    Returns
    -------
    dets   : ndarray (B, N, 5)
    labels : ndarray (B, N)
    """
    np_input = tensor.numpy()
    result = compiled(np_input)

    # Try to pick outputs by name, fall back to index order
    out_names = {out.any_name for out in compiled.outputs}

    def _get(keys: list[str], fallback_idx: int) -> np.ndarray:
        for k in keys:
            if k in out_names:
                return result[k]
        return result[compiled.outputs[fallback_idx]]

    dets = _get(["boxes", "dets"], 0)    # (B, N, 5)
    labels = _get(["labels"], 1)          # (B, N)
    return dets, labels.astype(np.int64)


def run_ov_without_nms(
    compiled: ov.CompiledModel,
    tensor: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Run OV model WITHOUT NMS.

    The non-NMS export path returns (B, K, 5) + (B, K) — static top-K.

    Parameters
    ----------
    tensor : (B, C, H, W) — supports any batch size >= 1

    Returns
    -------
    dets   : ndarray (B, K, 5)
    labels : ndarray (B, K)
    """
    np_input = tensor.numpy()
    result = compiled(np_input)

    out_names = {out.any_name for out in compiled.outputs}

    def _get(keys: list[str], fallback_idx: int) -> np.ndarray:
        for k in keys:
            if k in out_names:
                return result[k]
        return result[compiled.outputs[fallback_idx]]

    dets = _get(["boxes", "bboxes", "dets"], 0)    # (B, K, 5)
    labels = _get(["labels"], 1)                    # (B, K)
    return dets, labels.astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing (NMS applied to no-NMS outputs for mAP)
# ─────────────────────────────────────────────────────────────────────────────

def apply_nms_to_raw(
    bboxes: np.ndarray,
    scores: np.ndarray,
    score_thr: float,
    iou_thr: float,
    max_dets: int = 100,
    pre_top_k: int = 5000,
    max_output_boxes_per_class: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply post-processing identical to YOLOX's multiclass_nms export path.

    Mirrors the logic in multiclass_nms() exactly:
      1. global pre_top_k filter by max-class score
      2. per-class score threshold + NMS (max_output_boxes_per_class cap)
      3. global keep_top_k (max_dets) by score

    Parameters match the hardcoded values in export_by_feat():
      pre_top_k=5000, max_output_boxes_per_class=200, keep_top_k=max_per_img=100

    Returns
    -------
    boxes  : (N, 4)
    scores : (N,)
    labels : (N,)
    """
    t_boxes = torch.from_numpy(bboxes).float()   # (anchors, 4)
    t_scores = torch.from_numpy(scores).float()  # (anchors, num_classes)

    # Step 1: global pre_top_k — keep top-pre_top_k anchors by max class score
    if pre_top_k > 0 and t_boxes.shape[0] > pre_top_k:
        max_scores, _ = t_scores.max(dim=-1)
        _, topk_inds = max_scores.topk(pre_top_k)
        t_boxes = t_boxes[topk_inds]
        t_scores = t_scores[topk_inds]

    # Step 2: per-class score threshold + NMS
    num_classes = t_scores.shape[1]
    all_boxes, all_scores, all_labels = [], [], []
    for cls_idx in range(num_classes):
        cls_scores = t_scores[:, cls_idx]
        mask = cls_scores > score_thr
        if not mask.any():
            continue
        cls_boxes = t_boxes[mask]
        cls_scores_f = cls_scores[mask]
        keep = torchvision.ops.nms(cls_boxes, cls_scores_f, iou_thr)
        # cap per-class detections the same way max_output_boxes_per_class does
        keep = keep[:max_output_boxes_per_class]
        all_boxes.append(cls_boxes[keep])
        all_scores.append(cls_scores_f[keep])
        all_labels.append(torch.full((keep.numel(),), cls_idx, dtype=torch.long))

    if not all_boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)

    boxes_cat = torch.cat(all_boxes)
    scores_cat = torch.cat(all_scores)
    labels_cat = torch.cat(all_labels)

    # Step 3: global keep_top_k by score
    if len(scores_cat) > max_dets:
        _, top_idx = scores_cat.topk(max_dets)
        boxes_cat = boxes_cat[top_idx]
        scores_cat = scores_cat[top_idx]
        labels_cat = labels_cat[top_idx]

    return boxes_cat.numpy(), scores_cat.numpy(), labels_cat.numpy()


def apply_postprocess_nms(
    dets: np.ndarray,
    labels: np.ndarray,
    iou_thr: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Post-processing NMS on top-K model outputs.
    The model outputs top-K boxes by score (without NMS). This function
    applies per-class NMS to remove overlapping duplicates, matching what
    model_api / SSD adapter does at inference time.

    Parameters
    ----------
    dets   : (K, 5) [x1, y1, x2, y2, score]
    labels : (K,)   class labels
    iou_thr: IoU threshold for NMS

    Returns
    -------
    boxes  : (N, 4)
    scores : (N,)
    labels : (N,)
    """
    if len(dets) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)

    t_boxes = torch.from_numpy(dets[:, :4]).float()
    t_scores = torch.from_numpy(dets[:, 4]).float()
    t_labels = torch.from_numpy(labels).long()

    # Per-class NMS
    all_boxes, all_scores, all_labels = [], [], []
    for cls_id in t_labels.unique():
        mask = t_labels == cls_id
        cls_boxes = t_boxes[mask]
        cls_scores = t_scores[mask]
        keep = torchvision.ops.nms(cls_boxes, cls_scores, iou_thr)
        all_boxes.append(cls_boxes[keep])
        all_scores.append(cls_scores[keep])
        all_labels.append(t_labels[mask][keep])

    if not all_boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)

    return (
        torch.cat(all_boxes).numpy(),
        torch.cat(all_scores).numpy(),
        torch.cat(all_labels).numpy(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# COCO GT loading
# ─────────────────────────────────────────────────────────────────────────────

def load_coco_gt(
    ann_file: str,
    image_dir: str,
) -> dict[str, dict[str, Any]]:
    """Parse a COCO-format annotation JSON and return a mapping
    filename → {"boxes": ndarray(N,4 xyxy float32), "labels": ndarray(N int64)}.
    """
    with open(ann_file) as f:
        coco = json.load(f)

    id2file = {img["id"]: img["file_name"] for img in coco["images"]}
    gt: dict[str, dict] = {}

    for ann in coco["annotations"]:
        fname = id2file[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        xyxy = [x, y, x + w, y + h]
        label = ann["category_id"] - 1  # 0-indexed

        entry = gt.setdefault(fname, {"boxes": [], "labels": []})
        entry["boxes"].append(xyxy)
        entry["labels"].append(label)

    # Convert to arrays
    for fname in gt:
        gt[fname]["boxes"] = np.array(gt[fname]["boxes"], dtype=np.float32)
        gt[fname]["labels"] = np.array(gt[fname]["labels"], dtype=np.int64)

    return gt


# ─────────────────────────────────────────────────────────────────────────────
# mAP helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_pred_dict(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict:
    return {
        "boxes": torch.from_numpy(boxes).float(),
        "scores": torch.from_numpy(scores).float(),
        "labels": torch.from_numpy(labels).long(),
    }


def make_target_dict(boxes: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "boxes": torch.from_numpy(boxes).float(),
        "labels": torch.from_numpy(labels).long(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison loop
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(args: argparse.Namespace) -> None:
    device = args.device

    # ── Load torch model ────────────────────────────────────────────────────
    torch_model = load_torch_model(args.checkpoint, args.model_name, args.num_classes)
    torch_model = torch_model.to(device)
    data_params: DataInputParams = torch_model.data_input_params
    swap_rgb = args.model_name != "yolox_tiny"
    print(f"[Torch] model_name={args.model_name}, "
          f"input_size={data_params.input_size}, "
          f"mean={data_params.mean}, std={data_params.std}, swap_rgb={swap_rgb}")

    # ── Load OV model ────────────────────────────────────────────────────────
    ov_compiled = load_ov_model(args.ov_model, device=args.ov_device, with_nms=args.with_nms)

    # ── Collect images ───────────────────────────────────────────────────────
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths = sorted(
        p for p in Path(args.images_dir).iterdir()
        if p.suffix.lower() in img_exts
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {args.images_dir}")
    print(f"\nFound {len(image_paths)} image(s) in {args.images_dir}")

    # ── Load GT (optional) ──────────────────────────────────────────────────
    gt_map: dict | None = None
    if args.annotations:
        gt_map = load_coco_gt(args.annotations, args.images_dir)
        print(f"Loaded GT for {len(gt_map)} images from {args.annotations}")

    # ── Metric accumulators ─────────────────────────────────────────────────
    metric_torch = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric_ov = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    # ── Batch loop ──────────────────────────────────────────────────────────
    batch_size = getattr(args, "batch_size", 1)
    print(f"\n{'─'*80}")
    print(f"{'Image':<40} {'Torch dets':>10} {'OV dets':>10} {'Δboxes_mean':>14}")
    print(f"{'─'*80}")

    for batch_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_start : batch_start + batch_size]

        # ── Preprocess batch ────────────────────────────────────────────────
        tensors: list[torch.Tensor] = []
        scales: list[float] = []
        pad_tls: list[tuple[int, int]] = []
        ori_hws: list[tuple[int, int]] = []
        img_names: list[str] = []

        for img_path in batch_paths:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"[WARN] Cannot read {img_path}, skipping.")
                continue
            t, scale, pad_tl, ori_hw = preprocess_image(img_bgr, data_params, swap_rgb)
            tensors.append(t)
            scales.append(scale)
            pad_tls.append(pad_tl)
            ori_hws.append(ori_hw)
            img_names.append(img_path.name)

        if not tensors:
            continue

        # Stack to (B, C, H, W)
        batch_tensor = torch.cat(tensors, dim=0)
        cur_bs = batch_tensor.shape[0]

        # ── Batch inference ─────────────────────────────────────────────────
        if args.with_nms:
            t_dets_batch, t_labels_batch = run_torch_with_nms(
                torch_model, batch_tensor, device=device,
            )  # (B, K, 5), (B, K)
            ov_dets_batch, ov_labels_batch = run_ov_with_nms(ov_compiled, batch_tensor)
        else:
            t_dets_batch, t_labels_batch = run_torch_without_nms(
                torch_model, batch_tensor, device=device,
            )  # (B, K, 5), (B, K)
            ov_dets_batch, ov_labels_batch = run_ov_without_nms(ov_compiled, batch_tensor)

        # ── Per-image post-processing ────────────────────────────────────────
        for i in range(cur_bs):
            img_name = img_names[i]
            scale = scales[i]
            pad_tl = pad_tls[i]
            ori_hw = ori_hws[i]

            t_dets_i = t_dets_batch[i]     # (K, 5)
            t_lbls_i = t_labels_batch[i]   # (K,)
            ov_dets_i = ov_dets_batch[i]
            ov_lbls_i = ov_labels_batch[i]

            # Filter padding / zero-score rows
            t_valid = t_dets_i[:, 4] > 0
            ov_valid = ov_dets_i[:, 4] > 0
            t_dets_i = t_dets_i[t_valid]
            t_lbls_i = t_lbls_i[t_valid].astype(np.int64)
            ov_dets_i = ov_dets_i[ov_valid]
            ov_lbls_i = ov_lbls_i[ov_valid].astype(np.int64)

            if args.with_nms:
                t_boxes  = t_dets_i[:, :4]
                t_scores = t_dets_i[:, 4]
                t_labels = t_lbls_i
                ov_boxes  = ov_dets_i[:, :4]
                ov_scores = ov_dets_i[:, 4]
                ov_labels = ov_lbls_i
            else:
                # Apply per-class NMS to deduplicate top-K outputs
                t_boxes, t_scores, t_labels = apply_postprocess_nms(
                    t_dets_i, t_lbls_i, iou_thr=args.iou_thr,
                )
                ov_boxes, ov_scores, ov_labels = apply_postprocess_nms(
                    ov_dets_i, ov_lbls_i, iou_thr=args.iou_thr,
                )

            # Rescale to original image space
            t_boxes_orig  = rescale_boxes_letterbox(t_boxes,  scale, pad_tl, ori_hw)
            ov_boxes_orig = rescale_boxes_letterbox(ov_boxes, scale, pad_tl, ori_hw)

            # Per-image summary line
            delta_str = "n/a"
            if len(t_boxes_orig) > 0 and len(ov_boxes_orig) > 0:
                min_n = min(len(t_boxes_orig), len(ov_boxes_orig))
                delta = np.abs(t_boxes_orig[:min_n] - ov_boxes_orig[:min_n]).mean()
                delta_str = f"{delta:.4f}"
            print(
                f"{img_name:<40} {len(t_boxes_orig):>10} {len(ov_boxes_orig):>10} {delta_str:>14}"
            )

            # Accumulate mAP
            if gt_map is not None:
                if img_name in gt_map:
                    target = make_target_dict(gt_map[img_name]["boxes"], gt_map[img_name]["labels"])
                    metric_torch.update([make_pred_dict(t_boxes_orig, t_scores, t_labels)], [target])
                    metric_ov.update([make_pred_dict(ov_boxes_orig, ov_scores, ov_labels)], [target])
                else:
                    print(f"  [WARN] No GT found for {img_name}")

    print(f"{'─'*80}")

    # ── Print mAP results ───────────────────────────────────────────────────
    if gt_map is not None:
        print("\n── Torch mAP ──────────────────────────────")
        t_map = metric_torch.compute()
        for k, v in t_map.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                print(f"  {k:<30}: {v.item():.4f}")

        print("\n── OpenVINO mAP ────────────────────────────")
        ov_map_res = metric_ov.compute()
        for k, v in ov_map_res.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                print(f"  {k:<30}: {v.item():.4f}")

        print("\n── Delta (Torch − OV) ──────────────────────")
        for k in t_map:
            tv = t_map[k]
            ov_v = ov_map_res[k]
            if isinstance(tv, torch.Tensor) and tv.numel() == 1:
                print(f"  {k:<30}: {tv.item() - ov_v.item():+.4f}")
    else:
        print("\n(No --annotations provided; mAP not computed.)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare OTX YOLOX Torch vs OpenVINO model outputs and mAP."
    )
    # Required
    parser.add_argument("--checkpoint", required=True,
                        help="Path to OTX Lightning checkpoint (.ckpt)")
    parser.add_argument("--model_name", required=True,
                        choices=["yolox_tiny", "yolox_s", "yolox_l", "yolox_x"],
                        help="YOLOX variant name")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Number of detection classes")
    parser.add_argument("--ov_model", required=True,
                        help="Path to OpenVINO model XML file")
    parser.add_argument("--images_dir", required=True,
                        help="Directory with test images")
    # Optional
    parser.add_argument("--annotations", default=None,
                        help="COCO-format JSON annotations file (for mAP)")
    parser.add_argument("--input_size", nargs=2, type=int, default=None,
                        metavar=("H", "W"),
                        help="Override input size (H W). Default: from checkpoint.")
    parser.add_argument("--device", default="cpu",
                        help="Torch device: cpu or cuda")
    parser.add_argument("--ov_device", default="CPU",
                        help="OpenVINO inference device: CPU, GPU, NPU (default: CPU). "
                             "NOTE: NPU requires a model exported WITHOUT NMS (--no_nms), "
                             "because NMS produces dynamic-length outputs that the VPUX "
                             "compiler cannot handle.")
    parser.add_argument("--score_thr", type=float, default=0.01,
                        help="Score threshold (used when running without NMS)")
    parser.add_argument("--iou_thr", type=float, default=0.65,
                        help="IoU threshold for NMS (used when running without NMS)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of images to process per forward pass (default: 1). "
                             "Values > 1 require the model to have a dynamic batch dimension "
                             "(i.e. exported with --no_nms or with dynamic batch support). "
                             "NPU always requires batch_size=1.")

    nms_group = parser.add_mutually_exclusive_group()
    nms_group.add_argument("--with_nms", dest="with_nms", action="store_true", default=True,
                            help="Use NMS output mode (default)")
    nms_group.add_argument("--no_nms", dest="with_nms", action="store_false",
                            help="Use raw (no NMS) output mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"NMS mode:   {'WITH NMS' if args.with_nms else 'WITHOUT NMS'}")
    print(f"OV device:  {args.ov_device.upper()}")
    print(f"Batch size: {args.batch_size}")
    run_comparison(args)


if __name__ == "__main__":
    main()
