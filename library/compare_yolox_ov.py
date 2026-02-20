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
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import tv_tensors

# ── path setup ───────────────────────────────────────────────────────────────
LIB_DIR = Path(__file__).parent / "geti_tune_clean" / "library"
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

    Returns
    -------
    dets   : ndarray (N, 5) [x1, y1, x2, y2, score]  (model-input coords)
    labels : ndarray (N,)
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

    x = inner.extract_feat(tensor)
    outs = inner.bbox_head(x)  # (cls_scores, bbox_preds, objectnesses)

    dets_batch, labels_batch = inner.bbox_head.export_by_feat(
        *outs,
        batch_img_metas=[meta_info],
        rescale=False,
        with_nms=True,
    )
    # dets_batch: [1, N, 5], labels_batch: [1, N]
    dets = dets_batch[0].cpu().numpy()     # (N, 5)
    labels = labels_batch[0].cpu().numpy()  # (N,)
    # Remove padding rows (score == 0 from _select_nms_index)
    valid = dets[:, 4] > 0
    return dets[valid], labels[valid].astype(np.int64)


@torch.no_grad()
def run_torch_without_nms(
    model: YOLOX,
    tensor: torch.Tensor,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run YOLOX torch model WITHOUT NMS via export_by_feat.

    Returns
    -------
    bboxes : ndarray (anchors, 4) [x1, y1, x2, y2]
    scores : ndarray (anchors, num_classes)
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

    x = inner.extract_feat(tensor)
    outs = inner.bbox_head(x)

    bboxes_batch, scores_batch = inner.bbox_head.export_by_feat(
        *outs,
        batch_img_metas=[meta_info],
        rescale=False,
        with_nms=False,
    )
    # bboxes_batch: [1, anchors, 4], scores_batch: [1, anchors, num_classes]
    return bboxes_batch[0].cpu().numpy(), scores_batch[0].cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# OpenVINO inference
# ─────────────────────────────────────────────────────────────────────────────

def load_ov_model(xml_path: str, device: str = "CPU") -> ov.CompiledModel:
    """Compile an OpenVINO IR model.

    Supports CPU, GPU, and NPU devices.
    For NPU, LATENCY performance hint is set automatically (required by the NPU plugin).
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
        config["PERFORMANCE_HINT"] = "LATENCY"
        # Ensure static input shape for NPU (batch=1 is fixed at export time anyway)
        input_shape = model.inputs[0].partial_shape
        if input_shape.is_dynamic:
            static_shape = [d.get_length() if not d.is_dynamic else 1
                            for d in input_shape]
            model.reshape({model.inputs[0]: static_shape})
            print(f"[OV] NPU: reshaped input to static {static_shape}")

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
    Returns (N,5) and (N,) arrays for the first (and only) batch item.
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

    dets = dets[0]                        # (N, 5)
    labels = labels[0]                    # (N,)
    valid = dets[:, 4] > 0
    return dets[valid], labels[valid].astype(np.int64)


def run_ov_without_nms(
    compiled: ov.CompiledModel,
    tensor: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Run OV model WITHOUT NMS.

    Expected outputs: 'boxes'/'bboxes' [B,anchors,4] and 'labels'/'scores' [B,anchors,C].
    Returns (anchors, 4) and (anchors, C) arrays for the first batch item.
    """
    np_input = tensor.numpy()
    result = compiled(np_input)

    out_names = {out.any_name for out in compiled.outputs}

    def _get(keys: list[str], fallback_idx: int) -> np.ndarray:
        for k in keys:
            if k in out_names:
                return result[k]
        return result[compiled.outputs[fallback_idx]]

    bboxes = _get(["boxes", "bboxes"], 0)    # (B, anchors, 4)
    scores = _get(["labels", "scores"], 1)   # (B, anchors, C)
    return bboxes[0], scores[0]


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing (NMS applied to no-NMS outputs for mAP)
# ─────────────────────────────────────────────────────────────────────────────

def apply_nms_to_raw(
    bboxes: np.ndarray,
    scores: np.ndarray,
    score_thr: float,
    iou_thr: float,
    max_dets: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply score-threshold + per-class NMS to raw (no-NMS) outputs.

    Returns
    -------
    boxes  : (N, 4)
    scores : (N,)
    labels : (N,)
    """
    import torchvision

    t_boxes = torch.from_numpy(bboxes).float()   # (anchors, 4)
    t_scores = torch.from_numpy(scores).float()  # (anchors, num_classes)

    all_boxes, all_scores, all_labels = [], [], []
    num_classes = t_scores.shape[1]
    for cls_idx in range(num_classes):
        cls_scores = t_scores[:, cls_idx]
        mask = cls_scores > score_thr
        if not mask.any():
            continue
        cls_boxes = t_boxes[mask]
        cls_scores_f = cls_scores[mask]
        keep = torchvision.ops.nms(cls_boxes, cls_scores_f, iou_thr)
        all_boxes.append(cls_boxes[keep])
        all_scores.append(cls_scores_f[keep])
        all_labels.append(torch.full((keep.numel(),), cls_idx, dtype=torch.long))

    if not all_boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)

    boxes_np = torch.cat(all_boxes).numpy()
    scores_np = torch.cat(all_scores).numpy()
    labels_np = torch.cat(all_labels).numpy()

    # Global top-k
    if len(scores_np) > max_dets:
        top_idx = np.argsort(-scores_np)[:max_dets]
        boxes_np, scores_np, labels_np = boxes_np[top_idx], scores_np[top_idx], labels_np[top_idx]

    return boxes_np, scores_np, labels_np


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
    ov_compiled = load_ov_model(args.ov_model, device=args.ov_device)

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

    # ── Per-image loop ───────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"{'Image':<40} {'Torch dets':>10} {'OV dets':>10} {'Δboxes_mean':>14}")
    print(f"{'─'*80}")

    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] Cannot read {img_path}, skipping.")
            continue

        # ── Preprocess ──────────────────────────────────────────────────────
        tensor, scale, pad_tl, ori_hw = preprocess_image(img_bgr, data_params, swap_rgb)

        # ── Torch inference ─────────────────────────────────────────────────
        if args.with_nms:
            t_dets, t_labels = run_torch_with_nms(torch_model, tensor, device=device)
            # dets: (N,5) [x1,y1,x2,y2,score]
            t_boxes = t_dets[:, :4]
            t_scores = t_dets[:, 4]
        else:
            t_bboxes_raw, t_scores_raw = run_torch_without_nms(torch_model, tensor, device=device)
            t_boxes, t_scores, t_labels = apply_nms_to_raw(
                t_bboxes_raw, t_scores_raw, args.score_thr, args.iou_thr
            )

        # ── OV inference ────────────────────────────────────────────────────
        if args.with_nms:
            ov_dets, ov_labels = run_ov_with_nms(ov_compiled, tensor)
            ov_boxes = ov_dets[:, :4]
            ov_scores = ov_dets[:, 4]
        else:
            ov_bboxes_raw, ov_scores_raw = run_ov_without_nms(ov_compiled, tensor)
            ov_boxes, ov_scores, ov_labels = apply_nms_to_raw(
                ov_bboxes_raw, ov_scores_raw, args.score_thr, args.iou_thr
            )

        # ── Rescale to original image space ─────────────────────────────────
        t_boxes_orig = rescale_boxes_letterbox(t_boxes, scale, pad_tl, ori_hw)
        ov_boxes_orig = rescale_boxes_letterbox(ov_boxes, scale, pad_tl, ori_hw)

        # ── Per-image summary ────────────────────────────────────────────────
        delta_str = "n/a"
        if len(t_boxes_orig) > 0 and len(ov_boxes_orig) > 0:
            min_n = min(len(t_boxes_orig), len(ov_boxes_orig))
            delta = np.abs(t_boxes_orig[:min_n] - ov_boxes_orig[:min_n]).mean()
            delta_str = f"{delta:.4f}"
        print(f"{img_path.name:<40} {len(t_boxes_orig):>10} {len(ov_boxes_orig):>10} {delta_str:>14}")

        # ── Accumulate mAP (if GT provided) ─────────────────────────────────
        if gt_map is not None:
            fname = img_path.name
            if fname in gt_map:
                target = make_target_dict(gt_map[fname]["boxes"], gt_map[fname]["labels"])
                metric_torch.update([make_pred_dict(t_boxes_orig, t_scores, t_labels)], [target])
                metric_ov.update([make_pred_dict(ov_boxes_orig, ov_scores, ov_labels)], [target])
            else:
                print(f"  [WARN] No GT found for {fname}")

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
                        help="OpenVINO inference device: CPU, GPU, NPU (default: CPU)")
    parser.add_argument("--score_thr", type=float, default=0.01,
                        help="Score threshold (used when running without NMS)")
    parser.add_argument("--iou_thr", type=float, default=0.65,
                        help="IoU threshold for NMS (used when running without NMS)")

    nms_group = parser.add_mutually_exclusive_group()
    nms_group.add_argument("--with_nms", dest="with_nms", action="store_true", default=True,
                            help="Use NMS output mode (default)")
    nms_group.add_argument("--no_nms", dest="with_nms", action="store_false",
                            help="Use raw (no NMS) output mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"NMS mode: {'WITH NMS' if args.with_nms else 'WITHOUT NMS'}")
    print(f"OV device: {args.ov_device.upper()}")
    run_comparison(args)


if __name__ == "__main__":
    main()
