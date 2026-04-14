#!/usr/bin/env bash
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# XPU smoke benchmark – all Geti-supported models, seed 0, 1 dataset per task.
#
# Tasks: Detection, Instance Segmentation, Multi-class Classification
#        (H-label and Multi-label excluded)
#
# Usage:
#   cd /home/fst/training_extensions/library
#   bash tests/perf_v2/run_xpu.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Dataset root – all relative dataset paths are resolved against this
# ---------------------------------------------------------------------------
DATA_ROOT="/home/fst/bench_data"

# ---------------------------------------------------------------------------
# General settings
# ---------------------------------------------------------------------------
OUTPUT_ROOT="${OUTPUT_ROOT:-otx-benchmark-xpu}"
EVAL_UPTO="${EVAL_UPTO:-export}"
DEVICE=xpu
PYTHON="${PYTHON:-python}"

# ---------------------------------------------------------------------------
# Datasets (name must match the DatasetInfo.name in the task module)
# ---------------------------------------------------------------------------
DETECTION_DATASET="wgisd_small"
INSTANCE_SEG_DATASET="wgisd_small"
MULTICLASS_DATASET="multiclass_small_flowers"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
DETECTION_MODELS=(
    atss_mobilenetv2
    ssd_mobilenetv2
    deim_dfine_x
    deim_dfine_l
    deim_dfine_m
    deimv2_l
    deimv2_m
    deimv2_s
    rtdetr_50
    rfdetr_large
    rfdetr_medium
    rfdetr_small
    yolox_tiny
    yolox_s
    yolox_l
    yolox_x
)

INSTANCE_SEG_MODELS=(
    maskrcnn_r50
    rfdetr_seg_medium
    rfdetr_seg_large
    rfdetr_seg_small
    rfdetr_seg_xlarge
    maskrcnn_efficientnetb2b
    maskrcnn_swint
)

MULTICLASS_MODELS=(
    efficientnet_b0
    efficientnet_v2
    mobilenet_v3_large
    deit_tiny
    dino_v2
    efficientnet_b3
)

# ---------------------------------------------------------------------------
# Helper – one call per task; run.py iterates all models internally
# ---------------------------------------------------------------------------
run_task() {
    local task="$1"
    local dataset="$2"   # display only; run.py uses DATASET_COLLECTIONS

    echo ""
    echo ">>> [${task}]  dataset=${dataset}  seed=0  (1 repeat)"

    "$PYTHON" -m tests.perf_v2.run \
        --task        "$task"        \
        --data-root   "$DATA_ROOT"   \
        --output-root "$OUTPUT_ROOT" \
        --num-repeat  1              \
        --eval-upto   "$EVAL_UPTO"   \
        --device      "$DEVICE"      \
        --summary-file-root "$OUTPUT_ROOT" \
        --deterministic false
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_ROOT"

TOTAL=$(( ${#DETECTION_MODELS[@]} + ${#INSTANCE_SEG_MODELS[@]} + ${#MULTICLASS_MODELS[@]} ))

echo "============================================================"
echo " OTX XPU benchmark  (${TOTAL} models total)"
echo "   seed        : 0 (num-repeat=1)"
echo "   eval-upto   : ${EVAL_UPTO}"
echo "   data-root   : ${DATA_ROOT}"
echo "   output-root : ${OUTPUT_ROOT}"
echo "------------------------------------------------------------"
echo "   detection       : ${DETECTION_DATASET}   (${#DETECTION_MODELS[@]} models)"
echo "   instance_seg    : ${INSTANCE_SEG_DATASET}   (${#INSTANCE_SEG_MODELS[@]} models)"
echo "   multiclass_cls  : ${MULTICLASS_DATASET}   (${#MULTICLASS_MODELS[@]} models)"
echo "============================================================"

# ── Object Detection ──────────────────────────────────────────────────────
echo ""
echo "=== Object Detection (${#DETECTION_MODELS[@]} models) ==="
run_task "DETECTION" "$DETECTION_DATASET"

# ── Instance Segmentation ─────────────────────────────────────────────────
echo ""
echo "=== Instance Segmentation (${#INSTANCE_SEG_MODELS[@]} models) ==="
run_task "INSTANCE_SEGMENTATION" "$INSTANCE_SEG_DATASET"

# ── Multi-class Classification ────────────────────────────────────────────
echo ""
echo "=== Multi-class Classification (${#MULTICLASS_MODELS[@]} models) ==="
run_task "MULTI_CLASS_CLS" "$MULTICLASS_DATASET"

echo ""
echo "All done. Results in: ${OUTPUT_ROOT}"
