# Hyperparameter Exposure Proposal

> Scope: Object Detection, Instance Segmentation, Multi-class / Multi-label / H-label Classification.
> Augmentation parameters are out of scope (handled separately).

---

## Current State

The global `base.yaml` exposes only:
```yaml
hyperparameters:
  training:
    max_epochs: 200
```

Individual model manifests may expose `learning_rate` and, in a few detection models, `input_size_width` / `input_size_height`.  Everything else — batch size, weight decay, scheduler settings, gradient clipping, warmup — is invisible to users.

---

## Proposed Parameters

### 1. Universal — all tasks, all models

These map to the same recipe key path regardless of model family.

| Manifest key | Recipe path | Notes |
|---|---|---|
| `training.max_epochs` | `overrides.max_epochs` | Already in global base.yaml. |
| `training.batch_size` | `overrides.data.train_subset.batch_size` | Det default 4–8, inst-seg 4, cls 64. |
| `training.learning_rate` | `model.init_args.optimizer.init_args.lr` | Already exposed per-model; move to base. |
| `training.weight_decay` | `model.init_args.optimizer.init_args.weight_decay` | SGD default ≈0.0001, AdamW default ≈0.0001–0.05. |
| `training.num_warmup_steps` | `model.init_args.scheduler.init_args.num_warmup_steps` | Number of LR scheduler steps during which the learning rate linearly ramps up from 0 to the configured `lr`. 0 = no warmup. |
| `training.early_stopping_patience` | `callbacks[EarlyStoppingWithWarmup].init_args.patience` | Epochs with no improvement before stopping. Det/inst-seg default 10, cls default 5. |
| `training.gradient_clip_val` | `overrides.gradient_clip_val` | Max gradient norm before clipping. SGD-based det/inst-seg models: 35.0, RFDetr family: 0.1, classification models: not set (null). 0 = disabled. |

**Proposed addition to global `base.yaml`:**
```yaml
hyperparameters:
  training:
    max_epochs: 200
    batch_size: ~               # task-level base.yaml sets the default
    learning_rate: ~            # per-model manifest sets the default
    weight_decay: ~             # per-model manifest sets the default
    num_warmup_steps: ~         # per-model manifest sets the default
    early_stopping_patience: ~  # task-level base.yaml sets the default
    gradient_clip: ~  # per-model manifest sets the default; null = disabled
```

---

### 2. Detection & Instance Segmentation — all models

| Manifest key | Recipe path | Default values |
|---|---|---|
| `training.input_size_width` | `overrides.data.input_size[1]` | YOLOX/RTMDet/RTDetr/DFine/DEIM: 640, RFDetr-det: 576, SSD: 864, MaskRCNN: 1024, RTMDet-inst: 640, RFDetr-seg: 576 |
| `training.input_size_height` | `overrides.data.input_size[0]` | same as width (square by default) |

`input_size` is already partially exposed in some detection manifests (`rtdetr_50`, etc.).  It should be promoted to `detection/base.yaml` and `instance_segmentation/base.yaml` with task-appropriate defaults.

---

### 3. ReduceLROnPlateau scheduler — per-model

Used by: **all YOLOX, ATSS, SSD, RTMDet, RTDetr, DEIM, DFine, RFDetr, MaskRCNN (all variants), RTMDet-inst, RFDetr-seg, EfficientNet-B0/V2, MobileNetV3, MobileNetV4, DeiT, DINOv2**.  This is the majority of models.

| Manifest key | Recipe path | Typical defaults |
|---|---|---|
| `training.scheduler.factor` | `model.init_args.scheduler.init_args.main_scheduler_callable.init_args.factor` | Det: 0.1–0.5; Cls: 0.5 |
| `training.scheduler.patience` | `model.init_args.scheduler.init_args.main_scheduler_callable.init_args.patience` | Det: 4–10; RFDetr: 10; Cls: 3 |

Using a nested `scheduler` key groups these together and keeps the manifest readable.  Example per-model manifest block:
```yaml
hyperparameters:
  training:
    scheduler:
      factor: 0.1    # multiply LR by this when plateau is detected
      patience: 4    # epochs without improvement before reducing LR
```

Recipe defaults by model family:
```
YOLOX / ATSS / SSD / RTMDet-det / RTMDet-inst / MaskRCNN-r50 / MaskRCNN-EffNetB2b:  factor=0.1  patience=4
RTDetr / DFine:                                                                        factor=0.1  patience=6
DEIM:                                                                                  factor=0.5  patience=10
RFDetr-det / RFDetr-seg:                                                               factor=0.1  patience=10
EfficientNet / MobileNet / MobileNetV4 / DeiT / DINOv2:                               factor=0.5  patience=3
```

---

### 4. CosineAnnealingLR scheduler — per-model

Used by: **tv_efficientnet_b3, tv_efficientnet_v2_l, tv_mobilenet_v3_small** (TorchVision classification models only).

| Manifest key | Recipe path | Current default |
|---|---|---|
| `training.cosine_t_max` | `model.init_args.scheduler.init_args.main_scheduler_callable.init_args.T_max` | 100000 |
| `training.cosine_eta_min` | `model.init_args.scheduler.init_args.main_scheduler_callable.init_args.eta_min` | 0 |

`T_max` controls the cycle length in optimizer steps.  At typical batch sizes and epoch counts it effectively means "decay to `eta_min` by end of training".  Exposing it directly is brittle; consider computing it from `max_epochs × steps_per_epoch` in the application layer instead of exposing the raw value.

---

### 5. SGD optimizer — per-model

Used by: **YOLOX, ATSS, SSD, MaskRCNN-r50, MaskRCNN-EffNetB2b, RTMDet-inst-tiny, EfficientNet-B0/V2, MobileNetV3/V4, tv_efficientnet_b3, tv_efficientnet_v2_l, tv_mobilenet_v3_small**.

| Manifest key | Recipe path | Default |
|---|---|---|
| `training.momentum` | `model.init_args.optimizer.init_args.momentum` | 0.9 (uniform across all SGD models) |

Only worth exposing for power users; 0.9 is nearly always optimal.

---

### 6. AdamW optimizer — per-model

Used by: **RTDetr (all), RTMDet-det, DEIM (all), DFine, RFDetr (all), MaskRCNN-SwinT, MaskRCNN-r50_tv, RFDetr-seg (all), DeiT, DINOv2**.

No additional parameters beyond `learning_rate` and `weight_decay` (already in §1) need exposure.  The `betas` values are universally `[0.9, 0.999]` and should not be surfaced.

---

## Summary Table — What to Add Where

### Global `base.yaml`
```yaml
hyperparameters:
  training:
    max_epochs: 200               # already present
    batch_size: ~                 # NEW — default set by task-level base.yaml
    weight_decay: ~               # NEW — default set per-model manifest
    num_warmup_steps: ~           # NEW — default set per-model manifest
    early_stopping_patience: ~    # NEW — default set by task-level base.yaml
    gradient_clip_val: ~          # NEW — default set per-model manifest; null = disabled
```

### Task-level `detection/base.yaml` and `instance_segmentation/base.yaml`
```yaml
hyperparameters:
  training:
    input_size_width: 640    # det default; inst-seg default 1024
    input_size_height: 640
```

### Per-model manifests — ReduceLROnPlateau models
```yaml
hyperparameters:
  training:
    learning_rate: <model-specific>
    weight_decay: <model-specific>
    num_warmup_steps: <model-specific>
    gradient_clip_val: <model-specific>   # null for classification
    scheduler:
      factor: <model-specific>
      patience: <model-specific>
```

### Per-model manifests — CosineAnnealingLR models (tv_ cls only)
```yaml
hyperparameters:
  training:
    learning_rate: 0.01
    weight_decay: 0.0001
    # cosine_t_max: omit — compute from max_epochs in app layer
```

### Per-model manifests — SGD models (optional/advanced)
```yaml
hyperparameters:
  training:
    momentum: 0.9   # low priority; only if advanced controls are desired
```

---

## Parameters Intentionally Excluded

| Parameter | Reason |
|---|---|
| `optimizer.betas` | Always `[0.9, 0.999]` across all AdamW models; no user benefit. |
| `num_workers` | Infrastructure concern, not a training hyperparameter. |
| `warmup_iters` / `warmup_epochs` (EarlyStopping) | Internal burn-in guard; `num_warmup_steps` in the scheduler is the user-facing knob. |
| `min_delta` (EarlyStopping) | Very sensitive; changing it risks premature stopping. |
| `min_lr` / `min_lrschedule_patience` | Only in a handful of models; not meaningful to general users. |
| `cosine_t_max` | Better computed server-side from `max_epochs`; raw step count is confusing. |
| `AdaptiveTrainScheduling` params | Internal adaptive logic; exposing would confuse users. |
| `EMAWeightAveraging.decay` | Only RFDetr; very sensitive parameter, default 0.993 is optimal. |
| `AugmentationSchedulerCallback` params | DEIM-specific policy epochs; covered by augmentation section. |
| `gradient_clip_val` for classification | Not used by any classification recipe. |
