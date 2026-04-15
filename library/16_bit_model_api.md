# Plan: Unified Intensity/Scaling Support in ModelAPI

## Problem

ModelAPI has **no concept of input dtype or intensity scaling** — it hardcodes `uint8` assumptions in three places:


1. `embed_preprocessing` default `dtype=int` → `Type.u8`

2. `opset.pad` constant hardcoded as `np.uint8` in graph-based resize helpers

3. `pad_value` parameter range capped at `[0, 255]`


Meanwhile Geti Tune already has a rich `IntensityConfig` system (`ScaleToUnit`, `WindowLevel`, `PercentileClip`, `RangeScale`). The Geti application backend carries duplicate workarounds (`FP32OpenvinoAdapter`, `needs_float32_input`, manual `/255.0` scaling) to bridge the gap.

  https://github.com/open-edge-platform/training_extensions/blob/fb292b062f0247c7811f71258cd2af12d8a8affa/library/src/getitune/data/augmentation/intensity.py

---

## Part A: ModelAPI Changes

### A1. Add `intensity_mode` and `input_dtype` parameters to the registry

**File:** `model_api/src/model_api/models/parameters.py` — `IMAGE_PREPROCESSING` group

```python

"input_dtype": StringValue(default_value="u8", choices=("u8", "f32", "u16")),

"intensity_mode": StringValue(default_value="none", choices=("none", "scale_to_unit", "window", "percentile", "range_scale")),

"intensity_max_value": NumericalValue(float, default_value=None, min=0.0),

"intensity_window_center": NumericalValue(float, default_value=None),

"intensity_window_width": NumericalValue(float, default_value=None),

"intensity_percentile_low": NumericalValue(float, default_value=1.0, min=0.0, max=100.0),

"intensity_percentile_high": NumericalValue(float, default_value=99.0, min=0.0, max=100.0),

"intensity_scale_factor": NumericalValue(float, default_value=1.0),

"intensity_min_value": NumericalValue(float, default_value=0.0),

```


These are auto-read from IR `rt_info["model_info"]` via `Model._load_config`.


**What maps to the OV graph vs. Python-side:**

| Mode            | Embeddable in OV graph?          | Approach                                               |
| --------------- | -------------------------------- | ------------------------------------------------------ |
| `scale_to_unit` | Yes (static)                     | `divide(input, max_value)` in PrePostProcessor         |
| `window`        | Yes (static)                     | `(x - low) / (high - low) + clamp` in PrePostProcessor |
| `range_scale`   | Yes (static)                     | `multiply + clamp + normalize` in PrePostProcessor     |
| `percentile`    | No (dynamic, per-image quantile) | Python-side in `InputTransform` only                   |

### A2. Plumb `input_dtype` through `ImageModel` into `embed_preprocessing`

**File:** `model_api/src/model_api/models/image_model.py` — constructor (~line 86)

Map `self.params.input_dtype` to the `dtype` kwarg:


- `"u8"` → `dtype=int` (current default, `Type.u8`)

- `"f32"` → `dtype=float` (`Type.f32`)

- `"u16"` → new branch (`Type.u16`)


Pass it to `inference_adapter.embed_preprocessing(…, dtype=…)`.

### A3. Add intensity preprocessing to `embed_preprocessing`

**File:** `model_api/src/model_api/adapters/openvino_adapter.py`

Before resize,  mean/scale normalization, insert intensity-mapping OV graph nodes based on `intensity_mode`:

- `"scale_to_unit"` → `ppp.input().preprocess().scale([max_value])` (reusing OV PPP scale, applied before the existing mean/scale)

- `"window"` → custom preprocess node: `clamp((x - low) / (high - low), 0, 1)`

- `"range_scale"` → custom preprocess node: `clamp(x * factor, min, max)` then normalize

For `"percentile"` mode, set a flag so `InputTransform` handles it in Python.

Minor: Also add `elif dtype == "u16"` → `Type.u16` in `OpenvinoAdapter.embed_preprocessing`.

### A4. Fix hardcoded `np.uint8` pad constant in graph utils

**File:** `model_api/src/model_api/adapters/utils.py`

Replace `opset.constant(pad_value, dtype=np.uint8)` with dtype-aware logic — infer the element type from the input node and cast the pad constant accordingly.

Widen `pad_value` max from 255 to 65535. Eliminates the `_patch_pad_constant_type` monkey-patch.

### A5. Extend `InputTransform` for Python-side intensity modes

**File:** `model_api/src/model_api/adapters/utils.py` — `InputTransform` class

Add optional `intensity_fn` callable that runs before mean/scale. Handles `"percentile"` and the NPU/ONNX fallback path.


```python

class InputTransform:

def __init__(self, reverse_input_channels, mean_values, scale_values, intensity_fn=None):

self.intensity_fn = intensity_fn

# ... existing logic ...



def __call__(self, inputs):

if self.intensity_fn:

inputs = self.intensity_fn(inputs)

# ... existing BGR→RGB + mean/scale ...

```



### A6. Extend `setup_python_preprocessing_pipeline`

**File:** `model_api/src/model_api/adapters/utils.py`

Add `intensity_mode` and related params. Insert Python-side intensity transform before `InputTransform`.

## Part B: Geti Tune Library Changes

### B1. Embed intensity metadata in Geti Tune exporter rt_info

**File:** `library/src/getitune/backend/native/exporter/base.py` — `_extend_model_metadata`

Write `input_dtype`, `intensity_mode`, `intensity_max_value`, etc. from `IntensityConfig` into the IR. Extend `DataInputParams` to carry intensity config.

### B2. Remove workarounds once ModelAPI is updated


## Part C: Geti Application Changes

### C1. Add image bit-depth detection on media import

**File:** `app/services/import/base_import.py`

"Probe" actual pixel dtype via PIL `mode` on import. Store `pixel_dtype` (nullable string: `"uint8"`, `"uint16"`) on the Media DB record.

**Schema change** — `app/db/schema.py`: add `pixel_dtype` column (nullable string) to media table. Existing rows default to `None` (treated as `uint8`).

### C2. Add intensity configuration to training config

**File:** `app/models/training_configuration/configuration.py`

```python

@dataclass

class IntensityConfiguration:

mode: str = "scale_to_unit"

max_value: float | None = None # None = auto from dataset pixel_dtype

window_center: float | None = None

window_width: float | None = None

percentile_low: float = 1.0

percentile_high: float = 99.0

scale_factor: float = 1.0

min_value: float = 0.0

repeat_channels: int = 0

```

Add to `AlgorithmConfiguration` as a new `intensity` field.


### C3. Expose intensity settings in the API


No new endpoint needed — the existing PATCH endpoint with dot-notation already works:


```json

{

"intensity.mode": "window",

"intensity.window_center": 40.0,

"intensity.window_width": 400.0,

"intensity.repeat_channels": 3

}

```



### C4. Add smart defaults based on dataset bit depth



When creating a default training config, inspect the project's media `pixel_dtype`:



- All `uint8` → `mode="scale_to_unit"`, `max_value=255.0`

- Any `uint16` → `mode="scale_to_unit"`, `max_value=65535.0`

- Mixed → warn user, default to majority type



### C5. Wire intensity config into Geti Tune training launch



**File:** `app/execution/training/getitune_trainer.py`



Map Geti `IntensityConfiguration` → Geti Tune `IntensityConfig` when building the Geti Tune config dict.



### C6. Fix inference to use model-embedded intensity preprocessing

Once ModelAPI handles intensity natively, remove manual `/255.0` workaround. For backward compat with old IRs, keep `needs_float32_input()` as a temporary shim that injects `configuration={"input_dtype": "f32", "intensity_mode": "scale_to_unit", "intensity_max_value": 255.0}`.

### C7. UI changes (frontend spec)

1. **Auto-detected preset** based on dataset analysis

2. **Intensity mode dropdown** — Scale to unit | Window/Level | Percentile clip | Range scale

3. **Mode-specific fields** — conditionally shown (max_value, center/width, low%/high%, scale_factor/min/max)

4. **Channel repeat** — checkbox + count for grayscale→RGB


## Notes

### Backward compatibility (simple UINT8 case)

Old IRs default to `input_dtype="u8"`, `intensity_mode="none"` → current behavior preserved.

### Why all four intensity modes in ModelAPI?

Three (`scale_to_unit`, `window`, `range_scale`) are static and embeddable in the OV graph → zero caller-side preprocessing, identical behavior everywhere. `percentile` stays Python-side but the config is in rt_info so ModelAPI applies it consistently.


## Data Flow After Implementation


```

[Geti UI] User uploads uint16 TIFF thermal images

→ Geti detects pixel_dtype="uint16", auto-sets intensity.mode="scale_to_unit", max=65535

→ User switches to "range_scale" with thermal params → saved to DB



[Training] Geti → Geti Tune IntensityConfig(mode="range_scale", ...)

→ RangeScale applied during training

→ Exporter writes input_dtype, intensity_mode, intensity params to rt_info



[ModelAPI load] Reads all params from rt_info

→ Builds OV graph: u16 input → resize (u16 pad) → range_scale nodes → mean/scale → model



[Inference] Raw uint16 TIFF → ModelAPI → embedded preprocessing handles everything

→ No workarounds needed



[Old IR fallback] No input_dtype → defaults to u8 → current behavior unchanged

```
