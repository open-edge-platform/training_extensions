# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from loguru import logger

from app.models.media import MediaType
from app.models.model_revision import ModelFormat

if TYPE_CHECKING:
    from app.services import MediaService


# Filename of the model weights inside the archive, depending on the format.
_MODEL_FILENAME_BY_FORMAT: dict[ModelFormat, str] = {
    ModelFormat.OPENVINO: "model.xml",
    ModelFormat.ONNX: "model.onnx",
}


@dataclass(frozen=True)
class DemoFile:
    """A single auxiliary file to be added to the downloaded model archive."""

    name: str
    data: bytes


class DemoFilesService:
    def __init__(self, media_service: MediaService):
        self._media_service: MediaService = media_service

    def build_demo_files(self, project_id: UUID, model_format: ModelFormat) -> list[DemoFile]:
        """Build the auxiliary deployment files to bundle in with the model archive.

        Only OpenVINO (.xml/.bin) and ONNX (.onnx) variants are supported - for any other
        format, an empty list is returned (PyTorch checkpoints are typically used for
        fine-tuning rather than direct deployment).

        Args:
            project_id: Project that owns the model.
            model_format: Format of the model variant being downloaded.

        Returns:
            The list of files (name + bytes) to add to the zip archive.
        """
        if model_format not in _MODEL_FILENAME_BY_FORMAT:
            return []

        model_filename = _MODEL_FILENAME_BY_FORMAT[model_format]
        files: list[DemoFile] = []

        sample_image = self._pick_sample_image(project_id=project_id)
        if sample_image is not None:
            files.append(DemoFile(name="image.jpg", data=sample_image))
        else:
            logger.warning(
                "No suitable sample image found in project {}; the model archive will not include 'image.jpg'.",
                project_id,
            )

        files.append(DemoFile(name="demo.py", data=_DEMO.format(model_filename=model_filename).encode("utf-8")))
        files.append(
            DemoFile(name="demo_async.py", data=_DEMO_ASYNC.format(model_filename=model_filename).encode("utf-8"))
        )
        files.append(DemoFile(name="requirements.txt", data=_REQUIREMENTS.encode("utf-8")))
        files.append(DemoFile(name="README.md", data=_README.format(model_filename=model_filename).encode("utf-8")))
        return files

    def _pick_sample_image(self, project_id: UUID) -> bytes | None:
        """Pick a sample image from the project's dataset and return its bytes.

        Videos are excluded; if no image is available, returns None.
        """
        try:
            from app.services.media_service import MediaFilters  # local import to avoid cycle at module load
        except Exception:  # pragma: no cover - defensive
            return None

        try:
            media_list = self._media_service.list_media(
                project_id=project_id,
                filters=MediaFilters(limit=1, offset=0),
                exclude_types=[MediaType.VIDEO, MediaType.VIDEO_FRAME],
            )
        except Exception:
            logger.exception("Failed to list media to pick a sample image for project {}", project_id)
            return None

        for media in media_list:
            try:
                path: Path = self._media_service.get_media_binary_path(project_id=project_id, media=media)
                if path.exists():
                    return path.read_bytes()
            except Exception:
                logger.exception("Failed to read sample image {} for project {}", media.id, project_id)
                continue
        return None


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_DEMO = '''\
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Synchronous inference demo for a model exported from Geti.

Loads a sample image, runs inference with OpenVINO Model API, then saves an
output image with the overlayed predictions to result.jpg.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from model_api.models import Model
from PIL import Image

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "{model_filename}"
IMAGE_PATH = HERE / "image.jpg"
OUTPUT_PATH = HERE / "result.jpg"


def overlay_predictions(image_bgr: np.ndarray, result) -> np.ndarray:
    """Render predictions on top of the input image using model_api visualizers."""
    # model_api visualizers consume PIL/RGB images.
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    try:
        from model_api.visualizer import Visualizer

        visualizer = Visualizer()
        rendered = visualizer.show(image=image_pil, result=result)
        return cv2.cvtColor(np.array(rendered), cv2.COLOR_RGB2BGR)
    except Exception:
        # Fallback: print the result and return the original image untouched.
        print("Visualization not available, printing raw result instead:")
        print(result)
        return image_bgr


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {{MODEL_PATH}}")
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Sample image not found: {{IMAGE_PATH}}")

    print(f"Loading model from {{MODEL_PATH}}...")
    model = Model.create_model(str(MODEL_PATH))

    print(f"Loading image from {{IMAGE_PATH}}...")
    image_bgr = cv2.imread(str(IMAGE_PATH))
    if image_bgr is None:
        raise RuntimeError(f"Failed to decode image: {{IMAGE_PATH}}")

    print("Running synchronous inference...")
    result = model(image_bgr)
    print("Predictions:")
    print(result)

    output = overlay_predictions(image_bgr, result)
    cv2.imwrite(str(OUTPUT_PATH), output)
    print(f"Saved annotated result to {{OUTPUT_PATH}}")


if __name__ == "__main__":
    main()
'''


_DEMO_ASYNC = '''\
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Asynchronous inference demo for a model exported from Geti.

Uses OpenVINO Model API's AsyncPipeline to submit the sample image
asynchronously and retrieve the prediction once it is ready. The resulting
image with the overlayed predictions is saved to result_async.jpg.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from model_api.models import Model
from model_api.pipelines import AsyncPipeline
from PIL import Image

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "{model_filename}"
IMAGE_PATH = HERE / "image.jpg"
OUTPUT_PATH = HERE / "result_async.jpg"


def overlay_predictions(image_bgr: np.ndarray, result) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    try:
        from model_api.visualizer import Visualizer

        visualizer = Visualizer()
        rendered = visualizer.show(image=image_pil, result=result)
        return cv2.cvtColor(np.array(rendered), cv2.COLOR_RGB2BGR)
    except Exception:
        print("Visualization not available, printing raw result instead:")
        print(result)
        return image_bgr


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {{MODEL_PATH}}")
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Sample image not found: {{IMAGE_PATH}}")

    print(f"Loading model from {{MODEL_PATH}}...")
    model = Model.create_model(str(MODEL_PATH))

    print(f"Loading image from {{IMAGE_PATH}}...")
    image_bgr = cv2.imread(str(IMAGE_PATH))
    if image_bgr is None:
        raise RuntimeError(f"Failed to decode image: {{IMAGE_PATH}}")

    print("Running asynchronous inference...")
    pipeline = AsyncPipeline(model)
    pipeline.submit_data(image_bgr, id=0)
    pipeline.await_all()
    result, _meta = pipeline.get_result(0)
    print("Predictions:")
    print(result)

    output = overlay_predictions(image_bgr, result)
    cv2.imwrite(str(OUTPUT_PATH), output)
    print(f"Saved annotated result to {{OUTPUT_PATH}}")


if __name__ == "__main__":
    main()
'''


_REQUIREMENTS = """\
# Runtime dependencies to run the demo scripts.
openvino>=2024.0
openvino-model-api>=0.2.0
opencv-python-headless>=4.9
numpy>=1.26
pillow>=10.0
"""


_README = """\
# Geti exported model

This archive contains a model exported from Geti, along with a couple of
ready-to-run inference demos.

## Contents

| File | Description |
| ---- | ----------- |
| `{model_filename}` (+ `model.bin` for OpenVINO IR) | The exported model weights. |
| `image.jpg` | A sample image taken from the dataset used to train the model. |
| `demo.py` | Minimal **synchronous** inference example. |
| `demo_async.py` | Minimal **asynchronous** inference example. |
| `requirements.txt` | Python dependencies required by the demos. |
| `README.md` | This file. |

## Setup

The recommended way to set up a clean environment is with
[`uv`](https://docs.astral.sh/uv/) - a fast Python package manager.

### Option 1 - one-shot with `uvx` (no environment to manage)

If you just want to run the demos without creating a project, you can use
`uv run` with inline dependencies:

```bash
# From the directory where this README lives
uv run --with-requirements requirements.txt demo.py
uv run --with-requirements requirements.txt demo_async.py
```

`uv` will transparently create a temporary virtual environment, install the
dependencies and execute the script.

### Option 2 - create a persistent virtual environment

```bash
# Create and activate a virtual environment (Python 3.10+)
uv venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\\Scripts\\activate

# Install the runtime dependencies
uv pip install -r requirements.txt
```

### Option 3 - plain pip

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

## Running the demos

Once the environment is ready, simply run:

```bash
# Synchronous inference - writes the annotated result to result.jpg
python demo.py

# Asynchronous inference - writes the annotated result to result_async.jpg
python demo_async.py
```

Both scripts load `image.jpg`, run inference on it with OpenVINO Model API and
save an output image with the predicted bounding boxes / labels / masks
overlayed on top.

## Notes

* The demos default to running on CPU. To run on a different device (e.g. an
  Intel GPU), edit the scripts and pass device="GPU" to
  Model.create_model.
* For ONNX models, OpenVINO Model API reads the `.onnx` file directly - no
  additional conversion is required.
* These demos are intentionally minimal. For production deployment, refer to
  the [OpenVINO Model API documentation](https://github.com/open-edge-platform/model_api).
"""
