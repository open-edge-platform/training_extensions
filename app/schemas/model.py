from enum import Enum

from app.schemas.base import BaseIDNameModel


class ModelFormat(str, Enum):
    OPENVINO = "openvino_ir"
    ONNX = "onnx"


class Model(BaseIDNameModel):
    """
    Base model schema that includes common fields for all models.
    This can be extended by other schemas to include additional fields.
    """

    format: ModelFormat = ModelFormat.OPENVINO

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "76e07d18-196e-4e33-bf98-ac1d35dca4cb",
                "name": "YOLO-X for Vehicle Detection",
                "format": "openvino_ir",
            }
        }
    }
