import numpy as np
from model_api.models.result import DetectionResult
from model_api.visualizer.layout import Flatten
from model_api.visualizer.primitive import BoundingBox, Label
from model_api.visualizer.scene import DetectionScene
from PIL import Image


class DetectionVisualizer:
    def overlay_predictions(original_image: np.ndarray, predictions: DetectionResult) -> np.ndarray:
        image_pil = Image.fromarray(original_image)
        detection_scene = DetectionScene(image=image_pil, result=predictions, layout=Flatten(BoundingBox, Label))
        rendered_detections_pil = detection_scene.render()
        return np.array(rendered_detections_pil)
