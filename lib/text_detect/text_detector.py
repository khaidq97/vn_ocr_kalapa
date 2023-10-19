
from ..utils.tools import sorted_bbox
from .yolov5 import Yolov5Detect
from ..configs import settings as cfg

class TextDetector:
    def __init__(self, cfg=cfg) -> None:
        self.model = Yolov5Detect(
            weight=cfg.TEXT_DETECT_MODEL_PATH,
        )
        
    def __call__(self, img):
        boxes = self.model(img)[0].tolist()
        boxes = sorted_bbox(boxes)
        return boxes