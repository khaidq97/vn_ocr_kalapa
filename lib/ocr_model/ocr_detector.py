from .vietocr import VIETOCR
from ..configs import settings as cfg

class OCRDetector:
    def __init__(self, cfg=cfg):
        self.model = VIETOCR(cfg.OCR_MODEL_PATH)
        
    def __call__(self, image):
        result = self.model(image)
        return result