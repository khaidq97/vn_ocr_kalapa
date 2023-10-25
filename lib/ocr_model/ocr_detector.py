from .vietocr_model import VIETOCR
from .crnn_model.crnn_model import CRNNModel
from ..configs import settings as cfg

class OCRDetector:
    def __init__(self, cfg=cfg):
        # self.model = VIETOCR(cfg.OCR_MODEL_PATH)
        self.model = CRNNModel(
            model_path=cfg.OCR_MODEL_PATH,
            vocab_path='/home/khai/Desktop/COMPETITIONS/vn_ocr_kalapa/assets/vocab.txt',
            )
        
    def __call__(self, image):
        result = self.model.run(image)
        return result