from .ocr_model.ocr_engine import CRNNModelONNX
from .post_correction.post_corrector import PostCorrector
from .configs import settings as cfg

class Controller:
    def __init__(self, cfg=cfg):
        self.ocr_engine = CRNNModelONNX(
            model_path=cfg.OCR_MODEL_PATH,
            vocab_path=cfg.OCR_VOCAB_PATH,
            device=cfg.DEVICE,
        )
        
        self.post_corrector = PostCorrector()
    
    def __call__(self, image):
        result = self.ocr_engine.run(image)[0]
        result = self.post_corrector.run(result)
        return result