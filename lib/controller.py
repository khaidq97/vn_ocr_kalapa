from .configs import settings as cfg
from .ocr_model.ocr_engine import CRNNModelONNX
from .post_correction.post_corrector import PostCorrector

class Controller:
    def __init__(self, cfg=cfg):
        self.ocr_engine = CRNNModelONNX(
            model_path=cfg.OCR_MODEL_PATH,
            vocab_path=cfg.OCR_VOCAB_PATH,
            device=cfg.DEVICE,
            half=True
        )
        self.post_corrector = PostCorrector(cfg=cfg)
    
    def __call__(self, image):
        result = self.ocr_engine.run(image)[0]
        result = self.post_corrector(result)
        return result