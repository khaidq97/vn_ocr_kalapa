from .regex_corrector import RegexCorrector
from ..configs import settings as cfg

class PostCorrector:
    def __init__(self, cfg=cfg):
        self.regex_corrector = RegexCorrector(cfg.REGEX_CORRECTOR_DATA_PATH)
        
    def run(self, text):
        text = self.regex_corrector.run(text)
        return text