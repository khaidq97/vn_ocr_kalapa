from .ngram_post_correct import NgramPostCorrector
from ..configs import settings as cfg

class PostCorrector:
    def __init__(self, cfg=cfg):
        self.ngram_post_corrector = NgramPostCorrector(cfg.NGRAM_DATA_PATH)
        
    def __call__(self, sentence):
        sentence = self.ngram_post_corrector.correct(sentence)
        return sentence