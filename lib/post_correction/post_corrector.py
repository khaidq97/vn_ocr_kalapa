from ..configs import settings as cfg
from .ngram_post_correct import NgramPostCorrector
from .replace_corrector import ReplaceCorrector
from .sentence_correct import SentenceCorrector

class PostCorrector:
    def __init__(self, cfg=cfg):
        self.ngram_post_corrector = NgramPostCorrector(cfg.NGRAM_DATA_PATH)
        self.replace_post_corrector = ReplaceCorrector(cfg.REPLACE_DATA_PATH)
        self.sentence_correct = SentenceCorrector(cfg.SENTENCE_DATA_PATH)
        
        
    def __call__(self, sentence):
        sentence = self.replace_post_corrector.correct(sentence)
        # sentence = self.sentence_correct.correct(sentence)
        sentence = self.ngram_post_corrector.correct(sentence)
        return sentence