import copy 
import pandas as pd
import numpy as np
from nltk.util import ngrams
from fuzzywuzzy import fuzz
from unidecode import unidecode

def has_number(input_str):
    return any(char.isdigit() for char in input_str)

def correct_sentence(sentences_list,
                     sentences_tracking_list, 
                     correct_sentence,
                     threshold=70,
                     use_ngram_1=False):
    correct_sentence_list = correct_sentence.split()
    n_gram = len(correct_sentence_list)
    if not use_ngram_1 and n_gram < 2:
        return sentences_list, [], sentences_tracking_list
    
    sentence_list = copy.deepcopy(sentences_list)
    sentences_tracking_list_ = copy.deepcopy(sentences_tracking_list)
    n_gram_list = list(ngrams(sentence_list, n_gram))
    
    scores = []
    indexs = []
    for i, data in enumerate(n_gram_list):
        text = ' '.join(data)
        score = fuzz.ratio(unidecode(copy.deepcopy(text).lower()), unidecode(copy.deepcopy(correct_sentence).lower()))
        if score >= threshold and sentences_tracking_list_[i] and not has_number(text):
            scores.append(score)
            indexs.append(i)
    score = 0
    if len(indexs):
        id = indexs[np.argmax(scores)]
        score = max(scores)
        
        sentence_list[id:id+n_gram] = correct_sentence_list
        sentences_tracking_list_[id:id+n_gram] = [False] * n_gram
        
    return sentence_list, [score], sentences_tracking_list_

class SentenceCorrector:
    def __init__(self, data_path, threshold=85):
        self.database = self._load_data(data_path)
        self.threshold = threshold
        
    def correct(self, sentence):
        sentences_list = sentence.split()
        sentences_tracking_list = [True] * len(sentences_list)
        # check level 1
        sentences_list, level_1, sentences_tracking_list = self.correct_level(
                                                                            sentences_list=sentences_list,
                                                                            sentences_tracking_list=sentences_tracking_list,
                                                                            level_list=self.database)
        return ' '.join(sentences_list)
        
    def _load_data(self, data_path):
        df = pd.read_csv(str(data_path))
        return df['text'].values.tolist()
    
    def correct_level(self, sentences_list,sentences_tracking_list, level_list, use_ngram_1=False):
        sentence_list, score_list, correct_text_list, sentences_tracking = [], [], [], []
        for i, level in enumerate(level_list):
            sentence_, scores, sentences_tracking_list_ = correct_sentence(
                sentences_list=sentences_list,
                sentences_tracking_list=sentences_tracking_list,
                correct_sentence=level,
                threshold=75,
                use_ngram_1=use_ngram_1)
            if len(scores) > 0:
                sentence_list.append(sentence_)
                score_list.append(max(scores))
                correct_text_list.append(level)
                sentences_tracking.append(sentences_tracking_list_)
        max_score = 0  
        correct_text = ''
        sentence_tracking = sentences_tracking_list
        sentence_list_ = sentences_list
        if len(score_list):
            max_id = np.argmax(score_list)
            max_score = score_list[max_id]
        if max_score>=self.threshold:
            sentence_list_ = sentence_list[max_id]
            correct_text = correct_text_list[max_id]
            sentence_tracking = sentences_tracking[max_id]
        return sentence_list_, correct_text, sentence_tracking
    
if __name__ == '__main__':
    corrector = SentenceCorrector(data_path='/home/khai/Desktop/COMPETITIONS/vn_ocr_kalapa/vn_ocr_kalapa/assets/sentence_correct.csv')
    
    text = 'Q88 Cath Gến Nân Đồn Phường 55 Quận 9 Hồ Chí Minh'
    print(text)
    print(corrector.correct(text))