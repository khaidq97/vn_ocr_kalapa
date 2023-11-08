import re
import copy 
import pandas as pd 
import numpy as np
from nltk.util import ngrams
from fuzzywuzzy import fuzz
from unidecode import unidecode

def has_number(input_str):
    return any(char.isdigit() for char in input_str)

def correct_sentence(sentence,
                     fail_text,
                     correct_text, 
                     threshold=90):
    sentence_list = sentence.split()
    correct_text_list = correct_text.split()
    max_n = len(correct_text_list)
    for n_gram in range(1, max_n+1):
        n_gram_list = list(ngrams(sentence_list, n_gram))
        for data in n_gram_list:
            text_ = ' '.join(data)
            score = fuzz.ratio(unidecode(copy.deepcopy(text_).lower()), unidecode(copy.deepcopy(fail_text).lower()))
            # print(fail_text, '|', text_, score)
            if score >= threshold and not has_number(text_):
                print('replace: ', text_, correct_text)
                sentence = sentence.replace(text_, correct_text)
    return sentence
    
class ReplaceCorrector:
    def __init__(self, data_path):
        self.database = self._load_data(data_path)
        
    def correct(self, sentence):
        for key, value in self.database.values.tolist():
            sentence = correct_sentence(
                sentence, 
                fail_text=key,
                correct_text=value, 
                threshold=95
            )
        return sentence
        
    def _load_data(self, data_path):
        df = pd.read_csv(str(data_path))
        return df
    
if __name__ == '__main__':
    corrector = ReplaceCorrector(data_path='/home/khai/Desktop/COMPETITIONS/vn_ocr_kalapa/vn_ocr_kalapa/assets/replace_correction.csv')
    text = 'Ấp Sác Song Ngọc Thành Giồng Ring'
    
    print(text)
    print(corrector.correct(text))