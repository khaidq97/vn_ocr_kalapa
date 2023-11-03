import copy 
import pandas as pd
import numpy as np
from nltk.util import ngrams
from fuzzywuzzy import fuzz

def correct_sentence(sentence, 
                     correct_sentence,
                     threshold=70,
                     use_ngram_1=False):
    correct_sentence_list = correct_sentence.split()
    n_gram = len(correct_sentence_list)
    if not use_ngram_1 and n_gram < 2:
        return sentence, []
    
    sentence_list = copy.deepcopy(sentence).split()
    n_gram_list = list(ngrams(sentence_list, n_gram))
    
    scores = []
    for i, data in enumerate(n_gram_list):
        text = ' '.join(data)
        score = fuzz.ratio(text.lower(), correct_sentence.lower())
        if score >= threshold:
            sentence_list[i:i+n_gram] = correct_sentence_list
            scores.append(score)
    return ' '.join(sentence_list), scores

class Database:
    def __init__(self, data_path):
        self.df = pd.read_csv(str(data_path))
        self.level1 = self.get_level1_list()
        self.level2 = self.get_level2_list()
        self.level3 = self.get_level3_list()
        
    def level1_to_level2_list(self, name):
        df = copy.deepcopy(self.df)
        level2_list = df[df['level_1'] == name]['level_2'].unique().tolist()
        return level2_list
    
    def level2_to_level3_list(self, name):
        df = copy.deepcopy(self.df)
        level3_list = df[df['level_2'] == name]['level_3'].unique().tolist()
        return level3_list
    
    def level3_to_level1_level2(self, name):
        df = copy.deepcopy(self.df)
        level1 = df[df['level_3'] == name]['level1'].unique().tolist()
        level2 = df[df['level_3'] == name]['level_2'].unique().tolist()
        return level1, level2
    
    def level2_to_level3(self, name):
        df = copy.deepcopy(self.df)
        level3_list = df[df['level_2'] == name]['level_3'].unique().tolist()
        return level3_list
    
    def get_level1_list(self):
        df = copy.deepcopy(self.df)
        level1_list = df['level_1'].unique().tolist()
        return level1_list
    
    def get_level2_list(self):
        df = copy.deepcopy(self.df)
        level2_list = df['level_2'].unique().tolist()
        return level2_list
    
    def get_level3_list(self):
        df = copy.deepcopy(self.df)
        level3_list = df['level_3'].unique().tolist()
        return level3_list

class NgramPostCorrector:
    def __init__(self, data_path, threshold=70):
        self.database = Database(data_path)
        self.threshold = threshold
        
    def correct(self, sentence):
        # check level 1
        sentence, correct_text = self.correct_level(
            sentence=sentence,
            level_list=self.database.level1)
        # check level 2
        if correct_text != '':
            sentence, correct_text = self.correct_level(
                sentence=sentence,
                level_list=self.database.level1_to_level2_list(correct_text),
                use_ngram_1=False)
        # check level 3
        if correct_text != '':
            sentence, correct_text = self.correct_level(
                sentence=sentence,
                level_list=self.database.level2_to_level3_list(correct_text))
        return sentence
        
    def correct_level(self, sentence, level_list, use_ngram_1=False):
        sentence_list, score_list, correct_text_list = [], [], []
        for i, level in enumerate(level_list):
            sentence_, scores = correct_sentence(
                sentence=sentence,
                correct_sentence=level,
                threshold=70,
                use_ngram_1=use_ngram_1)
            if len(scores) > 0:
                sentence_list.append(sentence_)
                score_list.append(max(scores))
                correct_text_list.append(level)
        max_score = 0  
        correct_text = ''
        if len(score_list):
            max_id = np.argmax(score_list)
            max_score = score_list[max_id]
        if max_score>=self.threshold:
            sentence = sentence_list[max_id]
            correct_text = correct_text_list[max_id]
        return sentence, correct_text
        
    
    
if __name__ == '__main__':
    # corrector  = NgramPostCorrector(data_path='data/sorted.csv')
    corrector  = NgramPostCorrector(data_path='/media/khaidq@kaopiz.local/hdd1/OCR/KALAPA_OCR_VN/vn_ocr_kalapa/assets/postcorrection.csv')
    
    text = 'Tấn Tài Tp Phan Rang Tháp Chàm Ninh Thuận'
   
    print(text)
    print(corrector.correct(text))
    # print(corrector.correct_level1(text))
    # print(corrector.correct_level2(text))
    # print(corrector.correct_level(text,
    #                               level_list=corrector.database.level3))