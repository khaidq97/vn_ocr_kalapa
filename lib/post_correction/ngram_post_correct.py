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

class Database:
    def __init__(self, data_path):
        self.df = pd.read_csv(str(data_path))
        self.level1 = self.get_level1_list()
        self.level2 = self.get_level2_list()
        self.level3 = self.get_level3_list()
        
    def level1_to_level2_list(self, level_1):
        df = copy.deepcopy(self.df)
        level2_list = df[df['level_1'] == level_1]['level_2'].unique().tolist()
        return level2_list
    
    def level2_to_level3_list(self, level_1, level_2):
        df = copy.deepcopy(self.df)
        level3_list = df.query(f'level_2 == @level_2 and level_1 == @level_1')['level_3'].unique().tolist()
        return level3_list
    
    def level3_to_level4_list(self, level_2, level_3):
        df = copy.deepcopy(self.df)
        level4_list = df.query(f'level_2 == @level_2 and level_3 == @level_3')['level_4'].unique().tolist()
        return level4_list
    
    def level3_to_level1_level2(self, name):
        df = copy.deepcopy(self.df)
        level1 = df[df['level_3'] == name]['level1'].unique().tolist()
        level2 = df[df['level_3'] == name]['level_2'].unique().tolist()
        return level1, level2
    
    def level2_to_level3(self, name):
        df = copy.deepcopy(self.df)
        level3_list = df[df['level_2'] == name]['level_3'].unique().tolist()
        return level3_list
    
    def level3_to_level4(self, name):
        df = copy.deepcopy(self.df)
        level3_list = df[df['level_3'] == name]['level_4'].unique().tolist()
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
    def __init__(self, data_path, threshold=85):
        self.database = Database(data_path)
        self.threshold = threshold
        
    def correct(self, sentence):
        sentences_list = sentence.split()
        sentences_tracking_list = [True] * len(sentences_list)
        # check level 1
        sentences_list, level_1, sentences_tracking_list = self.correct_level(
                                                                            sentences_list=sentences_list,
                                                                            sentences_tracking_list=sentences_tracking_list,
                                                                            level_list=self.database.level1)
        # check level 2
        level_2 = ''
        if level_1 != '':
            sentences_list, level_2, sentences_tracking_list = self.correct_level(
                                                                        sentences_list=sentences_list,
                                                                        sentences_tracking_list=sentences_tracking_list,
                                                                        level_list=self.database.level1_to_level2_list(level_1))
            # print(self.database.level1_to_level2_list(level_1), level_1)
            # print(level_1)
        # check level 3
        level_3 = ''
        if level_2 != '':
            sentences_list, level_3, sentences_tracking_list = self.correct_level(
                                                                            sentences_list=sentences_list,
                                                                            sentences_tracking_list=sentences_tracking_list,
                                                                            level_list=self.database.level2_to_level3_list(level_1, level_2))
            # print(self.database.level2_to_level3_list(level_1, level_2))
            # print(level_1, level_2)
        # check level 4
        if level_3!='':
            level_list = []
            for x in self.database.level3_to_level4_list(level_2, level_3):
                if isinstance(x, str):
                    level_list.append(x)
            # print(level_1, level_2, level_3, level_list)
            if len(level_list):
                sentences_list, level_4, sentences_tracking_list = self.correct_level(
                                                                                    sentences_list=sentences_list,
                                                                                    sentences_tracking_list=sentences_tracking_list,
                                                                                    level_list=level_list)
        return ' '.join(sentences_list)
        
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
    corrector  = NgramPostCorrector(data_path='/home/khai/Desktop/COMPETITIONS/vn_ocr_kalapa/vn_ocr_kalapa/assets/postcorrection.csv')
    
    text = 'Xóm D Lau Võ Miếu Thanh Sơn Phú Thọ'
   
    print(text)
    print(corrector.correct(text))
    # print(corrector.correct_level1(text))
    # print(corrector.correct_level2(text))
    # print(corrector.correct_level(text,
    #                               level_list=corrector.database.level3))