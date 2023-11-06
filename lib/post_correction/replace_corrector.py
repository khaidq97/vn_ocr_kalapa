import re
import pandas as pd 

class ReplaceCorrector:
    def __init__(self, data_path):
        self.database = self._load_data(data_path)
        
    def correct(self, sentence):
        for key, value in self.database.values.tolist():
            sentence = re.sub(key, value, sentence)
        return sentence
        
    def _load_data(self, data_path):
        df = pd.read_csv(str(data_path))
        return df