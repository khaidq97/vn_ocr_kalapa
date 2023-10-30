import re
import pandas as pd

class RegexCorrector:
    def __init__(self, data_path):
        self.data = self.__load_data(data_path)
        
    def __load_data(self, data_path):
        df = pd.read_csv(str(data_path))
        data = {}
        for i in range(len(df)):
            data[df.iloc[i]['wrong']] = df.iloc[i]['correct']
        return data
    
    def run(self, text):
        for wrong, correct in self.data.items():
            text = re.sub(wrong, correct, text)
        return text 