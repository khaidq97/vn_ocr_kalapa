from PIL import Image 
import cv2
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

class VIETOCR:
    def __init__(self, model_path, device='cpu'):
        config = Cfg.load_config_from_name('vgg_transformer') # vgg_transformer vgg_seq2seq
        config['weights'] = model_path
        config['device'] = device
        config['predictor']['beamsearch'] = False
        self.model = Predictor(config)
        
    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        result = self.model.predict(image)
        return result