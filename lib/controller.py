from .ocr_model.ocr_detector import OCRDetector

class Controller:
    def __init__(self):
        self.ocr_detector = OCRDetector()
    
    def __call__(self, image):
        result = self.ocr_detector(image)
        return result