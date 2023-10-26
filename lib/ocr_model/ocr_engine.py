import cv2 
import torch
import numpy as np
import onnxruntime as ort
from PIL import Image
from torch.autograd import Variable
from torch.nn.functional import softmax
from ..utils.utils import strLabelConverter, resizePadding, load_vocab

class CRNNModelONNX:
    def __init__(self,
                 model_path,
                 vocab_path,
                 imgH=64,
                 imgW=512,
                 device='cpu',
                 half=False):
        self.device = device
        self.imgW = imgW
        self.imgH = imgH
        self.half = half
        alphabet = load_vocab(vocab_path)
        self.converter = strLabelConverter(alphabet, ignore_case=False)
        
        providers = ['CPUExecutionProvider'] if device=='cpu' else ['CUDAExecutionProvider']
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.output_names = [i.name for i in self.session.get_outputs()]
        self.input_names = self.session.get_inputs()[0].name
        
    def run(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = resizePadding(image, self.imgW, self.imgH)
        image = image.view(1, *image.size()).detach().cpu().numpy()
        if self.half:
            image = image.astype(np.float16)
        preds = self.session.run(self.output_names, {self.input_names: image})[0].astype(np.float32)
        preds = torch.from_numpy(preds)
        
        values, prob = softmax(preds, dim=-1).max(2)
        preds_idx = (prob > 0).nonzero()
        sent_prob = values[preds_idx[:,0], preds_idx[:, 1]].mean().item()

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
        return sim_pred, sent_prob