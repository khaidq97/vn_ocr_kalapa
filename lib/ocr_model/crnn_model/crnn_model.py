import cv2 
import torch
from PIL import Image
from torch.autograd import Variable
from .crnn import CRNN
from .utils import strLabelConverter, resizePadding, load_vocab
from torch.nn.functional import softmax

class CRNNModel:
    def __init__(self,
                 model_path,
                 vocab_path,
                 imgH=32,
                 imgW=512,
                 nc=3,
                 nh=256,
                 device='cpu'):
        self.device = device
        self.imgW = imgW
        self.imgH = imgH
        alphabet = load_vocab(vocab_path)
        nclass = len(alphabet) + 1 
        model = CRNN(imgH, nc, nclass, nh).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = model
        self.model.eval()
        self.converter = strLabelConverter(alphabet, ignore_case=False)
        
    @torch.no_grad()
    def run(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = resizePadding(image, self.imgW, self.imgH)
        image = image.view(1, *image.size()).to(self.device)
        preds = self.model(image)
        
        values, prob = softmax(preds, dim=-1).max(2)
        preds_idx = (prob > 0).nonzero()
        sent_prob = values[preds_idx[:,0], preds_idx[:, 1]].mean().item()

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
        
        return sim_pred