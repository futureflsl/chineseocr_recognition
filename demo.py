from crnn.network_torch import CRNN
from PIL import Image
import torch
import numpy as np
from torch.autograd import Variable
from crnn.utils import strLabelConverter, resizeNormalize
from crnn.network_torch import CRNN
from collections import OrderedDict


class CRNNManager(object):
    def __init__(self, weights):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        alphabetChinese = self.load_chars('./data/mychars.txt')
        self.model = CRNN(32, 1, len(alphabetChinese) + 1, 256, 1, lstmFlag=True)

        trainWeights = torch.load(weights, map_location=lambda storage, loc: storage)
        modelWeights = OrderedDict()
        for k, v in trainWeights.items():
            name = k.replace('module.', '')  # remove `module.`
            modelWeights[name] = v
        # load params
        self.model.load_state_dict(modelWeights)
        self.model.to(self.device)
        self.model.eval()
        self.converter = strLabelConverter(alphabetChinese)

    def load_chars(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            chars = f.read().replace('\n', '')
        return chars

    def ocr(self, pil_img):
        image = pil_img.convert('L')
        scale = image.size[1] * 1.0 / 32
        w = image.size[0] / scale
        w = int(w)
        transformer = resizeNormalize((w, 32))
        image = transformer(image)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(self.device)
        image = image.view(1, 1, *image.size())
        image = Variable(image)
        preds = self.model(image)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_pred = self.converter.decode(preds)
        return sim_pred


if __name__ == '__main__':
    om = CRNNManager('/home/fut/Downloads/chineseocr_recongnition/outputs/modellstm.pth')
    img = Image.open('/home/fut/Downloads/chineseocr_recongnition/data/images/9_IMG_20211229.jpg')
    print(om.ocr(img))
