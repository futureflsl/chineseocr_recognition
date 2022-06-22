import os
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
# from warpctc_pytorch import CTCLoss
from torch.nn import CTCLoss
from crnn.network_torch import CRNN
from crnn.util import strLabelConverter
from dataset import PathDataset, randomSequentialSampler, alignCollate
from dataset import resizeNormalize
from crnn.util import loadData
from generic_utils import Progbar


def load_chars(file):
    with open(file, 'r', encoding='utf-8') as f:
        chars = f.read().replace('\n', '')
    return chars


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def trainBatch(net, criterion, optimizer, cpu_images, cpu_texts):
    # data = train_iter.next()
    # cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)

    loadData(text, t)
    loadData(length, l)
    preds = net(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    net.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


def predict(im):
    """
    预测
    """
    image = im.convert('L')
    scale = image.size[1] * 1.0 / 32
    w = image.size[0] / scale
    w = int(w)
    transformer = resizeNormalize((w, 32))

    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred


def val(net, dataset, max_iter=100):
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    i = 0
    n_correct = 0
    N = len(dataset)
    # print('dataset len is:',N)
    max_iter = min(max_iter, N)
    # print('max_iter is:', max_iter)
    for i in range(max_iter):
        im, label = dataset[np.random.randint(0, N)]
        if im.size[0] > 1024:
            continue

        pred = predict(im)
        # print('pred is:',pred)
        # print('label is:',label)
        if pred.strip() == label:
            n_correct += 1

    accuracy = n_correct / float(max_iter)
    return accuracy


batchSize = 32
workers = 1
imgH = 32
imgW = 280
keep_ratio = True
cuda = True
ngpu = 1
nh = 256
lr = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
alphabetChinese = load_chars('./data/mychars.txt')
traindataset = PathDataset('./data/labels/train.txt')
testdataset = PathDataset('./data/labels/val.txt')
sampler = randomSequentialSampler(traindataset, batchSize)
train_loader = torch.utils.data.DataLoader(
    traindataset, batch_size=batchSize,
    shuffle=False, sampler=None,
    num_workers=int(workers),
    collate_fn=alignCollate(imgH=imgH, imgW=imgW, keep_ratio=keep_ratio))

train_iter = iter(train_loader)
model = CRNN(32, 1, len(alphabetChinese) + 1, 256, 1, lstmFlag=True)
model.apply(weights_init)
preWeightDict = torch.load('pretrained/ocr-lstm.pth', map_location=lambda storage, loc: storage)  ##加入项目训练的权重
modelWeightDict = model.state_dict()
for k, v in preWeightDict.items():
    name = k.replace('module.', '')  # remove `module.`
    if 'rnn.1.embedding' not in name:  ##不加载最后一层权重
        modelWeightDict[name] = v
model.load_state_dict(modelWeightDict)
optimizer = optim.Adadelta(model.parameters(), lr=lr)
converter = strLabelConverter(alphabetChinese)
criterion = CTCLoss()

image = torch.FloatTensor(batchSize, 3, imgH, imgH)
text = torch.IntTensor(batchSize * 5)
length = torch.IntTensor(batchSize)

if torch.cuda.is_available():
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[0])  ##转换为多GPU训练模型
    image = image.cuda()
    criterion = criterion.cuda()

nepochs = 10
acc = 0
interval = len(train_loader) // 2  ##评估模型
# 冻结预训练模型层参数
for i in range(nepochs):
    print('epoch:{}/{}'.format(i, nepochs))
    n = len(train_loader)
    pbar = Progbar(target=n)
    train_iter = iter(train_loader)
    loss = 0
    for j in range(n):
        for p in model.named_parameters():
            p[1].requires_grad = True
            if 'rnn.1.embedding' in p[0]:
                p[1].requires_grad = True
            else:
                p[1].requires_grad = False  ##冻结模型层

        model.train()
        cpu_images, cpu_texts = train_iter.next()
        cost = trainBatch(model, criterion, optimizer, cpu_images, cpu_texts)

        loss += cost.cpu().detach().numpy()

        if (j + 1) % interval == 0:
            curAcc = val(model, testdataset, max_iter=1024)
            if curAcc > acc:
                acc = curAcc
                torch.save(model.state_dict(), 'outputs/best.pth')

        pbar.update(j + 1, values=[('loss', loss / ((j + 1) * batchSize)), ('acc', acc)])
    if (i + 1) % 5 == 0:
        torch.save(model.state_dict(), 'outputs/last_{}.pth'.format(i + 1))

nepochs = 10
acc = 0
interval = len(train_loader) // 2  ##评估模型
# 释放模型层参数
for i in range(10, 10 + nepochs):
    print('epoch:{}/{}'.format(i, nepochs))
    n = len(train_loader)
    pbar = Progbar(target=n)
    train_iter = iter(train_loader)
    loss = 0
    for j in range(n):
        for p in model.named_parameters():
            p[1].requires_grad = True

        model.train()
        cpu_images, cpu_texts = train_iter.next()
        cost = trainBatch(model, criterion, optimizer, cpu_images, cpu_texts)

        loss += cost.cpu().detach().numpy()

        if (j + 1) % interval == 0:
            curAcc = val(model, testdataset, max_iter=1024)
            if curAcc > acc:
                acc = curAcc
                torch.save(model.state_dict(), 'outputs/modellstm.pth')

        pbar.update(j + 1, values=[('loss', loss / ((j + 1) * batchSize)), ('acc', acc)])
