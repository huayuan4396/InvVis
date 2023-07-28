import glob

import torch
from torchvision import transforms
from dataloader import *
from torch.autograd import Variable
from models.modules.DHN import DHN
from torch.utils.data import DataLoader
from DHNutils import SaveImageFromTensor, GetImgFromPatch


# load pretrained model
print("loading net ...")
net = DHN(inputSize=GetOption("inputSizeTest"),
          blockNum=GetOption("blockNum"),
          embedChannelNum=GetOption("embedChannelNum"),
          learningRate=10 ** GetOption("log10lr"),
          betas=(GetOption("betas1"), GetOption("betas2")),
          eps=GetOption("eps"),
          weightDecay=GetOption("weightDecay"),
          isTrain=False)


def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)


# make data image, qr code image, cover image same size
def padTensor(t, shape):
    if t.shape[2] > shape[2]:
        t = t[:, :, :shape[2], :]
    if t.shape[3] > shape[3]:
        t = t[:, :, :, :shape[3]]

    # row interpolation
    r = t.shape[2]
    sRow = t[:, :, r - 1:r, :]
    for i in range(shape[2] - r):
        m = (shape[2] - r - 1 - i) / (shape[2] - r - 1)
        t = torch.cat([t, sRow * m], dim=2)

    # col interpolation
    c = t.shape[3]
    sCol = t[:, :, :, c - 1:c]
    for i in range(shape[3] - c):
        m = (shape[3] - c - 1 - i) / (shape[3] - c - 1)
        t = torch.cat([t, sCol * m], dim=3)

    return t


load(GetOption("pretrainedModelDir"))

if torch.cuda.is_available():
    net.cuda()


# testing
net.eval()

imgTransform = transforms.Compose([transforms.ToTensor()])

with torch.no_grad():
    img = Image.open("./data/test/cover.png")
    img = to_rgb(img)
    img = imgTransform(img)
    img = img.unsqueeze(0)
    img = img[:, :, :img.shape[2] - img.shape[2] % 2, :img.shape[3] - img.shape[3] % 2]

    tarShape = img.shape

    di = Image.open("./data/test/data_image.png")
    di = to_rgb(di)
    di = imgTransform(di)
    di = di.unsqueeze(0)
    di = padTensor(di, tarShape)[:, :3, :, :]

    qr = Image.open("./data/test/qr_image.png")
    qr = to_rgb(qr)
    qr = imgTransform(qr)
    qr = qr.unsqueeze(0)
    qr = padTensor(qr, tarShape)[:, :1, :, :] * GetOption("qrMul")

    netInput = torch.cat([img, di, qr], dim=1)

    sz = GetOption("inputSizeTest")
    hc = (netInput.shape[2] - 1) // sz + 1
    wc = (netInput.shape[3] - 1) // sz + 1

    embedList = []
    restoreList = []

    for i in range(hc):
        for j in range(wc):
            nowInput = netInput[:, :, i * sz:min((i + 1) * sz, netInput.shape[2]), j * sz:min((j + 1) * sz, netInput.shape[3])]
            embedLossS, embedLossF, restoreLossTS, restoreLossTF, restoreLossBS, embed, restoreB = net(nowInput.cuda())
            embedList.append(embed)
            restoreList.append(restoreB)
            nowInput.cpu()
            torch.cuda.empty_cache()


    embed = None
    restore = None

    for i in range(hc):
        row = None
        for j in range(wc):
            now = embedList[i * wc + j]
            if j == 0:
                row = now
            else:
                row = torch.cat([row, now], dim=3)
        if i == 0:
            embed = row
        else:
            embed = torch.cat([embed, row], dim=2)

    for i in range(hc):
        row = None
        for j in range(wc):
            now = restoreList[i * wc + j]
            if j == 0:
                row = now
            else:
                row = torch.cat([row, now], dim=3)
        if i == 0:
            restore = row
        else:
            restore = torch.cat([restore, row], dim=2)

    SaveImageFromTensor(embed, GetOption("resultImgDir") + "embed" + GetOption("imgExt"), needUnnormalize=False)
    SaveImageFromTensor(restore[:, :3, :, :], GetOption("resultImgDir") + "restore_data_image" + GetOption("imgExt"), needUnnormalize=False)
    SaveImageFromTensor(restore[:, 3:, :, :] / GetOption("qrMul"), GetOption("resultImgDir") + "restore_qr_image" + GetOption("imgExt"), needUnnormalize=False)
    print("done")