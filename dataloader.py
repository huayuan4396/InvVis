from skimage import io
from PIL import Image
from random import randint
from torch.utils.data import Dataset
from DHNutils import GetPerlinNoise
import numpy as np
import torch
from DHNutils import GetOption, SaveImageFromTensor


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image



class DHNDataset(Dataset):
    def __init__(self, imgNameList, dataNameList, qrNameList, imgTransform=None, dataTransform=None, qrTransform=None, isTrain=True):
        self.imgNameList = imgNameList
        self.dataNameList = dataNameList
        self.dataTypeNum = len(dataNameList)
        self.qrNameList = qrNameList
        self.imgTransform = imgTransform
        self.dataTransform = dataTransform
        self.qrTransform = qrTransform
        self.isTrain = isTrain

    def __len__(self):
        return len(self.imgNameList)

    def __getitem__(self, idx):
        # img = io.imread(self.imgNameList[idx])[:, :, :3]
        # img2 = io.imread(self.imgNameList[np.random.randint(0, len(self.imgNameList))])[:, :, :3]
        img = Image.open(self.imgNameList[idx])
        img = to_rgb(img)
        typeIdx = idx % self.dataTypeNum

        randData = io.imread(self.dataNameList[typeIdx][np.random.randint(0, len(self.dataNameList[typeIdx]) - 1)])
        if len(randData.shape) == 2:
            pass
        elif len(randData.shape) == 3:
            randData = randData[:, :, 0]
        randData = Image.fromarray(randData)


        if self.isTrain:
            if self.imgTransform:
                img = self.imgTransform(img)
            if self.dataTransform:
                randData = self.dataTransform(randData)

            # randData = randData + torch.randn_like(randData) * 0.05

            randDataMin = torch.min(randData)
            randDataMax = torch.max(randData)
            randData = (randData - randDataMin) / (randDataMax - randDataMin + 0.00001)

            for i in range(GetOption("embedChannelNum") - 2):
                now = io.imread(self.dataNameList[typeIdx][np.random.randint(0, len(self.dataNameList[typeIdx]) - 1)])
                if len(now.shape) == 2:
                    pass
                elif len(now.shape) == 3:
                    now = now[:, :, 0]
                now = Image.fromarray(now)

                if self.dataTransform:
                    now = self.dataTransform(now)

                # now = now + torch.randn_like(now) * 0.05

                nowMin = torch.min(now)
                nowMax = torch.max(now)
                now = (now - nowMin) / (nowMax - nowMin + 0.00001)

                randData = torch.cat([randData, now], dim=0)

            qr = io.imread(self.qrNameList[np.random.randint(0, len(self.qrNameList) - 1)])
            if len(qr.shape) == 2:
                pass
            elif len(qr.shape) == 3:
                qr = qr[:, :, 0]
            qr = Image.fromarray(qr)

            if self.qrTransform:
                qr = self.qrTransform(qr)

            randData = torch.cat([randData, qr * GetOption("qrMul")], dim=0)


            return {"image": img, "randData": randData}

        else:
            randData = io.imread(self.dataNameList[typeIdx][idx % len(self.dataNameList[typeIdx])])
            if len(randData.shape) == 2:
                pass
            elif len(randData.shape) == 3:
                randData = randData[:, :, 0]
            randData = Image.fromarray(randData)
            
            if self.imgTransform:
                img = self.imgTransform(img)
            if self.dataTransform:
                randData = self.dataTransform(randData)

            # randData = randData + torch.randn_like(randData) * 0.05

            randDataMin = torch.min(randData)
            randDataMax = torch.max(randData)
            randData = (randData - randDataMin) / (randDataMax - randDataMin + 0.00001)

            for i in range(GetOption("embedChannelNum") - 2):
                now = io.imread(self.dataNameList[typeIdx][(idx + i + 1) % len(self.dataNameList[typeIdx])])
                if len(now.shape) == 2:
                    pass
                elif len(now.shape) == 3:
                    now = now[:, :, 0]
                now = Image.fromarray(now)

                if self.dataTransform:
                    now = self.dataTransform(now)

                # now = now + torch.randn_like(now) * 0.05

                nowMin = torch.min(now)
                nowMax = torch.max(now)
                now = (now - nowMin) / (nowMax - nowMin + 0.00001)

                randData = torch.cat([randData, now], dim=0)

            qr = io.imread(self.qrNameList[idx % len(self.qrNameList)])
            if len(qr.shape) == 2:
                pass
            elif len(qr.shape) == 3:
                qr = qr[:, :, 0]
            qr = Image.fromarray(qr)

            if self.qrTransform:
                qr = self.qrTransform(qr)

            return {"image": img, "dataImg": randData, "qrImg": qr * GetOption("qrMul")}




