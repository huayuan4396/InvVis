import torch.nn as nn
import models.pytorch_ssim as pyssim
import torch
from torchvision import utils
import matplotlib.pyplot as plt
from thirdparty.perlin_numpy_master.perlin_numpy import generate_fractal_noise_2d
import yaml
import numpy as np
import random
import string
import qrcode
from torchvision import transforms
from random import randint


L1Loss = nn.L1Loss(reduce=True, size_average=False)
SSIMLoss = pyssim.SSIM(window_size=11, size_average=True)
MSELoss = nn.MSELoss(reduce=True, size_average=False)
BCELoss = nn.BCELoss(size_average=False)


def GetSSIMLoss(target, pred):
    return 1 - SSIMLoss(target, pred)


def GetL1Loss(target, pred):
    return L1Loss(target, pred)


def GetMSELoss(target, pred):
    return MSELoss(target, pred)


def GetBCELoss(target, pred):
    return BCELoss(target, pred)


def Differentiable_Round(x, alpha):
    m = torch.floor(x) + 0.5
    r = x - m
    z = torch.tanh(torch.Tensor([alpha / 2.0])) * 2.0
    y = m.cuda() + (torch.tanh(alpha * r.cuda()) / z.cuda()).cuda()

    return y



def GetPerlinNoise(shape):
    resList = [2, 4, 8]
    res = resList[np.random.randint(0, 2)]
    octave = np.random.randint(1, 8)
    np.random.seed(np.random.randint(0, 10000000))
    noise = generate_fractal_noise_2d((shape[0], shape[1]), (res, res), octave)
    return noise


def SaveNoise(shape, name, dire="./data/train/scatter/"):
    img = torch.rand(shape)
    img = img.unsqueeze(0).unsqueeze(0)
    SaveImageFromTensor(img, dire + name + GetOption("imgExt"), needUnnormalize=False)


def SavePerlinNoise(shape, name, dire="./data/train/Perlin/"):
    img = GetPerlinNoise(shape)
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    SaveImageFromTensor(img, dire + name + GetOption("imgExt"), needUnnormalize=False)


def InitializeModel(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = 0.01 * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5':
                param.data.fill_(0.)




def Unnormalize(tensor: torch.Tensor, mean, std, inplace: bool = False) -> torch.Tensor:
    """Unnormalize a tensor image with mean and standard deviation.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if tensor.ndim < 3:
        raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def GetOption(optionName):
    with open("config.yml", mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    return opt[optionName]


def SaveImageFromTensor(tensor, filename, needUnnormalize=False):
    assert tensor.shape[0] == 1
    tensor = tensor.clone().detach()
    tensor = tensor.to(torch.device('cpu'))
    if needUnnormalize:
        tensor = Unnormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    utils.save_image(tensor, filename)


def SaveScatterImg(name):
    # img = torch.randn([512 * 512, 2]).numpy()
    # img = img.tolist()
    # img.sort(key=lambda x: x[0])
    #
    # tmp = []
    # nowLen = 0
    # cnt = 0
    #
    # while nowLen < 512 * 512:
    #     # step = randint(16, 32)
    #     step = 512
    #     if nowLen + step > 512 * 512:
    #         step = 512 * 512 - nowLen
    #     tmp += sorted(img[nowLen:nowLen + step][:], key=lambda x: x[1] * (1 if cnt % 2 == 0 else -1))
    #     nowLen += step
    #     cnt += 1
    #
    # img, tmp = tmp, img

    K = 7
    size = 512 // (K + 1)

    img = torch.randn([size * size, 2]).numpy()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = img.tolist()
    img.sort(key=lambda x: x[0])


    tmp = []

    # while nowLen < 512 * 64:
    #     # step = randint(16, 32)
    #     step = 512
    #     if nowLen + step > 512 * 64:
    #         step = 512 * 64 - nowLen
    #     tmp += sorted(img[nowLen:nowLen + step][:], key=lambda x: x[1] * (1 if cnt % 2 == 0 else -1))
    #     nowLen += step
    #     cnt += 1

    for i in range(size):
        nowRow = sorted(img[i * size:(i + 1) * size][:], key=lambda x: x[1])
        tmpRow = []
        for j in range(size):
            now = nowRow[j:j + 1][:]

            if j == size - 1:
                for k in range(K + 1):
                    tmpRow += now
                break

            nx = nowRow[j + 1:j + 2][:]

            for k in range(K + 1):
                # print(np.array(now).shape)
                # print(np.array(nx).shape)
                tmpRow += ((K + 1 - k) / (K + 1) * np.array(now) + k / (K + 1) * np.array(nx)).tolist()

        nowRow, tmpRow = tmpRow, nowRow

        if i == size - 1:
            for j in range(K + 1):
                tmp += nowRow
            break

        nxRow = sorted(img[(i + 1) * size:(i + 2) * size][:], key=lambda x: x[1])
        tmpRow = []
        for j in range(size):
            now = nxRow[j:j + 1][:]

            if j == size - 1:
                for k in range(K + 1):
                    tmpRow += now
                break

            nx = nxRow[j + 1:j + 2][:]

            for k in range(K + 1):
                tmpRow += ((K + 1 - k) / (K + 1) * np.array(now) + k / (K + 1) * np.array(nx)).tolist()

        nxRow, tmpRow = tmpRow, nxRow

        for j in range(K + 1):
            tmp += ((K + 1 - j) / (K + 1) * np.array(nowRow) + j / (K + 1) * np.array(nxRow)).tolist()

    img, tmp = tmp, img

    img = np.array(img).astype(np.float32)
    # print(img.shape)
    imgX = torch.from_numpy(img).reshape(512, 512, 2).permute(2, 0, 1)[0:1, ...].unsqueeze(0)
    imgY = torch.from_numpy(img).reshape(512, 512, 2).permute(2, 0, 1)[1:2, ...].unsqueeze(0)
    SaveImageFromTensor(imgX, GetOption("dataDir") + "train/Scatter_7/" + name + "X" + GetOption("imgExt"), needUnnormalize=False)
    SaveImageFromTensor(imgY, GetOption("dataDir") + "train/Scatter_7/" + name + "Y" + GetOption("imgExt"), needUnnormalize=False)


def SaveScatterImgV2(name, m):
    img = torch.randn([512 // m * 512 // m, 2]).numpy()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = img.tolist()

    tmp = []

    # while nowLen < 512 * 64:
    #     # step = randint(16, 32)
    #     step = 512
    #     if nowLen + step > 512 * 64:
    #         step = 512 * 64 - nowLen
    #     tmp += sorted(img[nowLen:nowLen + step][:], key=lambda x: x[1] * (1 if cnt % 2 == 0 else -1))
    #     nowLen += step
    #     cnt += 1

    for i in range(512 // m):
        nowRow = img[i * 512 // m:(i + 1) * 512 // m][:]
        tmpRow = []
        for j in range(512 // m):
            now = nowRow[j:j + 1][:]

            if j == 512 // m - 1:
                for k in range(m):
                    tmpRow += now
                break

            nx = nowRow[j + 1:j + 2][:]

            for k in range(m):
                tmpRow += ((m - k) / m * np.array(now) + k / m * np.array(nx)).tolist()

        nowRow, tmpRow = tmpRow, nowRow

        if i == 512 // m - 1:
            for j in range(m):
                tmp += nowRow
            break

        nxRow = img[(i + 1) * 512 // m:(i + 2) * 512 // m][:]
        tmpRow = []
        for j in range(512 // m):
            now = nxRow[j:j + 1][:]

            if j == 512 // m - 1:
                for k in range(m):
                    tmpRow += now
                break

            nx = nxRow[j + 1:j + 2][:]

            for k in range(m):
                tmpRow += ((m - k) / m * np.array(now) + k / m * np.array(nx)).tolist()

        nxRow, tmpRow = tmpRow, nxRow

        for j in range(m):
            tmp += ((m - j) / m * np.array(nowRow) + j / m * np.array(nxRow)).tolist()

    img, tmp = tmp, img

    img = np.array(img).astype(np.float32)
    # print(img.shape)
    imgX = torch.from_numpy(img).reshape(512, 512, 2).permute(2, 0, 1)[0:1, ...].unsqueeze(0)
    imgY = torch.from_numpy(img).reshape(512, 512, 2).permute(2, 0, 1)[1:2, ...].unsqueeze(0)
    SaveImageFromTensor(imgX, GetOption("dataDir") + "train/Scatter2/" + name + "X" + GetOption("imgExt"), needUnnormalize=False)
    SaveImageFromTensor(imgY, GetOption("dataDir") + "train/Scatter2/" + name + "Y" + GetOption("imgExt"), needUnnormalize=False)


def Fade(t):
    return 3 * t * t - 2 * t * t * t


def Interpolation(a, b, t):
    return (1 - t) * a + t * b


def SaveScatterImgV3(name, m):
    img = torch.randn([512 // m + 1,  512 // m + 1]).numpy()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = img.tolist()

    ret = []

    for i in range(512):
        for j in range(512):
            if i % m == 0 and j % m == 0:
                ret.append(img[i // m][j // m])
            else:
                ul = img[i // m][j // m]
                ur = img[i // m][(j - 1) // m + 1]
                dl = img[(i - 1) // m + 1][j // m]
                dr = img[(i - 1) // m + 1][(j - 1) // m + 1]
                di = (i - i // m * m) / m
                dj = (j - j // m * m) / m
                ret.append(Interpolation(Interpolation(ul, ur, Fade(dj)), Interpolation(dl, dr, Fade(dj)), Fade(di)))

    img = np.array(ret).astype(np.float32)
    # print(img.shape)
    img = torch.from_numpy(img).reshape(512, 512, 1).permute(2, 0, 1).unsqueeze(0)
    SaveImageFromTensor(img, GetOption("dataDir") + "train/Scatter2/" + name + "sc" + GetOption("imgExt"), needUnnormalize=False)


def GetRandomStr(length):
    letters = string.ascii_lowercase + "0123456789:;'\\,.<>[]{}-=_+|?!@#$%^&*()"
    rand_string = ''.join(random.choice(letters) for i in range(length))
    return rand_string


def SaveQRCode(name):
    QRC = qrcode.QRCode(
        version=40,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=40,
    )
    s = GetRandomStr(randint(800, 1200))
    QRC.add_data(s)
    QRC.make(fit=False)
    qr = QRC.make_image()

    qrImg = torch.from_numpy(np.array(qr)).unsqueeze(0).unsqueeze(0).float()
    tResize = transforms.Resize([512, 512])
    qrImg = tResize(qrImg)
    SaveImageFromTensor(qrImg, GetOption("dataDir") + "train/QR/" + name + GetOption("imgExt"), needUnnormalize=False)


def GetImgFromPatch(imgSize, patchSize, patchList):
    ih, iw = imgSize[:2]
    ph, pw = patchSize[:2]
    hc = (ih - 1) // ph + 1
    wc = (iw - 1) // pw + 1
    ret = None

    for i in range(hc):
        row = patchList[i * wc]
        for j in range(1, wc):
            row = torch.cat([row, patchList[i * wc + j]], dim=3)
        if i == 0:
            ret = row
        else:
            ret = torch.cat([ret, row], dim=2)

    return ret





