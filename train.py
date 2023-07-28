import glob

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import DHNutils
from dataloader import *
from torch.autograd import Variable
from DHNutils import GetOption
from models.modules.DHN import DHN
from torch.utils.data import DataLoader
import torch.optim as optim
from natsort import natsorted
import math
import time
from DHNutils import SaveImageFromTensor

# prepare training data
print("preparing training data")

# trainImgNameList = natsorted(
#     sorted(glob.glob(GetOption("dataDir") + GetOption("trainImgDir") + "*" + GetOption("imgExt"))))
trainImgNameList = natsorted(sorted(glob.glob("../VIS30K/VIS30K/*/" + "*Info*" + GetOption("imgExt"))))
validateImgNameList = natsorted(
    sorted(glob.glob(GetOption("dataDir") + GetOption("validateImgDir") + "*" + GetOption("imgExt"))))
sciDataNameList = natsorted(
    sorted(glob.glob(GetOption("dataDir") + GetOption("sciDataDir") + "*" + GetOption("imgExt"))))
perlinDataNameList = natsorted(
    sorted(glob.glob(GetOption("dataDir") + GetOption("perlinDataDir") + "*" + GetOption("imgExt"))))
scatterDataNameList = natsorted(
    sorted(glob.glob(GetOption("dataDir") + GetOption("scatterDataDir") + "*" + GetOption("imgExt"))))
qrDataNameList = natsorted(
    sorted(glob.glob(GetOption("dataDir") + GetOption("qrDataDir") + "*" + GetOption("imgExt"))))

trainDataSet = DHNDataset(trainImgNameList,
                          [sciDataNameList, perlinDataNameList, scatterDataNameList],
                          qrDataNameList,
                          imgTransform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                           transforms.RandomVerticalFlip(),
                                                           transforms.Resize(224),
                                                           transforms.RandomCrop(
                                                               (GetOption("inputSize"), GetOption("inputSize"))),
                                                           transforms.ToTensor()]),
                          dataTransform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                            transforms.RandomVerticalFlip(),
                                                            transforms.RandomCrop((GetOption("inputSize"), GetOption("inputSize"))),
                                                            transforms.ToTensor()]),
                          qrTransform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                          transforms.RandomVerticalFlip(),
                                                          transforms.RandomCrop((GetOption("inputSize"), GetOption("inputSize"))),
                                                          transforms.ToTensor()]),
                          isTrain=True)

print("data type num: " + str(trainDataSet.dataTypeNum))

trainDataLoader = DataLoader(trainDataSet, batch_size=GetOption("batchSize"), shuffle=True, num_workers=16,
                             drop_last=True)

validateImgTransform = transforms.Compose([transforms.RandomCrop((GetOption("inputSize"), GetOption("inputSize"))),
                                           transforms.ToTensor()])
validateDataTransform = transforms.Compose([transforms.Resize((GetOption("inputSize"), GetOption("inputSize"))),
                                            transforms.ToTensor()])
validateQRTransform = transforms.Compose([transforms.RandomCrop((GetOption("inputSize"), GetOption("inputSize"))),
                                          transforms.ToTensor(),
                                          transforms.RandomErasing(p=0.5, scale=(0.2, 0.95), ratio=(0.1, 0.95))])

print("training image number %d" % len(trainDataSet))

# define logging parameters
iteratorNum = 0
writer = SummaryWriter()


def computePSNR(origin, pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin / 1.0 - pred / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)


# define network
net = DHN(inputSize=GetOption("inputSize"),
          blockNum=GetOption("blockNum"),
          embedChannelNum=GetOption("embedChannelNum"),
          learningRate=10 ** GetOption("log10lr"),
          betas=(GetOption("betas1"), GetOption("betas2")),
          eps=GetOption("eps"),
          weightDecay=GetOption("weightDecay"),
          isTrain=True)
if torch.cuda.is_available():
    net.cuda()
# InitializeModel(net)

# define optimizer
params_trainable = (list(filter(lambda p: p.requires_grad, list(net.parameters()))))
optimizer = optim.Adam(params_trainable, net.learningRate, betas=net.betas, eps=net.eps, weight_decay=net.weightDecay)
lrScheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.9)

# load(GetOption("modelDir") + "MVSN_epoch160.pth")

# training
net.train()

f = False

for ec in range(GetOption("epochNum")):
    epochCount = ec
    st = time.time()

    totalLossHistory = []
    perceptionLossHistory = []
    embedLossSHistory = []
    embedLossFHistory = []
    restoreBSLossHistory = []
    restoreBFLossHistory = []

    for i, data in enumerate(trainDataLoader):
        if i > 500:
            break
        net.train()
        net.isTrain = True
        iteratorNum += 1

        img = data["image"].cuda()
        randData = data["randData"].cuda()

        netInput = torch.cat((img, randData), dim=1)
        assert netInput.shape[1] == 3 + GetOption("embedChannelNum")

        embedLossS, embedLossF, restoreLossTS, restoreLossTF, restoreLossBS, embed, restoreB = net(netInput)

        if net.isHaar:
            totalLoss = restoreLossBS * 3.2 + embedLossS * 2.0 + embedLossF * 1.0
        else:
            totalLoss = restoreLossBS * 3.2 + embedLossS * 2.0
        totalLoss.backward()
        optimizer.step()
        optimizer.zero_grad()

        totalLossHistory.append([totalLoss.item(), 0.])
        embedLossSHistory.append([embedLossS.item(), 0.])
        if net.isHaar:
            embedLossFHistory.append([embedLossF.item(), 0.])
        restoreBSLossHistory.append([restoreLossBS.item(), 0.])

    epochTotalLoss = np.mean(np.array(totalLossHistory), axis=0)
    epochEmbedLossS = np.mean(np.array(embedLossSHistory), axis=0)
    if net.isHaar:
        epochEmbedLossF = np.mean(np.array(embedLossFHistory), axis=0)
    epochRestoreBSLoss = np.mean(np.array(restoreBSLossHistory), axis=0)

    writer.add_scalar("epochLoss/total", epochTotalLoss[0], epochCount + 1)
    writer.add_scalar("embedLoss/embedS", epochEmbedLossS[0], epochCount + 1)
    if net.isHaar:
        writer.add_scalar("embedLoss/embedF", epochEmbedLossF[0], epochCount + 1)
    writer.add_scalar("restoreLoss/restoreS", epochRestoreBSLoss[0], epochCount + 1)

    sp = time.time()

    if epochCount > 0 and (epochCount + 1) % 10 == 0:
        torch.save({'opt': optimizer.state_dict(),
                    'net': net.state_dict()}, GetOption("modelDir") + "MVSN_epoch%d.pth" % (epochCount + 1))
    lrScheduler.step()

    with torch.no_grad():
        psnr_s = []
        psnr_c = []
        mse_s = []
        net.eval()
        net.isTrain = False

        img = io.imread(GetOption("dataDir") + GetOption("validateImgDir") + "white" + GetOption("imgExt"))[:, :, :3]
        img = Image.fromarray(img)
        dataX = io.imread(GetOption("dataDir") + GetOption("validateImgDir") + "x" + GetOption("imgExt"))[:, :, 0]
        dataX = Image.fromarray(dataX)
        dataY = io.imread(GetOption("dataDir") + GetOption("validateImgDir") + "y" + GetOption("imgExt"))[:, :, 0]
        dataY = Image.fromarray(dataY)
        dataZ = io.imread(GetOption("dataDir") + GetOption("validateImgDir") + "z" + GetOption("imgExt"))[:, :, 0]
        dataZ = Image.fromarray(dataZ)
        dataW = io.imread(GetOption("dataDir") + GetOption("validateImgDir") + "qr" + GetOption("imgExt"))[:, :, 0]
        dataW = Image.fromarray(dataW)
        img = validateImgTransform(img).cuda()
        dataX = validateDataTransform(dataX)
        dataY = validateDataTransform(dataY)
        dataZ = validateDataTransform(dataZ)
        dataW = validateQRTransform(dataW)
        # randData = torch.cat([dataX, dataY, dataZ, dataW * GetOption("qrMul")], dim=0).cuda()
        randData = torch.cat([dataX, dataY, dataW * GetOption("qrMul")], dim=0).cuda()
        randData = randData.unsqueeze(0)
        img = img.unsqueeze(0)
        netInput = torch.cat([img, randData], dim=1)
        embedLossS, embedLossF, restoreLossTS, restoreLossTF, restoreLossBS, embed, restoreB = net(netInput)
        # if epochCount > 0 and (epochCount + 1) % 5 == 0:
            # SaveImageFromTensor(embed, GetOption("resultImgDir") + "epoch" + str(epochCount + 1) + "_embed_" + GetOption("imgExt"), needUnnormalize=False)
            # SaveImageFromTensor(img, GetOption("resultImgDir") + "epoch" + str(epochCount + 1) + "_init_" + GetOption("imgExt"), needUnnormalize=False)
            # # SaveImageFromTensor(imp, GetOption("resultImgDir") + "epoch" + str(epochCount + 1) + "_imp_" + GetOption("imgExt"), needUnnormalize=False)
            # SaveImageFromTensor(torch.abs(embed - img), GetOption("resultImgDir") + "epoch" + str(epochCount + 1) + "_embedLoss_" + GetOption("imgExt"), needUnnormalize=False)
            # SaveImageFromTensor(restoreB[:, 0:1, :, :], GetOption("resultImgDir") + "epoch" + str(epochCount + 1) + "_restoreBX_" + GetOption("imgExt"), needUnnormalize=False)
            # SaveImageFromTensor(restoreB[:, 1:2, :, :], GetOption("resultImgDir") + "epoch" + str(epochCount + 1) + "_restoreBY_" + GetOption("imgExt"), needUnnormalize=False)
            # SaveImageFromTensor(restoreB[:, 2:3, :, :], GetOption("resultImgDir") + "epoch" + str(epochCount + 1) + "_restoreBZ_" + GetOption("imgExt"), needUnnormalize=False)
            # # SaveImageFromTensor(restoreB[:, 3:4, :, :] / GetOption("qrMul"), GetOption("resultImgDir") + "epoch" + str(epochCount + 1) + "_restoreBW_" + GetOption("imgExt"), needUnnormalize=False)
            # SaveImageFromTensor(torch.abs(restoreB[:, 0:1, :, :] - randData[:, 0:1, :, :]), GetOption("resultImgDir") + "epoch" + str(epochCount + 1) + "_restoreLossX_" + GetOption("imgExt"), needUnnormalize=False)
            # SaveImageFromTensor(torch.abs(restoreB[:, 1:2, :, :] - randData[:, 1:2, :, :]), GetOption("resultImgDir") + "epoch" + str(epochCount + 1) + "_restoreLossY_" + GetOption("imgExt"), needUnnormalize=False)
            # SaveImageFromTensor(torch.abs(restoreB[:, 2:3, :, :] - randData[:, 2:3, :, :]), GetOption("resultImgDir") + "epoch" + str(epochCount + 1) + "_restoreLossZ_" + GetOption("imgExt"), needUnnormalize=False)
            # # SaveImageFromTensor(torch.abs(restoreB[:, 3:4, :, :] - randData[:, 3:4, :, :]) / GetOption("qrMul"), GetOption("resultImgDir") + "epoch" + str(epochCount + 1) + "_restoreLossW_" + GetOption("imgExt"), needUnnormalize=False)

        # if epochCount == 0:
            # SaveImageFromTensor(dataX.unsqueeze(0), GetOption("resultImgDir") + "dataX" + GetOption("imgExt"), needUnnormalize=False)
            # SaveImageFromTensor(dataY.unsqueeze(0), GetOption("resultImgDir") + "dataY" + GetOption("imgExt"), needUnnormalize=False)
            # SaveImageFromTensor(dataZ.unsqueeze(0), GetOption("resultImgDir") + "dataZ" + GetOption("imgExt"), needUnnormalize=False)
            # SaveImageFromTensor(dataW.unsqueeze(0), GetOption("resultImgDir") + "dataW" + GetOption("imgExt"), needUnnormalize=False)

        mse_s.append(DHNutils.GetL1Loss(restoreB, randData).item())

        restoreB = restoreB.cpu().numpy().squeeze() * 255
        np.clip(restoreB, 0, 255)
        randData = randData.cpu().numpy().squeeze() * 255
        np.clip(randData, 0, 255)
        img = img.cpu().numpy().squeeze() * 255
        np.clip(img, 0, 255)
        embed = embed.cpu().numpy().squeeze() * 255
        np.clip(embed, 0, 255)
        psnr_temp = computePSNR(restoreB, randData)
        psnr_s.append(psnr_temp)
        psnr_temp_c = computePSNR(img, embed)
        psnr_c.append(psnr_temp_c)

        writer.add_scalars("PSNR_S", {"average psnr": np.mean(psnr_s)}, epochCount + 1)
        writer.add_scalars("PSNR_C", {"average psnr": np.mean(psnr_c)}, epochCount + 1)

        print("epoch: %d, lr: %.12f, epochLoss: %.4f, PSNR_C = %.4f, ResLossB: %.4f, time: %.1f" % (epochCount + 1, optimizer.state_dict()['param_groups'][0]['lr'], epochTotalLoss[0], np.mean(psnr_c), np.mean(mse_s), sp - st), end=", ")
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

print("done")
