import torch
from models.modules.FlowBlock import FlowBlock
from models.modules.DenseBlock import DenseBlock
import torch.nn as nn
from DHNutils import GetOption
import DHNutils
from models.modules.HAARLayer import HAARLayer
from torchvision import transforms
import random


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


class Quant(nn.Module):
    def __init__(self):
        super(Quant, self).__init__()

    def forward(self, x, isRound):
        x = x * 255.0
        if not isRound:
            noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5).cuda()
            # noise = torch.zeros_like(x).cuda()
            output = x + noise
            output = torch.clamp(output, 0, 255.)
        else:
            output = x.round() * 1.0
            output = torch.clamp(output, 0, 255.)
        return output / 255.0


class EnhanceBlock(nn.Module):
    def __init__(self, inChannelNum, middleChannelNum1, middleChannelNum2):
        super(EnhanceBlock, self).__init__()

        self.stage1 = nn.Sequential(
            DenseBlock(inChannelNum, middleChannelNum1),
            nn.Conv2d(middleChannelNum1, middleChannelNum1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(middleChannelNum1, middleChannelNum1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(middleChannelNum1, middleChannelNum1, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.stage2 = nn.Sequential(
            DenseBlock(middleChannelNum1, middleChannelNum2),
            nn.Conv2d(middleChannelNum2, middleChannelNum2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(middleChannelNum2, middleChannelNum2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(middleChannelNum2, middleChannelNum2, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.stage3 = nn.Sequential(
            DenseBlock(middleChannelNum2, middleChannelNum1),
            nn.Conv2d(middleChannelNum1, middleChannelNum1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(middleChannelNum1, middleChannelNum1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(middleChannelNum1, middleChannelNum1, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.stage4 = nn.Sequential(
            DenseBlock(middleChannelNum1, inChannelNum)
        )

    def forward(self, x):
        st1 = self.stage1(x)
        st2 = self.stage2(st1)
        st3 = self.stage3(st2)
        st4 = self.stage4(st3)

        return x + st4


class DHN(nn.Module):
    def __init__(self, inputSize, blockNum, embedChannelNum, learningRate, betas, eps, weightDecay, isTrain):
        super(DHN, self).__init__()
        self.inputSize = inputSize
        self.isTrain = isTrain
        self.blockNum = blockNum
        self.embedChannelNum = embedChannelNum
        self.isHaar = True
        self.splitNum1 = 4 * 3
        self.splitNum2 = embedChannelNum * 4
        self.learningRate = learningRate
        self.betas = betas
        self.eps = eps
        self.weightDecay = weightDecay

        self.enhanceBlockForward = EnhanceBlock(3 + self.embedChannelNum, 64, 256)
        self.enhanceBlockBackward = EnhanceBlock(3 + self.embedChannelNum, 64, 256)
        # self.enhanceBlockForward = None
        # self.enhanceBlockBackward = None

        if self.isHaar:
            self.flowBlock = FlowBlock(blockNum, self.splitNum1, self.splitNum2)
        else:
            self.flowBlock = FlowBlock(blockNum, self.splitNum1 // 4, self.splitNum2 // 4)

        self.quant = Quant()
        self.haar = HAARLayer(3)

    def encode(self, x):
        assert x.shape[1] == 3 + GetOption("embedChannelNum")
        xT = x.narrow(1, 0, self.splitNum1 // 4)
        xB = x.narrow(1, self.splitNum1 // 4, self.splitNum2 // 4)


        e = self.enhanceBlockForward(x)
        # e = x.clone()
        eT = e.narrow(1, 0, self.splitNum1 // 4)
        eB = e.narrow(1, self.splitNum1 // 4, self.splitNum2 // 4)\

        hT = self.haar(eT, False)

        B, C, H, W = xB.shape
        hB = eB.reshape(B, C, H // 2, 2, W // 2, 2)
        hB = hB.permute(0, 1, 3, 5, 2, 4)
        hB = hB.reshape(B, C * 4, H // 2, W // 2)
        h = torch.cat([hT, hB], dim=1)
        assert h.shape[1] == 12 + self.embedChannelNum * 4

        middle = self.flowBlock(h, False)

        embed = self.haar(middle.narrow(1, 0, self.splitNum1), True)
        embed = self.quant(embed, isRound=not self.isTrain)
        assert embed.shape[1] == 3

        return xT, xB, hT, hB, embed


    def decode(self, embed):
        noise = torch.zeros(embed.shape[0], self.embedChannelNum * 4, embed.shape[2] // 2, embed.shape[3] // 2).cuda()
        restoreInputImg = self.haar(embed, False)

        restoreInputData = noise
        restoreInput = torch.cat([restoreInputImg, restoreInputData], dim=1)

        end = self.flowBlock(restoreInput, True)
        endT = end.narrow(1, 0, self.splitNum1)
        endB = end.narrow(1, self.splitNum1, self.splitNum2)
        assert endT.shape[1] == 3 * 4 and endB.shape[1] == 4 * self.embedChannelNum

        restoreT = end.narrow(1, 0, self.splitNum1)
        restoreT = self.haar(restoreT, True)

        B, C, H, W = endB.shape
        restoreB = endB.reshape(B, C // 4, 2, 2, H, W)
        restoreB = restoreB.permute(0, 1, 4, 2, 5, 3)
        restoreB = restoreB.reshape(B, C // 4, H * 2, W * 2)

        restore = torch.cat([restoreT, restoreB], dim=1)
        restore = self.enhanceBlockBackward(restore)
        restoreT = restore.narrow(1, 0, self.splitNum1 // 4)
        restoreB = restore.narrow(1, self.splitNum1 // 4, self.splitNum2 // 4)

        assert restoreT.shape[1] == 3 and restoreB.shape[1] == self.embedChannelNum

        return restoreT, restoreB, endT


    def encodeNoHaar(self, x):
        xT = x.narrow(1, 0, self.splitNum1 // 4)
        xB = x.narrow(1, self.splitNum1 // 4, self.splitNum2 // 4)

        e = self.enhanceBlockForward(x)

        middle = self.flowBlock(e, False)

        embed = middle.narrow(1, 0, self.splitNum1 // 4)
        embed = self.quant(embed, isRound=not self.isTrain)
        assert embed.shape[1] == 3

        return xT, xB, 0, 0, embed


    def decodeNoHaar(self, embed):
        noise = torch.zeros(embed.shape[0], self.embedChannelNum, embed.shape[2], embed.shape[3]).cuda()
        restoreInputImg = embed.clone()

        restoreInputData = noise
        restoreInput = torch.cat([restoreInputImg, restoreInputData], dim=1)

        end = self.flowBlock(restoreInput, True)

        restore = self.enhanceBlockBackward(end)
        restoreT = restore.narrow(1, 0, self.splitNum1 // 4)
        restoreB = restore.narrow(1, self.splitNum1 // 4, self.splitNum2 // 4)

        assert restoreT.shape[1] == 3 and restoreB.shape[1] == self.embedChannelNum

        return restoreT, restoreB, 0


    def forward(self, x):
        if self.isHaar:
            xT, xB, hT, hB, embed = self.encode(x)
            restoreT, restoreB, endT = self.decode(embed)
            embedLossS = DHNutils.GetL1Loss(xT, embed) + DHNutils.GetMSELoss(xT, embed)
            embedF = self.haar(embed, False)
            embedLossF = DHNutils.GetMSELoss(hT.narrow(1, 0, 3), embedF.narrow(1, 0, 3)) + DHNutils.GetL1Loss(hT.narrow(1, 0, 3), embedF.narrow(1, 0, 3))

            restoreLossTS = DHNutils.GetMSELoss(xT, restoreT) + DHNutils.GetL1Loss(xT, restoreT)
            restoreLossTF = 0

            mul = 2.0
            restoreLossBS = DHNutils.GetL1Loss(xB[:, 0:self.embedChannelNum - 1, :, :], restoreB[:, 0:self.embedChannelNum - 1, :, :])
            restoreLossBS += DHNutils.GetMSELoss(xB[:, 0:self.embedChannelNum - 1, :, :], restoreB[:, 0:self.embedChannelNum - 1, :, :])
            restoreLossBS += DHNutils.GetL1Loss(mul * xB[:, self.embedChannelNum - 1:, :, :], mul * restoreB[:, self.embedChannelNum - 1:, :, :])
            restoreLossBS += DHNutils.GetMSELoss(mul * xB[:, self.embedChannelNum - 1:, :, :], mul * restoreB[:, self.embedChannelNum - 1:, :, :])

        else:
            xT, xB, hT, hB, embed = self.encodeNoHaar(x)
            restoreT, restoreB, endT = self.decodeNoHaar(embed)
            embedLossS = DHNutils.GetL1Loss(xT, embed) + DHNutils.GetMSELoss(xT, embed)
            embedLossF = 0.0
            restoreLossTS = DHNutils.GetMSELoss(xT, restoreT) + DHNutils.GetL1Loss(xT, restoreT)
            restoreLossTF = 0
            mul = 1.5
            restoreLossBS = DHNutils.GetL1Loss(xB[:, 0:self.embedChannelNum - 1, :, :], restoreB[:, 0:self.embedChannelNum - 1, :, :])
            restoreLossBS += DHNutils.GetMSELoss(xB[:, 0:self.embedChannelNum - 1, :, :], restoreB[:, 0:self.embedChannelNum - 1, :, :])
            restoreLossBS += DHNutils.GetL1Loss(mul * xB[:, self.embedChannelNum - 1:, :, :], mul * restoreB[:, self.embedChannelNum - 1:, :, :])
            restoreLossBS += DHNutils.GetMSELoss(mul * xB[:, self.embedChannelNum - 1:, :, :], mul * restoreB[:, self.embedChannelNum - 1:, :, :])

        return embedLossS, embedLossF, restoreLossTS, restoreLossTF, restoreLossBS, embed, restoreB





