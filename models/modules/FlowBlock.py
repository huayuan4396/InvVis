import torch
import torch.nn as nn
from models.modules.DenseBlock import DenseBlock
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape

    def get_weight(self, isRev):
        w_shape = self.w_shape
        if not isRev:
            weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
        else:
            weight = torch.inverse(self.weight.double()).float().view(w_shape[0], w_shape[1], 1, 1)
        return weight

    def forward(self, x, isRev):
        weight = self.get_weight(isRev)
        if not isRev:
            z = F.conv2d(x, weight)
            return z
        else:
            z = F.conv2d(x, weight)
            return z


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Bottleneck, self).__init__()
        # P = ((S-1)*W-S+F)/2, with F = filter size, S = stride
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        initialize_weights_xavier([self.conv1, self.conv2], 0.1)
        initialize_weights(self.conv3, 0)

    def forward(self, x):
        conv1 = self.lrelu(self.conv1(x))
        conv2 = self.lrelu(self.conv2(conv1))
        conv3 = self.conv3(conv2)
        return conv3


class FlowStep(nn.Module):
    def __init__(self, splitLen1, splitLen2, clamp=2.0):
        super(FlowStep, self).__init__()
        self.splitLen1 = splitLen1
        self.splitLen2 = splitLen2
        self.clamp = clamp

        # affine coupling function
        self.F = DenseBlock(self.splitLen2, self.splitLen1)
        self.G = DenseBlock(self.splitLen1, self.splitLen2)
        self.H = DenseBlock(self.splitLen1, self.splitLen2)

    def forward(self, x, isRev):
        x1, x2 = (x.narrow(1, 0, self.splitLen1), x.narrow(1, self.splitLen1, self.splitLen2))

        if not isRev:
            y1 = x1 + self.F(x2)
            s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(s)) + self.G(y1)
        else:
            s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)


class CouplingLayer(nn.Module):
    def __init__(self, split_len1, split_len2, kernal_size, clamp=1.0):
        super(CouplingLayer, self).__init__()
        self.split_len1 = split_len1
        self.split_len2 = split_len2
        self.clamp = clamp

        self.G1 = Bottleneck(self.split_len1, self.split_len2, kernal_size)
        self.G2 = Bottleneck(self.split_len2, self.split_len1, kernal_size)
        self.H1 = Bottleneck(self.split_len1, self.split_len2, kernal_size)
        self.H2 = Bottleneck(self.split_len2, self.split_len1, kernal_size)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        if not rev:
            y1 = x1.mul(torch.exp(self.clamp * (torch.sigmoid(self.G2(x2)) * 2 - 1))) + self.H2(x2)
            y2 = x2.mul(torch.exp(self.clamp * (torch.sigmoid(self.G1(y1)) * 2 - 1))) + self.H1(y1)
        else:
            y2 = (x2 - self.H1(x1)).div(torch.exp(self.clamp * (torch.sigmoid(self.G1(x1)) * 2 - 1)))
            y1 = (x1 - self.H2(y2)).div(torch.exp(self.clamp * (torch.sigmoid(self.G2(y2)) * 2 - 1)))
        return torch.cat((y1, y2), 1)


class INV_block(nn.Module):
    def __init__(self, splitLen1, splitLen2, clamp=2.0):
        super().__init__()
        self.clamp = clamp
        self.splitLen1 = splitLen1
        self.splitLen2 = splitLen2
        # ρ
        self.r = DenseBlock(self.splitLen1, self.splitLen2)
        # η
        self.y = DenseBlock(self.splitLen1, self.splitLen2)
        # φ
        self.f = DenseBlock(self.splitLen2, self.splitLen1)

        self._initialize_weights(0.01)

    def _initialize_weights(self, scale):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.splitLen1),
                  x.narrow(1, self.splitLen1, self.splitLen2))

        if not rev:

            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1

        else:

            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = (x1 - t2)

        return torch.cat((y1, y2), 1)


# class FlowBlock(nn.Module):
#     def __init__(self, blockNum, splitLen1, splitLen2):
#         super(FlowBlock, self).__init__()
#
#         self.inv1 = INV_block(splitLen1, splitLen2)
#         self.inv2 = INV_block(splitLen1, splitLen2)
#         self.inv3 = INV_block(splitLen1, splitLen2)
#         self.inv4 = INV_block(splitLen1, splitLen2)
#         self.inv5 = INV_block(splitLen1, splitLen2)
#         self.inv6 = INV_block(splitLen1, splitLen2)
#         self.inv7 = INV_block(splitLen1, splitLen2)
#         self.inv8 = INV_block(splitLen1, splitLen2)
#
#         self.inv9 = INV_block(splitLen1, splitLen2)
#         self.inv10 = INV_block(splitLen1, splitLen2)
#         self.inv11 = INV_block(splitLen1, splitLen2)
#         self.inv12 = INV_block(splitLen1, splitLen2)
#         self.inv13 = INV_block(splitLen1, splitLen2)
#         self.inv14 = INV_block(splitLen1, splitLen2)
#         self.inv15 = INV_block(splitLen1, splitLen2)
#         self.inv16 = INV_block(splitLen1, splitLen2)
#
#     def forward(self, x, rev=False):
#
#         if not rev:
#             out = self.inv1(x)
#             out = self.inv2(out)
#             out = self.inv3(out)
#             out = self.inv4(out)
#             out = self.inv5(out)
#             out = self.inv6(out)
#             out = self.inv7(out)
#             out = self.inv8(out)
#
#             out = self.inv9(out)
#             out = self.inv10(out)
#             out = self.inv11(out)
#             out = self.inv12(out)
#             out = self.inv13(out)
#             out = self.inv14(out)
#             out = self.inv15(out)
#             out = self.inv16(out)
#
#         else:
#             out = self.inv16(x, rev=True)
#             out = self.inv15(out, rev=True)
#             out = self.inv14(out, rev=True)
#             out = self.inv13(out, rev=True)
#             out = self.inv12(out, rev=True)
#             out = self.inv11(out, rev=True)
#             out = self.inv10(out, rev=True)
#             out = self.inv9(out, rev=True)
#
#             out = self.inv8(out, rev=True)
#             out = self.inv7(out, rev=True)
#             out = self.inv6(out, rev=True)
#             out = self.inv5(out, rev=True)
#             out = self.inv4(out, rev=True)
#             out = self.inv3(out, rev=True)
#             out = self.inv2(out, rev=True)
#             out = self.inv1(out, rev=True)
#
#         return out

class FlowBlock(nn.Module):
    def __init__(self, blockNum, splitLen1, splitLen2):
        super(FlowBlock, self).__init__()
        self.moduleList = nn.ModuleList()

        for i in range(blockNum):
            self.moduleList.append(InvertibleConv1x1(splitLen1 + splitLen2))
            self.moduleList.append(INV_block(splitLen1, splitLen2))

            self.moduleList.append(InvertibleConv1x1(splitLen1 + splitLen2))
            self.moduleList.append(INV_block(splitLen1, splitLen2))

            self.moduleList.append(InvertibleConv1x1(splitLen1 + splitLen2))
            self.moduleList.append(INV_block(splitLen1, splitLen2))

            self.moduleList.append(InvertibleConv1x1(splitLen1 + splitLen2))
            self.moduleList.append(INV_block(splitLen1, splitLen2))


    def forward(self, x, isRev):
        if not isRev:
            for m in self.moduleList:
                x = m(x, isRev)
        else:
            for m in reversed(self.moduleList):
                x = m(x, isRev)

        return x



