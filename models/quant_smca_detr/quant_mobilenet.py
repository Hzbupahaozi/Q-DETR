import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math
import numpy as np
from ._quan_base_plus import *
from .lsq_plus import *

stage_out_channel = [32] + [16] + [24] * 2 + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3 + [320]
n_bit = 4

overall_channel = stage_out_channel

mid_channel = []
for i in range(len(stage_out_channel)-1):
    if i == 0:
        mid_channel += [stage_out_channel[i]]
    else:
        mid_channel += [6 * stage_out_channel[i]]

class conv2d_3x3(nn.Module):
    def __init__(self, inp, oup, stride, norm_layer):
        super(conv2d_3x3, self).__init__()

        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn1 = norm_layer(oup)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu6(out)

        return out


class quan_conv2d_1x1(nn.Module):
    def __init__(self, inp, oup, stride, norm_layer):
        super(quan_conv2d_1x1, self).__init__()

        self.conv1 = Conv2dLSQ(inp, oup, kernel_size=1, stride=stride, padding=0, nbits_w=n_bit, mode=Qmodes.kernel_wise)
        self.bn1 = norm_layer(oup)

    def forward(self, x):

        #out = self.quan1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu6(out)

        return out

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class bottleneck(nn.Module):
    def __init__(self, inp, oup, mid, stride, norm_layer):
        super(bottleneck, self).__init__()

        self.stride = stride
        self.inp = inp
        self.oup = oup

        self.bias11 = LearnableBias(inp)
        self.prelu1 = nn.PReLU(inp)
        self.bias12 = LearnableBias(inp)

        self.conv1 = Conv2dLSQ(inp, mid, kernel_size=1, stride=1, padding=0, nbits_w=n_bit, mode=Qmodes.kernel_wise)
        self.bn1 = norm_layer(mid)

        self.bias21 = LearnableBias(mid)
        self.prelu2 = nn.PReLU(mid)
        self.bias22 = LearnableBias(mid)
        #self.quan2 = LTQ(n_bit)
        #self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, stride=stride, padding=1, groups=mid)
        self.conv2 = Conv2dLSQ(mid, mid, kernel_size=3, padding=1, bias=False, stride=stride, groups=mid, nbits_w=n_bit, mode=Qmodes.kernel_wise)
        #self.conv2 = HardQuantizeConv(mid, mid, n_bit, 3, stride, 1, groups=mid)
        self.bn2 = norm_layer(mid)

        self.bias31 = LearnableBias(mid)
        self.prelu3 = nn.PReLU(mid)
        self.bias32 = LearnableBias(mid)
        #self.quan3 = LTQ(n_bit)
        #self.conv3 = nn.Conv2d(mid, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = Conv2dLSQ(mid, oup, kernel_size=1, stride=1, padding=0, nbits_w=n_bit, mode=Qmodes.kernel_wise)
        #self.conv3 = HardQuantizeConv(mid, oup, n_bit, 1, 1, 0)
        self.bn3 = norm_layer(oup)

    def forward(self, x):

        out = self.bias11(x)
        out = self.prelu1(out)
        out = self.bias12(out)
        #out = self.quan1(out)
        out = self.conv1(out)
        out = self.bn1(out)

        out = self.bias21(out)
        out = self.prelu2(out)
        out = self.bias22(out)
        #out = self.quan2(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.bias31(out)
        out = self.prelu3(out)
        out = self.bias32(out)
        #out = self.quan3(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.inp == self.oup and self.stride == 1:
            return (out + x)

        else:
            return out

class MobileNetV2(nn.Module):
    def __init__(self, input_size=224, num_classes=1000, n_bit=4, norm_layer=nn.BatchNorm2d):
        super(MobileNetV2, self).__init__()

        self._norm_layer = norm_layer
        self.feature = []

        for i in range(19):
            if i == 0:
                self.feature.append(conv2d_3x3(3, overall_channel[i], 2, self._norm_layer))
            elif i == 1:
                self.feature.append(bottleneck(overall_channel[i-1], overall_channel[i], mid_channel[i-1],1, self._norm_layer))
            elif i == 18:
                self.feature.append(quan_conv2d_1x1(overall_channel[i-1], 1280, 1,self._norm_layer))
            else:
                if stage_out_channel[i-1]!=stage_out_channel[i] and stage_out_channel[i]!=96 and stage_out_channel[i]!=320:
                    self.feature.append(bottleneck(overall_channel[i-1], overall_channel[i], mid_channel[i-1], 2,self._norm_layer))
                else:
                    self.feature.append(bottleneck(overall_channel[i-1], overall_channel[i], mid_channel[i-1], 1,self._norm_layer))

        self.feature = nn.Sequential(*self.feature)
        self.pool1 = nn.AvgPool2d(7)
        self.fc = nn.Linear(1280, 1000)

    def forward(self, x):

        x = self.feature(x)

        # for i, block in enumerate(self.feature):
        #     if i == 0 :
        #         x = block(x)
        #     elif i == 18 :
        #         x = block(x)
        #     else :
        #         x = block(x)

        x = self.pool1(x)
        x = x.view(-1, 1280)
        x = self.fc(x)

        return x


def mobilenet_v2(n_bit, pretrained, norm_layer, **kwargs):
    #quantize_downsample = True
    model = MobileNetV2(input_size=224, num_classes=1000,n_bit=n_bit,norm_layer=norm_layer)
    state_dict = torch.load('./pretrained/mb_v2_w4a4.pth', map_location='cpu')
    model.load_state_dict(state_dict['model'], strict=True)
    return model
