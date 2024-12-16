"""
Network definition
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetTrueFCN(nn.Module):
    """
    Functional implementation of the FCN network.
    """

    def __init__(self):
        super(NetTrueFCN, self).__init__()

        # 224x224
        self.conv1 = nn.Conv2d(in_channels=1, 
                                out_channels=32, 
                                kernel_size=3, 
                                stride=1, 
                                padding=1)
        self.conv1_bn = nn.BatchNorm2d(self.conv1.out_channels)
        self.conv1_maxpool = nn.MaxPool2d(2, padding=1, ceil_mode=True) # 1/2

        # 112x112
        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels, 
                                out_channels=48,
                                kernel_size=3, 
                                stride=1,
                                padding=1)
        self.conv2_bn = nn.BatchNorm2d(self.conv2.out_channels)
        self.conv2_maxpool = nn.MaxPool2d(2, ceil_mode=False) # 1/4

        # 56x56
        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels, 
                                out_channels=48,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.conv3_bn = nn.BatchNorm2d(self.conv3.out_channels)
        self.conv3_maxpool = nn.MaxPool2d(2, ceil_mode=False) # 1/8

        # 28x28
        self.conv4 = nn.Conv2d(in_channels=self.conv3.out_channels, 
                                out_channels=64,
                                kernel_size=3, 
                                stride=1,
                                padding=1)
        self.conv4_bn = nn.BatchNorm2d(self.conv4.out_channels)
        self.conv4_maxpool = nn.MaxPool2d(2, ceil_mode=False) # 1/16

        # 14x14
        self.conv5 = nn.Conv2d(in_channels=self.conv4.out_channels, 
                                out_channels=64,
                                kernel_size=3, 
                                stride=1,
                                padding=1)
        self.conv5_bn = nn.BatchNorm2d(self.conv5.out_channels)
        self.conv5_maxpool = nn.MaxPool2d(2, ceil_mode=False) # 1/32

        # 7x7
        self.conv6 = nn.Conv2d(in_channels=self.conv5.out_channels, 
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.conv6_bn = nn.BatchNorm2d(self.conv6.out_channels)

        self.map7x7 = nn.Conv2d(in_channels=self.conv6.out_channels,
                                out_channels=1,
                                kernel_size=1,
                                padding=0)
        self.map14x14 = nn.Conv2d(in_channels=self.conv4.out_channels,
                                out_channels=1,
                                kernel_size=1,
                                padding=0)
        self.map28x28 = nn.Conv2d(in_channels=self.conv3.out_channels,
                                out_channels=1,
                                kernel_size=1,
                                padding=0)
        self.map56x56 = nn.Conv2d(in_channels=self.conv2.out_channels,
                                out_channels=1,
                                kernel_size=1,
                                padding=0)

        self.upscore2 = nn.ConvTranspose2d(
            1, 1, 4, stride=2, bias=False)
        self.upscore2b = nn.ConvTranspose2d(
            1, 1, 4, stride=2, bias=False)
        self.upscore2c = nn.ConvTranspose2d(
            1, 1, 4, stride=2, bias=False)
        self.upscore4 = nn.ConvTranspose2d(
            1, 1, 4, stride=4, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            1, 1, 8, stride=8, bias=False)

    
    def forward(self, x, mask):
        # Convolutional layers
        y = F.elu(self.conv1_maxpool(self.conv1_bn(self.conv1(x)))) # 1/2
        y = F.elu(self.conv2_maxpool(self.conv2_bn(self.conv2(y)))) # 1/4
        y = F.elu(self.conv3_maxpool(self.conv3_bn(self.conv3(y)))) # 1/8
        pool28x28 = y
        y = F.elu(self.conv4_maxpool(self.conv4_bn(self.conv4(y)))) # 1/16
        pool14x14 = y
        y = F.elu(self.conv5_maxpool(self.conv5_bn(self.conv5(y)))) # 1/32
        y = F.elu(self.conv6_bn(self.conv6(y)))

        # Skip links and upsample
        out7x7 = self.map7x7(y)
        up7x7_14x14 = self.upscore2(out7x7)[:,:,1:-1,1:-1]

        out14x14 = self.map14x14(pool14x14)
        up14x14_28x28 = self.upscore2b(out14x14 + up7x7_14x14)[:,:,1:-1,1:-1]

        out28x28 = self.map28x28(pool28x28)

        up28x28_224x224 = self.upscore8(out28x28 + up14x14_28x28)
        y = up28x28_224x224

        # To help the network reach extreme scores (0 and 1), we stretch a bit
        # the sigmoid. This does not change anything in terms of loss computation.
        y = F.sigmoid(y*1.5 - 0.25)

        mm = mask.view(y.size()[0], y.size()[-2] * y.size()[-1])
        ysingle = torch.sum(y.view(y.size()[0], -1) * mm, dim=1)
        ysingle = ysingle / (mm.sum(dim=1)+1)

        # We return both the mask and the averaged prediction
        return y, ysingle