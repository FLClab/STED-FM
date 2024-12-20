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
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(64, 1)

    
    def forward(self, x):
        # Convolutional layers
        y = F.elu(self.conv1_maxpool(self.conv1_bn(self.conv1(x)))) # 1/2
        y = F.elu(self.conv2_maxpool(self.conv2_bn(self.conv2(y)))) # 1/4
        y = F.elu(self.conv3_maxpool(self.conv3_bn(self.conv3(y)))) # 1/8
        y = F.elu(self.conv4_maxpool(self.conv4_bn(self.conv4(y)))) # 1/16
        y = F.elu(self.conv5_maxpool(self.conv5_bn(self.conv5(y)))) # 1/32
        y = F.elu(self.conv6_bn(self.conv6(y)))
        y = torch.flatten(self.avg_pool(y), start_dim=1)
        y = F.sigmoid(self.fc(y))
        # We return both the mask and the averaged prediction
        return y