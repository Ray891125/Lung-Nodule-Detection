import torch.nn.functional as F
import torch.nn as nn
import torch
from net.layer import *
from net.Encoder import resnet
bn_momentum = 0.1
affine = True

class FeatureNet(nn.Module):
    def __init__(self, config):
        super(FeatureNet, self).__init__()
        self.back1 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(128, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        self.back2 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(128, momentum=bn_momentum),
            nn.ReLU(inplace=True))  

        self.back3 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        # upsampling
        self.reduce1 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm3d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

        self.reduce2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm3d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        # upsampling
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

        self.reduce3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm3d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))

        self.path3 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

        self.reduce4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm3d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x1, out1, out2, out3, out4 = x
        # x1, out1, out2, out3, out4 = self.scnet50(x)
        out4 = self.reduce1(out4)
        rev3 = self.path1(out4)
        out3 = self.reduce2(out3)
        comb3 = self.back3(torch.cat((rev3, out3), 1))
        rev2 = self.path2(comb3)
        out2 = self.reduce3(out2)
        comb2 = self.back2(torch.cat((rev2, out2), 1))
#         rev1 = self.path3(comb2)
#         out1 = self.reduce4(out1)
#         comb1 = self.back1(torch.cat((rev1, out1), 1))

        return [x1, out1, comb2], out1