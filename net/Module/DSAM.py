import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch


class DsamModule(nn.Module):
    def __init__(self, in_c):
        """
        inplanes: input channels
        planes: output channels
        dilation: ??
        groups: ??
        pooling_r: downsmapling rate before k2
        norm_layer:??
        """
        mid_c = in_c//2
        super(DsamModule, self).__init__()
        self.relu = nn.ReLU(True)  
        self.maxpool = nn.MaxPool3d(kernel_size=3,stride=1,padding=1)
        self.avgpool = nn.AvgPool3d(kernel_size=3,stride=1,padding=1)
        self.conv_1a  = nn.Sequential(
            nn.Conv3d(in_c, mid_c, kernel_size=1, padding='same'),
            nn.BatchNorm3d(mid_c)
        )
        self.conv_1b  = nn.Sequential(
            nn.Conv3d(in_c, mid_c, kernel_size=1, padding='same'),
            nn.BatchNorm3d(mid_c)
        )
        self.conv_2a  = nn.Sequential(
            nn.Conv3d(mid_c, mid_c, kernel_size=3, padding='same'),
            nn.BatchNorm3d(mid_c)
        )
        self.conv_2b  = nn.Sequential(
            nn.Conv3d(mid_c, mid_c, kernel_size=5, padding='same'),
            nn.BatchNorm3d(mid_c)
        )
        self.conv_3  = nn.Sequential(
            nn.Conv3d(mid_c, in_c, kernel_size=1, padding='same'),
            nn.BatchNorm3d(in_c)
        )
        
    def forward(self, x):
        residual = x
        
        x1 = self.relu(self.conv_1a(x))
        x2 = self.relu(self.conv_1b(x))
        x1 = self.relu(self.conv_2a(x1))
        x2 = self.relu(self.conv_2b(x1))
        x3 = self.relu(self.conv_3(torch.add(x1,x2)))
        
        y1 = self.maxpool(x3)
        y2 = self.avgpool(x3)
        weight = torch.sigmoid(torch.add(y1,y2))
        out = torch.mul(x, weight)
        out = torch.add(out,residual)
        return out