"""mobilenetv2 in pytorch
[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),
            # nn.ELU(0.25, inplace=True),
            # nn.RReLU(0.1,0.25, inplace=True),
            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x
        
        return residual

class mobilenetv2(nn.Module):

    def __init__(self, class_num, num_stages, num_channels_stage, num_channels_1x1, repeat, t):
        super().__init__()
        self.num_stages = num_stages
        self.num_channels_stage = num_channels_stage
        self.num_channels_1x1 = num_channels_1x1
        self.repeat = repeat
        self.t = t
        stride =2
        self.pre = self.conv1x1(pre=True)
        self.first_stage = LinearBottleNeck(self.num_channels_1x1[0], self.num_channels_stage[0], 1, 1)
        self.init_stages = self._init_make_stages(self.repeat, stride, self.t)
        self.last_stage = LinearBottleNeck(self.num_channels_stage[self.num_stages], self.num_channels_1x1[1], 1, 8)
        self.post = self.conv1x1(pre=False)
        self.conv2 = nn.Conv2d(self.num_channels_1x1[2], class_num, 1)
            
    def forward(self, x):
        x = self.pre(x)
        x = self.first_stage(x)
        x = self.init_stages(x)
        x = self.last_stage(x)
        x = self.post(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return x
    
    def conv1x1(self, pre=True):
        if pre == True:
            in_channels_1x1=3
            out_channels_1x1 = self.num_channels_1x1[0]
        else:
            in_channels_1x1=self.num_channels_1x1[1]
            out_channels_1x1 = self.num_channels_1x1[2]
        return nn.Sequential(
            nn.Conv2d(in_channels_1x1, out_channels_1x1, 1, padding=1),
            nn.BatchNorm2d(out_channels_1x1),
            nn.ReLU6(inplace=True)
        )
    def _init_make_stages(self, repeat, stride, t):
        
        bottleneck_stages = []
        
        for i in range(self.num_stages):
            if (self.num_stages <= len(self.num_channels_stage)-1):
                in_depth = self.num_channels_stage[i]
                out_depth = self.num_channels_stage[i+1]
            else:
                in_depth = self.num_channels_stage[len(self.num_channels_stage)]
                out_depth = self.num_channels_stage[len(self.num_channels_stage)]
            
            bottleneck_stages+=[
                self._make_stage(repeat,in_depth, out_depth,stride,t)
            ]
        return nn.Sequential(*bottleneck_stages)
        
        
        
    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))
        
        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1
        
        return nn.Sequential(*layers)
