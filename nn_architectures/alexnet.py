import torch 
import torch.nn as nn

NUM_CLASSES = 10

conv_channel_nums = [64,192,384,256,256]
#conv_kernel_sizes = [11,5,3,3,3] # Original AlexNet Config
conv_kernel_sizes = [3,3,3,3,3]
#conv_strides =[4,1,1,1,1] # Original AlexNet Config
conv_strides =[2,1,1,1,1]
#conv_padding = [2,2,1,1,1] # Original AlexNet Config
conv_padding = [1,1,1,1,1]

maxpool_kernel_size =2
maxpool_stride = 2
fc_channel_nums = [4096,4096]


class alexnet(nn.Module):
    def __init__ (self, in_channels=3, num_classes=10):
        super(alexnet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv = self._init_conv()
        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = self._init_fc()
    def forward(self,out):
        out = self.conv(out)
        #out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    def _init_conv(self):
        layers_conv = []
        in_channels_conv = self.in_channels
        for i in range(5):
            layers_conv+=[
                nn.Conv2d(in_channels_conv, conv_channel_nums[i], kernel_size=conv_kernel_sizes[i], stride=conv_strides[i], padding=conv_padding[i]),
                nn.ReLU(inplace=True)
            ]
            if not (i==2 or i==3):
                layers_conv+=[nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)]
            in_channels_conv = conv_channel_nums[i]
        return nn.Sequential(*layers_conv)
    
    def _init_fc(self):
        layers_fc = []
        in_channels_fc = 256*2*2
        for i in range(2):
            layers_fc+=[
                nn.Dropout(),
                nn.Linear(in_channels_fc, fc_channel_nums[i]),
                nn.ReLU(inplace=True)
            ]
            in_channels_fc = fc_channel_nums[i]
        layers_fc+=[nn.Linear(fc_channel_nums[1], self.num_classes)]
        return nn.Sequential(*layers_fc)
               
        
