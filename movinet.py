import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from mobilenet import SEBlock3D

class MoviNetBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expanded_channels, kernel_size, stride=1, use_se=True,batchnorm=True, nonlinearity=nn.Hardswish):
        super().__init__()

        self.expand = nn.Conv3d(in_channels, expanded_channels,kernel_size=1)

        self.conv = nn.Conv3d(
            expanded_channels,
            expanded_channels,
            kernel_size=kernel_size
            stride=stride
            padding=kernel_size//2
            groups=expanded_channels
        )

        self.project = nn.Conv3d(expanded_channels, out_channels,kernel_size=1)

        self.squeeze_excite = SEBlock3D(expanded_channels) if use_se else None

        self.batchnorm = nn.BatchNorm3d(out_channels) if batchnorm else None

        self.nonlinearity = nonlinearity

    def forward(self, x):
        x = self.expand(x)
        x = self.conv(x)
        if self.squeeze_excite is not None:
            x = self.squeeze_excite(x)
        x = self.project(x)
        x = self.batchnorm(x)
        x = self.nonlinearity(x)
        




#A2 takes 224x224 resolution video as specified in the paper, it also seems not too computationally expensive
class MoViNetA2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        #define our number of classes
        self.num_classes = num_classes

        #define our first block, a 3D convolutional layer with 16 filters, kernel size of 1x3x3, stride of 1x2x2, and padding of 0x1x1  
        self.block1 = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        #define our second block, a 3D convolutional layer with 16 filters, kernel size of 3x3x3, stride of 1x2x2, and padding of 1x1x1
        self.block2 = nn.Sequential(
            #1
            nn.Conv3d(in_channels=16, out_channels=40, kernel_size=(1,5,5), stride=(1,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(in_channels=16),
            nn.Hardswish(),
            #2
            nn.Conv3d(in_channels=16),
            nn.BatchNorm3d(),
            nn.Hardswish(),
            #3
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
        )

        self.block3 = nn.Sequential(
            #1
            nn.Conv3d(in_channels=40,out_channels=96),
            nn.BatchNorm3d(),
            nn.Hardswish(),
            #2
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
            #3
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
            #4
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
            #5
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
        )
        self.block4 = nn.Sequential(
            #1
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
            #2
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
            #3
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
            #4
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
            #5
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
        )
        self.block5 = nn.Sequential(
            #1
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
            #2
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
            #3
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
            #4
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
            #5
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
            #6
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
        )
        
        self.block6 = nn.Sequential(
            #1
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
            #2
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
            #3
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
            #4
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
            #5
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
            #6
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
            #7
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
        )

        self.convpool = nn.Sequential(
            nn.Conv3d(),
            nn.BatchNorm3d(),
            nn.Hardswish(),
            nn.AdaptiveAvgPool3d()
        )

        self.classifier = nn.Sequential(
            nn.Linear(640, 2048),
            nn.BatchNorm1d(),
            nn.Hardswish(),
            nn.Linear(2048, self.num_classes)
        )
            

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.convpool(x)
        x = self.classifier(x)