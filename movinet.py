import torch.nn as nn
from torch.nn import init
from mobilenet import SEBlock3D

class MoviNetBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expanded_channels, kernel_size,stride=1, use_se=True,batchnorm=True, nonlinearity=nn.Hardswish()):
        super().__init__()

        self.expand = nn.Conv3d(in_channels, expanded_channels,kernel_size=1)

        self.conv = nn.Conv3d(
            expanded_channels,
            expanded_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size[0]-1,kernel_size[1]//2,kernel_size[2]//2) if isinstance(kernel_size, tuple) else kernel_size//2,
            groups=expanded_channels
        )
        
        self.squeeze_excite = SEBlock3D(expanded_channels) if use_se else None
        self.project = nn.Conv3d(expanded_channels, out_channels,kernel_size=1)
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
        return x



#A2 takes 224x224 resolution video as specified in the paper
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
            MoviNetBottleneck(in_channels=16, out_channels=16, expanded_channels=40, kernel_size=(1,5,5)),
            #2
            MoviNetBottleneck(in_channels=16, out_channels=16, expanded_channels=40, kernel_size=3),
            #3
            MoviNetBottleneck(in_channels=16, out_channels=40, expanded_channels=96,kernel_size=3)
        )

        self.block3 = nn.Sequential(
            #1
            MoviNetBottleneck(in_channels=40,out_channels=40,expanded_channels=96,kernel_size=3),
            #2
            MoviNetBottleneck(in_channels=40,out_channels=40,expanded_channels=120,kernel_size=3),
            #3
            MoviNetBottleneck(in_channels=40,out_channels=40,expanded_channels=96,kernel_size=3),
            #4
            MoviNetBottleneck(in_channels=40,out_channels=40,expanded_channels=96,kernel_size=3),
            #5
            MoviNetBottleneck(in_channels=40,out_channels=72,expanded_channels=120,kernel_size=3)
        )
        self.block4 = nn.Sequential(
            #1
            MoviNetBottleneck(in_channels=72,out_channels=72,expanded_channels=240,kernel_size=(5,3,3)),
            #2
            MoviNetBottleneck(in_channels=72,out_channels=72,expanded_channels=155,kernel_size=3),
            #3
            MoviNetBottleneck(in_channels=72,out_channels=72,expanded_channels=240,kernel_size=3),
            #4
            MoviNetBottleneck(in_channels=72,out_channels=72,expanded_channels=192,kernel_size=3),
            #5
            MoviNetBottleneck(in_channels=72,out_channels=72,expanded_channels=240,kernel_size=3)
        )
        self.block5 = nn.Sequential(
            #1
            MoviNetBottleneck(in_channels=72,out_channels=72,expanded_channels=240,kernel_size=(5,3,3)),
            #2
            MoviNetBottleneck(in_channels=72,out_channels=72,expanded_channels=240,kernel_size=3),
            #3
            MoviNetBottleneck(in_channels=72,out_channels=72,expanded_channels=240,kernel_size=3),
            #4
            MoviNetBottleneck(in_channels=72,out_channels=72,expanded_channels=240,kernel_size=3),
            #5
            MoviNetBottleneck(in_channels=72,out_channels=72,expanded_channels=144,kernel_size=(1,5,5)),
            #6
            MoviNetBottleneck(in_channels=72,out_channels=144,expanded_channels=240,kernel_size=3)
        )
        
        self.block6 = nn.Sequential(
            #1
            MoviNetBottleneck(in_channels=144,out_channels=144,expanded_channels=480,kernel_size=(5,3,3)),
            #2
            MoviNetBottleneck(in_channels=144,out_channels=144,expanded_channels=384,kernel_size=(1,5,5)),
            #3
            MoviNetBottleneck(in_channels=144,out_channels=144,expanded_channels=384,kernel_size=(1,5,5)),
            #4
            MoviNetBottleneck(in_channels=144,out_channels=144,expanded_channels=480,kernel_size=(1,5,5)),
            #5
            MoviNetBottleneck(in_channels=144,out_channels=144,expanded_channels=480,kernel_size=(1,5,5)),
            #6
            MoviNetBottleneck(in_channels=144,out_channels=144,expanded_channels=480,kernel_size=3),
            #7
            MoviNetBottleneck(in_channels=144,out_channels=640,expanded_channels=576,kernel_size=(1,3,3))
        )

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=640,out_channels=640,kernel_size=1),
            nn.BatchNorm3d(640),
            nn.Hardswish(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(640, 2048),
            nn.BatchNorm1d(2048),
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
        x = self.conv(x)

        #dynamically get the length of the tensor, T, use for pooling
        T = x.shape[2]
        avg_pool_layer = nn.AdaptiveAvgPool3d((T,7,7))
        x = avg_pool_layer(x)

        x = self.classifier(x)
        return x

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
                if hasattr(module, "nonlinearity"):
                    if module.nonlinearity == 'relu':
                        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    elif module.nonlinearity == 'hardswish':
                        init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.BatchNorm3d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)