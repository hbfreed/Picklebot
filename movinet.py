import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from mobilenet import SEBlock3D

class CausalConv3d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, dilation=1, stream_buffer=None, **kwargs):
        super().__init__()
        
        #ensure kernel_size and dilation are tuples
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if not isinstance(dilation, tuple):
            dilation = (dilation, dilation, dilation)

        if stream_buffer is not None:
            self.stream_buffer =  stream_buffer #value to pad the temporal dimension with, if we have a stream buffer, use that
        else:
            self.stream_buffer = 0 #else, use 0.

        #calculate the padding needed to ensure the output is causal
        if kernel_size[0] % 2 == 0:
            p_left, p_right = (kernel_size[0]-2) // 2, kernel_size[0] // 2 #even 
        else:
            p_left, p_right = (kernel_size[0]-1) // 2, (kernel_size[0]-1) // 2 #odd
        
        self.p_left_causal, self.p_right_causal = p_left + p_right, 0 
        

        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size,
                                stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        pad = (0,0,0,0,self.p_left_causal,0) 
        x = F.pad(x,pad, "constant", self.stream_buffer) #pad with the temporal dimension on the left. Not a fan of this API! I'm sure there's a good reason. x = (B, C, T, H, W)
        x = self.conv3d(x) #apply the convolution
        
        return x



class MoviNetBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expanded_channels, kernel_size,stride=1, use_se=True,batchnorm=True, nonlinearity=nn.Hardswish(),bias=False,dropout=0,padding=None,dilation=1):
        super().__init__()

        self.expand = nn.Conv3d(in_channels, expanded_channels,kernel_size=1,bias=bias)

        default_padding = (kernel_size[0]-1, kernel_size[1]//2, kernel_size[2]//2) if isinstance(kernel_size, tuple) else kernel_size//2
        padding = default_padding if padding is None else padding

        self.conv = nn.Conv3d(
            expanded_channels,
            expanded_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=expanded_channels,
            bias=bias,
            dilation=dilation
        )
        
        self.squeeze_excite = SEBlock3D(expanded_channels) if use_se else None
        self.project = nn.Conv3d(expanded_channels, out_channels,kernel_size=1,bias=bias)
        self.batchnorm = nn.BatchNorm3d(out_channels) if batchnorm else None
        self.nonlinearity = nonlinearity
        self.dropout = nn.Dropout3d(p=dropout)

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
    def __init__(self, num_classes=2, buffer_size=2): 
        super().__init__()

        #define our number of classes
        self.num_classes = num_classes

        #initialize the stream buffers
        self.buffer_size = buffer_size

        #define our first block, a 3D convolutional layer with 16 filters, kernel size of 1x3x3, stride of 1x2x2, and padding of 0x1x1  
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1),bias=False), #should be able to leave out stream buffer here
            nn.BatchNorm3d(16),
            nn.Hardswish()
        )

        #Block2
        self.block2 = nn.Sequential(
            MoviNetBottleneck(16, 16, 40, kernel_size=(1,5,5), stride=(1,2,2), padding=(0,2,2)),
            MoviNetBottleneck(16, 16, 40, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            MoviNetBottleneck(16, 16, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        )
        #block 3
        self.block3 = nn.Sequential(
            MoviNetBottleneck(16, 40, 96, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1)),
            MoviNetBottleneck(40, 40, 120, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            MoviNetBottleneck(40, 40, 96, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            MoviNetBottleneck(40, 40, 96, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            MoviNetBottleneck(40, 40, 120, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        )
        #block 4
        self.block4 = nn.Sequential(
            MoviNetBottleneck(40, 72, 240, kernel_size=(5,3,3), stride=(1,2,2), padding=(2,1,1)),
            MoviNetBottleneck(72, 72, 160, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            MoviNetBottleneck(72, 72, 240, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            MoviNetBottleneck(72, 72, 192, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            MoviNetBottleneck(72, 72, 240, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        )
        #block 5
        self.block5 = nn.Sequential(
            MoviNetBottleneck(72, 72, 240, kernel_size=(5,3,3), stride=(1,1,1), padding=(2,1,1)),
            MoviNetBottleneck(72, 72, 240, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            MoviNetBottleneck(72, 72, 240, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            MoviNetBottleneck(72, 72, 240, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            MoviNetBottleneck(72, 72, 144, kernel_size=(1,5,5), stride=(1,1,1), padding=(0,2,2)),
            MoviNetBottleneck(72, 72, 240, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        )
        #block 6
        self.block6 = nn.Sequential(
            MoviNetBottleneck(72 , 144, 480, kernel_size=(5,3,3), stride=(1,2,2), padding=(2,1,1)),
            MoviNetBottleneck(144, 144, 384, kernel_size=(1,5,5), stride=(1,1,1), padding=(0,2,2)),
            MoviNetBottleneck(144, 144, 384, kernel_size=(1,5,5), stride=(1,1,1), padding=(0,2,2)),
            MoviNetBottleneck(144, 144, 480, kernel_size=(1,5,5), stride=(1,1,1), padding=(0,2,2)),
            MoviNetBottleneck(144, 144, 480, kernel_size=(1,5,5), stride=(1,1,1), padding=(0,2,2)),
            MoviNetBottleneck(144, 144, 480, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            MoviNetBottleneck(144, 144, 576, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1))
        )

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=144,out_channels=640,kernel_size=1,bias=False),
            nn.BatchNorm3d(640),
            nn.Hardswish(),
            nn.Dropout3d(0.2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            nn.Linear(640, 2048),
            nn.BatchNorm1d(2048),
            nn.Hardswish(),
            nn.Dropout(0.2),
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
        x = self.classifier(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)