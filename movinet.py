import math
import torch
import torch.nn as nn
from torch.nn import init
from mobilenet import SEBlock3D

class MoviNetBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expanded_channels, kernel_size,stride=1, use_se=True,batchnorm=True, nonlinearity=nn.Hardswish(),bias=False,dropout=0,padding_size=None):
        super().__init__()

        self.expand = nn.Conv3d(in_channels, expanded_channels,kernel_size=1,bias=bias)

        default_padding = (kernel_size[0]-1, kernel_size[1]//2, kernel_size[2]//2) if isinstance(kernel_size, tuple) else kernel_size//2
        padding = default_padding if padding_size is None else padding_size

        self.conv = nn.Conv3d(
            expanded_channels,
            expanded_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=expanded_channels,
            bias=bias
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
    def __init__(self, num_classes=2, subclip_length=50,buffer_size=2):
        super().__init__()

        #define our number of classes
        self.num_classes = num_classes

        #initialize the subclip length
        self.subclip_length=subclip_length

        #initialize the stream buffers
        self.buffer_size = buffer_size

        #define our first block, a 3D convolutional layer with 16 filters, kernel size of 1x3x3, stride of 1x2x2, and padding of 0x1x1  
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1),bias=False),
            nn.BatchNorm3d(16),
            nn.Hardswish()
        )

        #define our second block, a 3D convolutional layer with 16 filters, kernel size of 3x3x3, stride of 1x2x2, and padding of 1x1x1
        self.block2 = nn.Sequential(
            #1
            MoviNetBottleneck(in_channels=16, out_channels=16, expanded_channels=40, kernel_size=(1,5,5),stride=(1,2,2)),
            #2
            MoviNetBottleneck(in_channels=16, out_channels=16, expanded_channels=40, kernel_size=3),
            #3
            MoviNetBottleneck(in_channels=16, out_channels=40, expanded_channels=64,kernel_size=3,dropout=0.2)
        )

        self.block3 = nn.Sequential(
            #1
            MoviNetBottleneck(in_channels=40,out_channels=40,expanded_channels=96,kernel_size=3,stride=(1,2,2)),
            #2
            MoviNetBottleneck(in_channels=40,out_channels=40,expanded_channels=120,kernel_size=3,dropout=0.2),
            #3
            MoviNetBottleneck(in_channels=40,out_channels=40,expanded_channels=96,kernel_size=3),
            #4
            MoviNetBottleneck(in_channels=40,out_channels=40,expanded_channels=96,kernel_size=3,dropout=0.2),
            #5
            MoviNetBottleneck(in_channels=40,out_channels=72,expanded_channels=120,kernel_size=3)
        )
        self.block4 = nn.Sequential(
            #1
            MoviNetBottleneck(in_channels=72,out_channels=72,expanded_channels=240,kernel_size=(5,3,3),stride=(1,2,2),padding_size=(2,1,1),dropout=0.2),
            #2
            MoviNetBottleneck(in_channels=72,out_channels=72,expanded_channels=155,kernel_size=3),
            #3
            MoviNetBottleneck(in_channels=72,out_channels=72,expanded_channels=240,kernel_size=3,dropout=0.2),
            #4
            MoviNetBottleneck(in_channels=72,out_channels=72,expanded_channels=192,kernel_size=3),
            #5
            MoviNetBottleneck(in_channels=72,out_channels=72,expanded_channels=240,kernel_size=3,dropout=0.2)
        )
        self.block5 = nn.Sequential(
            #1
            MoviNetBottleneck(in_channels=72,out_channels=72,expanded_channels=240,kernel_size=(5,3,3),padding_size=(2,1,1),dropout=0.2),
            #2
            MoviNetBottleneck(in_channels=72,out_channels=72,expanded_channels=240,kernel_size=3),
            #3
            MoviNetBottleneck(in_channels=72,out_channels=72,expanded_channels=240,kernel_size=3,dropout=0.2),
            #4
            MoviNetBottleneck(in_channels=72,out_channels=72,expanded_channels=240,kernel_size=3),
            #5
            MoviNetBottleneck(in_channels=72,out_channels=72,expanded_channels=144,kernel_size=(1,5,5)),
            #6
            MoviNetBottleneck(in_channels=72,out_channels=144,expanded_channels=240,kernel_size=3,dropout=0.2)
        )
        
        self.block6 = nn.Sequential(
            #1
            MoviNetBottleneck(in_channels=144,out_channels=144,expanded_channels=480,kernel_size=(5,3,3),stride=(1,2,2),padding_size=(2,1,1),dropout=0.2),
            #2
            MoviNetBottleneck(in_channels=144,out_channels=144,expanded_channels=384,kernel_size=(1,5,5)),
            #3
            MoviNetBottleneck(in_channels=144,out_channels=144,expanded_channels=384,kernel_size=(1,5,5),dropout=0.2),
            #4
            MoviNetBottleneck(in_channels=144,out_channels=144,expanded_channels=480,kernel_size=(1,5,5)),
            #5
            MoviNetBottleneck(in_channels=144,out_channels=144,expanded_channels=480,kernel_size=(1,5,5),dropout=0.2),
            #6
            MoviNetBottleneck(in_channels=144,out_channels=144,expanded_channels=480,kernel_size=3),
            #7
            MoviNetBottleneck(in_channels=144,out_channels=640,expanded_channels=576,kernel_size=(1,3,3),dropout=0.2)
        )

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=640,out_channels=640,kernel_size=1,bias=False),
            nn.BatchNorm3d(640),
            nn.Hardswish(),
            nn.Dropout3d(0.2),
            nn.AdaptiveAvgPool3d((1,1,1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(640, 2048),
            nn.BatchNorm1d(2048),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(2048, self.num_classes)
        )
            

#I think, in this forward pass, we have implemented the stream buffer from the paper.
    def forward(self, x):
        batch_size, C, T, H, W = x.shape
        num_clips = math.ceil(T/self.subclip_length)
        
        # Initialize buffers for each block
        buffer1 = torch.zeros(batch_size, 3, self.buffer_size, H, W, device=x.device).requires_grad_(False)
        buffer2 = torch.zeros(batch_size, 16, self.buffer_size, H//2, W//2, device=x.device).requires_grad_(False)
        buffer3 = torch.zeros(batch_size, 40, self.buffer_size, H//4, W//4, device=x.device).requires_grad_(False)
        buffer4 = torch.zeros(batch_size, 72, self.buffer_size, H//8, H//8, device=x.device).requires_grad_(False)
        buffer5 = torch.zeros(batch_size, 72, self.buffer_size, H//8, H//8, device=x.device).requires_grad_(False)
        buffer6 = torch.zeros(batch_size, 144, self.buffer_size, H//16, H//16, device=x.device).requires_grad_(False)
        
        pooled_outputs = []

        for i in range(num_clips):
            start_idx = i * self.subclip_length
            end_idx = start_idx + self.subclip_length
            x_clip = x[:, :, start_idx:end_idx, :, :]

            # Block 1
            out = self.block1(x_clip)
            buffer1 = out[:, :, -self.buffer_size:, :, :].detach()
            # Block 2
            x_with_buffer = torch.cat((buffer1, out), dim=2)
            out = self.block2(x_with_buffer)
            buffer2 = out[:, :, -self.buffer_size:, :, :].detach()

            x_with_buffer = torch.cat((buffer2, out), dim=2)
            out = self.block3(x_with_buffer)
            buffer3 = out[:, :, -self.buffer_size:, :, :].detach()

            x_with_buffer = torch.cat((buffer3, out), dim=2)
            out = self.block4(x_with_buffer)
            buffer4 = out[:, :, -self.buffer_size:, :, :].detach()
            
            x_with_buffer = torch.cat((buffer4, out), dim=2)
            out = self.block5(x_with_buffer)
            buffer5 = out[:, :, -self.buffer_size:, :, :].detach()
            
            x_with_buffer = torch.cat((buffer5, out), dim=2)
            out = self.block6(x_with_buffer)
            buffer6 = out[:, :, -self.buffer_size:, :, :].detach()

            
            x_with_buffer = torch.cat((buffer6, out), dim=2)
            out = self.conv(x_with_buffer)
            
            pooled_outputs.append(out)

        out = torch.cat(pooled_outputs, dim=2)
        out = nn.AdaptiveAvgPool3d((1,1,1))(out)
        out = self.classifier(out)
        return out
    
    # def forward(self,x):
    #     x = self.block1(x)
    #     x = self.block2(x)
    #     x = self.block3(x)
    #     x = self.block4(x)
    #     x = self.block5(x)
    #     x = self.block6(x)
    #     x = self.conv(x)
    #     x = self.classifier(x)
    #     return x

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