import torch.nn as nn
import torch.nn.functional as F
from mobilenet import Bottleneck3D #for the MobileNetV2 bottleneck blocks, which are used in the MobileViTV1 architecture. We use blocks from Mobilenet V3.


class SeparableSelfAttention(nn.Module):
    def __init__(self):
        super.__init__()
        pass



class MobileViTV1Block(nn.Module):
    def __init__(self,in_channels,out_channels,stride,expanded_channels,nonlinearity):
        super.__init__()

        self.conv = nn.Conv3d(in_channels=in_channels,out_channels=expanded_channels,kernel_size=3,stride=stride,padding=1)
        self.unfold = nn.Unfold()
        self.transformer = nn.Transformer()


        self.nonlinearity = nonlinearity

    #conv nxn -> pw_conv -> unfold -> 
    def forward(self,x):
        pass

class MobileViTV2Block(nn.Module):
    #separable self atention has linear time complexity (O(n)) instead of quadratic (O(n^2)) like MHA
    def __init__(self,in_channels,out_channels,stride,expanded_channels,b):
        super.__init__()

        self.b = b #number of times to repeat the separable self-attention and feed-forward layers

        self.dw_conv = nn.Conv3d()

        self.pw_conv1 = nn.Conv3d()

        self.unfold = nn.Unfold()

        self.separable_SA = nn.Sequential()

        self.feedforward = nn.Sequential()

        self.fold = nn.Fold()

        self.pw_conv2 = nn.Conv3d()
    #dw_conv-> pw_conv -> unfold -> separable_SA -> feedforward -> fold -> pw_conv
    def forward(self,x): #The separable self-attention and feed-forward layers are repeated b times before applying the folding operation.

        x = self.dw_conv(x)
        x = self.pw_conv1(x)
        x = self.unfold(x)
        for _ in range(self.b):
            x = self.separable_SA(x)
            x = self.feedforward(x)
        
        x = self.fold(x)
        x = self.pw_conv2(x)
        return x 

class MobileVitV1(nn.Module): #implementing MobileVit-XXS (0.4 GFLOPs, 1.3M params according to the mobilevit V2 paper), uses expansion factor of 2 for all bottlenecks.
    def __init__(self):
        super.__init__()

        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels=3,out_channels=16,kernel_size=3,stride=2,padding=1),
            nn.SiLU(),
            nn.BatchNorm3d(num_features=16),
            Bottleneck3D(in_channels=16,out_channels=16,stride=1,expanded_channels=16,nonlinearity=nn.SiLU()),
            )
        self.block2 = nn.Sequential(
            Bottleneck3D(in_channels=16,out_channels=24,stride=2,expanded_channels=32,nonlinearity=nn.SiLU()),
            Bottleneck3D(in_channels=24,out_channels=24,stride=1,expanded_channels=48,nonlinearity=nn.SiLU()),
            Bottleneck3D(in_channels=24,out_channels=24,stride=1,expanded_channels=48,nonlinearity=nn.SiLU()),
            )
        self.block3 = nn.Sequential(
            Bottleneck3D(in_channels=24,out_channels=48,stride=2,expanded_channels=48,nonlinearity=nn.SiLU()),
            MobileViTV1Block(),
            )
        self.block4 = nn.Sequential(
            Bottleneck3D(in_channels=48,out_channels=64,stride=2,expanded_channels=96,nonlinearity=nn.SiLU()),
            MobileViTV1Block(),
            )
        self.block5 = nn.Sequential(
            Bottleneck3D(in_channels=64,out_channels=80,stride=2,expanded_channels=48,nonlinearity=nn.SiLU()),
            MobileViTV1Block(),
            nn.Conv3d(in_channels=80,out_channels=320,kernel_size=1,stride=1,padding=0),
            nn.SiLU(),
            nn.BatchNorm3d(num_features=320)
            )


    def forward(self, x):

        #conv 3x3, stride 2, 16 channels
        T = x.shape[2]
        avg_pool_layer = nn.AvgPool3d(kernel_size=(T,7,7),stride=1)
        x = avg_pool_layer(x)
        x = nn.Linear(in_features=320,out_features=2)(x)
        x = F.softmax(x,dim=1)
        return x

class MobileVitV2(nn.Module): #replaces "Multi-Headed Attention in MobileViTV1 with separable self-attention" (pg 6 of paper), the highest latency part Implementing v2-0.5 (0.5 GFLOPs, 1.4M params).
    def __init__(self,alpha=0.5):
        super.__init__()

        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels=3,out_channels=32*alpha,kernel_size=3,stride=2,padding=1),
            nn.SiLU(),
            nn.BatchNorm3d(num_features=32*alpha),
            Bottleneck3D(in_channels=32*alpha,out_channels=64*alpha,stride=1,expanded_channels=32*alpha*2,nonlinearity=nn.SiLU()),
            )
        self.block2 = nn.Sequential(
            Bottleneck3D(in_channels=64*alpha,out_channels=128*alpha,stride=2,expanded_channels=64*alpha*2,nonlinearity=nn.SiLU()),
            Bottleneck3D(in_channels=128*alpha,out_channels=128*alpha,stride=1,expanded_channels=128*alpha*2,nonlinearity=nn.SiLU()),
            Bottleneck3D(in_channels=128*alpha,out_channels=128*alpha,stride=1,expanded_channels=128*alpha*2,nonlinearity=nn.SiLU()),
            )
        self.block3 = nn.Sequential(
            Bottleneck3D(in_channels=128*alpha,out_channels=256*alpha,stride=2,expanded_channels=128*alpha*2,nonlinearity=nn.SiLU()),
            MobileViTV1Block(),
            )
        self.block4 = nn.Sequential(
            Bottleneck3D(in_channels=256*alpha,out_channels=384*alpha,stride=2,expanded_channels=256*alpha*2,nonlinearity=nn.SiLU()),
            MobileViTV1Block(),
            )
        self.block5 = nn.Sequential(
            Bottleneck3D(in_channels=384*alpha,out_channels=512*alpha,stride=2,expanded_channels=48,nonlinearity=nn.SiLU()),
            MobileViTV1Block(),
            )



    def forward(self, x,alpha=0.5):
        
        T = x.shape[2]
        avg_pool_layer = nn.AvgPool3d(kernel_size=(T,7,7),stride=1)
        x = avg_pool_layer(x)
        x = nn.Linear(in_features=512*alpha,out_features=2)(x)
        x = F.softmax(x,dim=1)
        return x

class MobileFormer(nn.Module): #might not be worth implementing; technically fewer FLOPs (52 mil) than MobileViT, but more parameters (3.6 M).
    def __init__(self):
        super.__init__()
        pass


    def forward(self, x):
        pass