'''
Implementing Mobilenet v3 as seen in 
"Searching for MobileNetV3" for video classification,
note that balls are 0 and strikes are 1.
'''
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class SEBlock3D(nn.Module):
    def __init__(self,channels):
        super().__init__()
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels,channels//4,kernel_size=1), 
            nn.ReLU(inplace=True),
            nn.Conv3d(channels//4,channels,kernel_size=1),
            nn.Hardsigmoid()
            )

    def forward(self,x):
        w = self.se(x)
        x = x * w
        return x


class SEBlock2D(nn.Module):
    def __init__(self,channels):
        super().__init__()
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels,channels//4,kernel_size=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4,channels,kernel_size=1),
            nn.Hardsigmoid()
            )
    
    def forward(self,x):
        w = self.se(x)
        x = x * w
        return x

#Bottleneck for Mobilenets
class Bottleneck3D(nn.Module):
    def __init__(self, in_channels, out_channels, expanded_channels, stride=1, use_se=False, kernel_size=3,nonlinearity=nn.Hardswish(),batchnorm=True,dropout=0,bias=False):
        super().__init__()

        #pointwise conv1x1x1 (reduce channels)
        self.pointwise_conv1 = nn.Conv3d(in_channels,expanded_channels,kernel_size=1,bias=bias)
        #depthwise (spatial filtering)
        #groups to preserve channel-wise information
        self.depthwise_conv = nn.Conv3d(
            expanded_channels,#in channels
            expanded_channels,#out channels
            groups=expanded_channels,
            kernel_size=(1,kernel_size,kernel_size),
            stride=stride,
            padding=kernel_size//2,
            bias=bias
            )
        #squeeze-and-excite (recalibrate channel wise)
        self.squeeze_excite = SEBlock3D(expanded_channels) if use_se else None 
        #pointwise conv1x1x1 (expansion to increase channels)
        self.pointwise_conv2 = nn.Conv3d(expanded_channels,out_channels,kernel_size=1,bias=bias)
        self.batchnorm = nn.BatchNorm3d(out_channels) if batchnorm else None
        self.nonlinearity = nonlinearity
        self.dropout = nn.Dropout3d(p=dropout)

    def forward(self,x):
        x = self.pointwise_conv1(x)
        x = self.depthwise_conv(x)
        if self.squeeze_excite is not None:
            x = self.squeeze_excite(x)
        x = self.pointwise_conv2(x)
        x = self.batchnorm(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        return x


#2D bottleneck for our 2d convnet with LSTM
class Bottleneck2D(nn.Module):
    def __init__(self, in_channels, out_channels, expanded_channels, stride=1, use_se=False, kernel_size=3,nonlinearity=nn.Hardswish(),batchnorm=True,dropout=0,bias=False):
        super().__init__()

        #pointwise conv1x1x1 (reduce channels)
        self.pointwise_conv1 = nn.Conv2d(in_channels,expanded_channels,kernel_size=1,dropout=dropout,bias=bias)
        #depthwise (spatial filtering)
        #groups to preserve channel-wise information
        self.depthwise_conv = nn.Conv2d(
            expanded_channels,#in channels
            expanded_channels,#out channels
            groups=expanded_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2,
            dropout=dropout,
            bias=bias
            )
        #squeeze-and-excite (recalibrate channel wise)
        self.squeeze_excite = SEBlock2D(expanded_channels) if use_se else None 
        #pointwise conv1x1x1 (expansion to increase channels)
        self.pointwise_conv2 = nn.Conv2d(expanded_channels,out_channels,kernel_size=1,bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels) if batchnorm else None
        self.nonlinearity = nonlinearity

    def forward(self,x):
        x = self.pointwise_conv1(x)
        x = self.depthwise_conv(x)
        if self.squeeze_excite is not None:
            x = self.squeeze_excite(x)
        x = self.pointwise_conv2(x)
        x = self.batchnorm(x)
        x = self.nonlinearity(x) 
        return x        


#MobileNetV3-Large 2D + LSTM for helping with the temporal dimension
class MobileNetLarge2D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.num_classes = num_classes

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if hasattr(module, "nonlinearity"):
                    if module.nonlinearity == 'relu':
                        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    elif module.nonlinearity == 'hardswish':
                        init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)

    #conv2d (h-swish): 224x224x3 -> 112x112x16
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,stride=2,kernel_size=3,padding=1),
            nn.BatchNorm2d(16),
            nn.Hardswish()
            )
    #3x3 bottlenecks1 (3, ReLU): 112x112x16 -> 56x56x24
        self.block2 = nn.Sequential(
            Bottleneck2D(in_channels=16,out_channels=16,expanded_channels=16,stride=1,nonlinearity=nn.ReLU()),
            Bottleneck2D(in_channels=16,out_channels=24,expanded_channels=64,stride=2,nonlinearity=nn.ReLU()),
            Bottleneck2D(in_channels=24,out_channels=24,expanded_channels=72,stride=1,nonlinearity=nn.ReLU())
            )
    #5x5 bottlenecks1 (3, ReLU, squeeze-excite): 56x56x24 -> 28x28x40
        self.block3 = nn.Sequential(
            Bottleneck2D(in_channels=24,out_channels=40,expanded_channels=72,stride=2,use_se=True,kernel_size=5,nonlinearity=nn.ReLU()),
            Bottleneck2D(in_channels=40,out_channels=40,expanded_channels=120,stride=1,use_se=True,kernel_size=5,nonlinearity=nn.ReLU()),
            Bottleneck2D(in_channels=40,out_channels=40,expanded_channels=120,stride=1,use_se=True,kernel_size=5,nonlinearity=nn.ReLU())
            )
    #3x3 bottlenecks2 (6, h-swish, last two get squeeze-excite): 28x28x40 -> 14x14x112
        self.block4 = nn.Sequential(
            Bottleneck2D(in_channels=40,out_channels=80,expanded_channels=240,stride=2),
            Bottleneck2D(in_channels=80,out_channels=80,expanded_channels=240,stride=1),
            Bottleneck2D(in_channels=80,out_channels=80,expanded_channels=184,stride=1),
            Bottleneck2D(in_channels=80,out_channels=80,expanded_channels=184,stride=1),
            Bottleneck2D(in_channels=80,out_channels=112,expanded_channels=480,stride=1,use_se=True),
            Bottleneck2D(in_channels=112,out_channels=112,expanded_channels=672,stride=1,use_se=True)
            )
    #5x5 bottlenecks2 (3, h-swish, squeeze-excite): 14x14x112 -> 7x7x160
        self.block5 = nn.Sequential(
            Bottleneck2D(in_channels=112,out_channels=160,expanded_channels=672,stride=2,use_se=True,kernel_size=5),
            Bottleneck2D(in_channels=160,out_channels=160,expanded_channels=960,stride=1,use_se=True,kernel_size=5),
            Bottleneck2D(in_channels=160,out_channels=160,expanded_channels=960,stride=1,use_se=True,kernel_size=5)
            )
    #conv3d (h-swish), avg pool 7x7: 7x7x960 -> 1x1x960
        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels=160,out_channels=960,stride=1,kernel_size=1),
            nn.BatchNorm2d(960),
            nn.Hardswish(),
            nn.AvgPool2d(kernel_size=7,stride=1)
            )
    #LSTM: 1x1x960 ->
        self.lstm = nn.LSTM(input_size=960,hidden_size=32,num_layers=5,batch_first=True) 
    #classifier: conv3d 1x1 NBN (2, first uses h-swish): 1x1x960 
        self.classifier = nn.Sequential(
            nn.Linear(32,self.num_classes) #2 classes for ball/strike
            )

    def forward(self,x):
        #reshape from N,C,T,H,W to N*T,C,H,W for 2d convolutions, we will keep N as 1
        x = x.reshape(x.shape[0]*x.shape[2],x.shape[1],x.shape[3],x.shape[4])
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        #reshape for LSTM
        x = x.permute(0,2,1,3).contiguous()
        x = x.view(x.shape[0],x.shape[1],x.shape[2])
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        x = self.classifier(x)
        x = F.softmax(x,dim=1).to(torch.float16)
        return x



#MobileNetV3-Small 2d with lstm for helping with the temporal dimension
class MobileNetSmall2D(nn.Module):
    def __init__(self,num_classes=2):
        super().__init__()

        self.num_classes = num_classes


    #conv3d (h-swish): 224x224x3 -> 112x112x16
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.Hardswish()
            )
    #3x3 bottlenecks (3, ReLU, first gets squeeze-excite): 112x112x16 -> 28x28x24
        self.block2 = nn.Sequential(
            Bottleneck2D(in_channels=16,out_channels=16,expanded_channels=16,stride=2,use_se=True,nonlinearity=nn.ReLU()),
            Bottleneck2D(in_channels=16,out_channels=24,expanded_channels=72,stride=2,nonlinearity=nn.ReLU()),
            Bottleneck2D(in_channels=24,out_channels=24,expanded_channels=88,stride=1,nonlinearity=nn.ReLU())
            )
    #5x5 bottlenecks (8, h-swish, squeeze-excite): 28x28x24 -> 7x7x96
        self.block3 = nn.Sequential(
            Bottleneck2D(in_channels=24,out_channels=40,expanded_channels=96,stride=2,use_se=True,kernel_size=5),
            Bottleneck2D(in_channels=40,out_channels=40,expanded_channels=240,stride=1,use_se=True,kernel_size=5),
            Bottleneck2D(in_channels=40,out_channels=40,expanded_channels=240,stride=1,use_se=True,kernel_size=5),
            Bottleneck2D(in_channels=40,out_channels=48,expanded_channels=120,stride=1,use_se=True,kernel_size=5),
            Bottleneck2D(in_channels=48,out_channels=48,expanded_channels=144,stride=1,use_se=True,kernel_size=5),
            Bottleneck2D(in_channels=48,out_channels=96,expanded_channels=288,stride=2,use_se=True,kernel_size=5),
            Bottleneck2D(in_channels=96,out_channels=96,expanded_channels=576,stride=1,use_se=True,kernel_size=5),
            Bottleneck2D(in_channels=96,out_channels=96,expanded_channels=576,stride=1,use_se=True,kernel_size=5)
            )
    #conv2d (h-swish), avg pool 7x7: 7x7x96 -> 1x1x576
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=96,out_channels=576,kernel_size=1,stride=1,padding=0),
            SEBlock2D(channels=576),
            nn.BatchNorm2d(576),
            nn.Hardswish(),
            nn.AvgPool2d(kernel_size=7,stride=1)
            )
    #LSTM: 1x1x576 ->
        self.lstm = nn.LSTM(input_size=576,hidden_size=64,num_layers=1,batch_first=True)
    #classifier: conv3d 1x1 NBN (2, first uses h-swish): 1x1x576
        self.classifier = nn.Sequential(
            nn.Linear(64,self.num_classes) #2 classes for ball/strike
            )

    def forward(self,x):
        #reshape from N,C,T,H,W to N*T,C,H,W for 2d convolutions, we will keep N as 1
        x = x.reshape(x.shape[0]*x.shape[2],x.shape[1],x.shape[3],x.shape[4])
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        #reshape for LSTM
        x = x.permute(0,2,1,3).contiguous()
        x = x.view(x.shape[0],x.shape[1],x.shape[2])
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        x = self.classifier(x)
        x = F.softmax(x,dim=1).to(torch.float16)
        return x
    
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if hasattr(module, "nonlinearity"):
                    if module.nonlinearity == 'relu':
                        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    elif module.nonlinearity == 'hardswish':
                        init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)



#mobilenet large 3d convolutions
class MobileNetLarge3D(nn.Module):
    def __init__(self,num_classes=2):
        super().__init__()

        self.num_classes = num_classes

    #conv3d (h-swish): 224x224x3 -> 112x112x16
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels=3,out_channels=16,stride=2,kernel_size=3,padding=1),
            nn.BatchNorm3d(16),
            nn.Hardswish()
            )
        
    #3x3 bottlenecks1 (3, ReLU): 112x112x16 -> 56x56x24
        self.block2 = nn.Sequential(
            Bottleneck3D(in_channels=16,out_channels=16,expanded_channels=16,stride=1,nonlinearity=nn.ReLU(),dropout=0.2),
            Bottleneck3D(in_channels=16,out_channels=24,expanded_channels=64,stride=2,nonlinearity=nn.ReLU(),dropout=0.2),
            Bottleneck3D(in_channels=24,out_channels=24,expanded_channels=72,stride=1,nonlinearity=nn.ReLU(),dropout=0.2)
            )
        
    #5x5 bottlenecks1 (3, ReLU, squeeze-excite): 56x56x24 -> 28x28x40
        self.block3 = nn.Sequential(
            Bottleneck3D(in_channels=24,out_channels=40,expanded_channels=72,stride=2,use_se=True,kernel_size=5,nonlinearity=nn.ReLU(),dropout=0.2),
            Bottleneck3D(in_channels=40,out_channels=40,expanded_channels=120,stride=1,use_se=True,kernel_size=5,nonlinearity=nn.ReLU(),dropout=0.2),
            Bottleneck3D(in_channels=40,out_channels=40,expanded_channels=120,stride=1,use_se=True,kernel_size=5,nonlinearity=nn.ReLU(),dropout=0.2)
            )
        
    #3x3 bottlenecks2 (6, h-swish, last two get squeeze-excite): 28x28x40 -> 14x14x112
        self.block4 = nn.Sequential(
            Bottleneck3D(in_channels=40,out_channels=80,expanded_channels=240,stride=2,dropout=0.2),
            Bottleneck3D(in_channels=80,out_channels=80,expanded_channels=240,stride=1,dropout=0.2),
            Bottleneck3D(in_channels=80,out_channels=80,expanded_channels=184,stride=1,dropout=0.2),
            Bottleneck3D(in_channels=80,out_channels=80,expanded_channels=184,stride=1,dropout=0.2),
            Bottleneck3D(in_channels=80,out_channels=112,expanded_channels=480,stride=1,use_se=True,dropout=0.2),
            Bottleneck3D(in_channels=112,out_channels=112,expanded_channels=672,stride=1,use_se=True,dropout=0.2)
            )
        
    #5x5 bottlenecks2 (3, h-swish, squeeze-excite): 14x14x112 -> 7x7x160
        self.block5 = nn.Sequential(
            Bottleneck3D(in_channels=112,out_channels=160,expanded_channels=672,stride=2,use_se=True,kernel_size=5,dropout=0.2),
            Bottleneck3D(in_channels=160,out_channels=160,expanded_channels=960,stride=1,use_se=True,kernel_size=5,dropout=0.2),
            Bottleneck3D(in_channels=160,out_channels=160,expanded_channels=960,stride=1,use_se=True,kernel_size=5,dropout=0.2)
            )
        
    #conv3d (h-swish), avg pool 7x7: 7x7x960 -> 1x1x960
        self.block6 = nn.Sequential(
            nn.Conv3d(in_channels=160,out_channels=960,stride=1,kernel_size=1),
            nn.BatchNorm3d(960),
            nn.Hardswish()
            )
        
    #classifier: conv3d 1x1 NBN (2, first uses h-swish): 1x1x960 
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1,1,1),
            nn.Conv3d(in_channels=960,out_channels=1280,kernel_size=1,stride=1,padding=0), #2 classes for ball/strike
            nn.Hardswish(),
            nn.Conv3d(in_channels=1280,out_channels=self.num_classes,kernel_size=1,stride=1,padding=0)
            )
        
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.classifier(x)
        x = F.softmax(x,dim=1)
        x = x.view(x.shape[0], self.num_classes)
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

#mobilenet small 3d convolutions
class MobileNetSmall3D(nn.Module):
    def __init__(self,num_classes=2):
        super().__init__()

        self.num_classes = num_classes

    #conv3d (h-swish): 224x224x3 -> 112x112x16
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels=3,out_channels=16,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm3d(16),
            nn.Hardswish()
            )

    #3x3 bottlenecks (3, ReLU, first gets squeeze-excite): 112x112x16 -> 28x28x24
        self.block2 = nn.Sequential(
            Bottleneck3D(in_channels=16,out_channels=16,expanded_channels=16,stride=2,use_se=True,nonlinearity=nn.LeakyReLU(),dropout=0.2),
            Bottleneck3D(in_channels=16,out_channels=24,expanded_channels=72,stride=2,nonlinearity=nn.LeakyReLU(),dropout=0.2),
            Bottleneck3D(in_channels=24,out_channels=24,expanded_channels=88,stride=1,nonlinearity=nn.LeakyReLU(),dropout=0.2)
            )
    #5x5 bottlenecks (8, h-swish, squeeze-excite): 28x28x24 -> 7x7x96
        self.block3 = nn.Sequential(
            Bottleneck3D(in_channels=24,out_channels=40,expanded_channels=96,stride=2,use_se=True,kernel_size=5,dropout=0.2),
            Bottleneck3D(in_channels=40,out_channels=40,expanded_channels=240,stride=1,use_se=True,kernel_size=5,dropout=0.2),
            Bottleneck3D(in_channels=40,out_channels=40,expanded_channels=240,stride=1,use_se=True,kernel_size=5,dropout=0.2), 
            Bottleneck3D(in_channels=40,out_channels=48,expanded_channels=120,stride=1,use_se=True,kernel_size=5,dropout=0.2),
            Bottleneck3D(in_channels=48,out_channels=48,expanded_channels=144,stride=1,use_se=True,kernel_size=5,dropout=0.2),
            Bottleneck3D(in_channels=48,out_channels=96,expanded_channels=288,stride=2,use_se=True,kernel_size=5,dropout=0.2),
            Bottleneck3D(in_channels=96,out_channels=96,expanded_channels=576,stride=1,use_se=True,kernel_size=5,dropout=0.2),
            Bottleneck3D(in_channels=96,out_channels=96,expanded_channels=576,stride=1,use_se=True,kernel_size=5,dropout=0.2)
            )
    #conv3d (h-swish), avg pool 7x7: 7x7x96 -> 1x1x576
        self.block4 = nn.Sequential(
            nn.Conv3d(in_channels=96,out_channels=576,kernel_size=1,stride=1,padding=0),
            SEBlock3D(channels=576),
            nn.BatchNorm3d(576),
            nn.Hardswish()
            )
    #conv3d 1x1, NBN, (2, first uses h-swish): 1x1x576
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1,1,1),
            nn.Conv3d(in_channels=576,out_channels=1024,kernel_size=1,stride=1,padding=0),
            nn.Hardswish(),
            nn.Conv3d(in_channels=1024,out_channels=self.num_classes,kernel_size=1,stride=1,padding=0),
            )

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        x = F.softmax(x,dim=1)
        x = x.view(x.shape[0], self.num_classes)
        return x
    

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
                if hasattr(module, "nonlinearity"):
                    if module.nonlinearity == 'relu' or 'leaky_relu':
                        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    elif module.nonlinearity == 'hardswish':
                        init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.BatchNorm3d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)