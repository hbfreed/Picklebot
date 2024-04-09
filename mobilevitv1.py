import torch
import torch.nn as nn
from mobilenet import Bottleneck2D, Bottleneck3D
from einops import rearrange
from einops.layers.torch import Reduce
from typing import Union, Tuple, List


#helpers
def conv_1x1_bn(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True)
    )

def conv_nxn_bn(in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, ...]]=3, stride: int=1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU()
    )


#classes
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int=8, dim_head: int=64, dropout: float=0.):
        super().__init__()
        inner_dim = dim_head * heads #512 by default
        self.heads = heads
        self.scale = dim_head ** -0.5 #normalize/scale the dot product

        self.norm = nn.LayerNorm(dim) #normalize the input
        self.attend = nn.Softmax(dim=-1) #softmax the attention scores
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) 


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1) #instead of using separate linear layers for q, k, and v, we use one linear layer and split the output into 3 parts

        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv) #split the heads, rearrange the dimensions (b: batch, p: patch, n: number of tokens/sequence length, h: number of heads d: dim_head)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale #dot product of q and k, scaled by the square root of the dimension of the head

        attn = self.attend(dots) #apply softmax to the attention scores
        attn = self.dropout(attn)

        out = torch.matmul(attn, v) #multiply the attention scores by the values
        out = rearrange(out, 'b p h n d -> b p n (h d)') #combine the heads
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float=0.):
        super().__init__()
        self.layers = nn.ModuleList([]) #initialize an empty list of layers to append to

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads, dim_head, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    
class MobileViTBlock(nn.Module):
    def __init__(self, dim: int, depth: int, channel: int, kernel_size: Union[int, Tuple[int, ...]], patch_size: int, mlp_dim: int, dropout: float=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 9, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2*channel, channel, kernel_size)


    def forward(self,x: torch.Tensor) -> torch.Tensor:
        y = x.clone() #clone the input tensor so we can use it later
        #local representations
        x = self.conv1(x)
        x = self.conv2(x)

        #transformer/global representations
        _,_, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        #combine local and global representations/fusion
        x = self.conv3(x)
        x = torch.cat((x, y), dim=1)
        x = self.conv4(x)
        return x
    

class MobileViTV1(nn.Module):

    def __init__(
            self,
            image_size: Tuple[int, ...], #image height and width, will want T dimension from video later
            dims: List[int],
            channels: List[int], 
            num_classes: int,
            expansion: int=4,
            kernel_size: Union[int, Tuple[int, ...]]=3,
            patch_size: Tuple[int, ...]=(2,2),
            depths: Tuple[int, ...]=(2,4,3), #number of transformer layers in each block, described in the paper
        ):
        super().__init__()
        assert len(dims) == 3, 'dims must be a tuple of 3'
        assert len(depths) == 3, 'depths must be a tuple of 3'

        ih,iw = image_size #image height and width, will want T dimension from video later
        ph,pw = patch_size #patch height and width, again will want a patch time dimension later
        assert ih % ph == 0 and iw % pw == 0, 'image height and width must be divisible by patch height and width'

        init_dim, *_, last_dim = channels

        self.conv1 = conv_nxn_bn(3, init_dim, stride=2)

        self.stem = nn.ModuleList([]) 
        self.stem.append(Bottleneck2D(channels[0],channels[1],expanded_channels=channels[0]*4,stride=1))
        self.stem.append(Bottleneck2D(channels[1],channels[2],expanded_channels=channels[1]*4,stride=2))
        self.stem.append(Bottleneck2D(channels[2],channels[3],expanded_channels=channels[2]*4,stride=1))
        self.stem.append(Bottleneck2D(channels[2],channels[3],expanded_channels=channels[2]*4,stride=1))

        self.trunk = nn.ModuleList([]) 
        self.trunk.append(nn.ModuleList([
            Bottleneck2D(channels[3],channels[4],expanded_channels=channels[3]*4,stride=2),
            MobileViTBlock(dims[0],depths[0],channels[5],kernel_size,patch_size, int(dims[0]*2))
        ]))

        self.trunk.append(nn.ModuleList([
            Bottleneck2D(channels[5],channels[6], expanded_channels=channels[7]*4,stride=2),
            MobileViTBlock(dims[1], depths[1], channels[7], kernel_size, patch_size, int(dims[1]*4))
        ]))

        self.trunk.append(nn.ModuleList([
            Bottleneck2D(channels[7],channels[8], expanded_channels=channels[7]*4,stride=2),
            MobileViTBlock(dims[2], depths[2], channels[9], kernel_size, patch_size, int(dims[2]*4))
        ]))

        self.to_logits = nn.Sequential(
            conv_1x1_bn(channels[-2], last_dim),
            Reduce('b c h w -> b c', reduction='mean'),
            nn.Linear(channels[-1], num_classes, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.conv1(x)

        for i,conv in enumerate(self.stem):
            x = conv(x)

        for i,(conv, attn) in enumerate(self.trunk):
            x = conv(x)
            x = attn(x)

        return self.to_logits(x)
