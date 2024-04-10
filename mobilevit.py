import torch
import torch.nn as nn
import torch.nn.functional as F
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


class LinearFeedForward(nn.Module): #source is from the huggingface implementation of the MobileViT V2 model on their github
    def __init__(
        self,
        embed_dim: int,
        ffw_dim: int,
        dropout: float = 0.,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=ffw_dim,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(
            in_channels=ffw_dim,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        return x

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

class LinearAttention(nn.Module):
    def __init__(self, embed_dim: int, dropout: float=0.) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.qkv_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=True,
            kernel_size=1
        )
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)
        self.to_out = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=True,
            kernel_size=1
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # (batch_size, embed_dim, num_pixels_in_patch, num_patches) -> (batch_size, 1+2*embed_dim, num_pixels_in_patch, num_patches)
        qkv = self.qkv_proj(hidden_states)

        #project the hidden states into q,k,v
        #q -> (batch_size, 1, num_pixels_in_patch, num_patches)
        # v,k -> (batch_size, embed_dim, num_pixels_in_patch, num_patches)
        q, k, v = torch.split(qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1)

        #softmax
        context_scores = self.attend(q)
        context_scores = self.dropout(context_scores)

        #compute the context vector
        #(batch_size, embed_dim, num_pixels_in_patch, num_patches) x (batch_size, 1, num_pixels_in_patch, num_patches) -> (batch_size, embed_dim, num_pixels_in_patch, num_patches)
        context_vector = k * context_scores
        #(batch_size, embed_dim, num_pixels_in_patch, num_patches)  -> (batch_size, embed_dim, num_pixels_in_patch, 1)
        context_vector  = torch.sum(context_vector, dim=-1, keepdim=True)

        #combine the context with the values
        #(batch_size, embed_dim, num_pixels_in_patch, num_patches) * (batch_size, embed_dim, num_pixels_in_patch, 1) -> (batch_size, embed_dim, num_pixels_in_patch, num_patches)
        out = F.relu(v) * context_vector.expand_as(v)

        return self.to_out(out) #reiterating, the out is (batch_size, embed_dim, num_pixels_in_patch, num_patches)
    


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, ffw_dim: int, dropout: float=0.) -> None:
        super().__init__()
        self.layers = nn.ModuleList([]) #initialize an empty list of layers to append to

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads, dim_head, dropout),
                FeedForward(dim, ffw_dim, dropout)
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    

class LinearTransformer(nn.Module):
    def __init__(self,dim: int, depth:int, ffw_dim: int, dropout: float = 0.,layer_norm_eps: float = 1e-05,ffw_dropout: float = 0.) -> None:
        super().__init__()
        self.layers = nn.ModuleList([]) #initialize an empty list of layers to append to
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
            nn.GroupNorm(num_groups=1,num_channels=dim, eps=layer_norm_eps),
            LinearAttention(embed_dim=dim, dropout=dropout),
            nn.Dropout(p=dropout),
            nn.GroupNorm(num_groups=1,num_channels=dim, eps=layer_norm_eps),
            LinearFeedForward(embed_dim=dim, ffw_dim=ffw_dim, dropout=ffw_dropout)
            ]))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for ln1, attn, dropout, ln2, ffw in self.layers:
            y = x.clone()
            x = ln1(x)
            x = attn(x)
            x = dropout(x)
            x = x + y
            x = ln2(x)
            x = ffw(x)
            x = x + y
        return x

class MobileViTBlock(nn.Module):
    def __init__(self, dim: int, depth: int, channel: int, kernel_size: Union[int, Tuple[int, ...]], patch_size: int, ffw_dim: int, dropout: float=0., use_linear_attention: bool=False) -> None:
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)
        if use_linear_attention:
            self.transformer = LinearTransformer(dim,depth,ffw_dim,dropout,layer_norm_eps=1e-05,ffw_dropout= 0.)
        else:
            self.transformer = Transformer(dim, depth, 4, 9, ffw_dim, dropout)


        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2*channel, channel, kernel_size)


    def forward(self,x: torch.Tensor) -> torch.Tensor:
        y = x.clone() #clone the input tensor so we can use it later
        #local representations
        x = self.conv1(x)
        x = self.conv2(x)

        #transformer/global representations
        _,_, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw) #fold
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw) #unfold
        #combine local and global representations/fusion
        x = self.conv3(x)
        x = torch.cat((x, y), dim=1)
        x = self.conv4(x)
        return x
    
# class MobileViTV2Block(nn.Module):
#     def __init__(self, dim: int, depth: int, channel: int, kernel_size: Union[int, Tuple[int, ...]], patch_size: int, ffw_dim: int, dropout: float=0.) -> None:
#         super().__init__()
#         self.ph, self.pw = patch_size

#         self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
#         self.conv2 = conv_1x1_bn(channel, dim)
        
#         self.transformer = LinearTransformer(dim=dim,depth=depth,ffw_dim=ffw_dim,dropout=dropout,layer_norm_eps=1e-05,ffn_dropout= 0.)

#         self.conv3 = conv_1x1_bn(dim, channel)
#         self.conv4 = conv_nxn_bn(2*channel, channel, kernel_size)

#     def forward(self,x: torch.Tensor) -> torch.Tensor:
#         y = x.clone() #clone the input tensor so we can use it later
#         #local representations
#         x = self.conv1(x)
#         x = self.conv2(x)

#         #transformer/global representations
#         _,_, h, w = x.shape
#         x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw) #fold
#         x = self.transformer(x)
#         x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw) #unfold

#         #combine local and global representations/fusion
#         x = self.conv3(x)
#         x = torch.cat((x, y), dim=1)
#         x = self.conv4(x)
#         return x

class MobileViT(nn.Module):
    def __init__(
            self,
            image_size: Tuple[int, ...], #image height and width, will want T dimension from video later
            dims: List[int],
            channels: List[int], 
            num_classes: int,
            expansion: int=4,
            kernel_size: Union[int, Tuple[int, ...]]=3,
            patch_size: Tuple[int, ...]=(2,2),
            depths: Tuple[int, ...]=(2,4,3),
            use_linear_attention: bool=False #number of transformer layers in each block, described in the paper
        ):
        super().__init__()
        assert len(dims) == 3, 'dims must be a tuple of 3'
        assert len(depths) == 3, 'depths must be a tuple of 3'

        ih,iw = image_size #image height and width, will want T dimension from video later
        ph,pw = patch_size #patch height and width, again will want a patch time dimension later
        assert ih % ph == 0 and iw % pw == 0, 'image height and width must be divisible by patch height and width'

        init_dim, *_, last_dim = channels

        self.conv1 = conv_nxn_bn(3, init_dim, stride=2) #3 channels to 16

        self.stem = nn.ModuleList([]) 
        self.stem.append(Bottleneck2D(channels[0],channels[1],expanded_channels=channels[0]*expansion,stride=1))
        self.stem.append(Bottleneck2D(channels[1],channels[2],expanded_channels=channels[1]*expansion,stride=2))
        self.stem.append(Bottleneck2D(channels[2],channels[3],expanded_channels=channels[2]*expansion,stride=1))
        self.stem.append(Bottleneck2D(channels[2],channels[3],expanded_channels=channels[2]*expansion,stride=1))

        self.trunk = nn.ModuleList([]) 
        self.trunk.append(nn.ModuleList([
            Bottleneck2D(channels[3],channels[4],expanded_channels=channels[3]*expansion,stride=2),
            MobileViTBlock(dims[0],depths[0],channels[5],kernel_size,patch_size, int(dims[0]*2),use_linear_attention=use_linear_attention)
        ]))

        self.trunk.append(nn.ModuleList([
            Bottleneck2D(channels[5],channels[6], expanded_channels=channels[7]*expansion,stride=2),
            MobileViTBlock(dims[1], depths[1], channels[7], kernel_size, patch_size, int(dims[1]*4),use_linear_attention=use_linear_attention)
        ]))

        self.trunk.append(nn.ModuleList([
            Bottleneck2D(channels[7],channels[8], expanded_channels=channels[7]*4,stride=2),
            MobileViTBlock(dims[2], depths[2], channels[9], kernel_size, patch_size, int(dims[2]*4),use_linear_attention=use_linear_attention)
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

    def initialize_weights(self) -> None:
        for module in self.modules():    
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)