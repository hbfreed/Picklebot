import torch
import torch.nn as nn
import torch.nn.functional as F
from mobilenet import Bottleneck3D
from flash_attn import flash_attn_func,flash_attn_qkvpacked_func,flash_attn_triton_qkvpacked_func
from einops import rearrange
from einops.layers.torch import Reduce
from typing import Union, Tuple, List

'''Put together with a lot of inspiration from the huggingface implementation and lucidrains, karpathy, and the flash attention team'''

#helpers
def conv_1x1_bn(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm3d(out_channels,affine=False),
        nn.SiLU()
    )

def conv_nxn_bn(in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, ...]]=3, stride: int=1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, 1, bias=False),
        nn.BatchNorm3d(out_channels,affine=False),
        nn.SiLU()
    )


#classes
class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, embed_dim: int, heads: int=8, dim_head: int=64, dropout: float=0.):
        super().__init__()
        inner_dim = dim_head * heads #512 by default
        self.heads = heads
        self.scale = dim_head ** -0.5 #normalize/scale the dot product

        self.norm = nn.LayerNorm(embed_dim,elementwise_affine=False) #normalize the input
        self.attend = nn.Softmax(dim=-1) #softmax the attention scores
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, embed_dim, bias=False),
            nn.Dropout(dropout)
        )
        self.flash = True #check if the scaled dot product attention function is available, the flash attention stuff is from karpathy nanoGPT
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x) #instead of using separate linear layers for q, k, and v, we use one linear layer and split the output into 3 parts
        if self.flash:
            # out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,dropout_p=self.dropout_p) #scaled dot product attention
            qkv = qkv.view(qkv.shape[0],qkv.shape[2],3,qkv.shape[1],qkv.shape[3]//3)
            out = flash_attn_triton_qkvpacked_func(qkv) #scaled dot product attention
            out = out.transpose(1,2)
            # out = flash_attn_func(q, k, v, dropout_p=self.dropout_p) #scaled dot product attention

        else:
            qkv = qkv.chunk(3, dim=-1) #instead of using separate linear layers for q, k, and v, we use one linear layer and split the output into 3 parts    
            q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv) #split the heads, rearrange the dimensions (b: batch, p: patch, n: number of tokens/sequence length, h: number of heads d: dim_head)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale #dot product of q and k, scaled by the square root of the dimension of the head
            attn = self.attend(dots) #apply softmax to the attention scores
            attn = self.dropout(attn)

            out = torch.matmul(attn, v) #multiply the attention scores by the values
            out = rearrange(out, 'b p h n d -> b p n (h d)') #combine the heads
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, embed_dim: int, depth: int, heads: int, dim_head: int, ffw_dim: int, dropout: float=0.) -> None:
        super().__init__()
        self.layers = nn.ModuleList([]) #initialize an empty list of layers to append to

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(embed_dim, heads, dim_head, dropout),
                FeedForward(embed_dim, ffw_dim, dropout)
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    

class MobileViTBlock(nn.Module):
    def __init__(self, embed_dim: int, depth: int, channel: int, kernel_size: Union[int, Tuple[int, ...]], patch_size: int, ffw_dim: int, dropout: float=0., use_linear_attention: bool=False) -> None:
        super().__init__()
        self.pt, self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, embed_dim)

        #vivit
        self.transformer = Transformer(embed_dim=embed_dim, depth=depth, heads=4, dim_head=8, ffw_dim=ffw_dim, dropout=dropout)

        self.conv3 = conv_1x1_bn(embed_dim, channel)
        self.conv4 = conv_nxn_bn(2*channel, channel, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        # Global representations
        _, _, t, h, w = x.shape
        t_pad = (self.pt - t % self.pt) % self.pt
        h_pad = (self.ph - h % self.ph) % self.ph
        w_pad = (self.pw - w % self.pw) % self.pw
        x = F.pad(x, (0, w_pad, 0, h_pad, 0, t_pad))
        _, _, t_padded, h_padded, w_padded = x.shape
        x = rearrange(x, 'b d (t pt) (h ph) (w pw) -> b (pt ph pw) (t h w) d', pt=self.pt, ph=self.ph, pw=self.pw, t=t_padded//self.pt, h=h_padded//self.ph, w=w_padded//self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (pt ph pw) (t h w) d -> b d (t pt) (h ph) (w pw)', pt=self.pt, ph=self.ph, pw=self.pw, t=t_padded//self.pt, h=h_padded//self.ph, w=w_padded//self.pw)
        x = x[:, :, :t, :h, :w]  # Remove padding

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x
    

class NanDetector(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        if torch.isnan(grad_output).any():
            raise RuntimeError(f"Found nan in the backward pass for input:\n{input}")
        if torch.isnan(input).any():
            raise RuntimeError(f"Found nan in the forward pass for input:\n{input}")
        return grad_output

class MobileViT(nn.Module):
    def __init__(
            self,
            dims: List[int],
            channels: List[int], 
            num_classes: int,
            expansion: int=4,
            kernel_size: Union[int, Tuple[int, ...]]=3,
            patch_size: Tuple[int, ...]=(2,2,2),
            depths: Tuple[int, ...]=(2,4,3),
            use_linear_attention: bool=False #number of transformer layers in each block, described in the paper
        ):
        super().__init__()
        assert len(dims) == 3, 'dims must be a tuple of 3'
        assert len(depths) == 3, 'depths must be a tuple of 3'


        init_dim, *_, last_dim = channels

        self.conv1 = conv_nxn_bn(3, init_dim, stride=2) #3 channels to 16

        self.stem = nn.ModuleList([]) 
        self.stem.append(Bottleneck3D(channels[0],channels[1],expanded_channels=channels[0]*expansion,stride=1))
        self.stem.append(Bottleneck3D(channels[1],channels[2],expanded_channels=channels[1]*expansion,stride=2))
        self.stem.append(Bottleneck3D(channels[2],channels[3],expanded_channels=channels[2]*expansion,stride=1))
        self.stem.append(Bottleneck3D(channels[2],channels[3],expanded_channels=channels[2]*expansion,stride=1))

        self.trunk = nn.ModuleList([]) 
        self.trunk.append(nn.ModuleList([
            Bottleneck3D(channels[3],channels[4],expanded_channels=channels[3]*expansion,stride=2),
            MobileViTBlock(dims[0],depths[0],channels[5],kernel_size,patch_size, int(dims[0]*2),use_linear_attention=use_linear_attention)
        ]))

        self.trunk.append(nn.ModuleList([
            Bottleneck3D(channels[5],channels[6], expanded_channels=channels[7]*expansion,stride=2),
            MobileViTBlock(dims[1], depths[1], channels[7], kernel_size, patch_size, int(dims[1]*4),use_linear_attention=use_linear_attention)
        ]))

        self.trunk.append(nn.ModuleList([
            Bottleneck3D(channels[7],channels[8], expanded_channels=channels[7]*4,stride=2),
            MobileViTBlock(dims[2], depths[2], channels[9], kernel_size, patch_size, int(dims[2]*4),use_linear_attention=use_linear_attention)
        ]))

        self.to_logits = nn.Sequential(
            conv_1x1_bn(channels[-2], last_dim),
            Reduce('b c t h w -> b c', reduction='mean'),
            nn.Linear(channels[-1], num_classes, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.conv1(x)

        for conv in self.stem:
            x = conv(x)

        for conv, attn in self.trunk:
            x = conv(x)
            x = attn(x)
        x = self.to_logits(x)
        return x

    def initialize_weights(self) -> None:
        for module in self.modules():    
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    print(module.bias.data)
                    module.bias.data.zero_()