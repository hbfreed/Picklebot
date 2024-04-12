import torch
import torch.nn as nn
import torch.nn.functional as F
from mobilenet import Bottleneck2D, Bottleneck3D
from einops import rearrange
from einops.layers.torch import Reduce
from typing import Union, Tuple, List

'''Put together with a lot of help and inspiration from the huggingface implementation and lucidrains (https://github.com/lucidrains/vit-pytorch/)'''

#helpers
def conv_1x1_bn(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU()
    )

def conv_nxn_bn(in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, ...]]=3, stride: int=1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU()
    )


#classes
class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
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

        self.norm = nn.LayerNorm(embed_dim) #normalize the input
        self.attend = nn.Softmax(dim=-1) #softmax the attention scores
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(dropout)
        ) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        # print(f'preqkv:{x.shape}')
        qkv = self.to_qkv(x).chunk(3, dim=-1) #instead of using separate linear layers for q, k, and v, we use one linear layer and split the output into 3 parts
        # print(f'qkv:{qkv[0].shape, qkv[1].shape, qkv[2].shape}')
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv) #split the heads, rearrange the dimensions (b: batch, p: patch, n: number of tokens/sequence length, h: number of heads d: dim_head)
        # print(f'qkv:{q.shape, k.shape, v.shape}')
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
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, embed_dim)

        #vivit 

        self.transformer = Transformer(embed_dim, depth, 4, 8, ffw_dim, dropout)

        self.conv3 = conv_1x1_bn(embed_dim, channel)
        self.conv4 = conv_nxn_bn(2*channel, channel, kernel_size)


    def forward(self,x: torch.Tensor) -> torch.Tensor:
        y = x.clone() #clone the input tensor so we can use it later
        #local representations
        x = self.conv1(x)
        x = self.conv2(x)

        #transformer/global representations
        _,_, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw) #unfold
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw) #fold
        #combine local and global representations/fusion
        x = self.conv3(x)
        x = torch.cat((x, y), dim=1)
        x = self.conv4(x)
        return x
    

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

        for conv in self.stem:
            x = conv(x)

        for conv, attn in self.trunk:
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


class LinearSelfAttention(nn.Module): 
    def __init__(self, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        
        self.qkv_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim), #q is a vector for each patch position, so 1, k is a matrix, v is a matrix
            bias=True,
            kernel_size=1
        )
        self.attn_dropout = nn.Dropout(p=dropout)
        self.out_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=True,
            kernel_size=1
        )
        self.embed_dim = embed_dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # (batch_size, embed_dim, num_pixels_in_patch, num_patches) -> (batch_size, 1+2*embed_dim, num_pixels_in_patch, num_patches)
        qkv = self.qkv_proj(hidden_states)

        #project the hidden states into q,k,v
        #q -> (batch_size, 1, num_pixels_in_patch, num_patches)
        # v,k -> (batch_size, embed_dim, num_pixels_in_patch, num_patches)
        query, key, value = torch.split(qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1)

        #softmax
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        #compute the context vector
        #(batch_size, embed_dim, num_pixels_in_patch, num_patches) x (batch_size, 1, num_pixels_in_patch, num_patches) -> (batch_size, embed_dim, num_pixels_in_patch, num_patches)
        context_vector = key * context_scores
        #(batch_size, embed_dim, num_pixels_in_patch, num_patches)  -> (batch_size, embed_dim, num_pixels_in_patch, 1)
        context_vector  = torch.sum(context_vector, dim=-1, keepdim=True)

        #combine the context with the values
        #(batch_size, embed_dim, num_pixels_in_patch, num_patches) * (batch_size, embed_dim, num_pixels_in_patch, 1) -> (batch_size, embed_dim, num_pixels_in_patch, num_patches)
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)

        return out #reiterating, the out is (batch_size, embed_dim, num_pixels_in_patch, num_patches)


class MobileViTV2FFN(nn.Module): #source is from the huggingface implementation of the MobileViT V2 model on their github
    def __init__(
        self,
        embed_dim: int,
        ffn_latent_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=ffn_latent_dim,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(
            in_channels=ffn_latent_dim,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        return x


class MobileViTV2TransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_latent_dim: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-05,
        ffn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layernorm_before = nn.GroupNorm(num_groups=1,num_channels=embed_dim, eps=layer_norm_eps)
        self.attention = LinearSelfAttention(embed_dim=embed_dim, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm_after= nn.GroupNorm(num_groups=1,num_channels=embed_dim, eps=layer_norm_eps)
        self.ffn = MobileViTV2FFN(embed_dim=embed_dim, ffn_latent_dim=ffn_latent_dim, dropout=ffn_dropout)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        layernorm_before = self.layernorm_before(hidden_states)
        attention_output = self.attention(layernorm_before)
        attention_output = self.dropout(attention_output)
        hidden_states = attention_output + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.ffn(layer_output)

        layer_output = layer_output + hidden_states
        return layer_output


class MobileViTV2Transformer(nn.Module):
    def __init__(self, n_layers: int, d_model: int, ffn_multiplier: int = 2) -> None:
        super().__init__()
        ffn_multiplier = ffn_multiplier
        
        ffn_dims = [ffn_multiplier * d_model] * n_layers

        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]

        self.layer = nn.ModuleList()

        for block_idx in range(n_layers):
            transformer_layer = MobileViTV2TransformerLayer(
                embed_dim=d_model,
                ffn_latent_dim=ffn_dims[block_idx],
                dropout=0.0,
            )
            self.layer.append(transformer_layer)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class MobileViTV2Layer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        attn_unit_dim: int,
        kernel_size: int = 3,
        patch_size: int = 2,
        n_attn_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.patch_width = patch_size
        self.patch_height = patch_size
        #self.patch_depth = patch_size

        cnn_out_dim = attn_unit_dim

        #depthwise
        self.convkxk = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            dilation=1,
            bias=False,
            groups=1
        )

        #pointwise
        self.conv1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=cnn_out_dim,
            kernel_size=1,
            bias=False,
            stride=1,
            padding=0,
            dilation=1,
            groups=1
        )

        self.transformer = MobileViTV2Transformer(n_layers=n_attn_blocks, d_model=attn_unit_dim)

        self.layernorm = nn.GroupNorm(num_groups=1,num_channels=attn_unit_dim, eps=1e-05)

        #fusion, pointwise
        self.conv_projection = nn.Conv2d(
            in_channels=cnn_out_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
            groups=1
        )


    def unfolding(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor,Tuple[int,int]]:
        batch_size, in_channels, img_height, img_width = feature_map.size() #, img_depth = feature_map.size()
        patches = F.unfold( #need to figure out a 3d version, should be possible
            feature_map,
            kernel_size=(self.patch_height, self.patch_width),
            stride=(self.patch_height, self.patch_width),
        )
        patches = patches.reshape(batch_size, in_channels,self.patch_height*self.patch_width, -1)


        return patches, (img_height, img_width) #, img_depth)
    
    def folding(self, patches: torch.Tensor, output_size: Tuple[int,int]) -> torch.Tensor:
        batch_size, in_dim, patch_size, n_patches = patches.shape
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)

        feature_map = nn.functional.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_height, self.patch_width),
            stride=(self.patch_height, self.patch_width),
        )

        return feature_map


    def forward(self, features: torch.Tensor) -> torch.Tensor:
        #local representation using convs
        # print(f"before_conv:{features.shape}")
        features = self.convkxk(features)
        # print(f"after convkxk:{features.shape}")
        features = self.conv1x1(features)
        # print(f"after conv1x1:{features.shape}")
        #feature map -> patches (unfold)
        patches, output_size = self.unfolding(features)
        #learn global representations
        patches = self.transformer(patches)
        #fold patches back into feature map
        patches = self.layernorm(patches)
        #patches -> feature map (fold)
        #(batch_size, patch_height, patch_width, input_dim) -> batch_size, input_dim, patch_height, patch_width
        features = self.folding(patches, output_size)
        features = self.conv_projection(features)
        return features


class MobileViTV2(nn.Module):
    def __init__(self,expand_ratio: int = 2, alpha: float=1.,num_classes: int=1000) -> None:
        super().__init__()

        #note that our downsampling is in the mobilevitv2layer when the stride is 2 (it defaults to 2, so we downsample there) 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=int(32*alpha), kernel_size=3, stride=2, padding=1, bias=False) #should be image_size//2 x image_size//2 x 32\alpha, i.e., downsample here
        self.bn1 = Bottleneck2D(in_channels=int(32*alpha),out_channels=int(64*alpha),expanded_channels=int(32*expand_ratio*alpha)) #should be image_size//2 x image_size//2 x 64\alpha

        self.bn2 = Bottleneck2D(in_channels=int(64*alpha),out_channels=int(128*alpha),expanded_channels=int(64*expand_ratio*alpha),stride=2) #downsample here, should be image_size//4 x image_size//4 x 128\alpha
        self.bn3 = Bottleneck2D(in_channels=int(128*alpha),out_channels=int(128*alpha),expanded_channels=int(128*expand_ratio*alpha)) #no downsample, should be image_size//4 x image_size//4 x 128\alpha
        self.bn4 = Bottleneck2D(in_channels=int(128*alpha),out_channels=int(128*alpha),expanded_channels=int(128*expand_ratio*alpha)) #repeat the above here
        
        self.bn5 = Bottleneck2D(in_channels=int(128*alpha),out_channels=int(256*alpha),expanded_channels=int(256*expand_ratio*alpha),stride=2) #down2 169,888 params
        self.transformer1 = MobileViTV2Layer(in_channels=int(256*alpha),attn_unit_dim=int(128*alpha),n_attn_blocks=2) #want B=2, so repeat the attention and ffw layers twice, d=128 \alpha, 371,616
        
        self.bn6 = Bottleneck2D(in_channels=int(256*alpha),out_channels=int(384*alpha), expanded_channels=int(384*expand_ratio*alpha),stride=2) #down2 1,758,626 params
        self.transformer2 = MobileViTV2Layer(in_channels=int(384*alpha),attn_unit_dim=int(192*alpha),n_attn_blocks=4) #B=4, so repeat the attention and ffw layers four times, d = 192\alpha, 4,274,214 params
        
        self.bn7 = Bottleneck2D(in_channels=int(384*alpha),out_channels=int(512*alpha), expanded_channels=int(512*expand_ratio*alpha),stride=2) #down2, 5,201,958 params
        self.transformer3 = MobileViTV2Layer(in_channels=int(512*alpha),attn_unit_dim=int(256*alpha),n_attn_blocks=3) #B=3, so repeat the attention and ffw layers three times, d = 256\alpha, 9208617 params

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) #9,208,617 params
        self.fc = nn.Linear(int(512*alpha), num_classes) #1000 classes, so output should be 1000 (in channels are technically 512\alpha, but we can just use 512 for now), 9,471,273 params


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        # print(f"after conv1:{x.shape}")
        x = self.bn1(x)
        # print(f"after bn1:{x.shape}")
        x = self.bn2(x)
        # print(f"after bn2:{x.shape}")
        x = self.bn3(x)
        # print(f"after bn3:{x.shape}")
        x = self.bn4(x)
        # print(f"after bn4:{x.shape}")
        x = self.bn5(x)
        # print(f"after bn5:{x.shape}")
        x = self.transformer1(x)
        # # print(f"after transformer1:{x.shape}")
        x = self.bn6(x)
        # # print(f"after bn6:{x.shape}")
        x = self.transformer2(x)
        # # print(f"after transformer2:{x.shape}")
        x = self.bn7(x)
        # # print(f"after bn7:{x.shape}")
        x = self.transformer3(x)
        # # print(f"after transformer3:{x.shape}")

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

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