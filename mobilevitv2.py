import torch
import torch.nn as nn
import torch.nn.functional as F
from mobilenet import Bottleneck3D, Bottleneck2D #for the MobileNetV2 bottleneck blocks, which are used in the MobileViTV1 architecture. We use blocks from Mobilenet V3.
from typing import Optional, Tuple, Union

'''Put together with a lot of help and inspiration from the huggingface implementation and lucidrains (https://github.com/lucidrains/vit-pytorch/)'''

class LinearSelfAttention(nn.Module): 
    def __init__(self, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        
        self.qkv_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
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
        out_channels: int,
        attn_unit_dim: int,
        kernel_size: int = 3,
        patch_size: int = 2,
        n_attn_blocks: int = 2,
        expansion_ratio: int = 2,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.patch_width = patch_size
        self.patch_height = patch_size
        #self.patch_depth = patch_size

        cnn_out_dim = attn_unit_dim

        if stride == 2:
            self.downsampling_layer = Bottleneck2D(
                in_channels=in_channels,
                out_channels=out_channels,
                expanded_channels=in_channels*expansion_ratio,
                stride=stride,
                kernel_size=kernel_size,
            )
            in_channels = out_channels
        else:
            self.downsampling_layer = None

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
            padding=1,
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
            padding=1,
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
        #reduce spatial dims if needed
        if self.downsampling_layer:
            features = self.downsampling_layer(features)
        #local representation using convs
        features = self.convkxk(features)
        features = self.conv1x1(features)
        #feature map -> patches (unfold)
        patches, output_size = self.unfolding(features)
        #learn global representations
        patches = self.transformer(patches)
        patches = self.layernorm(patches)
        #patches -> feature map (fold)
        #(batch_size, patch_height, patch_width, input_dim) -> batch_size, input_dim, patch_height, patch_width
        features = self.folding(patches, output_size)
        features = self.conv_projection(features)
        return features


class MobileViTV2(nn.Module):
    def __init__(self,expand_ratio: int = 2) -> None:
        super().__init__()

        #note that our downsampling is in the mobilevitv2layer when the stride is 2 (it defaults to 2, so we downsample there) 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False) #should be image_size//2 x image_size//2 x 32\alpha, i.e., downsample here
        self.bn1 = Bottleneck2D(in_channels=32,out_channels=64,expanded_channels=32*expand_ratio) #should be image_size//2 x image_size//2 x 64\alpha

        self.bn2 = Bottleneck2D(in_channels=64,out_channels=128,expanded_channels=64*expand_ratio,stride=2) #downsample here, should be image_size//4 x image_size//4 x 128\alpha
        self.bn3 = Bottleneck2D(in_channels=128,out_channels=128,expanded_channels=128*expand_ratio) #no downsample, should be image_size//4 x image_size//4 x 128\alpha
        self.bn3_5 = Bottleneck2D(in_channels=128,out_channels=128,expanded_channels=128*expand_ratio) #repeat the above here
        
        self.transformer1 = MobileViTV2Layer(in_channels=128,out_channels=256,attn_unit_dim=128,n_attn_blocks=2) #want B=2, so repeat the attention and ffw layers twice, d=128 \alpha
        
        self.transformer2 = MobileViTV2Layer(in_channels=256,out_channels=384,attn_unit_dim=192,n_attn_blocks=4) #B=4, so repeat the attention and ffw layers four times, d = 192\alpha
        
        self.transformer3 = MobileViTV2Layer(in_channels=384,out_channels=512,attn_unit_dim=256,n_attn_blocks=3) #B=3, so repeat the attention and ffw layers three times, d = 256\alpha

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000) #1000 classes, so output should be 1000 (in channels are technically 512\alpha, but we can just use 512 for now)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.bn2(x)
        x = self.bn3(x)
        x = self.bn3_5(x)
        print(f"shape after bottleneck 3.5: {x.shape}")

        x = self.transformer1(x)

        x = self.transformer2(x)

        x = self.transformer3(x)

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