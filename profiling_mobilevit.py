import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Reduce
from typing import Union, Tuple, List
from pytorch_memlab import profile
from mobilenet import Bottleneck3D

# Define your model and other functions as provided

# Profile the MobileViT forward and backward pass
@profile
def profile_model():
    model = MobileViT(
        dims=[64, 128, 256],
        channels=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        num_classes=1000
    ).to('cuda')
    
    # Dummy input for the forward pass
    input_tensor = torch.randn(1, 3, 16, 224, 224).to('cuda')  # Example input dimensions
    
    # Forward pass
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    
    # Dummy target and loss for the backward pass
    target = torch.randn_like(output)
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    
    # Backward pass
    loss.backward()

if __name__ == '__main__':
    profile_model()

