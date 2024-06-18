from mobilevit import MobileViT
import torch
import torch.profiler

dims = [96,120,144]
channels = [16,32,48,48,64,64,80,80,96,96,384]
model = MobileViT(dims,channels,num_classes=13).to(torch.bfloat16).to('cuda')
model = torch.compile(model)
A = torch.rand(12,3,200,224,224,device='cuda',dtype=torch.bfloat16)


with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    profile_memory=True,  # Enable memory profiling
    record_shapes=True,  # Record shapes of the tensors
    with_stack=True  # Record stack info
) as prof:
    model(A)  # Run your model

# Print the profiling results
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

