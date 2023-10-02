import torch
import torch.nn as nn
import torch.nn.init as init
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def

#calculate the accuracy of the model, 
def calculate_accuracy(outputs,labels):
    predicted_classes = torch.argmax(outputs,dim=1).to(labels.device)
    num_correct = torch.sum(predicted_classes == labels).item()
    return num_correct

#initialize the model
def initialize_mobilenet_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
            # Initialize convolutional layers with appropriate initialization
            init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
            # Initialize Batch Normalization layers with small values and biases to zero
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')

#define our pipeline
@pipeline_def
def video_pipeline(file_root, sequence_length, initial_prefetch_size):
    videos, labels = fn.readers.video(device="gpu", file_root=file_root, sequence_length=sequence_length,
                              shard_id=0, num_shards=1, random_shuffle=True, initial_fill=initial_prefetch_size,pad_sequences=True,
                              file_list_include_preceding_frame=False)
    return videos, labels