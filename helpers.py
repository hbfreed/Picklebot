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

#calculate the loss of the model, averaging every window_size batches
def average_for_plotting(loss_list,window_size=1000):
    partial_size = len(loss_list) % window_size
    if partial_size > 0:
        avg_losses = torch.tensor(loss_list[:-partial_size]).view(-1,1000).mean(1)
        avg_partial = torch.tensor(loss_list[-partial_size:]).view(-1,partial_size).mean(1)
        avg_losses = torch.cat((avg_losses, avg_partial))

    else:
        avg_losses = torch.tensor(loss_list).view(-1,1000).mean(1)
    return avg_losses
#define our pipeline
@pipeline_def
def video_pipeline(file_root, sequence_length, initial_prefetch_size):
    videos, labels = fn.readers.video(device="gpu", file_root=file_root, sequence_length=sequence_length,
                              shard_id=0, num_shards=1, random_shuffle=True, initial_fill=initial_prefetch_size,pad_sequences=True,
                              file_list_include_preceding_frame=False)
    return videos, labels
