import torch
import torch.nn.functional as F
#import nvidia.dali.fn as dalifn
#from nvidia.dali import pipeline_def as dali_pipeline_def
# import amd.rocal.fn as fn 
# from amd.rocal.pipeline import pipeline_def

#calculate the accuracy of the model
@torch.no_grad()
def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    correct = torch.sum(preds == labels)
    return correct

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
# @dali_pipeline_def
# def dali_video_pipeline(file_root, sequence_length, initial_prefetch_size,mean,std):
    # videos, labels = dalifn.readers.video(device="gpu", file_root=file_root, sequence_length=sequence_length,
#                             #   shard_id=0, num_shards=1, random_shuffle=True, initial_fill=initial_prefetch_size,pad_sequences=True,
#                               file_list_include_preceding_frame=False)
#     videos = dalifn.normalize(videos,mean=mean,stddev=std)
#     return videos, labels

# @pipeline_def
# def rocal_video_pipeline(file_root, sequence_length, initial_prefetch_size,mean,std):
#     videos, labels = fn.readers.video(device="gpu", file_root=file_root, sequence_length=sequence_length,
#                               shard_id=0, num_shards=1, random_shuffle=True, initial_fill=initial_prefetch_size,pad_sequences=True,
#                               file_list_include_preceding_frame=False)
#     videos = fn.normalize(videos,mean=mean,stddev=std)
#     return videos, labels

def calculate_accuracy_bce(outputs, labels, threshold=0.5):
    # Apply threshold to obtain predicted classes and move to CPU
    outputs = torch.sigmoid(outputs)
    preds = (outputs >= threshold).float().cpu()

    # Move labels to CPU
    labels = labels.cpu()

    # Calculate the number of correct predictions using a larger integer data type to prevent overflow
    num_correct = torch.sum(preds == labels, dtype=torch.int64).item()

    return num_correct
