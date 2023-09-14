import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
import torch.nn.functional as F

'''Balls are 0, strikes are 1'''

def pad_batch(video,pad):
        video = video.transpose(0,-1)
        video = F.pad(video,(0,pad),value=2)
        video = video.permute(1,-1,2,3,0) #switch back, want channels to be first, so with batches, N,C,T,H,W
        return video

def custom_collate(batch): #this custom collate pads our batch.
    padded_batch = torch.tensor([])
    labels = torch.tensor([])
    max_length = max(video[0].shape[0] for video in batch)
    for video in batch:
        padded = pad_batch(video[0], max_length - video[0].shape[0])
        padded_batch = torch.cat((padded_batch, padded.unsqueeze(0)), dim=0)
        labels = torch.cat((labels, torch.tensor([video[1]])), dim=0)

    return padded_batch, labels


class PicklebotDataset(Dataset):
    def __init__(self, annotations_file, video_dir, transform=None,target_transform=None):
        self.video_labels = pd.read_csv(annotations_file,engine='pyarrow',header=None)
        self.video_dir = video_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.video_labels)
        
    def __getitem__(self,idx):
        video_path = os.path.join(self.video_dir, self.video_labels.iloc[idx,0])
        video = read_video(video_path,output_format="TCHW",pts_unit='sec')[0]/255
        label = self.video_labels.iloc[idx,1]
        if self.transform:
            video = self.transform(video)
        if self.target_transform:
            label = self.target_transform(label)
        return video, label