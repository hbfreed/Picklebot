import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
import torch.nn.functional as F

'''Strikes are 1, balls are 2'''

def custom_collate(batch):
    # Find the maximum length of the videos in the batch
    max_length = max(video.shape[0] for video, _ in batch)
    # Create a list for the padded videos and labels
    padded_batch_list = []
    labels = []
    
    # Iterate over the batch to pad each video
    for video, label in batch:
        # Calculate the necessary padding for the temporal dimension
        pad_size = max_length - video.shape[0]
        
        # Pad the temporal dimension (T) of the video
        # Note: video is expected to be in TCHW format
        padded_video = F.pad(video, (0, 0, 0, 0, 0, 0,0, pad_size), value=0)
        
        # Add the padded video to the list
        padded_batch_list.append(padded_video)
        
        # Add the label to the list
        labels.append(label)
    
    # Stack the padded videos along a new dimension (batch dimension)
    padded_batch = torch.stack(padded_batch_list)
    
    # Convert the list of labels to a tensor
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_batch.transpose(1,2), labels


class PicklebotDataset(Dataset):
    def __init__(self, annotations_file, video_dir, transform=None,target_transform=None,dtype=torch.float32):
        self.video_labels = pd.read_csv(annotations_file,engine='pyarrow',header=None)
        self.video_dir = video_dir
        self.transform = transform
        self.target_transform = target_transform
        self.dtype = dtype

    def __len__(self):
        return len(self.video_labels)
        
    def __getitem__(self,idx):
        video_path = os.path.join(self.video_dir, self.video_labels.iloc[idx,0])
        video = ((read_video(video_path,output_format="TCHW",pts_unit='sec')[0]).to(self.dtype))/255
        label = self.video_labels.iloc[idx,1]
        if self.transform:
            video = self.transform(video)
        if self.target_transform:
            label = self.target_transform(label)
        return video, label