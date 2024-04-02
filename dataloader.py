import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import cv2


def custom_collate(batch):
    videos,labels = zip(*batch)
    max_length = max(video.shape[0] for video in videos)
    
    padded_videos = []
    for video in videos:
        pad_size = max_length - video.shape[0]
        padded_video = F.pad(video,(0,0,0,0,0,0,0,pad_size),value=0)
        padded_videos.append(padded_video)
    
    padded_batch = torch.stack(padded_videos).transpose(1,2)
    labels = torch.tensor(labels,dtype=torch.long)

    return padded_batch, labels


class PicklebotDataset(Dataset):
    def __init__(self, annotations_file, video_dir, transform=None, target_transform=None, dtype=torch.bfloat16):
        self.video_labels = pd.read_csv(annotations_file, engine='pyarrow', encoding='ISO-8859-1')
        self.video_dir = video_dir
        self.transform = transform
        self.target_transform = target_transform
        self.dtype = dtype

    def __len__(self):
        return self.video_labels.shape[0]
        
    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_labels['filename'][idx])
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1).to(self.dtype) / 255
            frames.append(frame)
        cap.release()
        video = torch.stack(frames)
        label = self.video_labels["zone"][idx]
        if self.transform:
            video = self.transform(video)
        if self.target_transform:
            label = self.target_transform(label)
        return video, label
