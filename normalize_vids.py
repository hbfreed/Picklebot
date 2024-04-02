import torch
from torch.utils.data import DataLoader
from dataloader import PicklebotDataset, custom_collate
from torchvision import transforms
from tqdm import tqdm
import time
from torchvision.io import write_video
import os

std = (0.2104, 0.1986, 0.1829)
mean = (0.3939, 0.3817, 0.3314)
batch_size = 1
dtype = torch.float32

#annotations paths
train_annotations_file = '/home/henry/Documents/PythonProjects/picklebot_2m/picklebot_130k_train.csv'
val_annotations_file = '/home/henry/Documents/PythonProjects/picklebot_2m/picklebot_130k_val.csv'


#video paths
video_paths = '/home/henry/Documents/PythonProjects/picklebot_2m/picklebot_130k_all_together'
video_out_paths = '/home/henry/Documents/PythonProjects/picklebot_2m/picklebot_130k_all_together_normalized'

#establish our normalization using transforms, 
#note that we are doing this in our dataloader as opposed to in the training loop like with dali
transform = transforms.Normalize(mean,std)

#dataset     
train_dataset = PicklebotDataset(train_annotations_file,video_paths,transform=transform,dtype=dtype)
train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=False,num_workers=36)
val_dataset = PicklebotDataset(val_annotations_file,video_paths,transform=transform,dtype=dtype)
val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=False,num_workers=36)

#test the dataloader

for video,path in tqdm(train_loader):
    video = video.squeeze(0).permute(0,2,3,1)
    video = video * 255
    write_video(f"{video_out_paths}/{os.path.basename(path[0])}",video.squeeze(0),fps=15)
    

# for video,_ in tqdm(val_loader):
'''
NEED TO MODIFY THE DATALOADER FOR THIS, BUT TRYING SOMETHING FIRST




'''