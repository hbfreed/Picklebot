import os
import pandas as pd
import torch
import cv2
from torch.utils.data import DataLoader, Dataset
from dataloader import custom_collate, PicklebotDataset
import time
from tqdm import tqdm
import numpy as np

dtype = torch.bfloat16

annotations_file = '/home/henry/Documents/PythonProjects/picklebot_2m/picklebot_130k_val.csv'
video_paths = '/home/henry/Documents/PythonProjects/picklebot_2m/picklebot_130k_all_together'
batch_size = 16

class PicklebotDataset(Dataset):
    def __init__(self, annotations_file, video_dir, transform=None, target_transform=None, dtype=torch.bfloat16, backend='opencv'):
        self.video_labels = pd.read_csv(annotations_file, engine='pyarrow', encoding='ISO-8859-1')
        self.video_dir = video_dir
        self.transform = transform
        self.target_transform = target_transform
        self.dtype = dtype
        self.backend = backend

    def __len__(self):
        return self.video_labels.shape[0]
        
    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_labels['filename'][idx])

        if self.backend == 'newcv':
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1024)

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            frames = np.empty((frame_count, frame_height, frame_width,3), dtype=np.uint8) #channel last so we only have to permute once at the end

            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames[i] = frame

            cap.release()
            video = torch.from_numpy(frames)
 
        elif self.backend == 'opencv':
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            video = torch.from_numpy(np.array(frames,dtype=np.uint8))

        elif self.backend == 'ffmpeg':
            import ffmpeg
            # Use ffmpeg-python to decode the video frames
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            width = int(video_info['width'])
            height = int(video_info['height'])

            out, _ = (
                ffmpeg
                .input(video_path)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
            )
            frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
            video = torch.from_numpy(frames)

            label = self.video_labels["zone"][idx]
            if self.transform:
                video = self.transform(video)
            if self.target_transform:
                label = self.target_transform(label)
            return video, label

        
        label = self.video_labels["zone"][idx]
        if self.transform:
            video = self.transform(video)
        if self.target_transform:
            label = self.target_transform(label)
        return video, label

for be in ['opencv']:#,newcv,'ffmpeg']:
    #start the timer
    start = time.time()    
    dataset = PicklebotDataset(annotations_file,video_paths,dtype=dtype,backend=be)
    loader = DataLoader(dataset, batch_size=batch_size,shuffle=False,collate_fn=custom_collate,num_workers=12,pin_memory=True)


    for i in tqdm(loader):
        i = i[0].to('cuda').permute(0,1,4,2,3).to(torch.bfloat16) / 255
        pass
    print(f'time for {be} if we do video.shape for max videos:',time.time()-start)
        