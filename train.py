#Train our models and save them to disk
#Usage: python train.py --config config/config.json --dataloader torchvision
import os
import time
import numpy as np
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from psutil import cpu_count
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from mobilenet import MobileNetSmall3D,MobileNetLarge3D
from movinet import MoViNetA2
from helpers import calculate_accuracy_bce, average_for_plotting, calculate_accuracy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def create_dataloader(dataloader,batch_size,mean,std):
    #create dataloader
    if dataloader == "torchvision":
        from torch.utils.data import DataLoader
        from dataloader import PicklebotDataset, custom_collate

        #video paths
        train_video_paths = '/workspace/picklebotdataset/train'
        val_video_paths = '/workspace/picklebotdataset/val'

        #annotations paths
        train_annotations_file = '/home/henry/Documents/PythonProjects/picklebotdataset/train_labels.csv'
        val_annotations_file = '/home/henry/Documents/PythonProjects/picklebotdataset/val_labels.csv'

        #video paths
        train_video_paths = '/home/henry/Documents/PythonProjects/picklebotdataset/train_all_together'
        val_video_paths = '/home/henry/Documents/PythonProjects/picklebotdataset/val_all_together'

        #establish our normalization using transforms, 
        #note that we are doing this in our dataloader as opposed to in the training loop like with dali
        transform = transforms.Normalize(mean,std)

        #dataset     
        train_dataset = PicklebotDataset(train_annotations_file,train_video_paths,transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,collate_fn=custom_collate,num_workers=24)
        val_dataset = PicklebotDataset(val_annotations_file,val_video_paths,transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=True,collate_fn=custom_collate,num_workers=24)


    elif dataloader == "dali":
        from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
        from helpers import video_pipeline

        #information for the dali pipeline
        sequence_length = 130 #longest videos in our dataset 
        initial_prefetch_size = 20


        #video paths
        train_video_paths = '/home/hankhome/Documents/PythonProjects/picklebotdataset/train'
        val_video_paths = '/home/hankhome/Documents/PythonProjects/picklebotdataset/val'

        num_train_videos = len(os.listdir(train_video_paths + '/' + 'balls')) + len(os.listdir(train_video_paths + '/' + 'strikes'))
        num_val_videos = len(os.listdir(val_video_paths + '/' + 'balls')) + len(os.listdir(val_video_paths + '/' + 'strikes'))

        #multiply mean and val by 255 to convert to 0-255 range
        mean = (torch.tensor(mean)*255)[None,None,None,:]
        std = (torch.tensor(std)*255)[None,None,None,:]

        print("Building pipelines...")

        #build our pipelines
        train_pipe = video_pipeline(batch_size=batch_size, num_threads=cpu_count()//2, device_id=0, file_root=train_video_paths,
                                    sequence_length=sequence_length,initial_prefetch_size=initial_prefetch_size,mean=mean*255,std=std*255)
        val_pipe = video_pipeline(batch_size=batch_size, num_threads=cpu_count()//2, device_id=0, file_root=val_video_paths,
                                sequence_length=sequence_length,initial_prefetch_size=initial_prefetch_size,mean=mean,std=std)

        train_pipe.build()
        val_pipe.build()


        train_loader = DALIClassificationIterator(train_pipe, auto_reset=True,last_batch_policy=LastBatchPolicy.PARTIAL, size=num_train_videos)
        val_loader = DALIClassificationIterator(val_pipe, auto_reset=True,last_batch_policy=LastBatchPolicy.PARTIAL, size=num_val_videos)

        

    elif dataloader == "rocal":
        raise NotImplementedError("rocAL dataloader not implemented yet")
        #this will need testing, but here's a stab at it
        from amd.rocal.plugin.pytorch import ROCALClassificationIterator

        # Define the parameters for the ROCAL video reader
        video_folder_path = '/path/to/your/mp4/videos'
        shuffle = True
        file_list = '/path/to/your/file_list.txt'  # A text file listing the videos and their labels

        # Create a file list in the format required by ROCAL
        # Each line should have the format: video_filepath label_index
        # Example:
        # /path/to/video1.mp4 0
        # /path/to/video2.mp4 1
        # ...
        # You can create this file manually or write a script to generate it based on your dataset structure.
        # Make sure the file paths and labels are correct regarding your dataset.

        # Define a ROCAL pipeline for loading and augmenting videos
        class VideoPipe():
            def __init__(self, batch_size, num_threads, device_id, data, shuffle):
                self.input = ops.VideoReader(device="gpu", file_root=data, sequence_length=16,
                                            shard_id=device_id, random_shuffle=shuffle)
                self.crop = ops.Crop(device="gpu", crop=crop_size)
                self.uniform = ops.Uniform(range=(0.0, 1.0))
                self.transpose = ops.Transpose(device="gpu", perm=[3, 0, 1, 2])  # Change from DHWC to CDHW

            def define_graph(self):
                video, labels = self.input(name="Reader")
                video = self.crop(video, crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
                video = self.transpose(video)
                return video, labels

        # Instantiate the pipeline
        device_id = torch.cuda.current_device()
        pipe = VideoPipe(batch_size=batch_size, num_threads=cpu_count(), device_id=device_id,
                        data=video_folder_path, shuffle=shuffle)
        pipe.build()

        # Create the ROCALClassificationIterator
        data_loader = ROCALClassificationIterator(pipe, size=pipe.epoch_size("Reader"))

    else:
        raise ValueError(f"Invalid dataloader: {dataloader}")
    
    return train_loader, val_loader

def load_config(config_path):
    with open(config_path) as config_file:
        config = json.load(config_file)
    return config

def extract_features_labels(output,dataloader):
    if dataloader == "torchvision":
        features = output[0].to(device)
        labels = output[1].unsqueeze(1).to(device)

    elif dataloader == "dali":
        features = output[0]["data"].float().to(device)
        features = features/255 #normalize to 0-1
        features = features.permute(0,-1,1,2,3) #move channels to front
        labels = output[0]["label"].to(device)
        labels = labels.float()
    return features,labels

@torch.no_grad()
def estimate_loss(model,val_loader,criterion,dataloader):
    model.eval()

    val_correct = 0
    val_samples = 0
    val_loss = 0
    for batch_idx, output in enumerate(val_loader):
        features,labels = extract_features_labels(output,dataloader)
        outputs = model(features)
        if criterion == "CE":
            val_correct += calculate_accuracy(outputs,labels)
        elif criterion == "BCE":
            val_correct += calculate_accuracy_bce(outputs,labels)
        val_samples += labels.size(0)
        val_loss += criterion(outputs,labels).item()
    val_loss /= len(val_loader)
    val_accuracy = val_correct/val_samples
    return val_loss, val_accuracy

def train(config, dataloader="torchvision"):

    #hyperparameters
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_name = config["model_name"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    max_iters = config["max_iters"]
    eval_interval = config["eval_interval"]
    weight_decay = config["weight_decay"]
    std = tuple(config["std"]) #(0.2104, 0.1986, 0.1829)
    mean = tuple(config["mean"]) #(0.3939, 0.3817, 0.3314)
    use_autocast = config["use_autocast"]
    compile = config["compile"]
    criterion = config["criterion"]
    checkpoint = config["checkpoint"]

    print(f"Training model: {model_name} Using device: {device}")

    #create model
    valid_models = {"MoViNetA2":MoViNetA2,"MobileNetLarge3D":MobileNetLarge3D,"MobileNetSmall3D":MobileNetSmall3D}

    if model_name in valid_models:
        if criterion == "CE":
            model = valid_models[model_name](num_classes=2).to(device)
        elif criterion == "BCE":
            model = valid_models[model_name](num_classes=1).to(device)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    model.initialize_weights()

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    #create optimizer
    optimizer = optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)

    #create scheduler
    scheduler = CosineAnnealingLR(optimizer,T_max=max_iters)

    #create loss function
    valid_losses = {"CE":nn.CrossEntropyLoss(),"BCE":nn.BCEWithLogitsLoss()}
    if criterion in valid_losses:
        criterion = valid_losses[criterion]        

    else:
        raise ValueError(f"Invalid criterion: {criterion}")

    #create scaler for mixed precision training
    if use_autocast:
        scaler = GradScaler()

    #create tensorboard writer
    run_name = f"{model_name}_{criterion}"
    writer = SummaryWriter(f"runs/{run_name}")

    if checkpoint is not None:
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Loaded checkpoint at epoch {start_epoch}")

    #compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2 and a modern gpu (seems like mostly V/A/H 100s work best), these lines are straight from Karpathy 
        print("compilation complete!")
    
    #create dataloader
    train_loader, val_loader = create_dataloader(dataloader,batch_size,mean,std)

    
    #train the model

    #training loop
    start_time = time.time()
    print(f"Training... started: {time.ctime(start_time)}")
    train_losses = torch.tensor([])
    train_percent = torch.tensor([])
    val_losses = []
    val_percent = []

    try:
        for iter in range(max_iters):
            model.train()
            train_correct = 0
            train_samples = 0
            batch_loss_list = []
            batch_percent_list = []
            
            for batch_idx, output in tqdm(enumerate(train_loader)):
                features,labels = extract_features_labels(output,dataloader)

                optimizer.zero_grad(set_to_none=True)

                if use_autocast:
                    with autocast(dtype=dtype):
                        outputs = model(features)
                        loss = criterion(outputs,labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(features)
                    loss = criterion(outputs,labels)
                    loss.backward()
                    optimizer.step()

                #calculate accuracy
                if criterion == "CE":
                    train_correct += calculate_accuracy(outputs,labels)
                elif criterion == "BCE":
                    train_correct += calculate_accuracy_bce(outputs,labels)
                train_samples += labels.size(0)

                #append to lists
                batch_loss_list.append(loss.item())
                batch_percent_list.append(train_correct/train_samples)

                #write to tensorboard
                writer.add_scalar("training loss",loss.item(),(iter+1)*batch_idx)
                writer.add_scalar("training accuracy",train_correct/train_samples,(iter+1)*batch_idx)
                writer.flush()
        
            scheduler.step() #update learning rate
            train_losses = torch.cat((train_losses,average_for_plotting(batch_loss_list).unsqueeze(1)))
            train_percent = torch.cat((train_percent,average_for_plotting(batch_percent_list).unsqueeze(1)))
            elapsed_time = time.time() - start_time
            remaining_iters = max_iters - iter
            avg_time_per_iter = elapsed_time / (iter + 1)
            estimated_remaining_time = remaining_iters * avg_time_per_iter

            if iter % eval_interval == 0 or iter == max_iters - 1:
                val_loss, val_accuracy = estimate_loss(model,val_loader,criterion,dataloader=dataloader)
                val_losses.append(val_loss)
                val_percent.append(val_accuracy)

                print(f"Step {iter}: Train Loss: {train_losses[-1].mean().item():.4f}, Val Loss: {val_losses[-1]:.4f}")
                print(f"Step {iter}: Train Accuracy: {(train_percent[-1].mean().item())*100:.2f}%, Val Accuracy: {val_percent[-1]*100:.2f}%")
                writer.add_scalar('val loss', val_losses[-1], iter)
                writer.add_scalar('val accuracy', val_percent[-1], iter)
                torch.save(model.state_dict(), f'checkpoints/{model_name}{iter}.pth')

            tqdm.write(f"Iter [{iter+1}/{max_iters}] - Elapsed Time: {elapsed_time:.2f}s - Remaining Time: [{estimated_remaining_time:.2f}]")

            if iter == max_iters - 1:
                print("Training completed:")
                print(f"Final Train Loss: {train_losses[-1].mean().item():.4f}")
                print(f"Final Val Loss: {val_losses[-1]:.4f}")
                print(f"Final Train Accuracy: {(train_percent[-1].mean().item())*100:.2f}%")
                print(f"Final Val Accuracy: {val_percent[-1]*100:.2f}%")


            

    except KeyboardInterrupt:
        print(f"Keyboard interrupt,\nFinal Train Loss: {train_losses[-1].mean().item():.4f}")
        print(f"Final Val Loss: {val_losses[-1]:.4f}")
        print(f"Final Train Accuracy: {(train_percent[-1].mean().item())*100:.2f}%")
        print(f"Final Val Accuracy: {val_percent[-1]*100:.2f}%")
    finally:
        torch.save(model.state_dict(), f'checkpoints/{run_name}_finished.pth')
        with open(f'statistics/{run_name}_finished_train_losses.npy', 'wb') as f:
            np.save(f, train_losses.numpy())
        with open(f'statistics/{run_name}_finished_val_losses.npy', 'wb') as f:
            np.save(f, np.array(val_losses))
        with open(f'statistics/{run_name}_finished_train_percent.npy', 'wb') as f:
            np.save(f, train_percent.numpy())
        with open(f'statistics/{run_name}_finished_val_percent.npy', 'wb') as f:
            np.save(f, np.array(val_percent))
        print(f"Model and statistics saved!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a model with the specified config")
    parser.add_argument("--config","-c", type=str, required=True, help="Path to config file")
    parser.add_argument("--dataloader", "-d", type=str, required=False, help="Choose a dataloader from torchvision, dali, or rocal")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dataloader is not None:
        dataloader = args.dataloader
    else:
        dataloader = "torchvision"

    train(config,dataloader)
