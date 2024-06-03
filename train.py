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
from decimal import Decimal
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from psutil import cpu_count
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from mobilenet import MobileNetSmall3D,MobileNetLarge3D
from movinet import MoViNetA2
from mobilevit import MobileViT
from helpers import calculate_accuracy_bce, average_for_plotting, calculate_accuracy
#classification 0 is zone 1, classification 1 is zone 2, etc.


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if device == 'cuda' else torch.float32

def state_dict_converter(state_dict):
    for key in list(state_dict.keys()):
        if key.startswith("_orig_mod."):
            new_key = key.replace("_orig_mod.", "")
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict

def create_dataloader(dataloader,batch_size,mean,std,train_annotations_file,val_annotations_file,video_paths):
    #create dataloader
    if dataloader == "torchvision":
        from torch.utils.data import DataLoader
        from dataloader import PicklebotDataset, custom_collate
        #from torchvision import transforms may want to put this back in one day
        

        #establish our normalization using transforms, 
        #note that we are doing this in our dataloader as opposed to in the training loop like with dali
        #transform = transforms.Normalize(mean,std)

        #dataset     
        train_dataset = PicklebotDataset(train_annotations_file,video_paths,dtype=dtype,backend='opencv') #may want to add transform=transform back
        train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=False,collate_fn=custom_collate,num_workers=16,pin_memory=True) 
        val_dataset = PicklebotDataset(val_annotations_file,video_paths,dtype=dtype,backend='opencv') #may want to add transform=transform back
        val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=False,collate_fn=custom_collate,num_workers=16,pin_memory=True)


    elif dataloader == "dali":
        from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
        from helpers import dali_video_pipeline

        #information for the dali pipeline
        sequence_length = 130 #longest videos in our dataset 
        initial_prefetch_size = 20


        #video paths
        train_video_paths = '/home/henry/Documents/PythonProjects/picklebotdataset/train'
        val_video_paths = '/home/henry/Documents/PythonProjects/picklebotdataset/val'

        num_train_videos = len(os.listdir(train_video_paths + '/' + 'balls')) + len(os.listdir(train_video_paths + '/' + 'strikes'))
        num_val_videos = len(os.listdir(val_video_paths + '/' + 'balls')) + len(os.listdir(val_video_paths + '/' + 'strikes'))

        #multiply mean and val by 255 to convert to 0-255 range
        mean = (torch.tensor(mean)*255)[None,None,None,:]
        std = (torch.tensor(std)*255)[None,None,None,:]

        print("Building DALI pipelines...")

        #build our pipelines
        train_pipe = dali_video_pipeline(batch_size=batch_size, num_threads=cpu_count()//2, device_id=0, file_root=train_video_paths,
                                    sequence_length=sequence_length,initial_prefetch_size=initial_prefetch_size,mean=mean*255,std=std*255)
        val_pipe = dali_video_pipeline(batch_size=batch_size, num_threads=cpu_count()//2, device_id=0, file_root=val_video_paths,
                                sequence_length=sequence_length,initial_prefetch_size=initial_prefetch_size,mean=mean,std=std)

        train_pipe.build()
        val_pipe.build()


        train_loader = DALIClassificationIterator(train_pipe, auto_reset=True,last_batch_policy=LastBatchPolicy.PARTIAL, size=num_train_videos)
        val_loader = DALIClassificationIterator(val_pipe, auto_reset=True,last_batch_policy=LastBatchPolicy.PARTIAL, size=num_val_videos)

        

    elif dataloader == "rocal":
        from amd.rocal.plugin.pytorch import ROCALClassificationIterator
        from helpers import rocal_video_pipeline

        #information for the dali pipeline
        sequence_length = 130 #longest videos in our dataset 
        initial_prefetch_size = 20


        #video paths
        train_video_paths = '/home/henry/Documents/PythonProjects/picklebotdataset/train'
        val_video_paths = '/home/henry/Documents/PythonProjects/picklebotdataset/val'

        num_train_videos = len(os.listdir(train_video_paths + '/' + 'balls')) + len(os.listdir(train_video_paths + '/' + 'strikes'))
        num_val_videos = len(os.listdir(val_video_paths + '/' + 'balls')) + len(os.listdir(val_video_paths + '/' + 'strikes'))

        #multiply mean and val by 255 to convert to 0-255 range
        mean = (torch.tensor(mean)*255)[None,None,None,:]
        std = (torch.tensor(std)*255)[None,None,None,:]

        print("Building rocAL pipelines...")

        #build our pipelines
        train_pipe = rocal_video_pipeline(batch_size=batch_size, num_threads=cpu_count()//2, device_id=0, file_root=train_video_paths,
                                    sequence_length=sequence_length,initial_prefetch_size=initial_prefetch_size,mean=mean*255,std=std*255)
        val_pipe = rocal_video_pipeline(batch_size=batch_size, num_threads=cpu_count()//2, device_id=0, file_root=val_video_paths,
                                sequence_length=sequence_length,initial_prefetch_size=initial_prefetch_size,mean=mean,std=std)

        train_pipe.build()
        val_pipe.build()


        train_loader = ROCALClassificationIterator(train_pipe, auto_reset=True, size=num_train_videos)
        val_loader = ROCALClassificationIterator(val_pipe, auto_reset=True, size=num_val_videos)
    return train_loader, val_loader


def load_config(config_path):
    with open(config_path) as config_file:
        config = json.load(config_file)
    return config

def extract_features_labels(output,dataloader):
    if dataloader == "torchvision":
        features = output[0].to(device,non_blocking=True).permute(0,-1,1,2,3).to(torch.bfloat16) / 255 #doing these operations on the gpu makes loading 2x faster, investigating if we're better off doing this on the cpu to keep the gpu working on the model
        labels = output[1].unsqueeze(1).to(device,non_blocking=True)

    elif dataloader == "dali":
        features = output[0]["data"].float().to(device,non_blocking=True)
        features = features/255 #normalize to 0-1
        features = features.permute(0,-1,1,2,3) #move channels to front
        labels = output[0]["label"].to(device,non_blocking=True)
        labels = labels.float()
    return features,labels

@torch.no_grad()
def estimate_loss(model,val_loader,criterion,dataloader,use_autocast):
    print("Evaluating...")
    model.eval()
    if str(criterion) == "CrossEntropyLoss()":
        accuracy_calc = calculate_accuracy
    elif str(criterion) == "BCEWithLogitsLoss()":
        accuracy_calc = calculate_accuracy_bce
    val_correct = 0
    val_samples = 0
    val_loss = 0
    for output in tqdm(val_loader):
        features,labels = extract_features_labels(output,dataloader)
        if use_autocast:
            with autocast(dtype=dtype):
                outputs = model(features)
                if str(criterion) == "CrossEntropyLoss()":
                    labels = labels.to(torch.long).squeeze(1)
                val_correct += accuracy_calc(outputs,labels)
                val_samples += labels.size(0)
                val_loss += criterion(outputs,labels).item()
        else:
            outputs = model(features)
            if str(criterion) == "CrossEntropyLoss()":
                labels = labels.to(torch.long).squeeze(1)
            val_correct += accuracy_calc(outputs,labels)
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
    train_annotations_file = config["train_annotations_file"]
    val_annotations_file = config["val_annotations_file"]
    video_paths = config["video_paths"]
    num_classes = config["num_classes"]

    print(f"Training model: {model_name} Using device: {device} with dtype: {dtype}")

    #create model
    valid_models = {"MoViNetA2":MoViNetA2,"MobileNetLarge3D":MobileNetLarge3D,"MobileNetSmall3D":MobileNetSmall3D,"MobileViT":MobileViT}

    if model_name in valid_models:
        if model_name == "MobileViT":
            dims = config["dims"]
            channels = config["channels"]
            model = valid_models[model_name](dims=dims,channels=channels,num_classes=num_classes).to(device,non_blocking=True)
            accuracy_calc = calculate_accuracy
        else:
            model = valid_models[model_name](num_classes=13).to(device,non_blocking=True)
            accuracy_calc = calculate_accuracy
        
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    model.initialize_weights()

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    #create optimizer
    optimizer = optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)

    #create scheduler
    eta_min = float(Decimal(str(learning_rate))/Decimal('10')) #lr/10
    scheduler = CosineAnnealingLR(optimizer,T_max=max_iters,eta_min=eta_min)

    #create loss function
    valid_losses = {"CE":nn.CrossEntropyLoss(),"BCE":nn.BCEWithLogitsLoss()}
    if criterion in valid_losses:
        criterion = valid_losses[criterion]        

    else:
        raise ValueError(f"Invalid criterion: {criterion}")

    #create scaler for mixed precision training
    if use_autocast:
        scaler = GradScaler('cuda')

    #create tensorboard writer
    run_name = f"{model_name}_{criterion}"
    writer = SummaryWriter(f"runs/{run_name}")

    if checkpoint is not None:
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(state_dict_converter(checkpoint))
        start_epoch = config["checkpoint"]
        print(f"Loaded checkpoint at epoch {start_epoch}")

    #create dataloader
    train_loader, val_loader = create_dataloader(dataloader,batch_size,mean,std,train_annotations_file,val_annotations_file,video_paths)
    
    #compile the model
    if compile:
        print("Compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2 and a modern gpu (seems like mostly V/A/H 100s work best, but it absolutely speeds up my 7900xtx)
        print("Compilation complete!")
    

    
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
                    with autocast('cuda',enabled=True,dtype=dtype):
                        outputs = model(features)
                        if str(criterion) == "CrossEntropyLoss()":
                            labels = labels.to(torch.long).squeeze(1)
                        loss = criterion(outputs,labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(features)
                    if str(criterion) == "CrossEntropyLoss()":
                        labels = labels.to(torch.long).squeeze(1)
                    loss = criterion(outputs,labels)
                    loss.backward()
                    optimizer.step()


                train_correct += accuracy_calc(outputs,labels)
                train_samples += labels.size(0)

                #append to lists
                batch_loss_list.append(loss.item())
                batch_percent_list.append(train_correct/train_samples)

                #write to tensorboard
                writer.add_scalar("training loss",loss.item(),(iter+1)*batch_idx)
                writer.add_scalar("training accuracy",train_correct/train_samples,(iter+1)*batch_idx)
        
            scheduler.step() #update learning rate
            train_losses = torch.cat((train_losses,average_for_plotting(batch_loss_list).unsqueeze(1)))
            train_percent = torch.cat((train_percent,average_for_plotting(batch_percent_list).unsqueeze(1)))
            elapsed_time = time.time() - start_time
            remaining_iters = max_iters - iter
            avg_time_per_iter = elapsed_time / (iter + 1)
            estimated_remaining_time = remaining_iters * avg_time_per_iter

            if iter % eval_interval == 0 or iter == max_iters - 1:
                val_loss, val_accuracy = estimate_loss(model,val_loader,criterion,use_autocast=use_autocast,dataloader=dataloader)
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
            np.save(f, train_losses.cpu().numpy())
        with open(f'statistics/{run_name}_finished_val_losses.npy', 'wb') as f:
            np.save(f, np.array(val_losses))
        with open(f'statistics/{run_name}_finished_train_percent.npy', 'wb') as f:
            np.save(f, train_percent.cpu().numpy())
        with open(f'statistics/{run_name}_finished_val_percent.npy', 'wb') as f:
            np.save(f, (val_percent))
        print(f"Model and statistics saved!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a model with the specified config")
    parser.add_argument("--config","-C", type=str, required=True, help="Path to config file")
    parser.add_argument("--dataloader", "-D", type=str, required=False, help="Choose a dataloader from torchvision, dali, or rocal")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dataloader is not None:
        dataloader = args.dataloader
    else:
        dataloader = "torchvision"
    
    def profile():
        train(config,dataloader)

    import cProfile
    profiler = cProfile.Profile()
    profiler.runcall(profile)

    import pstats
    from pstats import SortKey

    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.TIME)  # Sort by time
    stats.dump_stats('train_stats.prof')
