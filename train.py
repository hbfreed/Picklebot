'''The big difference between this file and cloud_train is that this file uses tensorboardx 
instead of plotting in the notebook'''
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from psutil import cpu_count
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataloader import PicklebotDataset, custom_collate
from mobilenet import MobileNetLarge2D, MobileNetSmall2D, MobileNetSmall3D, MobileNetLarge3D
from helpers import calculate_accuracy 

'''strikes are 0, balls are 1 since we pad with 0s and cross entropy loss has to ignore something.'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#hyperparameters
learning_rate = 1e-5 #the paper quotes rmsprop with 0.1 lr, but we have a tiny batch size, and are using AdamW
batch_size = 4 #the paper quotes 128 images/chip, but with video we have to change this
max_iters = 1000
eval_interval = 2
weight_decay = 0.0005
momentum = 0.9
eps = np.sqrt(0.002) #From the pytorch blog post, "a reasonable approximation can be taken with the formula PyTorch_eps = sqrt(TF_eps)."
std = (0.2104, 0.1986, 0.1829)
mean = (0.3939, 0.3817, 0.3314)
use_autocast = False
compile = False

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
train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,collate_fn=custom_collate,num_workers=cpu_count())
val_dataset = PicklebotDataset(val_annotations_file,val_video_paths,transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=True,collate_fn=custom_collate,num_workers=cpu_count())

#define model, initialize weights 
model = MobileNetSmall3D()
model.initialize_weights()
model = model.to(device)

#for multi-gpu
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# optimizer, loss function
# optimizer = optim.RMSprop(params=model.parameters(),lr=learning_rate,weight_decay=weight_decay,momentum=momentum,eps=eps) #starting with AdamW for now. 
optimizer = optim.AdamW(params=model.parameters(),lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
if use_autocast:
    scaler = GradScaler()
model_name = 'mobilenetsmall_3D_local' 
# model.load_state_dict(torch.load(f'{model_name}.pth')) #if applicable, load the model from the last checkpoint
writer = SummaryWriter(f'runs/{model_name}')

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2 and a modern gpu, these lines were lifted from karpathy
    print("compilation complete!")


#estimate loss using the val set, and calculate accuracy
@torch.no_grad()
def estimate_loss():
    #evaluate the model
    model.eval()
    val_losses = [] 
    val_correct = 0
    val_samples = 0

    #calculate the loss
    for val_features,val_labels in val_loader:
        val_features = val_features.to(device)
        val_labels = val_labels.to(torch.int64) #waiting to move to device until after forward pass, idk if this matters
        # val_labels = val_labels.expand(val_features.shape[2]) #this is only for our lstm T -> batch size, a lame hack
        
        val_outputs = model(val_features)
        
        val_loss = criterion(val_outputs,val_labels.to(device))
        
        val_losses.append(val_loss.item())
        
        val_correct += calculate_accuracy(val_outputs,val_labels)
        val_samples += len(val_labels)

    avg_val_loss = np.mean(val_losses)
    val_accuracy = val_correct / val_samples
    return avg_val_loss, val_accuracy

#try except block so we can manually early stop while saving the model
#training loop
start_time = time.time()
train_losses = []
train_percent = []
val_losses = []
val_percent = []

try:
    for iter in range(max_iters):
        
        model.train()
        train_correct = 0
        train_samples = 0
        batch_loss_list = []

        #forward pass
        for batch_idx, (features,labels) in tqdm(enumerate(train_loader)):
            
            labels = labels.to(torch.int64)
            # labels = labels.expand(features.shape[2]) #this is a hack to make the labels the same shape as the outputs when we're using LSTM, so we can calculate the loss, but is lame.
            features = features.to(device)

            #zero the gradients
            optimizer.zero_grad(set_to_none=True)

            if use_autocast:    
                with autocast():
                    outputs = model(features)
                    loss = criterion(outputs,labels.to(device))
                
                #backprop & update weights

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                outputs = model(features)
                loss = criterion(outputs,labels.to(device))

                #backprop & update weights
                loss.backward()
                optimizer.step()

            batch_loss_list.append(loss.item()) #append the loss of the batch to our list to be averaged and plotted later, this is dataset size / batch size long
            batch_correct = calculate_accuracy(outputs,labels) #number of correct predictions in the batch
            train_correct += batch_correct #this is the total number of correct predictions so far
            train_samples += len(labels) #this is the total number of samples so far
            writer.add_scalar('training loss', batch_loss_list[-1], iter*len(train_loader) + batch_idx)

        train_losses.append(np.mean(batch_loss_list))
        train_percent.append(train_correct / train_samples)
        elapsed = time.time() - start_time
        remaining_iters = max_iters - iter
        avg_time_per_iter = elapsed / (iter + 1)
        estimated_remaining_time = remaining_iters * avg_time_per_iter


        if iter % eval_interval == 0 or iter == max_iters - 1:
            #evaluate the model, call the estimate_loss function
            val_loss, val_accuracy = estimate_loss()
        
            val_losses.append(val_loss)
            writer.add_scalar('validation loss', val_losses[-1], iter)
            train_percent.append(sum(train_correct)/len(train_correct)*batch_size)
            writer.add_scalar('training accuracy', train_percent[-1], iter)
            val_percent.append(val_accuracy)
            writer.add_scalar('validation accuracy', val_percent[-1], iter)

        tqdm.write(f"Iter [{iter+1}/{max_iters}] - Elapsed Time: {elapsed:.2f}s  Remaining Time: [{estimated_remaining_time:.2f}]")

        if iter == max_iters -1:
            print("Training completed:") 
            print(f"Final loss: {train_losses[-1]:.4f},")
            print(f"Final val loss: {val_losses[-1]:.4f}, ")
            print(f"Final train accuracy: {train_percent[-1]*100:.2f}%, ")
            print(f"Final val accuracy: {val_percent[-1]*100:.2f}%") 
            
except KeyboardInterrupt:
    print(f"Keyboard interrupt:\nFinal train loss: {train_losses[-1]:.4f}, ")
    print(f"Final val loss: {val_losses[-1]:.4f}, ")
    print(f"Final train accuracy: {train_percent[-1]*100:.2f}%, ")
    print(f"Final val accuracy: {val_percent[-1]*100:.2f}%")

finally:
    torch.save(model.state_dict(), f'{model_name}.pth')
    with open(f'{model_name}_train_losses.npy', 'wb') as f:
        np.save(f, np.array(train_losses))
    with open(f'{model_name}_val_losses.npy', 'wb') as f:
        np.save(f, np.array(val_losses))
    with open(f'{model_name}_train_percent.npy', 'wb') as f:
        np.save(f, np.array(train_percent))
    with open(f'{model_name}_val_percent.npy', 'wb') as f:
        np.save(f, np.array(val_percent))
    print(f"Model saved!") 
    
