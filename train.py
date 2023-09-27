import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
from psutil import cpu_count
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataloader import PicklebotDataset, custom_collate
from mobilenet import MobileNetLarge2D, MobileNetSmall2D, MobileNetSmall3D, MobileNetLarge3D
from helpers import calculate_accuracy, initialize_mobilenetv3_weights

'''Balls are 0, strikes are 1'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#hyperparameters

learning_rate = 3e-4 #the paper quotes rmsprop with 0.1 lr, but we have a tiny batch size
batch_size = 2 #the paper quotes 128 images/chip, but our hardware isn't good enough
max_iters = 20
eval_interval = 5
weight_decay=0.0005
momentum=0.9
eps=np.sqrt(0.002) #From the pytorch blog post, "a reasonable approximation can be taken with the formula PyTorch_eps = sqrt(TF_eps)."

#annotations paths
train_annotations_file = '/home/henry/Documents/PythonProjects/picklebotdataset/train_small_files.csv' #NEED TO CHANGE THIS TO GO BACK TO REAL TRAIN AND TEST'''
val_annotations_file = '/home/henry/Documents/PythonProjects/picklebotdataset/val_small_files.csv' #NEED TO CHANGE THIS TO GO BACK TO REAL TRAIN AND TEST'''
test_annotations_file = '/home/henry/Documents/PythonProjects/picklebotdataset/test_labels.csv'

#video paths
train_video_paths = '/home/henry/Documents/PythonProjects/picklebotdataset/train'
val_video_paths = '/home/henry/Documents/PythonProjects/picklebotdataset/val'
test_video_paths = '/home/henry/Documents/PythonProjects/picklebotdataset/test'

#dataset     

transform = transforms.Normalize((0.5,), (0.5,))

train_dataset = PicklebotDataset(train_annotations_file,train_video_paths,transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,collate_fn=custom_collate,num_workers=cpu_count())
val_dataset = PicklebotDataset(val_annotations_file,val_video_paths,transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=True,collate_fn=custom_collate,num_workers=cpu_count())
test_dataset = PicklebotDataset(test_annotations_file,test_video_paths,transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True,collate_fn=custom_collate,num_workers=cpu_count())

#model, optimizer, loss function
model = MobileNetLarge2D(num_classes=2)

#initialize the weights
initialize_mobilenetv3_weights(model)

#optimizer = optim.RMSprop(params=model.parameters(),lr=learning_rate,weight_decay=weight_decay,momentum=momentum,eps=eps) #starting with AdamW for now. 
optimizer = optim.AdamW(params=model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=2)#ignore_index=0 was ignoring the label 0!
model_name = 'mobilenetlarge2d_overfit'
model = model.to(device)
# model.load_state_dict(torch.load(f'{model_name}.pth')) #if applicable, load the model from the last checkpoint
writer = SummaryWriter(f'runs/{model_name}')


@torch.no_grad()
def estimate_loss():
    #evaluate the model
    model.eval()
    total_val_loss = 0.0
    num_val_batches = 0
    val_correct = 0
    val_accuracy = []
    #calculate the loss
    for val_features,val_labels in tqdm(val_loader):
        val_features = val_features.to(device)
        val_labels = val_labels.to(torch.int64) #waiting to move to device until after forward pass, idk if this matters, but i imagine it could save gpu memory
        val_labels = val_labels.expand(val_features.shape[2]) #this is only for our lstm T -> batch size, a lame hack
        val_outputs = model(val_features)
        val_loss = criterion(val_outputs,val_labels.to(device))
        total_val_loss += val_loss.item()
        num_val_batches += 1        
        val_correct += calculate_accuracy(val_outputs,val_labels)
    avg_val_loss = total_val_loss / num_val_batches
    val_accuracy = val_correct / len(val_dataset)
    return avg_val_loss, val_accuracy

#try except block so we can manually early stop while saving the model
#training loop
start_time = time.time()
train_losses = []
val_losses = []
train_correct = []
train_percent = []
val_percent = []

try:
    for iter in range(max_iters):
        model.train()
        #forward pass
        for batch_idx, (features,labels) in tqdm(enumerate(train_loader)):
            
            labels = labels.to(torch.int64)
            labels = labels.expand(features.shape[2]) #this is a hack to make the labels the same shape as the outputs when we're using LSTM, so we can calculate the loss, but is lame.
            
            features = features.to(device) 
            print(features.shape) 
            #zero the gradients
            optimizer.zero_grad(set_to_none=True)
            
            outputs = model(features)
            loss = criterion(outputs,labels.to(device))
    
            train_correct.append(calculate_accuracy(outputs,labels))
            train_losses.append(loss.item())   

            writer.add_scalar('training loss', loss.item(), batch_idx + iter*len(train_loader))

            loss.backward()
            optimizer.step()
            

        if iter % eval_interval == 0 or iter == max_iters - 1:
            #evaluate the model, call the estimate_loss function
            val_loss, val_accuracy = estimate_loss()
        
            val_losses.append(val_loss)
            writer.add_scalar('validation loss', val_losses[-1], iter)
            train_percent.append(sum(train_correct)/len(train_correct)*batch_size)
            writer.add_scalar('training accuracy', train_percent[-1], iter)
            val_percent.append(val_accuracy)
            writer.add_scalar('validation accuracy', val_percent[-1], iter)

        elapsed = time.time() - start_time
        remaining_iters = max_iters - iter
        avg_time_per_iter = elapsed / (iter + 1)
        estimated_remaining_time = remaining_iters * avg_time_per_iter
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
    
