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
from visdom import Visdom
from dataloader import PicklebotDataset, custom_collate
from mobilenet import MobileNetLarge, MobileNetSmall, MobileNetSmallNoLSTM,MobileNetLargeNoLSTM


'''Balls are 0, strikes are 1'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#hyperparameters

learning_rate = 1e-5 #the paper quotes rmsprop with 0.1 lr, but we have a tiny batch size
batch_size = 2 #the paper quotes 128 images/chip, but our hardware isn't good enough
max_iters = 10
eval_interval = 1
weight_decay=0.0005
momentum=0.9
eps=np.sqrt(0.002) #From the pytorch blog post, "a reasonable approximation can be taken with the formula PyTorch_eps = sqrt(TF_eps)."

#annotations paths
train_annotations_file = '/home/henry/Documents/PythonProjects/picklebotdataset/train_labels.csv'
val_annotations_file = '/home/henry/Documents/PythonProjects/picklebotdataset/val_labels.csv'
test_annotations_file = '/home/henry/Documents/PythonProjects/picklebotdataset/test_labels.csv'

#video paths
train_video_paths = '/home/henry/Documents/PythonProjects/picklebotdataset/train'
val_video_paths = '/home/henry/Documents/PythonProjects/picklebotdataset/val'
test_video_paths = '/home/henry/Documents/PythonProjects/picklebotdataset/test'

#dataset     

transform = transforms.Normalize((0.5,), (0.5,))

train_dataset = PicklebotDataset(train_annotations_file,train_video_paths,transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,collate_fn=custom_collate,num_workers=cpu_count()//2)
val_dataset = PicklebotDataset(val_annotations_file,val_video_paths,transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=True,collate_fn=custom_collate,num_workers=cpu_count()//2)
test_dataset = PicklebotDataset(test_annotations_file,test_video_paths,transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True,collate_fn=custom_collate,num_workers=cpu_count()//2)


#model, optimizer, loss function
model = MobileNetSmallNoLSTM()
optimizer = optim.RMSprop(params=model.parameters(),lr=learning_rate,weight_decay=weight_decay,momentum=momentum,eps=eps) #starting with AdamW for now. 
#optimizer = optim.AdamW(params=model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=2)#ignore_index=0 was ignoring the label 0!
vis = Visdom()
scaler = GradScaler()
model_name = 'mobilenetsmallnolstm_test' 
model = model.to(device)

@torch.no_grad()
def estimate_loss():
    #evaluate the model
    model.eval()
    total_val_loss = 0.0
    num_val_batches = 0
    val_accuracy = []
    #calculate the loss
    for val_features,val_labels in val_loader:
        val_features = val_features.transpose(1,2).to(device)
        #val_labels = val_labels.to(torch.int64) #waiting to move to device until after forward pass, idk if this matters
        val_outputs = model(val_features)
        val_loss = criterion(val_outputs,val_labels.to(device))
        total_val_loss += val_loss.item()
        num_val_batches += 1        
        val_correct += calculate_accuracy(val_outputs,val_labels)
    avg_val_loss = total_val_loss / num_val_batches
    val_accuracy = val_correct / len(val_dataset)
    return avg_val_loss, val_accuracy

def calculate_accuracy(outputs,labels):
    predicted_classes = torch.argmax(outputs,dim=1).to(labels.device)
    num_correct = torch.sum(predicted_classes == labels).item()
    return num_correct

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
            #transpose, our convolutions like channel to be index 1, so [batch,channel,time,height,width]
            features = features.to(device) 
            
            #zero the gradients
            optimizer.zero_grad(set_to_none=True)
            
            # with autocast():
            outputs = model(features)
            loss = criterion(outputs,labels.to(device))

            train_correct.append(calculate_accuracy(outputs,labels))
            train_losses.append(loss.item())   

            loss.backward()

            
            optimizer.step()



            if batch_idx %  1797 == 0:

                print(f" Batch {batch_idx}: loss: {loss:.4f} accuracy: {sum(train_correct)/(len(train_correct)*batch_size)*100:.2f}%")
                vis.line(
                    X=np.arange(batch_idx+1),
                    Y=np.array(train_losses),
                    win='Loss Plot',
                    opts=dict(
                        title='Train Losses',
                        legend=['Training Loss'],
                        xlabel='Batch Number',
                        ylabel='Loss',
                    )
                )

        if iter % eval_interval == 0 or iter == max_iters - 1:
            #evaluate the model, call the estimate_loss function
            val_loss, val_accuracy = estimate_loss()
        
            val_losses.append(val_loss)

            train_percent.append(sum(train_correct)/len(train_correct)*batch_size)
            val_percent.append(sum(val_accuracy)/len(val_accuracy)*batch_size)


            vis.text(f"step {iter}: train loss:  {loss:.4f}, val loss: {val_loss:.4f}",win='loss_text')
            vis.line(
                X=np.arange(iter+1),
                Y=np.concatenate((np.array(train_losses).reshape(-1, 1), np.array(val_losses).reshape(-1, 1)), axis=1),
                win='Loss Plot',
                opts=dict(
                    title='Train and Validation Losses',
                    legend=['Training Loss', 'Validation Loss'],
                    xlabel='Iteration',
                    ylabel='Loss',
                    linecolor=np.array([[0, 0, 255], [255, 0, 0]]),
                )
            )
            vis.text(f"step {iter}: train accuracy:  {train_percent[-1]*100:.2f}%, test accuracy: {val_percent[-1]*100:.2f}%",win='accuracy_text')
            vis.line(
                X=np.arange(iter+1),
                Y=np.concatenate((np.array(train_percent).reshape(-1, 1), np.array(val_percent).reshape(-1, 1)), axis=1),
                win='Accuracy Plot',
                opts=dict(
                    title='Train and Validation Accuracy',
                    legend=['Training Accuracy', 'Validation Accuracy'],
                    xlabel='Iteration',
                    ylabel='Accuracy',
                    linecolor=np.array([[0, 0, 255], [255, 0, 0]]),
                )
            )

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
    print(f"Keyboard interrupt:\nFinal loss: {loss:.4f},\nFinal train loss: {train_losses[-1]:.4f}, ")
    print(f"Final val loss: {val_losses[-1]:.4f}, ")
    print(f"Final train accuracy: {train_percent[-1]*100:.2f}%, ")
    print(f"Final val accuracy: {val_percent[-1]*100:.2f}%")

finally:
    torch.save(model.state_dict(), f'{model_name}.pth')
    print(f"Model saved!") 
    
