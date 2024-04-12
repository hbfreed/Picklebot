from mobilevit import MobileViT
from mobilevitv2 import MobileViTV2
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
torch.multiprocessing.set_sharing_strategy('file_system')
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16
model = MobileViT(
    image_size=(256,256),
    dims=[96,120,144],
    channels=[16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384], #this is "MobileViT-S", the largest mobilevit model (lol)
    num_classes=1000
    ).to(device)
# model = MobileViTV2().to(device)
#print the number of parameters in the model
print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())}")
model = model.to(torch.bfloat16)
model = torch.compile(model)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
num_epochs = 50 
batch_size = 256
num_threads = 48
crop_size = 256

scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

def custom_collate(batch, memory_format):
    """Based on fast_collate from the APEX example
       https://github.com/NVIDIA/apex/blob/5b5d41034b506591a316c308c3d2cd14d5187e23/examples/imagenet/main_amp.py#L265
    """
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets

# Load the data
traindir = '/home/henry/Documents/imagenet/train/'
torch_dataset = datasets.ImageFolder(traindir, transforms.Compose([transforms.RandomResizedCrop(crop_size),transforms.RandomHorizontalFlip()]))
train_loader = torch.utils.data.DataLoader(torch_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_threads,
                                            pin_memory=True,
                                            collate_fn= lambda b: custom_collate(b,torch.channels_last),
                                            prefetch_factor=5,)
print('starting training...')
# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader):
        images = batch[0].to(device).to(torch.bfloat16)
        labels = batch[1].squeeze(-1).long().to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    scheduler.step()

    train_loss /= len(train_loader.dataset)

    # # Validation loop
    # model.eval()
    # val_loss = 0.0
    # val_accuracy = 0.0
    # with torch.no_grad():
    #     for batch in tqdm(val_loader):
    #         images = batch[0]["data"].to(device).to(torch.bfloat16)
    #         labels = batch[0]["label"].squeeze(-1).long().to(device)

    #         outputs = model(images)
    #         loss = criterion(outputs, labels)

    #         val_loss += loss.item() * images.size(0)
    #         _, predicted = torch.max(outputs, 1)
    #         val_accuracy += torch.sum(predicted == labels).item()

    # val_loss /= len(val_loader.dataset)
    # val_accuracy /= len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")#, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
