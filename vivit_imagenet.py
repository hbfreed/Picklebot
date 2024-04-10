from mobilevitv2 import MobileViTV2
from mobilevit import MobileViTV1
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
torch.multiprocessing.set_sharing_strategy('file_system')

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the images to a consistent size
    transforms.ToTensor(),  # Convert the images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image tensors
])

# Load the ImageNet-1K dataset from cache
dataset = load_dataset("imagenet-1k", split="train",trust_remote_code=True)
val_dataset = load_dataset("imagenet-1k", split="validation",trust_remote_code=True)

# Apply the transformations to the datasets
def transform_images(examples):
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
    return examples

dataset.set_transform(transform_images)
val_dataset.set_transform(transform_images)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileViTV1(
    image_size=(256,256),
    dims=[96,120,144],
    channels=[16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
    num_classes=1000
    ).to(device)
model = model.to(torch.bfloat16)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
num_epochs = 50 

scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

# Create data loaders
batch_size = 128
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x,num_workers=24,pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=lambda x: x,num_workers=24,pin_memory=True)
print('starting training...')
# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader):
        images = torch.stack([item["pixel_values"] for item in batch]).to(device).to(torch.bfloat16)
        labels = torch.tensor([item["label"] for item in batch]).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    scheduler.step()

    train_loss /= len(train_loader.dataset)

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            images = torch.stack([item["pixel_values"] for item in batch]).to(device).to(torch.bfloat16)
            labels = torch.tensor([item["label"] for item in batch]).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_accuracy += torch.sum(predicted == labels).item()

    val_loss /= len(val_loader.dataset)
    val_accuracy /= len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
