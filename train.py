import time
import json
import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import cProfile
import pstats
from pstats import SortKey
from torch.utils.data import DataLoader
from dataloader import PicklebotDataset, custom_collate
from mobilenet import MobileNetSmall3D, MobileNetLarge3D
from movinet import MoViNetA2
from mobilevit import MobileViT
import bitsandbytes as bnb

def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return None, None, None

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return local_rank, rank, world_size

def state_dict_converter(state_dict):
    for key in list(state_dict.keys()):
        if key.startswith("_orig_mod."):
            new_key = key.replace("_orig_mod.", "")
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict

def create_dataloader(config: dict) -> tuple[DataLoader, DataLoader]:
    train_dataset = PicklebotDataset(
        config['train_annotations_file'],
        config['video_paths'],
        backend='opencv'
    )
    val_dataset = PicklebotDataset(
        config['val_annotations_file'],
        config['video_paths'],
        backend='opencv'
    )
    
    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.is_initialized() else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=custom_collate,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader, train_sampler

def get_average(loss_list:list, window_size:int=1000) -> list:
    partial_size = len(loss_list) % window_size
    if partial_size > 0:
        avg_losses = torch.tensor(loss_list[:-partial_size]).view(-1,1000).mean(1)
        avg_partial = torch.tensor(loss_list[-partial_size:]).view(-1,partial_size).mean(1)
        avg_losses = torch.cat((avg_losses, avg_partial))
    else:
        avg_losses = torch.tensor(loss_list).view(-1,1000).mean(1)
    return avg_losses

def load_config(config_path):
    with open(config_path) as config_file:
        config = json.load(config_file)
    return config

def extract_features_labels(output, device):
    features = output[0].to(device, non_blocking=True).permute(0,-1,1,2,3).to(torch.bfloat16)/255
    labels = output[1].unsqueeze(1).to(device, non_blocking=True)
    return features, labels

@torch.no_grad()
def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    correct = torch.sum(preds == labels)
    return correct

def calculate_accuracy_bce(outputs, labels, threshold=0.5):
    outputs = torch.sigmoid(outputs)
    preds = (outputs >= threshold).float().cpu()
    labels = labels.cpu()
    num_correct = torch.sum(preds == labels, dtype=torch.int64).item()
    return num_correct

@torch.no_grad()
def estimate_loss(model, val_loader, criterion, device, use_autocast, dtype):
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
        features, labels = extract_features_labels(output, device)
        if use_autocast:
            with autocast(dtype=dtype, device_type='cuda'):
                outputs = model(features)
                if str(criterion) == "CrossEntropyLoss()":
                    labels = labels.to(torch.long).squeeze(1)
                val_correct += accuracy_calc(outputs, labels)
                val_samples += labels.size(0)
                val_loss += criterion(outputs, labels).item()
        else:
            outputs = model(features)
            if str(criterion) == "CrossEntropyLoss()":
                labels = labels.to(torch.long).squeeze(1)
            val_correct += accuracy_calc(outputs, labels)
            val_samples += labels.size(0)
            val_loss += criterion(outputs, labels).item()
    val_loss /= len(val_loader)
    val_accuracy = val_correct/val_samples
    return val_loss, val_accuracy

def initialize_model(config: dict, device) -> nn.Module:
    valid_models = {
        "MoViNetA2": MoViNetA2,
        "MobileNetLarge3D": MobileNetLarge3D,
        "MobileNetSmall3D": MobileNetSmall3D,
        "MobileViT": MobileViT
    }

    if config['model_name'] not in valid_models:
        raise ValueError(f"Invalid model name: {config['model_name']}")

    if config['model_name'] == "MobileViT":
        model = valid_models[config['model_name']](
            dims=config['dims'],
            channels=config['channels'],
            num_classes=config['num_classes']
        ).to(device, non_blocking=True)
    else:
        model = valid_models[config['model_name']](
            num_classes=config['num_classes']
        ).to(device, non_blocking=True)

    model.initialize_weights()

    if config['compile']:
        print("Compiling the model... (takes a ~minute)")
        model = torch.compile(model)
        print("Compilation complete!")

    return model

def train(config):
    # Set up distributed training
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if local_rank is not None else 'cuda')
    is_main_process = rank in [0, None]  # True for rank 0 or single GPU
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if torch.cuda.is_available() and not config.get('varying_input_size', False):
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)
        torch.backends.cudnn.deterministic = True

    # Create model and move it to GPU with DDP
    model = initialize_model(config, device)
    if local_rank is not None:
        model = DDP(model, device_ids=[local_rank])

    accuracy_calc = calculate_accuracy

    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    eta_min = config['learning_rate']/10
    scheduler = CosineAnnealingLR(optimizer, T_max=config['max_iters'], eta_min=eta_min)

    valid_losses = {"CE": nn.CrossEntropyLoss(), "BCE": nn.BCEWithLogitsLoss()}
    if config['criterion'] in valid_losses:
        criterion = valid_losses[config['criterion']]
    else:
        raise ValueError(f"Invalid criterion: {config['criterion']}")

    if config['use_autocast']:
        scaler = GradScaler('cuda')

    if is_main_process:
        run_name = f"{config['model_name']}_{criterion}"
        writer = SummaryWriter(f"runs/{run_name}")

    if config['checkpoint'] is not None:
        print("Loading checkpoint...")
        checkpoint = torch.load(config['checkpoint'])
        model.load_state_dict(state_dict_converter(checkpoint))
        start_epoch = config["checkpoint"]
        print(f"Loaded checkpoint at epoch {start_epoch}")

    train_loader, val_loader, train_sampler = create_dataloader(config)
    
    start_time = time.time()
    print(f"Training... started: {time.ctime(start_time)}")
    train_losses = torch.tensor([])
    train_percent = torch.tensor([])
    val_losses = []
    val_percent = []
    assert config['effective_batch_size'] % config['batch_size'] == 0, "Batch size must divide effective batch size"
    grad_accum_steps = config['effective_batch_size'] // config['batch_size']
    print(f"Using {grad_accum_steps} gradient accumulation steps for a total effective batch size of {config['effective_batch_size']}")

    try:
        for iter in range(config['max_iters']):
            if train_sampler:
                train_sampler.set_epoch(iter)
                
            model.train()
            train_correct = 0
            train_samples = 0
            batch_loss_list = []
            batch_percent_list = []
            
            for batch_idx, output in enumerate(train_loader):
                features, labels = extract_features_labels(output, device)

                if config['use_autocast']:
                    with autocast('cuda', enabled=True, dtype=dtype):
                        outputs = model(features)
                        if str(criterion) == "CrossEntropyLoss()":
                            labels = labels.to(torch.long).squeeze(1)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                else:
                    outputs = model(features)
                    if str(criterion) == "CrossEntropyLoss()":
                        labels = labels.to(torch.long).squeeze(1)
                    loss = criterion(outputs, labels)
                    loss.backward()

                train_correct += accuracy_calc(outputs, labels)
                train_samples += labels.size(0)

                batch_loss_list.append(loss.item())
                batch_percent_list.append(train_correct/train_samples)

                if (batch_idx+1) % grad_accum_steps == 0:
                    if config['use_autocast']:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if is_main_process:
                    writer.add_scalar("training loss", loss.item(), (iter+1)*batch_idx)
                    writer.add_scalar("training accuracy", train_correct/train_samples, (iter+1)*batch_idx)
        
            scheduler.step()
            if is_main_process:
                train_losses = torch.cat((train_losses, get_average(batch_loss_list).unsqueeze(1)))
                train_percent = torch.cat((train_percent, get_average(batch_percent_list).unsqueeze(1)))
            
            elapsed_time = time.time() - start_time
            remaining_iters = config['max_iters'] - iter
            avg_time_per_iter = elapsed_time / (iter + 1)
            estimated_remaining_time = remaining_iters * avg_time_per_iter

            if iter % config['eval_interval'] == 0 or iter == config['max_iters'] - 1:
                val_loss, val_accuracy = estimate_loss(model, val_loader, criterion, device, config['use_autocast'], dtype)
                if is_main_process:
                    val_losses.append(val_loss)
                    val_percent.append(val_accuracy)
                    print(f"Step {iter}: Train Loss: {train_losses[-1].mean().item():.4f}, Val Loss: {val_losses[-1]:.4f}")
                    print(f"Step {iter}: Train Accuracy: {(train_percent[-1].mean().item())*100:.2f}%, Val Accuracy: {val_percent[-1]*100:.2f}%")
                    writer.add_scalar('val loss', val_losses[-1], iter)
                    writer.add_scalar('val accuracy', val_percent[-1], iter)
                    
                    # Save model on main process only
                    save_path = f"checkpoints/{config['model_name']}_{iter}.pth"
                    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                    torch.save(state_dict, save_path)

            if is_main_process:
                tqdm.write(f"Iter [{iter+1}/{config['max_iters']}] - Elapsed Time: {elapsed_time:.2f}s - Remaining Time: [{estimated_remaining_time:.2f}]")

            if iter == config['max_iters'] - 1 and is_main_process:
                print("Training completed:")
                print(f"Final Train Loss: {train_losses[-1].mean().item():.4f}")
                print(f"Final Val Loss: {val_losses[-1]:.4f}")
                print(f"Final Train Accuracy: {(train_percent[-1].mean().item())*100:.2f}%")
                print(f"Final Val Accuracy: {val_percent[-1]*100:.2f}%")

    except KeyboardInterrupt:
        if is_main_process:
            print(f"Keyboard interrupt,\nFinal Train Loss: {train_losses[-1].mean().item():.4f}")
            print(f"Final Val Loss: {val_losses[-1]:.4f}")
            print(f"Final Train Accuracy: {(train_percent[-1].mean().item())*100:.2f}%")
            print(f"Final Val Accuracy: {val_percent[-1]*100:.2f}%")
    finally:
        if is_main_process:
            torch.save(model.state_dict(), f'checkpoints/{run_name}_finished.pth')
            print(f"Model and statistics saved!")
        
        # Clean up distributed training
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    print(f"Using {dtype} on {device}")
    torch.set_float32_matmul_precision('high') if device == 'cuda' else torch.set_float32_matmul_precision('highest')
    print(f"using {torch.get_float32_matmul_precision()} precision")

    parser = argparse.ArgumentParser(description="Train a model with the specified config")
    parser.add_argument("config", type=str, help="Path to config file")
    args = parser.parse_args()
    config = load_config(args.config)
    
    def profile():
        train(config)
    
    profiler = cProfile.Profile()
    profiler.runcall(profile)
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.TIME)  # Sort by time
    stats.dump_stats('train_stats.prof')
