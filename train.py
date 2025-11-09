#!/usr/bin/env python3
"""
Offline training pipeline for TronNet policy.
Loads episode .npz files and performs supervised imitation learning with safe-masked cross-entropy loss.
"""

import os
import argparse
import glob
import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set CPU threads
torch.set_num_threads(max(1, os.cpu_count() // 2))


class TronNet(nn.Module):
    """Neural network for Tron policy learning.
    
    Input: (batch_size, 6, 18, 20) state tensor
    Output: (batch_size, 4) logits for UP, DOWN, LEFT, RIGHT
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 18 * 20, 128), nn.ReLU(),
            nn.Linear(128, 4)  # UP, DOWN, LEFT, RIGHT
        )
    
    def forward(self, x):
        return self.net(x)


class EpisodeDataset(Dataset):
    """Dataset for loading episode .npz files.
    
    Each sample is a single tick from an episode:
    - x: state tensor (6, 18, 20)
    - y: action label (0-3)
    - safe: safety mask (4,) indicating which moves are safe
    """
    def __init__(self, episode_files: List[str], filter_unsafe: bool = True):
        """
        Args:
            episode_files: List of paths to .npz episode files
            filter_unsafe: If True, skip ticks where safe.sum() == 0
        """
        self.episode_files = episode_files
        self.filter_unsafe = filter_unsafe
        self.samples = []
        
        # Load all episodes and extract samples
        for ep_file in episode_files:
            try:
                data = np.load(ep_file, allow_pickle=False)
                states = data['states']  # (T, 6, 18, 20)
                actions = data['actions']  # (T,)
                safe = data['safe']  # (T, 4)
                
                T = len(states)
                for t in range(T):
                    # Filter out ticks where no moves are safe
                    if self.filter_unsafe and safe[t].sum() == 0:
                        continue
                    
                    self.samples.append({
                        'state': states[t].astype(np.float32),
                        'action': int(actions[t]),
                        'safe': safe[t].astype(np.uint8)
                    })
            except Exception as e:
                # Skip corrupted files
                continue
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'x': torch.from_numpy(sample['state']),
            'y': torch.tensor(sample['action'], dtype=torch.long),
            'safe': torch.from_numpy(sample['safe'])
        }


def masked_cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor, 
                              safe_mask: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Compute cross-entropy loss with safety masking.
    
    Args:
        logits: (N, 4) logits from model
        targets: (N,) target action indices
        safe_mask: (N, 4) binary mask indicating safe moves (1=safe, 0=unsafe)
    
    Returns:
        loss: Scalar loss value
        valid_samples: Number of samples used in loss
        total_samples: Total number of samples
    """
    N = logits.shape[0]
    total_samples = N
    
    # Mask unsafe moves by setting their logits to large negative value
    logits_masked = logits.clone()
    logits_masked[safe_mask == 0] = -1e9
    
    # Check if target action is safe for each sample
    batch_indices = torch.arange(N, device=logits.device)
    target_safe = safe_mask[batch_indices, targets] == 1
    
    # Set target to -1 (ignore_index) for unsafe targets
    targets_masked = targets.clone()
    targets_masked[~target_safe] = -1
    
    # Compute cross-entropy loss (ignores targets with value -1)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    loss = criterion(logits_masked, targets_masked)
    
    valid_samples = target_safe.sum().item()
    
    return loss, valid_samples, total_samples


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, 
                    safe_mask: torch.Tensor) -> Tuple[float, int, int]:
    """Compute accuracy on safe-masked logits.
    
    Args:
        logits: (N, 4) logits from model
        targets: (N,) target action indices
        safe_mask: (N, 4) binary mask indicating safe moves
    
    Returns:
        accuracy: Accuracy on valid samples
        correct: Number of correct predictions
        valid_samples: Number of valid samples
    """
    # Mask unsafe moves
    logits_masked = logits.clone()
    logits_masked[safe_mask == 0] = -1e9
    
    # Get predictions (argmax over masked logits)
    preds = torch.argmax(logits_masked, dim=1)
    
    # Only count samples where target is safe
    N = logits.shape[0]
    batch_indices = torch.arange(N, device=logits.device)
    target_safe = safe_mask[batch_indices, targets] == 1
    
    if target_safe.sum() == 0:
        return 0.0, 0, 0
    
    correct = (preds[target_safe] == targets[target_safe]).sum().item()
    valid_samples = target_safe.sum().item()
    accuracy = correct / valid_samples if valid_samples > 0 else 0.0
    
    return accuracy, correct, valid_samples


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                device: torch.device) -> Tuple[float, float, float]:
    """Train for one epoch.
    
    Returns:
        avg_loss: Average loss
        accuracy: Accuracy on valid samples
        invalid_pct: Percentage of samples with invalid (unsafe) targets
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_valid = 0
    total_samples = 0
    num_batches = 0
    
    for batch in dataloader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        safe = batch['safe'].to(device)
        
        # Forward pass
        logits = model(x)
        
        # Compute loss
        loss, valid_samples, batch_total = masked_cross_entropy_loss(logits, y, safe)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_samples += batch_total
        
        # Compute accuracy
        acc, correct, valid = compute_accuracy(logits, y, safe)
        total_correct += correct
        total_valid += valid
        
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = total_correct / total_valid if total_valid > 0 else 0.0
    invalid_pct = (1.0 - total_valid / total_samples) * 100 if total_samples > 0 else 0.0
    
    return avg_loss, accuracy, invalid_pct


def validate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    """Validate model.
    
    Returns:
        avg_loss: Average loss
        accuracy: Accuracy on valid samples
        invalid_pct: Percentage of samples with invalid (unsafe) targets
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_valid = 0
    total_samples = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            safe = batch['safe'].to(device)
            
            # Forward pass
            logits = model(x)
            
            # Compute loss
            loss, valid_samples, batch_total = masked_cross_entropy_loss(logits, y, safe)
            
            # Accumulate metrics
            total_loss += loss.item()
            total_samples += batch_total
            
            # Compute accuracy
            acc, correct, valid = compute_accuracy(logits, y, safe)
            total_correct += correct
            total_valid += valid
            
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = total_correct / total_valid if total_valid > 0 else 0.0
    invalid_pct = (1.0 - total_valid / total_samples) * 100 if total_samples > 0 else 0.0
    
    return avg_loss, accuracy, invalid_pct


def main():
    parser = argparse.ArgumentParser(description='Train TronNet policy on episode data')
    parser.add_argument('--data', type=str, default='data/episodes',
                       help='Directory containing episode .npz files')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--bs', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio (0.0-1.0)')
    parser.add_argument('--save', type=str, default='model.pt',
                       help='Path to save best model')
    parser.add_argument('--no-filter-unsafe', action='store_true',
                       help='Do not filter out ticks with no safe moves')
    
    args = parser.parse_args()
    
    # Find all episode files
    data_dir = Path(args.data)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data}")
    
    episode_files = sorted(glob.glob(str(data_dir / "*.npz")))
    if not episode_files:
        raise ValueError(f"No .npz files found in {args.data}")
    
    # Split into train/val by file
    random.shuffle(episode_files)
    n_val = int(len(episode_files) * args.val_split)
    val_files = episode_files[:n_val]
    train_files = episode_files[n_val:]
    
    if not train_files:
        raise ValueError("No training files after split")
    
    # Create datasets
    train_dataset = EpisodeDataset(train_files, filter_unsafe=not args.no_filter_unsafe)
    val_dataset = EpisodeDataset(val_files, filter_unsafe=not args.no_filter_unsafe) if val_files else None
    
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=0) if val_dataset else None
    
    # Initialize model
    device = torch.device('cpu')  # CPU-only as per requirements
    model = TronNet().to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc, train_invalid = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        if val_loader:
            val_loss, val_acc, val_invalid = validate(model, val_loader, device)
            
            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
            
            # Print metrics
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Invalid: {train_invalid:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Invalid: {val_invalid:.2f}%")
        else:
            # No validation set, save based on train loss
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                best_model_state = model.state_dict().copy()
            
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Invalid: {train_invalid:.2f}%")
    
    # Save best model
    if best_model_state is not None:
        torch.save(best_model_state, args.save)
        print(f"\nSaved best model to {args.save}")
    else:
        # Fallback: save current model
        torch.save(model.state_dict(), args.save)
        print(f"\nSaved model to {args.save}")


if __name__ == '__main__':
    main()

