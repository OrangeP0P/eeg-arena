"""
Checkpoint utilities for saving and loading model states
"""
import os
import torch
from datetime import datetime


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    """
    Save training checkpoint

    Args:
        model: PyTorch model
        optimizer: optimizer
        scheduler: learning rate scheduler
        epoch: current epoch
        metrics: dict of metrics
        save_path: path to save checkpoint

    Returns:
        str: path to saved checkpoint
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
    }

    torch.save(checkpoint, save_path)
    return save_path


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device='cuda'):
    """
    Load training checkpoint

    Args:
        model: PyTorch model
        optimizer: optimizer (optional)
        scheduler: learning rate scheduler (optional)
        checkpoint_path: path to checkpoint
        device: device to load model to

    Returns:
        dict: checkpoint content
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


def save_best_model(model, save_dir, dataset_name, metrics, suffix=''):
    """
    Save best model with metrics in filename

    Args:
        model: PyTorch model
        save_dir: directory to save
        dataset_name: dataset name
        metrics: dict of metrics
        suffix: optional suffix

    Returns:
        str: path to saved model
    """
    os.makedirs(save_dir, exist_ok=True)

    # Build filename from metrics
    if 'kappa' in metrics:
        fname = f"{dataset_name}_kappa{metrics['kappa']:.4f}_epoch{metrics.get('epoch', '')}{suffix}.pth"
    elif 'acc' in metrics:
        fname = f"{dataset_name}_acc{metrics['acc']:.4f}_epoch{metrics.get('epoch', '')}{suffix}.pth"
    elif 'r2' in metrics:
        fname = f"{dataset_name}_r2{metrics['r2']:.4f}_epoch{metrics.get('epoch', '')}{suffix}.pth"
    else:
        fname = f"{dataset_name}_best{suffix}.pth"

    save_path = os.path.join(save_dir, fname)
    torch.save(model.state_dict(), save_path)
    return save_path


__all__ = ['save_checkpoint', 'load_checkpoint', 'save_best_model']
