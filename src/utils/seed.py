"""Seed utilities for reproducibility."""

import random
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility.
    
    Sets seeds for:
    - Python random module
    - NumPy
    - PyTorch (CPU and CUDA)
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set random seed to {seed}")


def get_device(use_cuda: bool = True, device_id: int = 0) -> torch.device:
    """Get PyTorch device.
    
    Args:
        use_cuda: Whether to use CUDA if available
        device_id: CUDA device ID
        
    Returns:
        PyTorch device
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        logger.info(f"Using device: {device} ({torch.cuda.get_device_name(device_id)})")
    else:
        device = torch.device('cpu')
        logger.info("Using device: CPU")
    
    return device


