"""Optimizer and Learning Rate Scheduler creation with weight decay decoupling and warmup."""

import math
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR


def create_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01
) -> torch.optim.Optimizer:
    """
    Construct AdamW optimizer with decoupled weight decay for 1D biases and LayerNorms.
    Only parameters with requires_grad=True are included.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases and 1D parameters (e.g. LayerNorm scale/shift) do not undergo weight decay
        if param.ndim <= 1 or "bias" in name or "norm" in name or "ln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Filter out empty parameter groups
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g["params"]) > 0]

    if not optimizer_grouped_parameters:
        raise ValueError("No trainable parameters found in model to optimize!")

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    return optimizer


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.001
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_lr_ratio * lr, after a warmup period during which it increases linearly.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)

    return LambdaLR(optimizer, lr_lambda)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 0,
    min_lr: float = 1e-7,
    base_lr: float = 1e-4
) -> torch.optim.lr_scheduler.LRScheduler:
    """Instantiate Cosine schedule with linear warmup."""
    min_lr_ratio = min_lr / max(base_lr, 1e-12)
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=min_lr_ratio
    )
