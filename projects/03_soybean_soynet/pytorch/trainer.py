"""
SoyNet Trainer Module
======================
PyTorch training loop with mixed precision, early stopping, and LR scheduling.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

from . import config


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


class Trainer:
    """PyTorch trainer with mixed precision and early stopping."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor,
        device: torch.device,
        model_name: str,
        output_dir: Path
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function with class weights
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Mixed precision scaler
        self.scaler = GradScaler()
        
        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "learning_rate": [],
            "epoch_time": []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_path = self.output_dir / "model.pth"
        
    def _create_optimizer(self, lr: float) -> torch.optim.Optimizer:
        """Create AdamW optimizer."""
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=config.WEIGHT_DECAY
        )
    
    def _train_epoch(self, optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.float().to(self.device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            
            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    def _validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.float().to(self.device)
                
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    def _save_checkpoint(self, val_loss: float):
        """Save model if validation loss improved."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'val_loss': val_loss,
                'model_name': self.model_name
            }, self.best_model_path)
            return True
        return False
    
    def train(self, epochs: int, lr: float, phase_name: str = "Training") -> None:
        """
        Run training loop.
        
        Args:
            epochs: Number of epochs
            lr: Learning rate
            phase_name: Name for logging (e.g., "Phase 1: Frozen" or "Phase 2: Fine-tuning")
        """
        print(f"\n{'='*60}")
        print(f"{phase_name} - {self.model_name}")
        print(f"{'='*60}")
        
        optimizer = self._create_optimizer(lr)
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
            verbose=True
        )
        early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, mode='min')
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self._train_epoch(optimizer)
            
            # Validate
            val_loss, val_acc = self._validate()
            
            # Scheduler step
            scheduler.step(val_loss)
            
            # Get current LR
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record history
            epoch_time = time.time() - epoch_start
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_accuracy"].append(val_acc)
            self.history["learning_rate"].append(current_lr)
            self.history["epoch_time"].append(epoch_time)
            
            # Save best model
            saved = self._save_checkpoint(val_loss)
            save_str = " [SAVED]" if saved else ""
            
            # Log progress
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s{save_str}")
            
            # Early stopping
            if early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Save history
        self._save_history()
    
    def _save_history(self):
        """Save training history to JSON."""
        history_path = self.output_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved to: {history_path}")
    
    def load_best_model(self):
        """Load the best saved model."""
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from: {self.best_model_path}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: List[float],
    model_name: str,
    device: torch.device
) -> Tuple[nn.Module, Dict]:
    """
    Full training pipeline with two phases:
    1. Frozen backbone training
    2. Fine-tuning
    
    Returns:
        Trained model and training history
    """
    output_dir = config.get_model_dir(model_name)
    
    # Convert class weights to tensor
    weights_tensor = torch.FloatTensor(class_weights)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=weights_tensor,
        device=device,
        model_name=model_name,
        output_dir=output_dir
    )
    
    # Phase 1: Train with frozen backbone
    model.freeze_backbone()
    trainer.train(
        epochs=config.FREEZE_EPOCHS,
        lr=config.INITIAL_LR,
        phase_name="Phase 1: Frozen Backbone"
    )
    
    # Phase 2: Fine-tune
    model.unfreeze_backbone()
    trainer.train(
        epochs=config.FINE_TUNE_EPOCHS,
        lr=config.FINE_TUNE_LR,
        phase_name="Phase 2: Fine-tuning"
    )
    
    # Load best model
    trainer.load_best_model()
    
    return model, trainer.history
