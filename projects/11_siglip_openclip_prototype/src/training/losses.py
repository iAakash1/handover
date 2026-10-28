"""Supervised Multi-Positive Contrastive Learning Loss for Vision-Language Domain Adaptation."""

from typing import Any, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ConfigDict


class MultiPositiveContrastiveLoss(nn.Module):
    r"""
    Supervised Multi-Positive Multimodal Contrastive Loss.
    
    Mathematical Formulation:
    --------------------------
    Given a batch of B samples with normalized image embeddings z_i^I, normalized text embeddings z_i^T,
    and class labels y_i in {0, ..., C-1}:
    
    For anchor i, the set of positive keys is:
        P(i) = { j in {1, ..., B} : y_j = y_i }
        
    The cross-modal similarity matrix is:
        S_{i, j} = <z_i^I, z_j^T> / tau
        
    Image-to-Text Loss:
        L_i^{I -> T} = - 1 / |P(i)| * sum_{p in P(i)} [ S_{i, p} - logsumexp_a(S_{i, a}) ]
        
    Text-to-Image Loss:
        L_j^{T -> I} = - 1 / |P(j)| * sum_{p in P(j)} [ S_{p, j} - logsumexp_a(S_{a, j}) ]
        
    Total Symmetric Loss:
        L = 1 / (2 * B) * sum_{i=1}^B ( L_i^{I -> T} + L_i^{T -> I} )
    """

    def __init__(self, temperature: float = 0.07, eps: float = 1e-12):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-positive supervised contrastive loss.
        
        Args:
            image_embeddings: Tensor of shape (B, D), L2-normalized
            text_embeddings: Tensor of shape (B, D), L2-normalized
            labels: Tensor of shape (B,) containing class indices
            
        Returns:
            Scalar loss tensor
        """
        device = image_embeddings.device
        batch_size = image_embeddings.shape[0]

        if batch_size == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Normalize embeddings to guarantee unit hypersphere projection
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1, eps=self.eps)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1, eps=self.eps)

        # Compute cosine similarity matrix scaled by temperature: shape (B, B)
        # sim_matrix[i, j] = <image_i, text_j> / tau
        sim_matrix = torch.matmul(image_embeddings, text_embeddings.T) / self.temperature

        # Construct positive label mask: shape (B, B)
        # mask[i, j] = 1 if labels[i] == labels[j] else 0
        labels_view = labels.view(-1, 1)
        pos_mask = torch.eq(labels_view, labels_view.T).float().to(device)

        # Number of positives per row and column
        pos_counts_i2t = pos_mask.sum(dim=1).clamp(min=1.0)
        pos_counts_t2i = pos_mask.sum(dim=0).clamp(min=1.0)

        # 1. Image-to-Text Direction (row-wise log-softmax)
        # log_prob_i2t[i, j] = S_{i, j} - logsumexp_k(S_{i, k})
        log_prob_i2t = F.log_softmax(sim_matrix, dim=1)
        # Sum over positive pairs and normalize by |P(i)|
        loss_i2t = -(pos_mask * log_prob_i2t).sum(dim=1) / pos_counts_i2t

        # 2. Text-to-Image Direction (column-wise log-softmax)
        # log_prob_t2i[i, j] = S_{i, j} - logsumexp_k(S_{k, j})
        log_prob_t2i = F.log_softmax(sim_matrix.T, dim=1)
        # Sum over positive pairs and normalize by |P(j)|
        loss_t2i = -(pos_mask.T * log_prob_t2i).sum(dim=1) / pos_counts_t2i

        # Total symmetric loss
        total_loss = (loss_i2t.mean() + loss_t2i.mean()) / 2.0
        return total_loss


def create_loss(config: Union[ConfigDict, Dict[str, Any]]) -> nn.Module:
    """Create loss function based on config."""
    loss_cfg = config.get("loss", {}) if isinstance(config, dict) else getattr(config, "loss", {})
    if isinstance(loss_cfg, dict):
        temp = float(loss_cfg.get("temperature", 0.07))
    else:
        temp = float(getattr(loss_cfg, "temperature", 0.07))

    return MultiPositiveContrastiveLoss(temperature=temp)
