"""Abstract Base Class for Dual-Encoder Vision-Language Models."""

import abc
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import count_parameters


class BaseVLM(nn.Module, abc.ABC):
    """
    Common abstraction for Vision-Language Models (OpenCLIP, SigLIP, etc.).
    The rest of the training, inference, and evaluation pipeline interacts strictly
    via this common interface.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def encode_image(self, images: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """
        Extract image embeddings from image tensor.
        Args:
            images: Tensor of shape (batch_size, 3, height, width)
            normalize: If True, returns L2-normalized embeddings
        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        pass

    @abc.abstractmethod
    def encode_text(
        self,
        texts: Union[str, List[str]],
        device: Optional[torch.device] = None,
        normalize: bool = False
    ) -> torch.Tensor:
        """
        Tokenize and extract text embeddings from string or list of strings.
        Args:
            texts: A single string or list of strings
            device: Target device for token tensors
            normalize: If True, returns L2-normalized embeddings
        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        pass

    def normalize_embeddings(self, embeddings: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Apply L2 normalization across the embedding dimension: v / ||v||_2.
        Args:
            embeddings: Tensor of shape (..., embedding_dim)
            eps: Small epsilon for numerical stability
        Returns:
            L2 normalized tensor of the same shape
        """
        return F.normalize(embeddings, p=2, dim=-1, eps=eps)

    @abc.abstractmethod
    def get_embedding_dimension(self) -> int:
        """Return embedding dimension of the shared multimodal space."""
        pass

    @abc.abstractmethod
    def configure_trainable_layers(
        self,
        image_unfreeze_layers: int = 0,
        text_unfreeze_layers: int = 0
    ) -> Dict[str, Any]:
        """
        Configure layer freezing and partial fine-tuning.
        - 0 means entire encoder is frozen
        - N > 0 means unfreeze the final N transformer blocks / layers (+ projection head)
        - -1 means unfreeze all layers in that modality encoder
        """
        pass

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return all parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_parameter_summary(self) -> Dict[str, Union[int, float]]:
        """Return parameter counts and trainable percentage."""
        return count_parameters(self)

    @abc.abstractmethod
    def get_image_preprocess(self) -> Optional[Callable]:
        """Return default image preprocessing transform for the model."""
        pass
