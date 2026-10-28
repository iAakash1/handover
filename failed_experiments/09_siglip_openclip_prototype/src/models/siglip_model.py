"""SigLIP Dual-Encoder Vision-Language Model Wrapper using Hugging Face Transformers."""

from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.nn as nn
from transformers import AutoProcessor, SiglipModel

from src.models.base import BaseVLM
from src.utils import count_parameters


class SigLIPModelWrapper(BaseVLM):
    """
    SigLIP wrapper providing unified BaseVLM interface and partial layer unfreezing.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.model_name = model_name

        try:
            self.model = SiglipModel.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load SigLIP model '{model_name}': {e}") from e

        if device is not None:
            self.model = self.model.to(device)

        # Detect embedding dimension
        if hasattr(self.model.config, "projection_dim") and self.model.config.projection_dim:
            self.embedding_dim = self.model.config.projection_dim
        elif hasattr(self.model.config, "hidden_size"):
            self.embedding_dim = self.model.config.hidden_size
        else:
            self.embedding_dim = 768

    def _extract_tensor(self, output: Any) -> torch.Tensor:
        """Helper to extract tensor from HuggingFace BaseModelOutputWithPooling or raw tensor."""
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            return output.last_hidden_state[:, 0, :]
        if isinstance(output, (list, tuple)):
            return output[0]
        if isinstance(output, torch.Tensor):
            return output
        raise TypeError(f"Cannot extract tensor from output of type {type(output)}")

    def encode_image(self, images: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """Encode image tensor of shape (batch, 3, H, W)."""
        raw_out = self.model.get_image_features(pixel_values=images)
        embs = self._extract_tensor(raw_out)
        if normalize:
            embs = self.normalize_embeddings(embs)
        return embs

    def encode_text(
        self,
        texts: Union[str, List[str]],
        device: Optional[torch.device] = None,
        normalize: bool = False
    ) -> torch.Tensor:
        """Tokenize and encode text string or list of text strings."""
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(text=texts, padding="max_length", max_length=64, return_tensors="pt")
        
        target_device = device
        if target_device is None and next(self.parameters(), None) is not None:
            target_device = next(self.parameters()).device

        if target_device is not None:
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

        raw_out = self.model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None)
        )
        embs = self._extract_tensor(raw_out)
        if normalize:
            embs = self.normalize_embeddings(embs)
        return embs

    def get_embedding_dimension(self) -> int:
        return self.embedding_dim

    def get_image_preprocess(self) -> Optional[Callable]:
        return None

    def configure_trainable_layers(
        self,
        image_unfreeze_layers: int = 0,
        text_unfreeze_layers: int = 0
    ) -> Dict[str, Any]:
        """
        Configure partial fine-tuning:
        - Freezes all parameters by default.
        - Unfreezes last N image transformer layers and vision head.
        - Unfreezes last N text transformer layers and text head.
        """
        # Step 1: Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # Step 2: Configure Vision Model
        if image_unfreeze_layers == -1:
            for p in self.model.vision_model.parameters():
                p.requires_grad = True
        elif image_unfreeze_layers > 0:
            vision_model = self.model.vision_model
            # Unfreeze head and post layernorm
            if hasattr(vision_model, "head") and vision_model.head is not None:
                for p in vision_model.head.parameters():
                    p.requires_grad = True

            if hasattr(vision_model, "post_layernorm") and vision_model.post_layernorm is not None:
                for p in vision_model.post_layernorm.parameters():
                    p.requires_grad = True

            # Unfreeze last N encoder layers
            if hasattr(vision_model, "encoder") and hasattr(vision_model.encoder, "layers"):
                layers = vision_model.encoder.layers
                num_layers = len(layers)
                n = min(image_unfreeze_layers, num_layers)
                for layer in layers[num_layers - n:]:
                    for p in layer.parameters():
                        p.requires_grad = True

        # Step 3: Configure Text Model
        if text_unfreeze_layers == -1:
            for p in self.model.text_model.parameters():
                p.requires_grad = True
        elif text_unfreeze_layers > 0:
            text_model = self.model.text_model
            # Unfreeze head and final layer norm
            if hasattr(text_model, "head") and text_model.head is not None:
                for p in text_model.head.parameters():
                    p.requires_grad = True

            if hasattr(text_model, "final_layer_norm") and text_model.final_layer_norm is not None:
                for p in text_model.final_layer_norm.parameters():
                    p.requires_grad = True

            # Unfreeze last N encoder layers
            if hasattr(text_model, "encoder") and hasattr(text_model.encoder, "layers"):
                layers = text_model.encoder.layers
                num_layers = len(layers)
                n = min(text_unfreeze_layers, num_layers)
                for layer in layers[num_layers - n:]:
                    for p in layer.parameters():
                        p.requires_grad = True

        return count_parameters(self)
