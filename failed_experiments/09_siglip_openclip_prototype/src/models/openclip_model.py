"""OpenCLIP Dual-Encoder Vision-Language Model Wrapper."""

from typing import Any, Callable, Dict, List, Optional, Union
import open_clip
import torch
import torch.nn as nn

from src.models.base import BaseVLM
from src.utils import count_parameters


class OpenCLIPModelWrapper(BaseVLM):
    """
    OpenCLIP wrapper providing unified BaseVLM interface and partial layer unfreezing.
    """

    def __init__(
        self,
        architecture: str = "ViT-B-16",
        pretrained: str = "openai",
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.architecture = architecture
        self.pretrained = pretrained

        # Load OpenCLIP model, preprocess transforms, and tokenizer
        try:
            model, train_preprocess, val_preprocess = open_clip.create_model_and_transforms(
                architecture,
                pretrained=pretrained,
                device=device or "cpu"
            )
            tokenizer = open_clip.get_tokenizer(architecture)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create OpenCLIP model with architecture='{architecture}', pretrained='{pretrained}': {e}"
            ) from e

        self.model = model
        self.train_preprocess = train_preprocess
        self.val_preprocess = val_preprocess
        self.tokenizer = tokenizer

        # Determine embedding dimension
        if hasattr(self.model, "visual") and hasattr(self.model.visual, "output_dim"):
            self.embedding_dim = self.model.visual.output_dim
        elif hasattr(self.model, "text_projection") and self.model.text_projection is not None:
            self.embedding_dim = self.model.text_projection.shape[-1]
        else:
            # Fallback probe
            dummy_img = torch.zeros(1, 3, 224, 224)
            with torch.no_grad():
                out = self.model.encode_image(dummy_img)
                self.embedding_dim = out.shape[-1]

    def encode_image(self, images: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """Encode image tensor of shape (batch, 3, H, W)."""
        embs = self.model.encode_image(images)
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

        tokens = self.tokenizer(texts)
        if device is not None:
            tokens = tokens.to(device)
        elif next(self.parameters(), None) is not None:
            tokens = tokens.to(next(self.parameters()).device)

        embs = self.model.encode_text(tokens)
        if normalize:
            embs = self.normalize_embeddings(embs)
        return embs

    def get_embedding_dimension(self) -> int:
        return self.embedding_dim

    def get_image_preprocess(self) -> Optional[Callable]:
        return self.val_preprocess

    def configure_trainable_layers(
        self,
        image_unfreeze_layers: int = 0,
        text_unfreeze_layers: int = 0
    ) -> Dict[str, Any]:
        """
        Configure partial fine-tuning:
        - Freezes all parameters by default.
        - Unfreezes last N image transformer blocks and projection head.
        - Unfreezes last N text transformer blocks and projection head.
        """
        # Step 1: Freeze all parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # Step 2: Configure Image Encoder
        if image_unfreeze_layers == -1:
            # Unfreeze entire visual encoder
            for p in self.model.visual.parameters():
                p.requires_grad = True
        elif image_unfreeze_layers > 0:
            # Always unfreeze visual projections and final layer norm
            visual = self.model.visual
            if hasattr(visual, "proj") and visual.proj is not None:
                if isinstance(visual.proj, nn.Parameter):
                    visual.proj.requires_grad = True
                elif isinstance(visual.proj, nn.Module):
                    for p in visual.proj.parameters():
                        p.requires_grad = True

            if hasattr(visual, "ln_post") and visual.ln_post is not None:
                for p in visual.ln_post.parameters():
                    p.requires_grad = True

            if hasattr(visual, "head") and visual.head is not None:
                for p in visual.head.parameters():
                    p.requires_grad = True

            # Unfreeze the last N visual transformer blocks
            if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
                blocks = visual.transformer.resblocks
                num_blocks = len(blocks)
                n = min(image_unfreeze_layers, num_blocks)
                for block in blocks[num_blocks - n:]:
                    for p in block.parameters():
                        p.requires_grad = True
            elif hasattr(visual, "trunk") and hasattr(visual.trunk, "blocks"):
                blocks = visual.trunk.blocks
                num_blocks = len(blocks)
                n = min(image_unfreeze_layers, num_blocks)
                for block in blocks[num_blocks - n:]:
                    for p in block.parameters():
                        p.requires_grad = True

        # Step 3: Configure Text Encoder
        if text_unfreeze_layers == -1:
            # Unfreeze entire text model / transformer
            if hasattr(self.model, "transformer"):
                for p in self.model.transformer.parameters():
                    p.requires_grad = True
            if hasattr(self.model, "text_projection") and self.model.text_projection is not None:
                self.model.text_projection.requires_grad = True
            if hasattr(self.model, "ln_final") and self.model.ln_final is not None:
                for p in self.model.ln_final.parameters():
                    p.requires_grad = True
        elif text_unfreeze_layers > 0:
            # Always unfreeze text projection and final normalization
            if hasattr(self.model, "text_projection") and self.model.text_projection is not None:
                self.model.text_projection.requires_grad = True

            if hasattr(self.model, "ln_final") and self.model.ln_final is not None:
                for p in self.model.ln_final.parameters():
                    p.requires_grad = True

            # Unfreeze last N text transformer blocks
            if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "resblocks"):
                blocks = self.model.transformer.resblocks
                num_blocks = len(blocks)
                n = min(text_unfreeze_layers, num_blocks)
                for block in blocks[num_blocks - n:]:
                    for p in block.parameters():
                        p.requires_grad = True

        return count_parameters(self)
