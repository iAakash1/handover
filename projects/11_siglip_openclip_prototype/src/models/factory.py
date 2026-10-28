"""Model factory for instantiating pluggable Vision-Language Model backbones."""

from typing import Any, Dict, Optional, Union
import torch

from src.config import ConfigDict
from src.models.base import BaseVLM
from src.models.openclip_model import OpenCLIPModelWrapper
from src.models.siglip_model import SigLIPModelWrapper


def create_model(config: Union[ConfigDict, Dict[str, Any]], device: Optional[torch.device] = None) -> BaseVLM:
    """
    Factory function to instantiate the configured VLM backbone.
    
    Supported backbones in config.model.framework:
    - "open_clip"
    - "siglip"
    
    Raises ValueError with actionable error message for unsupported frameworks.
    """
    model_cfg = config.get("model", {}) if isinstance(config, dict) else getattr(config, "model", {})
    if isinstance(model_cfg, dict):
        framework = model_cfg.get("framework", "").lower().strip()
    else:
        framework = getattr(model_cfg, "framework", "").lower().strip()

    if not framework:
        raise ValueError("Model framework must be specified in config ('model.framework'). Supported: ['open_clip', 'siglip']")

    if framework == "open_clip":
        arch = model_cfg.get("architecture", "ViT-B-16") if isinstance(model_cfg, dict) else getattr(model_cfg, "architecture", "ViT-B-16")
        pretrained = model_cfg.get("pretrained", "openai") if isinstance(model_cfg, dict) else getattr(model_cfg, "pretrained", "openai")
        print(f"[ModelFactory] Creating OpenCLIP model: architecture='{arch}', pretrained='{pretrained}'")
        return OpenCLIPModelWrapper(architecture=arch, pretrained=pretrained, device=device)

    elif framework == "siglip":
        name = model_cfg.get("name", "google/siglip-base-patch16-224") if isinstance(model_cfg, dict) else getattr(model_cfg, "name", "google/siglip-base-patch16-224")
        print(f"[ModelFactory] Creating SigLIP model: name='{name}'")
        return SigLIPModelWrapper(model_name=name, device=device)

    else:
        raise ValueError(
            f"Unsupported VLM backbone framework: '{framework}'. "
            f"Supported frameworks are: ['open_clip', 'siglip']."
        )
