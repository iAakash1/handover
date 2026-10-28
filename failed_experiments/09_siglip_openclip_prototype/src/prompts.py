"""Disease-aware prompt bank loading, validation, sampling, and embedding aggregation."""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import yaml


class PromptBank:
    """Manages disease-aware prompts for tomato classes."""

    def __init__(self, prompts_file: Union[str, Path]):
        self.prompts_file = Path(prompts_file).resolve()
        if not self.prompts_file.exists():
            raise FileNotFoundError(f"Prompts file not found: {self.prompts_file}")

        with open(self.prompts_file, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Expected dictionary in prompts file, got {type(raw)}")

        # Normalize keys for robust matching
        self.prompts_by_class: Dict[str, List[str]] = {}
        for key, prompts in raw.items():
            if isinstance(prompts, list) and len(prompts) > 0:
                self.prompts_by_class[key] = [str(p).strip() for p in prompts]
            elif isinstance(prompts, str) and prompts.strip():
                self.prompts_by_class[key] = [prompts.strip()]

    def validate_classes(self, dataset_classes: List[str]) -> None:
        """Verify that every discovered class in the dataset has at least one valid prompt."""
        missing = []
        for c in dataset_classes:
            if not self.get_prompts(c):
                missing.append(c)
        if missing:
            raise ValueError(
                f"Missing prompt definitions for {len(missing)} dataset classes: {missing}. "
                f"Please add prompts for these classes in {self.prompts_file}."
            )

    def get_prompts(self, class_name: str) -> List[str]:
        """Retrieve prompts for a class with exact and case-insensitive fallback."""
        if class_name in self.prompts_by_class:
            return self.prompts_by_class[class_name]

        # Case-insensitive or normalized check
        lower_map = {k.lower(): v for k, v in self.prompts_by_class.items()}
        if class_name.lower() in lower_map:
            return lower_map[class_name.lower()]

        # Try stripped/standardized variants
        clean_name = class_name.replace(" ", "_").lower()
        for k, v in self.prompts_by_class.items():
            if k.replace(" ", "_").lower() == clean_name:
                return v

        return []

    def sample_prompt(self, class_name: str) -> str:
        """Sample a random positive prompt for training."""
        prompts = self.get_prompts(class_name)
        if not prompts:
            # Fallback generic prompt if somehow missing
            return f"a photo of tomato {class_name.replace('_', ' ')}"
        return random.choice(prompts)

    def build_class_text_embeddings(
        self,
        class_names: List[str],
        model: Any,
        device: torch.device
    ) -> torch.Tensor:
        """
        Encode all prompts for each class, aggregate prompt embeddings into a class-level
        normalized text representation.

        Returns:
            Tensor of shape (num_classes, embedding_dim), L2-normalized.
        """
        model.eval()
        class_embeddings = []

        with torch.no_grad():
            for c in class_names:
                prompts = self.get_prompts(c)
                if not prompts:
                    prompts = [f"a photo of tomato {c.replace('_', ' ')}"]

                # Encode all prompts for this class
                prompt_embs = model.encode_text(prompts, device=device)
                prompt_embs = model.normalize_embeddings(prompt_embs)

                # Mean aggregation over prompts
                class_emb = prompt_embs.mean(dim=0, keepdim=True)
                # Re-normalize class text representation
                class_emb = model.normalize_embeddings(class_emb)
                class_embeddings.append(class_emb)

        # Concatenate into (num_classes, embedding_dim)
        return torch.cat(class_embeddings, dim=0)


def load_prompt_bank(prompts_file: Union[str, Path]) -> PromptBank:
    """Helper to instantiate PromptBank."""
    return PromptBank(prompts_file)
