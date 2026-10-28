"""Multimodal PyTorch Dataset for Tomato Disease Images and Text Prompts."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.prompts import PromptBank


class TomatoMultimodalDataset(Dataset):
    """
    Multimodal Dataset returning:
    - Transformed Image Tensor
    - Integer Class Label
    - Class Name String
    - Disease-Aware Prompt String (randomly sampled positive prompt during training)
    - Image Filepath
    """

    def __init__(
        self,
        data_source: Union[str, Path, pd.DataFrame],
        transform: Optional[Callable] = None,
        prompt_bank: Optional[PromptBank] = None,
        is_train: bool = False
    ):
        super().__init__()
        if isinstance(data_source, (str, Path)):
            path = Path(data_source).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Split CSV file not found: {path}")
            self.df = pd.read_csv(path)
        elif isinstance(data_source, pd.DataFrame):
            self.df = data_source.copy()
        else:
            raise TypeError(f"data_source must be Path, str, or pd.DataFrame, got {type(data_source)}")

        # Verify required columns
        for col in ["image_path", "label", "class_name"]:
            if col not in self.df.columns:
                raise ValueError(f"Split data is missing required column: '{col}'")

        self.transform = transform
        self.prompt_bank = prompt_bank
        self.is_train = is_train

        # Pre-extract lists for fast indexed access
        self.image_paths = self.df["image_path"].tolist()
        self.labels = self.df["label"].tolist()
        self.class_names = self.df["class_name"].tolist()

        # Build unique class names list preserved in label order
        unique_classes = []
        for c in sorted(list(set(self.class_names))):
            unique_classes.append(c)
        self.classes = unique_classes

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        class_name = self.class_names[idx]

        # Load image safely
        full_path = Path(img_path)
        if not full_path.is_absolute():
            full_path = (Path.cwd() / full_path).resolve()

        try:
            with Image.open(full_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image at {full_path}: {e}") from e

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            # Fallback PIL to tensor if no transform
            from torchvision.transforms import functional as TF
            image_tensor = TF.to_tensor(image)

        # Get prompt
        if self.prompt_bank is not None:
            if self.is_train:
                prompt = self.prompt_bank.sample_prompt(class_name)
            else:
                # Default prompt for validation/test single prompt queries
                prompts = self.prompt_bank.get_prompts(class_name)
                prompt = prompts[0] if prompts else f"a photo of tomato {class_name}"
        else:
            prompt = f"a photo of tomato {class_name}"

        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "class_name": class_name,
            "prompt": prompt,
            "image_path": str(img_path)
        }


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom batch collate function."""
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    class_names = [item["class_name"] for item in batch]
    prompts = [item["prompt"] for item in batch]
    image_paths = [item["image_path"] for item in batch]

    return {
        "images": images,
        "labels": labels,
        "class_names": class_names,
        "prompts": prompts,
        "image_paths": image_paths
    }
