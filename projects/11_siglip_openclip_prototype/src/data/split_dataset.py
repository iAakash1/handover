"""Deterministic stratified dataset splitting with permanent persistence and reuse."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.dataset_validation import discover_dataset_classes
from src.utils import ensure_dir


def create_or_load_splits(
    dataset_root: Union[str, Path],
    splits_dir: Union[str, Path] = "data/splits",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    force_regenerate: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load existing split CSVs or create deterministic stratified splits if they don't exist.
    
    Split ratios must sum to 1.0 (within numerical tolerance).
    Saves:
      - splits_dir/train.csv
      - splits_dir/val.csv
      - splits_dir/test.csv
    
    Each CSV contains columns: [image_path, label, class_name]
    """
    s_dir = Path(splits_dir).resolve()
    ensure_dir(s_dir)

    train_path = s_dir / "train.csv"
    val_path = s_dir / "val.csv"
    test_path = s_dir / "test.csv"

    # Check if splits already exist and reuse them unless force_regenerate is True
    if not force_regenerate and train_path.exists() and val_path.exists() and test_path.exists():
        print(f"Loading existing deterministic splits from: {s_dir}")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        return train_df, val_df, test_df

    # Validate split ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-4):
        raise ValueError(
            f"Split ratios must sum to 1.0! Got train={train_ratio}, val={val_ratio}, test={test_ratio} (sum={total_ratio})"
        )

    print(f"Generating new deterministic stratified splits (train={train_ratio}, val={val_ratio}, test={test_ratio}, seed={seed})")
    
    class_names, class_to_files = discover_dataset_classes(dataset_root)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    all_records = []
    for c_name in class_names:
        files = class_to_files[c_name]
        if len(files) < 3:
            raise ValueError(
                f"Class '{c_name}' has only {len(files)} samples, which is too few for stratified train/val/test splitting."
            )
        for fpath in files:
            # Store path as relative to workspace root if possible, else str(fpath)
            try:
                rel_path = fpath.relative_to(Path.cwd())
                path_str = str(rel_path).replace("\\", "/")
            except ValueError:
                path_str = str(fpath).replace("\\", "/")

            all_records.append({
                "image_path": path_str,
                "label": class_to_idx[c_name],
                "class_name": c_name
            })

    full_df = pd.DataFrame(all_records)

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        full_df,
        test_size=val_test_ratio,
        stratify=full_df["label"],
        random_state=seed,
        shuffle=True
    )

    # Second split: val vs test
    relative_test_ratio = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df["label"],
        random_state=seed,
        shuffle=True
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Verify no intersection between splits
    train_set = set(train_df["image_path"])
    val_set = set(val_df["image_path"])
    test_set = set(test_df["image_path"])

    assert len(train_set.intersection(val_set)) == 0, "Data leakage detected between train and val splits!"
    assert len(train_set.intersection(test_set)) == 0, "Data leakage detected between train and test splits!"
    assert len(val_set.intersection(test_set)) == 0, "Data leakage detected between val and test splits!"

    # Save to disk
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved stratified splits to {s_dir}:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(full_df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(full_df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(full_df)*100:.1f}%)")

    return train_df, val_df, test_df
