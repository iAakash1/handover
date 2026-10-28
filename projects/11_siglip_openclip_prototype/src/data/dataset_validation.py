"""Dataset validation, corruption detection, integrity reporting, and distribution visualization."""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.utils import ensure_dir, save_json

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def discover_dataset_classes(dataset_root: Union[str, Path]) -> Tuple[List[str], Dict[str, List[Path]]]:
    """
    Discover class folders and collect all image file paths dynamically.
    Works if dataset contains subdirectories for classes, or nested folders.
    """
    root = Path(dataset_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {root}")

    # Discover direct subdirectories as classes
    class_dirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    # If no class subdirectories in root, check if there is an inner directory (e.g. data/raw/PlantVillage/PlantVillage)
    if not class_dirs:
        inner_dirs = [d for d in root.glob("*") if d.is_dir() and not d.name.startswith(".")]
        if inner_dirs:
            # Check inside first subdir
            nested_classes = [d for d in inner_dirs[0].iterdir() if d.is_dir() and not d.name.startswith(".")]
            if nested_classes:
                class_dirs = nested_classes

    if not class_dirs:
        raise ValueError(f"No class subdirectories found in dataset root: {root}")

    class_dirs.sort(key=lambda d: d.name)
    
    class_names = []
    class_to_files: Dict[str, List[Path]] = {}

    for c_dir in class_dirs:
        class_name = c_dir.name
        # Find all files with supported extensions
        images = [
            f for f in c_dir.glob("**/*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if images:
            class_names.append(class_name)
            class_to_files[class_name] = sorted(images)

    if not class_names:
        raise ValueError(f"No valid image files found under {root} with extensions {SUPPORTED_EXTENSIONS}")

    return class_names, class_to_files


def compute_file_hash(filepath: Path) -> str:
    """Compute MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_dataset(
    dataset_root: Union[str, Path],
    output_dir: Union[str, Path] = "outputs/dataset_reports",
    check_duplicates: bool = True,
    generate_sample_grid: bool = True
) -> Dict[str, Any]:
    """
    Perform rigorous dataset validation:
    - Verifies dataset structure and class discovery
    - Attempts to open and verify every single image
    - Identifies and records corrupted images
    - Calculates class distribution, sample counts, and imbalance ratio
    - Detects duplicate files
    - Saves JSON report, CSV distribution, and PNG bar chart & sample grid
    """
    out_dir = Path(output_dir).resolve()
    ensure_dir(out_dir)

    class_names, class_to_files = discover_dataset_classes(dataset_root)

    total_files = sum(len(files) for files in class_to_files.values())
    valid_files_by_class: Dict[str, List[Path]] = {c: [] for c in class_names}
    corrupted_files: List[Dict[str, str]] = []
    hash_to_files: Dict[str, List[str]] = {}

    print(f"Validating dataset at: {dataset_root}")
    print(f"Discovered {len(class_names)} classes with {total_files} total image files.")

    for class_name in class_names:
        files = class_to_files[class_name]
        for fpath in files:
            try:
                # Test opening and decoding image
                with Image.open(fpath) as img:
                    img.verify()
                # Also test actual RGB conversion
                with Image.open(fpath) as img:
                    _ = img.convert("RGB")
                valid_files_by_class[class_name].append(fpath)

                if check_duplicates:
                    fhash = compute_file_hash(fpath)
                    if fhash not in hash_to_files:
                        hash_to_files[fhash] = []
                    hash_to_files[fhash].append(str(fpath))

            except Exception as e:
                corrupted_files.append({
                    "path": str(fpath),
                    "class": class_name,
                    "error": str(e)
                })

    valid_count = sum(len(v) for v in valid_files_by_class.values())
    corrupted_count = len(corrupted_files)

    # Class distribution statistics
    class_counts = {c: len(valid_files_by_class[c]) for c in class_names}
    counts_array = np.array(list(class_counts.values()))
    
    min_count = int(counts_array.min()) if len(counts_array) > 0 else 0
    max_count = int(counts_array.max()) if len(counts_array) > 0 else 0
    mean_count = float(counts_array.mean()) if len(counts_array) > 0 else 0.0
    imbalance_ratio = float(max_count / min_count) if min_count > 0 else float("inf")

    # Duplicates detection
    duplicates = {h: paths for h, paths in hash_to_files.items() if len(paths) > 1}
    duplicate_count = sum(len(paths) - 1 for paths in duplicates.values())

    report = {
        "dataset_root": str(dataset_root),
        "total_files_discovered": total_files,
        "valid_images": valid_count,
        "corrupted_images": corrupted_count,
        "corrupted_file_list": corrupted_files,
        "num_classes": len(class_names),
        "classes": class_names,
        "class_counts": class_counts,
        "statistics": {
            "min_samples_per_class": min_count,
            "max_samples_per_class": max_count,
            "mean_samples_per_class": round(mean_count, 2),
            "imbalance_ratio": round(imbalance_ratio, 3),
        },
        "duplicates": {
            "duplicate_clusters": len(duplicates),
            "total_duplicate_copies": duplicate_count
        }
    }

    # Save JSON report
    report_path = out_dir / "validation_report.json"
    save_json(report, report_path)
    print(f"Saved validation report to: {report_path}")

    # Save CSV distribution
    df_dist = pd.DataFrame([
        {
            "class_name": c,
            "count": class_counts[c],
            "percentage": round(class_counts[c] / valid_count * 100, 2) if valid_count > 0 else 0
        }
        for c in class_names
    ])
    csv_path = out_dir / "class_distribution.csv"
    df_dist.to_csv(csv_path, index=False)
    print(f"Saved class distribution CSV to: {csv_path}")

    # Generate Class Distribution Bar Chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df_dist["class_name"], df_dist["count"], color="#2E86AB", edgecolor="#1B4965")
    plt.title("Tomato Disease Dataset - Class Sample Distribution", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Disease Class", fontsize=12, fontweight="bold")
    plt.ylabel("Number of Valid Images", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plot_path = out_dir / "class_distribution.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved class distribution plot to: {plot_path}")

    # Generate Sample Images Grid
    if generate_sample_grid and class_names:
        cols = min(4, len(class_names))
        rows = (len(class_names) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
        axes = np.array(axes).reshape(-1)

        for i, c in enumerate(class_names):
            ax = axes[i]
            if valid_files_by_class[c]:
                sample_img_path = valid_files_by_class[c][0]
                try:
                    img = Image.open(sample_img_path).convert("RGB")
                    ax.imshow(img)
                    ax.set_title(c.replace("_", " "), fontsize=10, fontweight="bold")
                except Exception:
                    ax.text(0.5, 0.5, "Image load error", ha="center", va="center")
            ax.axis("off")

        # Hide extra empty subplots
        for j in range(len(class_names), len(axes)):
            axes[j].axis("off")

        plt.suptitle("Sample Images per Tomato Class", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        sample_grid_path = out_dir / "sample_images.png"
        plt.savefig(sample_grid_path, dpi=200)
        plt.close()
        print(f"Saved sample image grid to: {sample_grid_path}")

    return report
